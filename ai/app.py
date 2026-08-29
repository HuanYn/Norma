from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ai.config import load_settings
from ai.evaluation import EvaluationService
from ai.index import AlbumIndexer
from ai.index.embedding import (
    EmbeddingProvider,
    EmbeddingProviderUnavailableError,
    create_embedding_provider,
    embedding_provider_capabilities,
)
from ai.jobs import PrepareJobManager
from ai.library import AlbumCatalogService
from ai.maintenance import CacheMaintenanceService
from ai.people import (
    FaceProviderUnavailableError,
    PeopleIndexer,
    canonical_face_provider_name,
    create_face_provider,
)
from ai.preferences import PreferenceService, PreferenceSuggestionAlreadyConsumedError
from ai.preferences.suggestion_service import (
    PreferenceSuggestionConflictError,
    PreferenceSuggestionNumericalError,
    PreferenceSuggestionService,
)
from ai.provider_runtime import EmbeddingWarmupManager
from ai.rag import (
    CitationValidationError,
    EvidenceIntegrityError,
    NoEvidenceError,
    ProviderFailureError,
    VLMInputBudgetError,
)
from ai.rag.providers import GroundedGenerationProvider
from ai.rag.service import (
    GroundedRAGService,
    RAGBusyError,
    RAGDriftError,
    RAGEvidenceTooLargeError,
)
from ai.rag.security import redact_local_paths
from ai.rag.transformers_runtime import (
    LocalVLMUnavailableError,
    create_local_qwen3vl_provider,
)
from ai.retrieval import RetrievalService
from ai.selection import ReplacementService, SelectionService
from ai.schemas import (
    AlbumEmbeddingResponse,
    AlbumIndexRequest,
    AlbumIndexResponse,
    AlbumListResponse,
    AlbumPhotoListResponse,
    AlbumRAGRequest,
    AlbumRAGResponse,
    AlbumSearchRequest,
    AlbumSearchResponse,
    AlbumSummary,
    CapabilitiesResponse,
    CacheGcRequest,
    CacheGcResponse,
    CacheQuotaRequest,
    CacheQuotaResponse,
    CacheUsageResponse,
    EmbeddingProviderListResponse,
    EmbeddingProviderStatusResponse,
    EvaluationCandidateResponse,
    EvaluationQueryCreateRequest,
    EvaluationQueryListResponse,
    EvaluationQuerySummary,
    EvaluationRunRequest,
    EvaluationRunResponse,
    HealthResponse,
    JobListResponse,
    JobResponse,
    MaintenanceRunListResponse,
    PeopleIndexResponse,
    PairwiseFeedbackRequest,
    PreferenceModelResponse,
    PreferencePairSuggestionRequest,
    PreferencePairSuggestionResponse,
    PreferenceStateResponse,
    PrepareJobRequest,
    RelevanceJudgmentBatchRequest,
    RelevanceJudgmentBatchResponse,
    SelectionRequest,
    SelectionHistoryResponse,
    SelectionReplacementRequest,
    SelectionReplacementResponse,
    SelectionResponse,
)
from ai.storage import Database


settings = load_settings()
database = Database(settings.database_path)
prepare_jobs: PrepareJobManager | None = None
embedding_warmup: EmbeddingWarmupManager | None = None

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("norma.ai")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global embedding_warmup, prepare_jobs
    database.initialize()
    prepare_jobs = PrepareJobManager(
        database,
        settings.data_dir,
        settings.embedding_provider,
        settings.face_provider,
        settings.embedding_device,
        settings.embedding_batch_size,
        settings.model_cache_dir,
    )
    prepare_jobs.start()
    embedding_warmup = EmbeddingWarmupManager(embedding_provider)
    if settings.prewarm_embedding:
        embedding_warmup.submit()
    logger.info("Norma AI worker ready; data_dir=%s", settings.data_dir)
    try:
        yield
    finally:
        prepare_jobs.shutdown()
        prepare_jobs = None
        embedding_warmup = None


app = FastAPI(
    title="Norma AI Worker",
    version="0.1.0",
    description="Local domain API for multimodal photo understanding and selection.",
    lifespan=lifespan,
)

_MEDIA_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")
_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})


def _derived_jpeg(*relative_parts: str) -> FileResponse:
    """Serve one generated JPEG without exposing the rest of ``data_dir``."""

    if not all(_MEDIA_COMPONENT_PATTERN.fullmatch(part) for part in relative_parts):
        raise HTTPException(status_code=404, detail="media not found")
    if Path(relative_parts[-1]).suffix.casefold() not in _JPEG_SUFFIXES:
        raise HTTPException(status_code=404, detail="media not found")

    data_root = settings.data_dir.resolve()
    allowed_root = data_root.joinpath(*relative_parts[:-1]).resolve()
    try:
        allowed_root.relative_to(data_root)
        candidate = allowed_root.joinpath(relative_parts[-1]).resolve(strict=True)
        candidate.relative_to(allowed_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=404, detail="media not found") from None

    if not candidate.is_file() or candidate.suffix.casefold() not in _JPEG_SUFFIXES:
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(
        candidate,
        media_type="image/jpeg",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@app.get(
    "/media/thumbnails/{album_id}/{filename}",
    response_class=FileResponse,
    include_in_schema=False,
)
def album_thumbnail(album_id: str, filename: str) -> FileResponse:
    return _derived_jpeg("thumbnails", album_id, filename)


@app.get(
    "/media/faces/{provider}/{album_id}/thumbnails/{filename}",
    response_class=FileResponse,
    include_in_schema=False,
)
def face_thumbnail(
    provider: str,
    album_id: str,
    filename: str,
) -> FileResponse:
    return _derived_jpeg("faces", provider, album_id, "thumbnails", filename)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        schema_version=database.current_version(),
        embedding_provider=embedding_provider().name,
        face_provider=canonical_face_provider_name(settings.face_provider),
    )


@app.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        embedding_provider=embedding_provider().name,
        milestones={
            "library": "cpu-fallback-indexer",
            "multimodal_index": embedding_provider().name,
            "people": "opencv-yunet-sface-constrained-prototype-clustering-v2",
            "selection": "contextual-utility+structured-cp-sat-or-greedy-v1",
            "preference": "bayesian-contextual-laplace-runtime-v1+legacy-logistic-v1",
            "library_lifecycle": "persistent-catalog-and-jobs-v1",
            "retrieval_evaluation": "human-relevance-metrics-v1",
            "cache_maintenance": "audited-quota-gc-v2",
            "provider_warmup": "background-idempotent-v1",
            "rag": "learned-openclip-retrieval+local-qwen3vl+citation-enforcement-v1",
            "video": "deferred",
            "world": "deferred",
        },
    )


def catalog_service() -> AlbumCatalogService:
    return AlbumCatalogService(database)


def embedding_provider() -> EmbeddingProvider:
    return create_embedding_provider(
        settings.embedding_provider,
        cache_dir=settings.model_cache_dir,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )


@app.get("/providers/embedding", response_model=EmbeddingProviderListResponse)
def list_embedding_providers() -> EmbeddingProviderListResponse:
    return EmbeddingProviderListResponse(
        items=embedding_provider_capabilities(
            settings.embedding_provider, settings.embedding_device
        )
    )


@app.get(
    "/providers/embedding/status",
    response_model=EmbeddingProviderStatusResponse,
)
def get_embedding_provider_status() -> EmbeddingProviderStatusResponse:
    return embedding_warmup_manager().status()


@app.post(
    "/providers/embedding/warmup",
    response_model=EmbeddingProviderStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def warmup_embedding_provider() -> EmbeddingProviderStatusResponse:
    return embedding_warmup_manager().submit()


def embedding_warmup_manager() -> EmbeddingWarmupManager:
    if embedding_warmup is None:
        raise RuntimeError("embedding warmup manager is not running")
    return embedding_warmup


def cache_maintenance_service() -> CacheMaintenanceService:
    return CacheMaintenanceService(
        database,
        settings.data_dir,
        model_cache_dir=settings.model_cache_dir,
        budget_bytes=settings.cache_budget_bytes,
    )


@app.post("/maintenance/cache/gc", response_model=CacheGcResponse)
def collect_generated_cache(request: CacheGcRequest) -> CacheGcResponse:
    try:
        return cache_maintenance_service().collect(request)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/maintenance/cache/usage", response_model=CacheUsageResponse)
def get_cache_usage() -> CacheUsageResponse:
    return cache_maintenance_service().usage()


@app.post("/maintenance/cache/enforce", response_model=CacheQuotaResponse)
def enforce_cache_quota(request: CacheQuotaRequest) -> CacheQuotaResponse:
    try:
        return cache_maintenance_service().enforce_quota(request)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/maintenance/runs", response_model=MaintenanceRunListResponse)
def list_maintenance_runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> MaintenanceRunListResponse:
    return cache_maintenance_service().list_runs(limit=limit, offset=offset)


def prepare_job_manager() -> PrepareJobManager:
    if prepare_jobs is None:
        raise RuntimeError("prepare job manager is not running")
    return prepare_jobs


@app.get("/albums", response_model=AlbumListResponse)
def list_albums(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> AlbumListResponse:
    return catalog_service().list_albums(limit=limit, offset=offset)


@app.get("/albums/{album_id}", response_model=AlbumSummary)
def get_album(album_id: str) -> AlbumSummary:
    try:
        return catalog_service().get_album(album_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/albums/{album_id}/photos", response_model=AlbumPhotoListResponse)
def list_album_photos(
    album_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    include_rejects: bool = False,
    sort: Literal["path", "quality", "capture_time"] = "path",
) -> AlbumPhotoListResponse:
    try:
        return catalog_service().list_photos(
            album_id,
            limit=limit,
            offset=offset,
            include_rejects=include_rejects,
            sort=sort,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/albums/{album_id}/selections", response_model=SelectionHistoryResponse)
def list_album_selections(
    album_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> SelectionHistoryResponse:
    try:
        return catalog_service().list_selections(album_id, limit=limit, offset=offset)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/jobs/prepare",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_prepare_job(request: PrepareJobRequest) -> JobResponse:
    try:
        return prepare_job_manager().submit(request)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404, detail=f"folder not found: {error}"
        ) from error
    except NotADirectoryError as error:
        raise HTTPException(
            status_code=400, detail=f"path is not a folder: {error}"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/jobs", response_model=JobListResponse)
def list_jobs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status_filter: Literal["queued", "running", "completed", "failed", "cancelled"]
    | None = Query(default=None, alias="status"),
) -> JobListResponse:
    return prepare_job_manager().list(
        limit=limit,
        offset=offset,
        status=status_filter,
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    try:
        return prepare_job_manager().get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str) -> JobResponse:
    try:
        return prepare_job_manager().cancel(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/albums/index", response_model=AlbumIndexResponse)
def index_album(request: AlbumIndexRequest) -> AlbumIndexResponse:
    try:
        return AlbumIndexer(database, settings.data_dir).index(
            Path(request.folder), request.name
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"文件夹不存在：{error}") from error
    except NotADirectoryError as error:
        raise HTTPException(
            status_code=400, detail=f"路径不是文件夹：{error}"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def retrieval_service() -> RetrievalService:
    return RetrievalService(
        database,
        settings.data_dir,
        embedding_provider(),
    )


def rag_generation_provider() -> GroundedGenerationProvider:
    return create_local_qwen3vl_provider(
        settings.local_vlm_model_dir,
        max_new_tokens=settings.vlm_max_new_tokens,
    )


def rag_service() -> GroundedRAGService:
    return GroundedRAGService(
        database,
        retrieval_service(),
        rag_generation_provider,
    )


def evaluation_service() -> EvaluationService:
    return EvaluationService(database, retrieval_service())


def people_indexer() -> PeopleIndexer:
    return PeopleIndexer(
        database,
        settings.data_dir,
        create_face_provider(
            settings.face_provider, cache_dir=settings.model_cache_dir
        ),
    )


def selection_service() -> SelectionService:
    return SelectionService(
        database,
        embedding_provider(),
    )


def preference_service() -> PreferenceService:
    return PreferenceService(
        database,
        embedding_provider(),
    )


def preference_suggestion_service() -> PreferenceSuggestionService:
    return PreferenceSuggestionService(
        database,
        embedding_provider(),
    )


def replacement_service() -> ReplacementService:
    return ReplacementService(
        database,
        embedding_provider(),
    )


@app.post("/albums/{album_id}/embed", response_model=AlbumEmbeddingResponse)
def embed_album(album_id: str) -> AlbumEmbeddingResponse:
    try:
        return retrieval_service().embed_album(album_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EmbeddingProviderUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/albums/search", response_model=AlbumSearchResponse)
def search_album(request: AlbumSearchRequest) -> AlbumSearchResponse:
    if bool(request.query) == bool(request.reference_photo_id):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of query or reference_photo_id",
        )
    try:
        return retrieval_service().search(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EmbeddingProviderUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/albums/{album_id}/rag", response_model=AlbumRAGResponse)
def run_grounded_rag(
    album_id: str,
    request: AlbumRAGRequest,
) -> AlbumRAGResponse:
    """Learned retrieval + local VLM with referential citation validation only."""

    try:
        return rag_service().run(album_id, request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=_safe_rag_error(error)) from error
    except (RAGDriftError, EvidenceIntegrityError) as error:
        raise HTTPException(status_code=409, detail=_safe_rag_error(error)) from error
    except (RAGEvidenceTooLargeError, VLMInputBudgetError) as error:
        raise HTTPException(status_code=413, detail=_safe_rag_error(error)) from error
    except RAGBusyError as error:
        raise HTTPException(status_code=429, detail=_safe_rag_error(error)) from error
    except (CitationValidationError, NoEvidenceError, ValueError) as error:
        raise HTTPException(status_code=422, detail=_safe_rag_error(error)) from error
    except (
        LocalVLMUnavailableError,
        ProviderFailureError,
        EmbeddingProviderUnavailableError,
    ) as error:
        raise HTTPException(status_code=503, detail=_safe_rag_error(error)) from error


def _safe_rag_error(error: BaseException) -> str:
    """Keep local loader/source paths out of RAG HTTP error details."""

    return redact_local_paths(str(error)) or type(error).__name__


@app.post("/evaluation/queries", response_model=EvaluationQuerySummary)
def create_evaluation_query(
    request: EvaluationQueryCreateRequest,
) -> EvaluationQuerySummary:
    try:
        return evaluation_service().create_query(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get(
    "/albums/{album_id}/evaluation/queries",
    response_model=EvaluationQueryListResponse,
)
def list_evaluation_queries(album_id: str) -> EvaluationQueryListResponse:
    try:
        return evaluation_service().list_queries(album_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get(
    "/evaluation/queries/{query_id}/candidates",
    response_model=EvaluationCandidateResponse,
)
def list_evaluation_candidates(
    query_id: str,
    limit: int = Query(default=50, ge=1, le=50),
) -> EvaluationCandidateResponse:
    try:
        return evaluation_service().candidates(query_id, limit=limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EmbeddingProviderUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.put(
    "/evaluation/queries/{query_id}/judgments",
    response_model=RelevanceJudgmentBatchResponse,
)
def upsert_relevance_judgments(
    query_id: str,
    request: RelevanceJudgmentBatchRequest,
) -> RelevanceJudgmentBatchResponse:
    try:
        return evaluation_service().upsert_judgments(query_id, request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post(
    "/albums/{album_id}/evaluation/runs",
    response_model=EvaluationRunResponse,
)
def run_retrieval_evaluation(
    album_id: str,
    request: EvaluationRunRequest,
) -> EvaluationRunResponse:
    try:
        return evaluation_service().run(album_id, request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EmbeddingProviderUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/evaluation/runs/{run_id}", response_model=EvaluationRunResponse)
def get_retrieval_evaluation(run_id: str) -> EvaluationRunResponse:
    try:
        return evaluation_service().get_run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/albums/{album_id}/people", response_model=PeopleIndexResponse)
def get_people(album_id: str) -> PeopleIndexResponse:
    try:
        return people_indexer().get(album_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/albums/{album_id}/people/index", response_model=PeopleIndexResponse)
def index_people(album_id: str) -> PeopleIndexResponse:
    try:
        return people_indexer().index(album_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FaceProviderUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/selections", response_model=SelectionResponse)
def create_selection(request: SelectionRequest) -> SelectionResponse:
    try:
        return selection_service().select(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EmbeddingProviderUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/selections/{selection_id}", response_model=SelectionResponse)
def get_selection(selection_id: str) -> SelectionResponse:
    try:
        return selection_service().get(selection_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/selections/{selection_id}/preference-pairs/suggest",
    response_model=PreferencePairSuggestionResponse,
)
def suggest_preference_pair(
    selection_id: str,
    request: PreferencePairSuggestionRequest,
) -> PreferencePairSuggestionResponse:
    try:
        return preference_suggestion_service().suggest(selection_id, request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PreferenceSuggestionConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except PreferenceSuggestionNumericalError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EmbeddingProviderUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/preferences/{user_id}", response_model=PreferenceStateResponse)
def get_preference_state(user_id: str) -> PreferenceStateResponse:
    return preference_service().get_state(user_id)


@app.post("/feedback/pairwise", response_model=PreferenceModelResponse)
def record_pairwise_feedback(
    request: PairwiseFeedbackRequest,
) -> PreferenceModelResponse:
    try:
        return preference_service().record_pairwise(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PreferenceSuggestionAlreadyConsumedError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EmbeddingProviderUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post(
    "/selections/{selection_id}/replace",
    response_model=SelectionReplacementResponse,
)
def replace_selection_photo(
    selection_id: str,
    request: SelectionReplacementRequest,
) -> SelectionReplacementResponse:
    try:
        return replacement_service().replace(selection_id, request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except EmbeddingProviderUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


web_dist = Path(__file__).resolve().parent / "web_dist"
if web_dist.joinpath("index.html").is_file():
    # Keep this mount last so API routes continue to take precedence.
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
else:

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def web_not_built() -> str:
        return """<!doctype html><html><body><h1>Norma</h1>
        <p>The web interface has not been built. Run <code>pnpm build</code>.</p>
        </body></html>"""
