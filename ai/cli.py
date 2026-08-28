from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import uvicorn

from ai.config import Settings, load_settings
from ai.evaluation import EvaluationService
from ai.index import AlbumIndexer
from ai.index.embedding import (
    create_embedding_provider,
    embedding_provider_capabilities,
)
from ai.jobs import get_persisted_job, list_persisted_jobs
from ai.library import AlbumCatalogService
from ai.maintenance import CacheMaintenanceService
from ai.people import PeopleIndexer, create_face_provider
from ai.preferences import PreferenceService
from ai.preferences.model import load_preference_model
from ai.retrieval import RetrievalService
from ai.schemas import (
    AlbumSearchRequest,
    CacheGcRequest,
    CacheQuotaRequest,
    EmbeddingProviderStatusResponse,
    EmbeddingProviderWarmupResponse,
    EvaluationQueryCreateRequest,
    EvaluationRunRequest,
    PairwiseFeedbackRequest,
    RelevanceJudgmentBatchRequest,
    RelevanceJudgmentInput,
    SelectionReplacementRequest,
    SelectionRequest,
)
from ai.selection import ReplacementService, SelectionService
from ai.storage import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="norma",
        description="Run Norma's local photo intelligence pipeline directly from Python.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Cache/database directory (default: NORMA_DATA_DIR or .norma/data)",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="Initialize the local SQLite database")
    commands.add_parser("albums", help="List indexed albums")
    commands.add_parser("providers", help="List embedding providers and availability")
    commands.add_parser("provider-status", help="Show active provider load state")
    commands.add_parser("warmup", help="Load and probe the active embedding provider")

    cache_gc = commands.add_parser(
        "cache-gc", help="Audit or delete old unreferenced generated cache files"
    )
    cache_gc.add_argument(
        "--apply", action="store_true", help="Delete eligible files; default is dry-run"
    )
    cache_gc.add_argument("--min-age-seconds", type=int, default=3600)

    commands.add_parser("cache-usage", help="Show cache/model/database disk usage")
    cache_enforce = commands.add_parser(
        "cache-enforce", help="Dry-run or enforce the configured cache budget"
    )
    cache_enforce.add_argument("--budget-gb", type=float)
    cache_enforce.add_argument("--apply", action="store_true")
    cache_enforce.add_argument("--min-age-seconds", type=int, default=3600)
    maintenance_runs = commands.add_parser(
        "maintenance-runs", help="List persisted maintenance audit records"
    )
    maintenance_runs.add_argument("--limit", type=int, default=50)
    maintenance_runs.add_argument("--offset", type=int, default=0)

    album = commands.add_parser("album", help="Show persisted album statistics")
    album.add_argument("album_id")

    photos = commands.add_parser("photos", help="List photos in an indexed album")
    photos.add_argument("album_id")
    photos.add_argument("--include-rejects", action="store_true")

    index = commands.add_parser("index", help="Index a JPG/JPEG folder read-only")
    index.add_argument("folder", type=Path)
    index.add_argument("--name")

    prepare = commands.add_parser(
        "prepare", help="Index an album and build semantic/people caches"
    )
    prepare.add_argument("folder", type=Path)
    prepare.add_argument("--name")
    prepare.add_argument(
        "--skip-people", action="store_true", help="Skip face detection/clustering"
    )

    embed = commands.add_parser("embed", help="Build the semantic cache for an album")
    embed.add_argument("album_id")

    search = commands.add_parser("search", help="Search an album using text")
    search.add_argument("album_id")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)

    image_search = commands.add_parser(
        "image-search", help="Find photos like a reference"
    )
    image_search.add_argument("album_id")
    image_search.add_argument("photo_id")
    image_search.add_argument("--limit", type=int, default=20)

    people = commands.add_parser("people", help="Detect faces and build people groups")
    people.add_argument("album_id")

    select = commands.add_parser("select", help="Create a constrained collection")
    select.add_argument("album_id")
    select.add_argument("prompt")

    feedback = commands.add_parser("feedback", help="Record a pairwise preference")
    feedback.add_argument("album_id")
    feedback.add_argument("preferred_photo_id")
    feedback.add_argument("rejected_photo_id")
    feedback.add_argument("--selection-id")
    feedback.add_argument("--user-id", default="local")

    replace = commands.add_parser("replace", help="Replace one selected photo")
    replace.add_argument("selection_id")
    replace.add_argument("remove_photo_id")

    selection = commands.add_parser(
        "show-selection", help="Read a persisted selection audit"
    )
    selection.add_argument("selection_id")

    preferences = commands.add_parser(
        "show-preferences", help="Read the current local preference weights"
    )
    preferences.add_argument("--user-id", default="local")

    history = commands.add_parser(
        "selection-history", help="List persisted selections for an album"
    )
    history.add_argument("album_id")
    history.add_argument("--limit", type=int, default=50)
    history.add_argument("--offset", type=int, default=0)

    jobs = commands.add_parser("jobs", help="List persisted background jobs")
    jobs.add_argument(
        "--status", choices=("queued", "running", "completed", "failed", "cancelled")
    )
    jobs.add_argument("--limit", type=int, default=50)
    jobs.add_argument("--offset", type=int, default=0)

    job = commands.add_parser("show-job", help="Show one persisted background job")
    job.add_argument("job_id")

    eval_add = commands.add_parser(
        "eval-add-query", help="Add a human relevance evaluation query"
    )
    eval_add.add_argument("album_id")
    eval_add.add_argument("query")
    eval_add.add_argument("--notes")

    eval_queries = commands.add_parser(
        "eval-queries", help="List evaluation queries for an album"
    )
    eval_queries.add_argument("album_id")

    eval_candidates = commands.add_parser(
        "eval-candidates", help="Rank candidates for manual relevance labeling"
    )
    eval_candidates.add_argument("query_id")
    eval_candidates.add_argument("--limit", type=int, default=50)

    eval_judge = commands.add_parser(
        "eval-judge", help="Set one 0..3 relevance judgment"
    )
    eval_judge.add_argument("query_id")
    eval_judge.add_argument("photo_id")
    eval_judge.add_argument("relevance", type=int, choices=range(4))
    eval_judge.add_argument("--annotator", default="local")

    eval_run = commands.add_parser(
        "eval-run", help="Compute and persist retrieval metrics"
    )
    eval_run.add_argument("album_id")
    eval_run.add_argument("--cutoffs", type=int, nargs="+", default=[1, 5, 10])
    eval_run.add_argument("--query-id", action="append", dest="query_ids")

    eval_show = commands.add_parser(
        "eval-show-run", help="Read a persisted retrieval evaluation report"
    )
    eval_show.add_argument("run_id")

    for name, help_text in (
        ("web", "Run the local Norma website"),
        ("serve", "Run the local FastAPI server"),
    ):
        serve = commands.add_parser(name, help=help_text)
        serve.add_argument("--host")
        serve.add_argument("--port", type=int)
        serve.add_argument("--reload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = _settings(args.data_dir)
    if args.command in {"serve", "web"}:
        if (
            args.command == "web"
            and not Path(__file__)
            .with_name("web_dist")
            .joinpath("index.html")
            .is_file()
        ):
            parser.error("web assets are missing; run `pnpm build` first")
        _serve(args, settings)
        return 0

    database = Database(settings.database_path)
    database.initialize()
    provider = create_embedding_provider(
        settings.embedding_provider,
        cache_dir=settings.model_cache_dir,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    try:
        payload = _dispatch(args, settings, database, provider)
    except (
        FileNotFoundError,
        NotADirectoryError,
        KeyError,
        ValueError,
        RuntimeError,
    ) as error:
        _print_json({"ok": False, "error": str(error)}, args.pretty, stream=sys.stderr)
        return 1
    _print_json({"ok": True, "result": payload}, args.pretty)
    return 0


def _dispatch(
    args: argparse.Namespace,
    settings: Settings,
    database: Database,
    provider: Any,
) -> Any:
    retrieval = RetrievalService(database, settings.data_dir, provider)
    evaluation = EvaluationService(database, retrieval)
    if args.command == "init":
        return {
            "database": str(database.path),
            "schema_version": database.current_version(),
        }
    if args.command == "albums":
        with database.connect() as connection:
            rows = connection.execute(
                """
                SELECT a.id, a.name, a.source_path, a.indexed_at, COUNT(p.id) AS photos
                FROM albums a LEFT JOIN photos p ON p.album_id = a.id
                GROUP BY a.id ORDER BY a.indexed_at DESC, a.id
                """
            ).fetchall()
        return [dict(row) for row in rows]
    if args.command == "providers":
        return embedding_provider_capabilities(settings.embedding_provider)
    if args.command == "provider-status":
        return EmbeddingProviderStatusResponse(
            provider=provider.name,
            dimension=provider.dimension,
            model_backed=provider.model_backed,
            loaded=provider.is_loaded,
            device=provider.runtime_device,
            warmup_state="ready" if provider.is_loaded else "idle",
        ).model_dump()
    if args.command == "warmup":
        loaded_before = provider.is_loaded
        started = time.perf_counter()
        provider.warmup()
        return EmbeddingProviderWarmupResponse(
            provider=provider.name,
            dimension=provider.dimension,
            model_backed=provider.model_backed,
            loaded_before=loaded_before,
            loaded_after=provider.is_loaded,
            device=provider.runtime_device,
            duration_ms=round((time.perf_counter() - started) * 1000),
        ).model_dump()
    if args.command == "cache-gc":
        return (
            CacheMaintenanceService(
                database,
                settings.data_dir,
                model_cache_dir=settings.model_cache_dir,
                budget_bytes=settings.cache_budget_bytes,
            )
            .collect(
                CacheGcRequest(
                    dry_run=not args.apply,
                    min_age_seconds=args.min_age_seconds,
                )
            )
            .model_dump()
        )
    if args.command in {"cache-usage", "cache-enforce", "maintenance-runs"}:
        maintenance = CacheMaintenanceService(
            database,
            settings.data_dir,
            model_cache_dir=settings.model_cache_dir,
            budget_bytes=settings.cache_budget_bytes,
        )
        if args.command == "cache-usage":
            return maintenance.usage().model_dump()
        if args.command == "cache-enforce":
            if args.budget_gb is not None and args.budget_gb <= 0:
                raise ValueError("budget-gb must be greater than zero")
            budget = (
                round(args.budget_gb * 1024**3) if args.budget_gb is not None else None
            )
            return maintenance.enforce_quota(
                CacheQuotaRequest(
                    budget_bytes=budget,
                    dry_run=not args.apply,
                    min_age_seconds=args.min_age_seconds,
                )
            ).model_dump()
        if args.limit < 1 or args.limit > 200 or args.offset < 0:
            raise ValueError("limit must be 1..200 and offset must be non-negative")
        return maintenance.list_runs(limit=args.limit, offset=args.offset).model_dump()
    if args.command == "album":
        return AlbumCatalogService(database).get_album(args.album_id).model_dump()
    if args.command == "photos":
        query = """
            SELECT id, absolute_path, quality_score, auto_reject,
                   reject_reason, similarity_group, embedding_path
            FROM photos WHERE album_id = ?
        """
        parameters: list[Any] = [args.album_id]
        if not args.include_rejects:
            query += " AND auto_reject = 0"
        query += " ORDER BY absolute_path"
        with database.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        if not rows:
            with database.connect() as connection:
                album = connection.execute(
                    "SELECT id FROM albums WHERE id = ?", (args.album_id,)
                ).fetchone()
            if album is None:
                raise KeyError(f"album not found: {args.album_id}")
        return [
            {
                "photo_id": row["id"],
                "filename": Path(row["absolute_path"]).name,
                "quality_score": row["quality_score"],
                "auto_reject": bool(row["auto_reject"]),
                "reject_reason": row["reject_reason"],
                "similarity_group": row["similarity_group"],
                "embedded": bool(row["embedding_path"]),
            }
            for row in rows
        ]
    if args.command == "index":
        result = AlbumIndexer(database, settings.data_dir).index(args.folder, args.name)
        return _compact_index(result)
    if args.command == "prepare":
        indexed = AlbumIndexer(database, settings.data_dir).index(
            args.folder, args.name
        )
        embedded = RetrievalService(database, settings.data_dir, provider).embed_album(
            indexed.album_id
        )
        people = None
        if not args.skip_people:
            people = PeopleIndexer(
                database,
                settings.data_dir,
                create_face_provider(
                    settings.face_provider, cache_dir=settings.model_cache_dir
                ),
            ).index(indexed.album_id)
        return {
            "album": _compact_index(indexed),
            "embedding": embedded.model_dump(),
            "people": people.model_dump() if people else None,
        }
    if args.command == "embed":
        return retrieval.embed_album(args.album_id).model_dump()
    if args.command == "search":
        return (
            RetrievalService(database, settings.data_dir, provider)
            .search(
                AlbumSearchRequest(
                    album_id=args.album_id,
                    query=args.query,
                    limit=args.limit,
                )
            )
            .model_dump()
        )
    if args.command == "image-search":
        return (
            RetrievalService(database, settings.data_dir, provider)
            .search(
                AlbumSearchRequest(
                    album_id=args.album_id,
                    reference_photo_id=args.photo_id,
                    limit=args.limit,
                )
            )
            .model_dump()
        )
    if args.command == "people":
        return (
            PeopleIndexer(
                database,
                settings.data_dir,
                create_face_provider(
                    settings.face_provider, cache_dir=settings.model_cache_dir
                ),
            )
            .index(args.album_id)
            .model_dump()
        )
    if args.command == "select":
        return (
            SelectionService(database, provider)
            .select(SelectionRequest(album_id=args.album_id, prompt=args.prompt))
            .model_dump()
        )
    if args.command == "feedback":
        return (
            PreferenceService(database, provider)
            .record_pairwise(
                PairwiseFeedbackRequest(
                    album_id=args.album_id,
                    preferred_photo_id=args.preferred_photo_id,
                    rejected_photo_id=args.rejected_photo_id,
                    selection_id=args.selection_id,
                    user_id=args.user_id,
                )
            )
            .model_dump()
        )
    if args.command == "replace":
        return (
            ReplacementService(database, provider)
            .replace(
                args.selection_id,
                SelectionReplacementRequest(remove_photo_id=args.remove_photo_id),
            )
            .model_dump()
        )
    if args.command == "show-selection":
        return SelectionService(database, provider).get(args.selection_id).model_dump()
    if args.command == "show-preferences":
        model = load_preference_model(database, args.user_id)
        return {
            "user_id": model.user_id,
            "comparisons": model.comparisons,
            "weights": model.weights,
        }
    if args.command == "selection-history":
        if args.limit < 1 or args.limit > 200 or args.offset < 0:
            raise ValueError("limit must be 1..200 and offset must be non-negative")
        return (
            AlbumCatalogService(database)
            .list_selections(args.album_id, limit=args.limit, offset=args.offset)
            .model_dump()
        )
    if args.command == "jobs":
        if args.limit < 1 or args.limit > 200 or args.offset < 0:
            raise ValueError("limit must be 1..200 and offset must be non-negative")
        return list_persisted_jobs(
            database,
            limit=args.limit,
            offset=args.offset,
            status=args.status,
        ).model_dump()
    if args.command == "show-job":
        return get_persisted_job(database, args.job_id).model_dump()
    if args.command == "eval-add-query":
        return evaluation.create_query(
            EvaluationQueryCreateRequest(
                album_id=args.album_id,
                query_text=args.query,
                notes=args.notes,
            )
        ).model_dump()
    if args.command == "eval-queries":
        return evaluation.list_queries(args.album_id).model_dump()
    if args.command == "eval-candidates":
        return evaluation.candidates(args.query_id, limit=args.limit).model_dump()
    if args.command == "eval-judge":
        return evaluation.upsert_judgments(
            args.query_id,
            RelevanceJudgmentBatchRequest(
                annotator=args.annotator,
                judgments=[
                    RelevanceJudgmentInput(
                        photo_id=args.photo_id,
                        relevance=args.relevance,
                    )
                ],
            ),
        ).model_dump()
    if args.command == "eval-run":
        return evaluation.run(
            args.album_id,
            EvaluationRunRequest(
                query_ids=args.query_ids,
                cutoffs=args.cutoffs,
            ),
        ).model_dump()
    if args.command == "eval-show-run":
        return evaluation.get_run(args.run_id).model_dump()
    raise RuntimeError(f"unhandled command: {args.command}")


def _settings(data_dir: Path | None) -> Settings:
    current = load_settings()
    if data_dir is None:
        return current
    return Settings(
        host=current.host,
        port=current.port,
        data_dir=data_dir.resolve(),
        log_level=current.log_level,
        embedding_provider=current.embedding_provider,
        face_provider=current.face_provider,
        embedding_device=current.embedding_device,
        embedding_batch_size=current.embedding_batch_size,
        model_cache_root=current.model_cache_root,
        prewarm_embedding=current.prewarm_embedding,
        cache_budget_bytes=current.cache_budget_bytes,
    )


def _serve(args: argparse.Namespace, settings: Settings) -> None:
    if args.data_dir:
        # ai.app reads this variable during import inside uvicorn.
        import os

        os.environ["NORMA_DATA_DIR"] = str(settings.data_dir)
    uvicorn.run(
        "ai.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )


def _compact_index(result: Any) -> dict[str, Any]:
    return {
        "album_id": result.album_id,
        "name": result.name,
        "source_path": result.source_path,
        "total": result.total,
        "rejected": result.rejected,
        "similar_groups": result.similar_groups,
        "duration_ms": result.duration_ms,
        "provider": result.provider,
        "errors": result.errors,
    }


def _print_json(
    value: Any,
    pretty: bool,
    *,
    stream: Any | None = None,
) -> None:
    output = stream or sys.stdout
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            default=str,
        ),
        file=output,
    )
