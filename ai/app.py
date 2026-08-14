from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from ai.config import load_settings
from ai.index import AlbumIndexer
from ai.index.embedding import create_embedding_provider
from ai.people import PeopleIndexer, create_face_provider
from ai.preferences import PreferenceService
from ai.preferences.model import load_preference_model
from ai.retrieval import RetrievalService
from ai.selection import ReplacementService, SelectionService
from ai.schemas import (
    AlbumEmbeddingResponse,
    AlbumIndexRequest,
    AlbumIndexResponse,
    AlbumSearchRequest,
    AlbumSearchResponse,
    CapabilitiesResponse,
    HealthResponse,
    PeopleIndexResponse,
    PairwiseFeedbackRequest,
    PreferenceModelResponse,
    PreferenceStateResponse,
    SelectionRequest,
    SelectionReplacementRequest,
    SelectionReplacementResponse,
    SelectionResponse,
)
from ai.storage import Database


settings = load_settings()
database = Database(settings.database_path)

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("norma.ai")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    database.initialize()
    logger.info("Norma AI worker ready; data_dir=%s", settings.data_dir)
    yield


app = FastAPI(
    title="Norma AI Worker",
    version="0.1.0",
    description="Local domain API for multimodal photo understanding and selection.",
    lifespan=lifespan,
)
app.mount("/media", StaticFiles(directory=settings.data_dir, check_dir=False), name="media")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", schema_version=database.current_version())


@app.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        milestones={
            "library": "cpu-fallback-indexer",
            "multimodal_index": "lightweight-semantic-v1",
            "people": "opencv-haar-dct-v1",
            "selection": "structured-cp-sat-or-greedy",
            "preference": "online-pairwise-logistic-v1",
            "video": "deferred",
            "world": "deferred",
        }
    )


@app.post("/albums/index", response_model=AlbumIndexResponse)
def index_album(request: AlbumIndexRequest) -> AlbumIndexResponse:
    try:
        return AlbumIndexer(database, settings.data_dir).index(
            Path(request.folder), request.name
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"文件夹不存在：{error}") from error
    except NotADirectoryError as error:
        raise HTTPException(status_code=400, detail=f"路径不是文件夹：{error}") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def retrieval_service() -> RetrievalService:
    return RetrievalService(
        database,
        settings.data_dir,
        create_embedding_provider(settings.embedding_provider),
    )


def people_indexer() -> PeopleIndexer:
    return PeopleIndexer(
        database,
        settings.data_dir,
        create_face_provider(settings.face_provider),
    )


def selection_service() -> SelectionService:
    return SelectionService(
        database,
        create_embedding_provider(settings.embedding_provider),
    )


def preference_service() -> PreferenceService:
    return PreferenceService(
        database,
        create_embedding_provider(settings.embedding_provider),
    )


def replacement_service() -> ReplacementService:
    return ReplacementService(
        database,
        create_embedding_provider(settings.embedding_provider),
    )


@app.post("/albums/{album_id}/embed", response_model=AlbumEmbeddingResponse)
def embed_album(album_id: str) -> AlbumEmbeddingResponse:
    try:
        return retrieval_service().embed_album(album_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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


@app.post("/albums/{album_id}/people/index", response_model=PeopleIndexResponse)
def index_people(album_id: str) -> PeopleIndexResponse:
    try:
        return people_indexer().index(album_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/selections", response_model=SelectionResponse)
def create_selection(request: SelectionRequest) -> SelectionResponse:
    try:
        return selection_service().select(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/selections/{selection_id}", response_model=SelectionResponse)
def get_selection(selection_id: str) -> SelectionResponse:
    try:
        return selection_service().get(selection_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/preferences/{user_id}", response_model=PreferenceStateResponse)
def get_preference_state(user_id: str) -> PreferenceStateResponse:
    model = load_preference_model(database, user_id)
    return PreferenceStateResponse(
        user_id=model.user_id,
        comparisons=model.comparisons,
        weights=model.weights,
    )


@app.post("/feedback/pairwise", response_model=PreferenceModelResponse)
def record_pairwise_feedback(
    request: PairwiseFeedbackRequest,
) -> PreferenceModelResponse:
    try:
        return preference_service().record_pairwise(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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
