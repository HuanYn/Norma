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
from ai.retrieval import RetrievalService
from ai.schemas import (
    AlbumEmbeddingResponse,
    AlbumIndexRequest,
    AlbumIndexResponse,
    AlbumSearchRequest,
    AlbumSearchResponse,
    CapabilitiesResponse,
    HealthResponse,
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
            "selection": "planned",
            "preference": "planned",
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
