from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from ai.config import load_settings
from ai.schemas import CapabilitiesResponse, HealthResponse
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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", schema_version=database.current_version())


@app.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        milestones={
            "library": "scaffolded",
            "multimodal_index": "planned",
            "selection": "planned",
            "preference": "planned",
            "video": "deferred",
            "world": "deferred",
        }
    )

