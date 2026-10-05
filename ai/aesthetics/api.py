from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from ai.aesthetics.jobs import AestheticsJobManager
from ai.aesthetics.provider import AestheticsProviderUnavailableError
from ai.aesthetics.service import AestheticsService
from ai.schemas import JobResponse


class AestheticsRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    force: bool = False


def create_aesthetics_router(
    get_service: Callable[[], AestheticsService],
    get_jobs: Callable[[], AestheticsJobManager],
) -> APIRouter:
    router = APIRouter(tags=["learned-aesthetics"])

    @router.get("/aesthetics/status")
    def assessment_status() -> dict[str, object]:
        return get_service().provider.status()

    @router.get("/albums/{album_id}/aesthetics")
    def cached_assessment(album_id: str) -> dict[str, object]:
        try:
            return get_service().cached(album_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from None

    @router.post("/albums/{album_id}/aesthetics/jobs", response_model=JobResponse, status_code=202)
    def start_assessment(album_id: str, request: AestheticsRunRequest) -> JobResponse:
        try:
            return get_jobs().submit(album_id, force=request.force)
        except KeyError as error:
            raise HTTPException(404, str(error)) from None
        except AestheticsProviderUnavailableError as error:
            raise HTTPException(503, str(error)) from None
        except ValueError as error:
            raise HTTPException(409, str(error)) from None

    @router.get("/aesthetics/jobs/{job_id}", response_model=JobResponse)
    def get_assessment_job(job_id: str) -> JobResponse:
        try:
            return get_jobs().get(job_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from None

    @router.post("/aesthetics/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_assessment_job(job_id: str) -> JobResponse:
        try:
            return get_jobs().cancel(job_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from None

    return router
