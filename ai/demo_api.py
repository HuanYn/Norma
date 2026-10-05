"""Opt-in demo endpoints. Remote generation never receives a client-supplied path."""

from __future__ import annotations

import io
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai.config import Settings
from ai.schemas import SelectionRequest, SelectionResponse
from ai.selection.service import SelectionService
from ai.rag.cloud_runtime import CloudVLMUnavailableError
from ai.rag.transformers_runtime import LocalVLMUnavailableError
from ai.index.embedding import EmbeddingProviderUnavailableError
from ai.selection.structured import (
    StructuredIntentError,
    StructuredProviderError,
    create_structured_parser,
)
from ai.storage import Database
from ai.video.client import VideoWorkerClient, VideoWorkerError
from ai.video.models import VideoOptions


class ParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=1000)
    allow_cloud: bool = False


class ModelSelectionRequest(SelectionRequest):
    allow_cloud: bool = Field(default=False, strict=True)


class LocalVideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    album_id: str = Field(min_length=1, max_length=128)
    photo_id: str = Field(min_length=1, max_length=128)
    upload_confirmed: Literal[True]
    options: VideoOptions

    @field_validator("upload_confirmed", mode="before")
    @classmethod
    def require_confirmation(cls, value):
        if value is not True:
            raise ValueError("Confirm this photo's upload to your video server")
        return value


@lru_cache(maxsize=1)
def _parser(settings: Settings):
    return create_structured_parser(settings)


def _client(settings: Settings) -> VideoWorkerClient:
    if not settings.video_base_url:
        raise HTTPException(503, "Video worker is not configured")
    token = ""
    if settings.video_worker_token_file:
        try:
            path = settings.video_worker_token_file
            if path.stat().st_size > 1024:
                raise ValueError("Invalid token file")
            token = path.read_text(encoding="utf-8").strip()
            if len(token) < 24 or not token.isascii():
                raise ValueError("Invalid token file")
        except (OSError, ValueError):
            raise HTTPException(
                503, "Video worker credential file is unavailable or invalid"
            ) from None
    try:
        return VideoWorkerClient(settings.video_base_url, token=token)
    except ValueError:
        raise HTTPException(
            503, "Invalid video worker endpoint configuration"
        ) from None


def _initialize_links(database: Database) -> None:
    with database.connect() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS demo_video_links_v1 (
            job_id TEXT PRIMARY KEY, album_id TEXT NOT NULL, photo_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")


def _image_bytes(database: Database, album_id: str, photo_id: str) -> bytes:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT absolute_path, file_size, source_mtime_ns FROM photos WHERE id=? AND album_id=?",
            (photo_id, album_id),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Indexed photo not found in this album")
    try:
        path = Path(row["absolute_path"])
        before = path.stat()
        if (
            before.st_size != row["file_size"]
            or before.st_mtime_ns != row["source_mtime_ns"]
        ):
            raise HTTPException(
                409, "Photo changed since indexing; refresh the album first"
            )
        if before.st_size > 64 * 1024 * 1024:
            raise HTTPException(413, "Source photo exceeds the demo size limit")
        with Image.open(path) as source:
            if min(source.size) < 32:
                raise HTTPException(422, "Photo is too small for video generation")
            if source.width * source.height > 40_000_000 or source.format not in {
                "JPEG",
                "PNG",
            }:
                raise HTTPException(
                    413, "Photo dimensions or format exceed demo limits"
                )
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            oriented.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            clean = Image.new("RGB", oriented.size)
            clean.paste(oriented)
            buffer = io.BytesIO()
            clean.save(buffer, format="JPEG", quality=90)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise HTTPException(
                409, "Photo changed during preparation; retry after refreshing"
            )
        return buffer.getvalue()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise HTTPException(400, "Could not safely decode the indexed photo") from None


def create_demo_router(
    get_database: Callable[[], Database],
    get_settings: Callable[[], Settings],
    get_selection_service: Callable[[], SelectionService],
) -> APIRouter:
    router = APIRouter(tags=["demo"])
    artifact_lock = threading.Lock()

    def parse(prompt: str, allow_cloud: bool):
        settings = get_settings()
        if settings.vlm_provider == "openai-compatible" and not allow_cloud:
            raise HTTPException(
                400, "Cloud text parsing requires explicit consent; no images are sent"
            )
        try:
            return _parser(settings).parse(prompt)
        except StructuredIntentError as error:
            raise HTTPException(422, str(error)) from None
        except StructuredProviderError as error:
            raise HTTPException(503, str(error)) from None
        except (LocalVLMUnavailableError, CloudVLMUnavailableError):
            raise HTTPException(
                503,
                "Selection model is unavailable; check its configured files or service",
            ) from None

    @router.post("/selection/parse")
    def parse_intent(request: ParseRequest):
        result = parse(request.prompt, request.allow_cloud)
        return {
            "status": result.status,
            "document": result.document.model_dump(),
            "provenance": result.provenance.model_dump(),
        }

    @router.post("/selections/structured", response_model=SelectionResponse)
    def structured_selection(request: ModelSelectionRequest):
        result = parse(request.prompt, request.allow_cloud)
        if result.status != "ready":
            raise HTTPException(
                422,
                {
                    "message": "Some requirements need clarification or are not supported",
                    "document": result.document.model_dump(),
                },
            )
        try:
            return get_selection_service().select(
                SelectionRequest.model_validate(
                    request.model_dump(exclude={"allow_cloud"})
                ),
                intent=result.to_selection_intent(),
                semantic_query=result.semantic_query,
                intent_provenance=result.provenance.model_dump(),
            )
        except KeyError:
            raise HTTPException(
                404, "Album or required evidence is unavailable"
            ) from None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        except EmbeddingProviderUnavailableError:
            raise HTTPException(503, "Embedding model is unavailable") from None

    @router.get("/demo/video/status")
    def video_status():
        settings = get_settings()
        if not settings.video_base_url:
            return {"configured": False, "reachable": False}
        try:
            return {
                "configured": True,
                "reachable": True,
                "worker": _client(settings).health(),
            }
        except VideoWorkerError as error:
            return {"configured": True, "reachable": False, "error": str(error)}

    @router.post("/demo/video/jobs", status_code=202)
    def submit_video(request: LocalVideoRequest):
        database = get_database()
        client = _client(get_settings())
        _initialize_links(database)
        content = _image_bytes(database, request.album_id, request.photo_id)
        try:
            job = client.submit(content, request.options, upload_confirmed=True)
        except VideoWorkerError as error:
            raise HTTPException(502, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        if not re.fullmatch(r"[0-9a-f]{32}", str(job.get("id", ""))):
            raise HTTPException(502, "Video worker returned an invalid job identifier")
        with database.connect() as connection:
            connection.execute(
                "INSERT INTO demo_video_links_v1(job_id,album_id,photo_id) VALUES (?,?,?)",
                (job["id"], request.album_id, request.photo_id),
            )
        return job

    def owned_job(job_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404, "Video job not found")
        database = get_database()
        _initialize_links(database)
        with database.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM demo_video_links_v1 WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Video job not found")

    @router.get("/demo/video/jobs/{job_id}")
    def video_job(job_id: str):
        owned_job(job_id)
        try:
            return _client(get_settings()).status(job_id)
        except VideoWorkerError as error:
            raise HTTPException(502, str(error)) from None

    @router.post("/demo/video/jobs/{job_id}/cancel")
    def cancel_video(job_id: str):
        owned_job(job_id)
        try:
            return _client(get_settings()).cancel(job_id)
        except VideoWorkerError as error:
            raise HTTPException(502, str(error)) from None

    @router.get("/demo/video/jobs/{job_id}/artifact")
    def video_artifact(job_id: str):
        owned_job(job_id)
        settings = get_settings()
        root = settings.data_dir / "video-artifacts"
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{job_id}.mp4"
        with artifact_lock:
            if target.exists():
                try:
                    if (
                        target.is_symlink()
                        or not 32 <= target.stat().st_size <= 256 * 1024 * 1024
                    ):
                        raise ValueError
                    with target.open("rb") as source:
                        if source.read(8)[4:8] != b"ftyp":
                            raise ValueError
                except (OSError, ValueError):
                    raise HTTPException(
                        409,
                        "Local video cache is invalid; remote artifact has been preserved",
                    ) from None
            else:
                try:
                    _client(settings).download(job_id, target)
                except VideoWorkerError as error:
                    raise HTTPException(502, str(error)) from None
        return FileResponse(
            target, media_type="video/mp4", filename=f"norma-{job_id}.mp4"
        )

    return router
