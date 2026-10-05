from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .generator import GenerationCancelled, MODEL_ID, VideoModelUnavailable, WanGenerator
from .models import VideoOptions, VideoSubmit, decode_image

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
MAX_BODY_BYTES = 12 * 1024 * 1024
logger = logging.getLogger("norma.video.worker")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkerSettings:
    data_dir: Path
    token: str = ""
    model_path: str = MODEL_ID
    allow_downloads: bool = False
    revision: str | None = None
    offload: str = "model"
    max_pending: int = 4
    max_jobs: int = 200

    def __post_init__(self):
        if self.token and (len(self.token) < 24 or any(ord(c) < 33 or ord(c) > 126 for c in self.token)):
            raise ValueError("Worker token must contain at least 24 printable non-space ASCII characters")
        if not 1 <= self.max_pending <= 32 or not 1 <= self.max_jobs <= 10000:
            raise ValueError("Invalid job limits")


class JobManager:
    def __init__(self, settings: WorkerSettings, generator=None):
        self.settings = settings
        self.root = settings.data_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.generator = generator or WanGenerator(
            settings.model_path, allow_downloads=settings.allow_downloads,
            revision=settings.revision, offload=settings.offload,
        )
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._cancellations: dict[str, threading.Event] = {}
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._restore()

    def _restore(self) -> None:
        for path in self.root.glob("*/job.json"):
            if not JOB_ID.fullmatch(path.parent.name) or path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                if path.stat().st_size > 128 * 1024:
                    continue
                job = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(job, dict) or job.get("id") != path.parent.name:
                    continue
                if job.get("status") not in TERMINAL:
                    job.update(status="interrupted", stage="interrupted", finished_at=utc_now(),
                               error="Worker restarted before this job finished; resubmit explicitly")
                self._jobs[job["id"]] = job
                self._persist(job)
            except (OSError, ValueError, TypeError):
                continue

    def _persist(self, job: dict) -> None:
        folder = self.root / job["id"]
        temporary = folder / "job.json.tmp"
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(folder / "job.json")

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Worker already shut down")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="norma-video", daemon=True)
                self._thread.start()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for job_id, event in self._cancellations.items():
                if self._jobs[job_id]["status"] not in TERMINAL:
                    self.cancel(job_id)
                    event.set()
            self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, request: VideoSubmit) -> dict:
        image = decode_image(request.image_base64)
        png = io.BytesIO()
        image.save(png, format="PNG")
        pixels = png.getvalue()
        with self._lock:
            if self._closed:
                raise HTTPException(503, "Worker is stopping")
            pending = sum(job["status"] not in TERMINAL for job in self._jobs.values())
            if pending >= self.settings.max_pending:
                raise HTTPException(429, "Video queue is full; wait for a job to finish")
            if len(self._jobs) >= self.settings.max_jobs:
                raise HTTPException(507, "Job retention limit reached; operator must archive old jobs")
            job_id = uuid.uuid4().hex
            folder = self.root / job_id
            folder.mkdir(mode=0o700)
            (folder / "input.png").write_bytes(pixels)
            options = VideoOptions.model_validate(request.model_dump(exclude={"image_base64", "upload_confirmed"}))
            job = {
                "id": job_id, "status": "queued", "stage": "queued", "created_at": utc_now(),
                "started_at": None, "finished_at": None, "step": None, "total_steps": None,
                "step_percent": None, "cancel_requested": False, "error": None,
                "model_id": MODEL_ID, "options": options.model_dump(),
                "input_sha256": hashlib.sha256(pixels).hexdigest(),
                "input_dimensions": list(image.size), "upload_confirmed": True,
                "metadata_removed": True, "crop_mode": "center_crop", "fps": 24,
                "result": None, "elapsed_seconds": None,
            }
            self._jobs[job_id] = job
            self._cancellations[job_id] = threading.Event()
            self._persist(job)
            self._queue.put(job_id)
            return self.get(job_id)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if not JOB_ID.fullmatch(job_id) or job_id not in self._jobs:
                raise HTTPException(404, "Unknown video job")
            return json.loads(json.dumps(self._jobs[job_id]))

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self.get(job_id)
            if job["status"] in TERMINAL:
                return job
            self._cancellations[job_id].set()
            original = self._jobs[job_id]
            original["cancel_requested"] = True
            if job["status"] == "queued":
                original.update(status="cancelled", stage="cancelled", finished_at=utc_now())
            self._persist(original)
            return self.get(job_id)

    def artifact(self, job_id: str) -> Path:
        job = self.get(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, "Video is not completed")
        path = self.root / job_id / "output.mp4"
        if not path.is_file() or path.is_symlink() or path.parent.is_symlink():
            raise HTTPException(404, "Video artifact is missing")
        return path

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            with self._lock:
                job = self._jobs[job_id]
                if job["status"] in TERMINAL:
                    continue
                if self._closed:
                    self.cancel(job_id)
                    continue
                job.update(status="running", stage="starting", started_at=utc_now())
                self._persist(job)
            started = time.monotonic()
            cancelled = self._cancellations[job_id]

            def progress(stage, step, total):
                if cancelled.is_set():
                    raise GenerationCancelled()
                with self._lock:
                    job.update(stage=stage, step=step, total_steps=total,
                               step_percent=round(100 * step / total, 1) if step is not None and total else None)
                    self._persist(job)

            try:
                from PIL import Image
                with Image.open(self.root / job_id / "input.png") as source:
                    frame = source.convert("RGB")
                output = self.root / job_id / "output.mp4"
                metadata = self.generator.generate(
                    frame, VideoOptions.model_validate(job["options"]), output, progress, cancelled,
                )
                if cancelled.is_set():
                    raise GenerationCancelled()
                if not output.is_file() or output.stat().st_size < 32:
                    raise RuntimeError("No video artifact")
                with output.open("rb") as stream:
                    if stream.read(12)[4:8] != b"ftyp":
                        raise RuntimeError("Invalid MP4 artifact")
                with self._lock:
                    if cancelled.is_set():
                        raise GenerationCancelled()
                    job.update(status="completed", stage="completed", result=metadata)
            except GenerationCancelled:
                with self._lock:
                    job.update(status="cancelled", stage="cancelled")
            except VideoModelUnavailable as exc:
                # Private operator log only; the HTTP response remains sanitized.
                # Do not log submitted images, prompts, headers or credentials.
                logger.exception("Video job %s failed during %s", job_id, job["stage"])
                with self._lock:
                    job.update(status="failed", stage="failed", error=str(exc))
            except Exception:
                logger.exception("Video job %s failed during %s", job_id, job["stage"])
                with self._lock:
                    job.update(status="failed", stage="failed",
                               error="Video generation failed; no completed video is available")
            finally:
                with self._lock:
                    job.update(finished_at=utc_now(), elapsed_seconds=round(time.monotonic() - started, 3))
                    self._persist(job)


class BodyLimitMiddleware:
    """Bound body bytes even when Content-Length is absent or false."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            if int(headers.get(b"content-length", b"0")) > MAX_BODY_BYTES:
                return await JSONResponse({"detail": "Request exceeds 12 MiB"}, 413)(scope, receive, send)
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        content = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            content.extend(message.get("body", b""))
            if len(content) > MAX_BODY_BYTES:
                return await JSONResponse({"detail": "Request exceeds 12 MiB"}, 413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(content), "more_body": False}
            return await receive()

        return await self.app(scope, bounded_receive, send)


def create_app(settings: WorkerSettings, *, generator=None) -> FastAPI:
    manager = JobManager(settings, generator=generator)

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        yield
        manager.close()

    app = FastAPI(title="Norma private video worker", lifespan=lifespan)
    app.state.video_manager = manager
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Default FastAPI errors echo supplied input, including image bytes.
        return JSONResponse({"detail": [{"loc": e["loc"], "type": e["type"], "msg": e["msg"]}
                                         for e in exc.errors()]}, status_code=422)

    def authenticate(request: Request):
        if settings.token:
            provided = request.headers.get("authorization", "")
            expected = f"Bearer {settings.token}"
            if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
                raise HTTPException(401, "Worker authorization required")

    @app.get("/health", dependencies=[Depends(authenticate)])
    def health():
        return {"service": "norma-video", "status": "ok", "model_id": MODEL_ID,
                "model_loaded": bool(getattr(manager.generator, "loaded", False)),
                "downloads_allowed": settings.allow_downloads, "single_worker": True,
                "note": "Health is not proof that CUDA, weights or generation are ready"}

    @app.post("/v1/video/jobs", status_code=202, dependencies=[Depends(authenticate)])
    def submit(request: VideoSubmit):
        try:
            return manager.submit(request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/video/jobs/{job_id}", dependencies=[Depends(authenticate)])
    def status(job_id: str):
        return manager.get(job_id)

    @app.post("/v1/video/jobs/{job_id}/cancel", dependencies=[Depends(authenticate)])
    def cancel(job_id: str):
        return manager.cancel(job_id)

    @app.get("/v1/video/jobs/{job_id}/artifact", dependencies=[Depends(authenticate)])
    def artifact(job_id: str):
        return FileResponse(manager.artifact(job_id), media_type="video/mp4", filename=f"norma-{job_id}.mp4")

    return app
