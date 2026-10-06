"""Authenticated loopback-only, single-session non-real-time world worker."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import logging
from pathlib import Path
import re
import threading
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ai.exploration.controls import ActionName
from ai.exploration.runtime import MAX_STEPS, HARD_MAX_STEPS
from ai.video.models import decode_image, MAX_BASE64_LENGTH


class NewSession(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_base64: str = Field(min_length=4, max_length=MAX_BASE64_LENGTH, repr=False)
    prompt: str = Field(min_length=1, max_length=1500)
    upload_confirmed: bool


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sequence: int = Field(ge=0, lt=HARD_MAX_STEPS)
    action: ActionName


class Sessions:
    def __init__(self, root: Path, runtime):
        self.root, self.runtime = root, runtime
        self.max_steps = getattr(runtime, "max_steps", MAX_STEPS)
        root.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.active = None
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="norma-world")
        # Runtime caches cannot be recovered from historical JSON after restart.
        for path in root.glob("*/session.json"):
            try:
                record = json.loads(path.read_text())
                if record["status"] not in {"closed", "failed", "interrupted"}:
                    record.update(status="interrupted", phase="worker_restarted")
                    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            except (ValueError, KeyError, OSError):
                logging.exception("Could not mark interrupted world session")

    def save(self, record):
        target = self.root / record["id"] / "session.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(target)

    def get(self, sid):
        with self.lock:
            if sid not in self.records:
                raise HTTPException(
                    404, "Unknown session; a restarted worker requires a new session"
                )
            return json.loads(json.dumps(self.records[sid]))

    def create(self, request: NewSession):
        if not request.upload_confirmed or not request.prompt.strip():
            raise HTTPException(
                422, "Explicit image upload consent and nonempty prompt required"
            )
        try:
            image = decode_image(request.image_base64)
        except ValueError:
            raise HTTPException(422, "Invalid JPEG/PNG image") from None
        with self.lock:
            if self.active:
                raise HTTPException(
                    409, "A world session is already active; close it first"
                )
            if len(self.records) >= 20:
                raise HTTPException(
                    429,
                    "Worker session budget reached; preserve results and restart the worker",
                )
            sid = uuid.uuid4().hex
            folder = self.root / sid
            folder.mkdir(mode=0o700)
            image.save(folder / "source.png")
            record = dict(
                id=sid,
                status="initializing",
                phase="queued",
                next_sequence=0,
                history=[],
                requests={},
                created_at=time.time(),
                last_action=time.time(),
                image_sha256=hashlib.sha256(
                    (folder / "source.png").read_bytes()
                ).hexdigest(),
                prompt=request.prompt,
                max_steps=self.max_steps,
            )
            self.records[sid] = record
            self.active = sid
            self.cancelled.clear()
            self.save(record)
            self.pool.submit(self.initialize, sid, image, request.prompt)
            return self.get(sid)

    def progress(self, sid, phase, done=0, total=0):
        with self.lock:
            self.records[sid].update(
                phase=phase, completed_chunks=done, total_chunks=total
            )

    def initialize(self, sid, image, prompt):
        try:
            result = self.runtime.begin(
                image, prompt, self.root / sid, lambda *v: self.progress(sid, *v)
            )
            with self.lock:
                if self.cancelled.is_set():
                    self.finish_close(sid)
                else:
                    self.records[sid].update(
                        status="ready",
                        phase="ready",
                        initialization=result,
                        last_action=time.time(),
                    )
                    self.save(self.records[sid])
        except Exception:
            self.fail(sid)

    def step(self, sid, request: Step):
        if request.sequence >= self.max_steps:
            raise HTTPException(422, "Session step limit reached")
        with self.lock:
            record = self.records.get(sid)
            if record is None:
                raise HTTPException(404, "Unknown session")
            key = str(request.sequence)
            if key in record["requests"]:
                if record["requests"][key] != request.action:
                    raise HTTPException(
                        409, "Conflicting replay for an already accepted sequence"
                    )
                return self.get(sid)  # Idempotent retry; never infer twice.
            if (
                self.active != sid
                or record["status"] != "ready"
                or request.sequence != record["next_sequence"]
            ):
                raise HTTPException(
                    409, "Session not ready or stale/out-of-order sequence"
                )
            record["requests"][key] = request.action
            record.update(status="generating", phase="queued", last_action=time.time())
            self.save(record)
            self.pool.submit(self.generate, sid, request)
            return self.get(sid)

    def generate(self, sid, request):
        try:
            result = self.runtime.step(
                request.action, self.cancelled.is_set, lambda *v: self.progress(sid, *v)
            )
            with self.lock:
                if self.cancelled.is_set():
                    self.finish_close(sid)
                    return
                record = self.records[sid]
                record["history"].append(result)
                record.update(
                    status="ready", phase="ready", next_sequence=request.sequence + 1
                )
                self.save(record)
        except InterruptedError:
            with self.lock:
                self.finish_close(sid)
        except Exception:
            self.fail(sid)

    def fail(self, sid):
        logging.exception("World operation failed; discard possibly mutated GPU state")
        with self.lock:
            self.runtime.close()
            self.records[sid].update(
                status="failed",
                phase="failed",
                error="Generation failed; start a new session. Earlier clips are preserved.",
            )
            self.active = None
            self.save(self.records[sid])

    def finish_close(self, sid):
        self.runtime.close()
        self.records[sid].update(status="closed", phase="closed")
        self.active = None
        self.save(self.records[sid])

    def close(self, sid):
        with self.lock:
            record = self.records.get(sid)
            if record is None:
                raise HTTPException(404, "Unknown session")
            if self.active == sid:
                self.cancelled.set()
                if record["status"] == "ready":
                    record.update(status="closing", phase="closing")
                    self.pool.submit(self.close_idle, sid)
                else:
                    record.update(status="closing", phase="closing")
            return self.get(sid)

    def close_idle(self, sid):
        with self.lock:
            if self.active == sid:
                self.finish_close(sid)

    def expire(self):
        with self.lock:
            if self.active:
                record = self.records[self.active]
                if (
                    record["status"] == "ready"
                    and time.time() - record["last_action"] > 600
                ):
                    self.close(self.active)


async def json_body(request, schema, limit):
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > limit:
            raise HTTPException(413, "Request body too large")
    try:
        return schema.model_validate_json(bytes(content))
    except ValueError:
        raise HTTPException(422, "Invalid request fields") from None


def create_app(root: Path, runtime, token: str):
    if len(token) < 24:
        raise ValueError("Private worker token required")
    manager = Sessions(root, runtime)

    @asynccontextmanager
    async def lifespan(app):
        stopped = threading.Event()

        def sweep():
            while not stopped.wait(30):
                manager.expire()

        thread = threading.Thread(target=sweep, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            if manager.active:
                manager.close(manager.active)
            manager.pool.shutdown(wait=True)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.sessions = manager

    def authorize(request: Request):
        value = request.headers.get("authorization", "")
        if not hmac.compare_digest(value.encode(), ("Bearer " + token).encode()):
            raise HTTPException(401, "Unauthorized")

    @app.get("/status", dependencies=[Depends(authorize)])
    def status():
        manager.expire()
        return {
            "active": bool(manager.active),
            "max_steps": manager.max_steps,
            "decode_mode": getattr(runtime, "decode_mode", "prefix"),
        }

    @app.post("/sessions", status_code=202, dependencies=[Depends(authorize)])
    async def create(request: Request):
        return manager.create(await json_body(request, NewSession, 12 * 1024 * 1024))

    @app.get("/sessions/{sid}", dependencies=[Depends(authorize)])
    def state(sid: str):
        manager.expire()
        return manager.get(sid)

    @app.post(
        "/sessions/{sid}/steps", status_code=202, dependencies=[Depends(authorize)]
    )
    async def step(sid: str, request: Request):
        return manager.step(sid, await json_body(request, Step, 2048))

    @app.post("/sessions/{sid}/close", dependencies=[Depends(authorize)])
    def close(sid: str):
        return manager.close(sid)

    @app.get("/sessions/{sid}/artifacts/{name}", dependencies=[Depends(authorize)])
    def artifact(sid: str, name: str):
        record = manager.get(sid)
        if not re.fullmatch(r"step-\d{3}\.mp4", name) or name not in {
            item["artifact"] for item in record["history"]
        }:
            raise HTTPException(404, "Unknown completed artifact")
        return FileResponse(root / sid / name, media_type="video/mp4", filename=name)

    return app
