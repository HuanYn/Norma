"""Local browser gateway: worker token never reaches browser, per-browser ownership."""

from __future__ import annotations

import hashlib
import asyncio
from pathlib import Path
import re
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
import httpx
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ai.exploration.server import NewSession, Step, json_body


def create_web(
    token,
    root: Path,
    *,
    transport=None,
    port=8768,
    workers=("http://127.0.0.1:8769",),
    cookie_name="norma_world_browser",
):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )
    owners: dict[str, set[str]] = {}
    current: dict[str, str] = {}
    locations: dict[str, str] = {}
    submission_lock = asyncio.Lock()
    cache = root / ".norma/world-interactive-cache"
    cache.mkdir(parents=True, exist_ok=True)

    def owner(request: Request, mutating=False):
        value = request.cookies.get(cookie_name, "")
        if value not in owners:
            raise HTTPException(401, "Open the world page first")
        if mutating:
            origin = request.headers.get("origin")
            allowed_origins = (
                {f"{request.url.scheme}://{request.url.netloc}"}
                if port is None
                else {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            )
            if request.headers.get("x-norma-world") != "1" or (
                origin and origin not in allowed_origins
            ):
                raise HTTPException(403, "Same-origin confirmation required")
        return value

    def owned(request, sid, mutating=False):
        browser = owner(request, mutating)
        if sid not in owners[browser] or not re.fullmatch(r"[a-f0-9]{32}", sid):
            raise HTTPException(404, "Unknown browser session")
        return browser

    async def upstream(method, path, body=None, binary=False, worker=None):
        credential = token() if callable(token) else token
        if not credential:
            raise HTTPException(503, "World worker credential is not configured")
        sid = path.split("/")[2] if path.startswith("/sessions/") else None
        endpoint = worker or locations.get(sid, workers[0])
        try:
            async with httpx.AsyncClient(
                base_url=endpoint,
                timeout=45,
                trust_env=False,
                transport=transport,
                follow_redirects=False,
                headers={"Authorization": "Bearer " + credential},
            ) as client:
                async with client.stream(method, path, json=body) as response:
                    data = bytearray()
                    async for block in response.aiter_bytes():
                        data.extend(block)
                        if len(data) > (
                            64 * 1024 * 1024 if binary else 2 * 1024 * 1024
                        ):
                            raise HTTPException(502, "Worker response exceeded limit")
                    if response.status_code >= 400:
                        raise HTTPException(
                            response.status_code
                            if response.status_code in {404, 409, 413, 422, 429}
                            else 502,
                            "World worker rejected the request; check session status",
                        )
                    if response.status_code not in {200, 202}:
                        raise HTTPException(502, "Unexpected worker response")
                    if binary:
                        return bytes(data)
                    import json

                    return json.loads(data)
        except (httpx.HTTPError, ValueError):
            raise HTTPException(
                503, "Private world worker unavailable; check SSH tunnel"
            ) from None

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        browser = request.cookies.get(cookie_name, "")
        if browser not in owners:
            if len(owners) >= 100:
                raise HTTPException(429, "Browser session limit reached")
            browser = uuid.uuid4().hex
            owners[browser] = set()
        response = HTMLResponse(
            (root / "scripts/templates/world_interactive.html").read_text(
                encoding="utf-8"
            )
        )
        response.set_cookie(
            cookie_name,
            browser,
            httponly=True,
            samesite="strict",
            max_age=86400,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/examples/{name}")
    def example(name: str, request: Request):
        owner(request)
        if name not in {"mountain", "temple"}:
            raise HTTPException(404)
        path = root / ".norma/user-world-demo-20261005/inputs" / (name + ".jpg")
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/current")
    async def restore(request: Request):
        browser = owner(request)
        sid = current.get(browser)
        if sid:
            try:
                return await upstream("GET", f"/sessions/{sid}")
            except HTTPException as error:
                if error.status_code != 404:
                    raise
                return {
                    "id": sid,
                    "status": "interrupted",
                    "history": [],
                    "next_sequence": 0,
                    "phase": "worker_restarted",
                    "max_steps": 7,
                }
        return {"status": "empty"}

    @app.post("/api/sessions", status_code=202)
    async def create(request: Request):
        browser = owner(request, True)
        payload = await json_body(request, NewSession, 12 * 1024 * 1024)
        if not payload.upload_confirmed:
            raise HTTPException(422, "Confirm this photo upload")
        async with submission_lock:
            if len(workers) == 1:
                endpoint = workers[0]
            else:
                endpoint = None
                for candidate in workers:
                    try:
                        status = await upstream("GET", "/status", worker=candidate)
                    except HTTPException:
                        continue
                    if status.get("active") is False:
                        endpoint = candidate
                        break
                if endpoint is None:
                    raise HTTPException(
                        409, "没有空闲探索服务；请结束旧场景，或检查第二个私有服务连接"
                    )
            # Never automatically retry a POST: an uncertain response may already
            # have allocated GPU state on the worker.
            result = await upstream(
                "POST", "/sessions", payload.model_dump(), worker=endpoint
            )
            sid = result["id"]
            if not re.fullmatch(r"[a-f0-9]{32}", sid):
                raise HTTPException(502, "Invalid worker session id")
            locations[sid] = endpoint
            owners[browser].add(sid)
            current[browser] = sid
        return result

    @app.get("/api/sessions/{sid}")
    async def state(sid: str, request: Request):
        owned(request, sid)
        return await upstream("GET", f"/sessions/{sid}")

    @app.post("/api/sessions/{sid}/steps", status_code=202)
    async def step(sid: str, request: Request):
        owned(request, sid, True)
        payload = await json_body(request, Step, 2048)
        return await upstream("POST", f"/sessions/{sid}/steps", payload.model_dump())

    @app.post("/api/sessions/{sid}/close")
    async def close(sid: str, request: Request):
        browser = owned(request, sid, True)
        try:
            result = await upstream("POST", f"/sessions/{sid}/close")
        except HTTPException as error:
            if error.status_code != 404:
                raise
            result = {"status": "closed", "id": sid, "history": []}
        if result["status"] in {"closed", "failed"}:
            current.pop(browser, None)
        return result

    @app.get("/api/sessions/{sid}/artifacts/{name}")
    async def artifact(sid: str, name: str, request: Request):
        owned(request, sid)
        if not re.fullmatch(r"step-\d{3}\.mp4", name):
            raise HTTPException(404)
        record = await upstream("GET", f"/sessions/{sid}")
        item = next((v for v in record["history"] if v["artifact"] == name), None)
        if item is None:
            raise HTTPException(404)
        folder = cache / sid
        folder.mkdir(exist_ok=True)
        target = folder / name
        if (
            not target.is_file()
            or hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]
        ):
            content = await upstream(
                "GET", f"/sessions/{sid}/artifacts/{name}", binary=True
            )
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise HTTPException(502, "Artifact integrity check failed")
            temporary = folder / (uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
        return FileResponse(target, media_type="video/mp4", filename=name)

    return app
