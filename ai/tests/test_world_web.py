"""Browser gateway isolation tests; mocked HTTP is not model-quality evidence."""

import hashlib

from fastapi.testclient import TestClient
import httpx
import pytest

from ai.exploration.web import create_web
from ai.tests.test_world_sessions import payload

SID = "a" * 32
HEADERS = {"X-Norma-World": "1", "Origin": "http://127.0.0.1:8768"}
CONTENT = b"protocol-test-not-a-video"


@pytest.fixture
def gateway(tmp_path):
    template = tmp_path / "scripts/templates/world_interactive.html"
    template.parent.mkdir(parents=True)
    template.write_text("<html>World</html>", encoding="utf-8")
    calls = []
    options = {"status": 200, "content": CONTENT}

    def handler(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-private-worker-token"
        if options["status"] != 200:
            return httpx.Response(
                options["status"],
                text="PRIVATE_SENTINEL",
                headers={"Location": "http://external.invalid"},
            )
        if "/artifacts/" in request.url.path:
            return httpx.Response(200, content=options["content"])
        return httpx.Response(
            200,
            json={
                "id": SID,
                "status": "ready",
                "history": [
                    {
                        "artifact": "step-000.mp4",
                        "sha256": hashlib.sha256(CONTENT).hexdigest(),
                    }
                ],
            },
        )

    app = create_web(
        "test-private-worker-token", tmp_path, transport=httpx.MockTransport(handler)
    )
    with TestClient(app) as client:
        yield client, app, calls, options


def begin(client):
    assert client.get("/").status_code == 200
    response = client.post("/api/sessions", json=payload(), headers=HEADERS)
    assert response.status_code == 202
    assert "test-private-worker-token" not in response.text


def test_cookie_csrf_and_browser_isolation(gateway):
    client, app, calls, _ = gateway
    assert client.get("/api/current").status_code == 401
    response = client.get("/")
    assert (
        "HttpOnly" in response.headers["set-cookie"]
        and "SameSite=strict" in response.headers["set-cookie"]
    )
    assert client.post("/api/sessions", json=payload()).status_code == 403
    assert (
        client.post(
            "/api/sessions",
            json=payload(),
            headers=HEADERS | {"Origin": "https://evil.invalid"},
        ).status_code
        == 403
    )
    assert len(calls) == 0
    begin(client)
    assert client.get("/api/current").json()["id"] == SID
    with TestClient(app) as stranger:
        stranger.get("/")
        assert stranger.get(f"/api/sessions/{SID}").status_code == 404
        assert (
            stranger.get(f"/api/sessions/{SID}/artifacts/step-000.mp4").status_code
            == 404
        )
        assert (
            stranger.post(
                f"/api/sessions/{SID}/close", json={}, headers=HEADERS
            ).status_code
            == 404
        )


def test_download_hash_cache_range_and_whitelist(gateway):
    client, _, calls, options = gateway
    begin(client)
    url = f"/api/sessions/{SID}/artifacts/step-000.mp4"
    assert client.get(url).content == CONTENT
    assert client.get(url, headers={"Range": "bytes=0-3"}).status_code == 206
    assert len([r for r in calls if "/artifacts/" in r.url.path]) == 1
    assert client.get(f"/api/sessions/{SID}/artifacts/source.png").status_code == 404
    assert client.get(f"/api/sessions/{SID}/artifacts/step-001.mp4").status_code == 404


def test_bad_artifact_not_published(gateway):
    client, _, _, options = gateway
    begin(client)
    options["content"] = b"wrong bytes"
    assert client.get(f"/api/sessions/{SID}/artifacts/step-000.mp4").status_code == 502


@pytest.mark.parametrize("status", [302, 500, 401])
def test_upstream_redirect_and_error_redacted(gateway, status):
    client, _, calls, options = gateway
    begin(client)
    options["status"] = status
    response = client.get(f"/api/sessions/{SID}")
    assert response.status_code == 502 and "PRIVATE_SENTINEL" not in response.text
    assert all(r.url.host == "127.0.0.1" for r in calls)


def test_restart_and_missing_consent(gateway):
    client, _, calls, options = gateway
    begin(client)
    before = len(calls)
    assert (
        client.post(
            "/api/sessions",
            json=payload() | {"upload_confirmed": False},
            headers=HEADERS,
        ).status_code
        == 422
    )
    assert len(calls) == before
    options["status"] = 404
    assert client.get("/api/current").json()["status"] == "interrupted"


def test_two_world_workers_route_actions_and_artifacts_independently(tmp_path):
    template = tmp_path / "scripts/templates/world_interactive.html"
    template.parent.mkdir(parents=True)
    template.write_text("World", encoding="utf-8")
    sessions, calls = {}, []

    def handler(request):
        port = request.url.port
        calls.append((port, request.method, request.url.path))
        if request.url.path == "/status":
            return httpx.Response(200, json={"active": port in sessions})
        if request.url.path == "/sessions":
            sid = ("a" if port == 8769 else "b") * 32
            sessions[port] = sid
            return httpx.Response(202, json={"id": sid, "status": "initializing"})
        sid = request.url.path.split("/")[2]
        assert sessions[port] == sid, "An action was sent to the other photo worker"
        return httpx.Response(200, json={"id": sid, "status": "ready"})

    app = create_web(
        "token",
        tmp_path,
        transport=httpx.MockTransport(handler),
        workers=("http://127.0.0.1:8769", "http://127.0.0.1:8770"),
    )
    with TestClient(app) as client:
        client.get("/")
        a = client.post("/api/sessions", json=payload(), headers=HEADERS).json()["id"]
        b = client.post("/api/sessions", json=payload(), headers=HEADERS).json()["id"]
        assert a != b
        for sid in (a, b):
            assert (
                client.post(
                    f"/api/sessions/{sid}/steps",
                    json={"sequence": 0, "action": "forward"},
                    headers=HEADERS,
                ).status_code
                == 202
            )
            assert client.get(f"/api/sessions/{sid}").status_code == 200
        assert (
            client.post("/api/sessions", json=payload(), headers=HEADERS).status_code
            == 409
        )
        assert len([c for c in calls if c[1:] == ("POST", "/sessions")]) == 2
