"""Session protocol tests with an explicitly fake renderer, not model evidence."""

import base64
import io
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ai.exploration.server import create_app

TOKEN = "test-world-token-not-a-real-secret-12345"
AUTH = {"Authorization": "Bearer " + TOKEN}


class FakeRuntime:
    def __init__(self):
        self.steps = []
        self.closed = 0
        self.gate = None
        self.failure = False

    def begin(self, image, prompt, folder, progress):
        self.folder = folder
        progress("encoding_input")
        return {"max_steps": 7}

    def step(self, action, cancelled, progress):
        if self.gate:
            self.gate.wait(2)
        if cancelled():
            raise InterruptedError()
        if self.failure:
            raise RuntimeError("Test-only failure")
        index = len(self.steps)
        self.steps.append(action)
        name = f"step-{index:03d}.mp4"
        (self.folder / name).write_bytes(b"fake-protocol-test-not-video")
        return {"sequence": index, "action": action, "artifact": name}

    def close(self):
        self.closed += 1


def payload():
    output = io.BytesIO()
    Image.new("RGB", (64, 64)).save(output, "PNG")
    return dict(
        image_base64=base64.b64encode(output.getvalue()).decode(),
        prompt="A landscape",
        upload_confirmed=True,
    )


def wait(client, sid, desired="ready"):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        record = client.get(f"/sessions/{sid}", headers=AUTH).json()
        if record["status"] == desired:
            return record
        time.sleep(0.01)
    raise AssertionError(record)


@pytest.fixture
def worker(tmp_path):
    runtime = FakeRuntime()
    with TestClient(create_app(tmp_path, runtime, TOKEN)) as client:
        yield client, runtime


def start(client):
    result = client.post("/sessions", json=payload(), headers=AUTH)
    assert result.status_code == 202
    sid = result.json()["id"]
    wait(client, sid)
    return sid


def test_auth_and_no_body_echo(worker):
    client, _ = worker
    assert client.get("/status").status_code == 401
    assert (
        client.post(
            "/sessions", json={"image_base64": "PRIVATE_SENTINEL"}, headers=AUTH
        ).status_code
        == 422
    )
    assert (
        "PRIVATE_SENTINEL"
        not in client.post(
            "/sessions", json={"image_base64": "PRIVATE_SENTINEL"}, headers=AUTH
        ).text
    )


@pytest.mark.parametrize(
    "change",
    [
        {"upload_confirmed": False},
        {"upload_confirmed": 1},
        {"prompt": " "},
        {"image_base64": "bad"},
        {"extra": True},
    ],
)
def test_new_session_validation(worker, change):
    client, _ = worker
    assert (
        client.post("/sessions", json=payload() | change, headers=AUTH).status_code
        == 422
    )


def test_single_session_sequence_and_replay(worker):
    client, runtime = worker
    sid = start(client)
    assert client.post("/sessions", json=payload(), headers=AUTH).status_code == 409
    url = f"/sessions/{sid}/steps"
    assert (
        client.post(
            url, json={"sequence": 1, "action": "right"}, headers=AUTH
        ).status_code
        == 409
    )
    assert (
        client.post(
            url, json={"sequence": 0, "action": "forward"}, headers=AUTH
        ).status_code
        == 202
    )
    assert wait(client, sid)["next_sequence"] == 1
    assert (
        client.post(
            url, json={"sequence": 0, "action": "forward"}, headers=AUTH
        ).status_code
        == 202
    )
    assert runtime.steps == ["forward"]
    assert (
        client.post(
            url, json={"sequence": 0, "action": "right"}, headers=AUTH
        ).status_code
        == 409
    )
    assert (
        client.post(
            url, json={"sequence": 1, "action": "right"}, headers=AUTH
        ).status_code
        == 202
    )
    assert wait(client, sid)["next_sequence"] == 2
    assert runtime.steps == ["forward", "right"]
    assert (
        client.get(f"/sessions/{sid}/artifacts/step-000.mp4", headers=AUTH).status_code
        == 200
    )
    assert (
        client.get(f"/sessions/{sid}/artifacts/step-006.mp4", headers=AUTH).status_code
        == 404
    )
    assert (
        client.get(f"/sessions/{sid}/artifacts/source.png", headers=AUTH).status_code
        == 404
    )


@pytest.mark.parametrize(
    "step",
    [
        {"sequence": True, "action": "forward"},
        {"sequence": 7, "action": "forward"},
        {"sequence": 0, "action": "jump"},
        {"sequence": 0, "action": "right", "arbitrary_path": "x"},
    ],
)
def test_step_validation(worker, step):
    client, _ = worker
    sid = start(client)
    assert (
        client.post(f"/sessions/{sid}/steps", json=step, headers=AUTH).status_code
        == 422
    )


def test_busy_and_cancel_discard_mutated_state(worker):
    client, runtime = worker
    sid = start(client)
    runtime.gate = threading.Event()
    client.post(
        f"/sessions/{sid}/steps", json={"sequence": 0, "action": "right"}, headers=AUTH
    )
    assert (
        client.post(
            f"/sessions/{sid}/steps",
            json={"sequence": 1, "action": "left"},
            headers=AUTH,
        ).status_code
        == 409
    )
    client.post(f"/sessions/{sid}/close", headers=AUTH)
    runtime.gate.set()
    assert wait(client, sid, "closed")["history"] == []
    assert runtime.closed == 1
    start(client)


def test_generation_failure_closes_cache(worker):
    client, runtime = worker
    sid = start(client)
    runtime.failure = True
    client.post(
        f"/sessions/{sid}/steps", json={"sequence": 0, "action": "right"}, headers=AUTH
    )
    record = wait(client, sid, "failed")
    assert runtime.closed == 1 and record["history"] == []
    assert "Test-only failure" not in record["error"]


def test_idle_expiration(worker):
    client, _ = worker
    sid = start(client)
    client.app.state.sessions.records[sid]["last_action"] = time.time() - 601
    client.get("/status", headers=AUTH)
    wait(client, sid, "closed")
