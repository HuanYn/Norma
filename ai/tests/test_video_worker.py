from __future__ import annotations

import base64
import io
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from ai.video.client import VideoWorkerClient
from ai.video.generator import GenerationCancelled, VideoModelUnavailable, WanGenerator
from ai.video.models import VideoOptions, decode_image
from ai.video.worker import JobManager, WorkerSettings, create_app


def picture() -> bytes:
    content = io.BytesIO()
    Image.new("RGB", (96, 64), "blue").save(content, "JPEG")
    return content.getvalue()


def payload(**changes):
    result = {"image_base64": base64.b64encode(picture()).decode(),
              "prompt": "Slow camera push in", "upload_confirmed": True}
    result.update(changes)
    return result


class FakeGenerator:
    loaded = False

    def __init__(self):
        self.calls = 0

    def generate(self, image, options, output, progress, cancelled):
        self.calls += 1
        progress("loading_model", None, None)
        progress("denoising", 1, options.num_inference_steps)
        # Explicit unit-test fixture, not a model output or playable demo video.
        output.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 40)
        return {"fixture": True, "fps": 24, "frame_count": options.num_frames}


def wait_terminal(client, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = client.get(f"/v1/video/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Test worker did not finish")


def test_lazy_health_never_loads_model(tmp_path):
    model = WanGenerator()
    with TestClient(create_app(WorkerSettings(tmp_path), generator=model)) as client:
        result = client.get("/health").json()
        assert result["model_loaded"] is False
        assert result["downloads_allowed"] is False
        assert "not proof" in result["note"]
    assert model.loaded is False


def test_submit_persist_generate_download_and_restart(tmp_path):
    fake = FakeGenerator()
    with TestClient(create_app(WorkerSettings(tmp_path), generator=fake)) as client:
        response = client.post("/v1/video/jobs", json=payload())
        assert response.status_code == 202
        job_id = response.json()["id"]
        final = wait_terminal(client, job_id)
        assert final["status"] == "completed"
        assert final["result"]["fixture"] is True
        assert final["step_percent"] == 5
        assert final["metadata_removed"] is True
        artifact = client.get(f"/v1/video/jobs/{job_id}/artifact")
        assert artifact.status_code == 200
        assert artifact.headers["content-type"] == "video/mp4"
        assert "image_base64" not in (tmp_path / job_id / "job.json").read_text()
        assert final["options"]["num_frames"] == 49
    manager = JobManager(WorkerSettings(tmp_path), generator=fake)
    assert manager.get(job_id)["status"] == "completed"
    assert fake.calls == 1
    manager.close()


@pytest.mark.parametrize("changes", [
    {"upload_confirmed": False}, {"upload_confirmed": 1}, {"prompt": "  "},
    {"width": 640, "height": 390}, {"num_frames": 48}, {"seed": -1},
    {"num_inference_steps": 500}, {"guidance_scale": 20},
    {"width": 1280, "height": 1280}, {"image_base64": "wrong@"},
    {"image_url": "https://example.com/private.jpg"}, {"image_path": "/etc/passwd"},
    {"fps": 100}, {"prompt": "text\x00bad"},
])
def test_invalid_requests_never_run_or_echo_image(tmp_path, changes):
    fake = FakeGenerator()
    with TestClient(create_app(WorkerSettings(tmp_path), generator=fake)) as client:
        request = payload(**changes)
        response = client.post("/v1/video/jobs", json=request)
        assert response.status_code == 422
        assert picture().hex() not in response.text
        assert request["image_base64"] not in response.text
    assert fake.calls == 0


def test_confirmation_required(tmp_path):
    request = payload()
    del request["upload_confirmed"]
    with TestClient(create_app(WorkerSettings(tmp_path), generator=FakeGenerator())) as client:
        assert client.post("/v1/video/jobs", json=request).status_code == 422


def test_authentication_covers_all_endpoints(tmp_path):
    token = "a-private-test-token-with-enough-length"
    with TestClient(create_app(WorkerSettings(tmp_path, token=token), generator=FakeGenerator())) as client:
        assert client.get("/health").status_code == 401
        assert client.post("/v1/video/jobs", json=payload()).status_code == 401
        assert client.get("/v1/video/jobs/" + "0" * 32).status_code == 401
        assert client.post("/v1/video/jobs/" + "0" * 32 + "/cancel").status_code == 401
        assert client.get("/v1/video/jobs/" + "0" * 32 + "/artifact").status_code == 401
        assert client.get("/health", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_model_error_is_explicit_no_artifact(tmp_path):
    class Missing:
        def generate(self, *args):
            raise VideoModelUnavailable("CUDA GPU unavailable; video generation has not run")
    with TestClient(create_app(WorkerSettings(tmp_path), generator=Missing())) as client:
        job_id = client.post("/v1/video/jobs", json=payload()).json()["id"]
        final = wait_terminal(client, job_id)
        assert final["status"] == "failed"
        assert "CUDA GPU unavailable" in final["error"]
        assert client.get(f"/v1/video/jobs/{job_id}/artifact").status_code == 409


def test_exception_does_not_leak_secrets(tmp_path):
    class Broken:
        def generate(self, *args):
            raise RuntimeError("secret-token /private/path")
    with TestClient(create_app(WorkerSettings(tmp_path), generator=Broken())) as client:
        job_id = client.post("/v1/video/jobs", json=payload()).json()["id"]
        final = wait_terminal(client, job_id)
        assert final["status"] == "failed"
        assert "secret-token" not in json.dumps(final)


def test_bounded_queue_cancel_pending_and_running(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class Blocking:
        def generate(self, image, options, output, progress, cancelled):
            entered.set()
            while not release.wait(0.01):
                if cancelled.is_set():
                    raise GenerationCancelled()

    with TestClient(create_app(WorkerSettings(tmp_path, max_pending=2), generator=Blocking())) as client:
        first = client.post("/v1/video/jobs", json=payload()).json()["id"]
        assert entered.wait(1)
        second = client.post("/v1/video/jobs", json=payload()).json()["id"]
        assert client.post("/v1/video/jobs", json=payload()).status_code == 429
        assert client.get(f"/v1/video/jobs/{first}").json()["step_percent"] is None
        assert client.post(f"/v1/video/jobs/{second}/cancel").json()["status"] == "cancelled"
        running = client.post(f"/v1/video/jobs/{first}/cancel").json()
        assert running["cancel_requested"] is True
        assert wait_terminal(client, first)["status"] == "cancelled"
        release.set()


def test_restart_marks_unfinished_interrupted_without_rerun(tmp_path):
    job_id = "a" * 32
    folder = tmp_path / job_id
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps({"id": job_id, "status": "running"}))
    fake = FakeGenerator()
    with TestClient(create_app(WorkerSettings(tmp_path), generator=fake)) as client:
        response = client.get(f"/v1/video/jobs/{job_id}").json()
        assert response["status"] == "interrupted"
        assert "resubmit" in response["error"]
    assert fake.calls == 0


def test_retention_cap_and_unknown_job(tmp_path):
    with TestClient(create_app(WorkerSettings(tmp_path, max_jobs=1), generator=FakeGenerator())) as client:
        job_id = client.post("/v1/video/jobs", json=payload()).json()["id"]
        wait_terminal(client, job_id)
        assert client.post("/v1/video/jobs", json=payload()).status_code == 507
        assert client.get("/v1/video/jobs/" + "0" * 32).status_code == 404
        assert client.get("/v1/video/jobs/not-an-id").status_code == 404


def test_body_limit_before_json_decode(tmp_path):
    with TestClient(create_app(WorkerSettings(tmp_path), generator=FakeGenerator())) as client:
        assert client.post("/v1/video/jobs", content=b"{}", headers={"Content-Length": "999999999"}).status_code == 413
        assert client.post("/v1/video/jobs", content=b"{}", headers={"Content-Length": "bad"}).status_code == 400


def test_decode_removes_metadata():
    exif = Image.Exif()
    exif[270] = "private description"
    data = io.BytesIO()
    Image.new("RGB", (60, 80), "red").save(data, "JPEG", exif=exif)
    result = decode_image(base64.b64encode(data.getvalue()).decode())
    assert result.getexif() == {}
    assert result.info == {}


def test_client_requires_confirmation_before_network(monkeypatch):
    client = VideoWorkerClient("http://127.0.0.1:8766")
    monkeypatch.setattr(client, "_request", lambda *args: pytest.fail("Unexpected network call"))
    with pytest.raises(ValueError, match="Confirm"):
        client.submit(picture(), VideoOptions(prompt="Pan left"))


def test_client_sends_clean_image_only(monkeypatch):
    client = VideoWorkerClient("http://127.0.0.1:8766")
    sent = []
    monkeypatch.setattr(client, "_request", lambda *args: sent.append(args) or {"id": "a" * 32})
    client.submit(picture(), VideoOptions(prompt="Pan left"), upload_confirmed=True)
    method, path, request = sent[0]
    assert method == "POST" and path == "/v1/video/jobs"
    assert request["upload_confirmed"] is True
    assert decode_image(request["image_base64"]).info == {}
    assert "image_path" not in request


@pytest.mark.parametrize("url", ["http://10.0.0.1:8766", "file:///tmp", "https://a:b@host", "https://host/path", "https://host?token=x"])
def test_client_rejects_unsafe_origins(url):
    with pytest.raises(ValueError):
        VideoWorkerClient(url)


def test_token_settings_and_option_bounds(tmp_path):
    with pytest.raises(ValueError):
        WorkerSettings(tmp_path, token="short")
    with pytest.raises(ValueError):
        WanGenerator(offload="arbitrary")
    with pytest.raises(ValidationError):
        VideoOptions(prompt="Pan left", num_frames="49")
