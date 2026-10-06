from pathlib import Path

import pytest

from scripts import install_demo_models
from scripts.reproduce_demo import image_hashes, wait_job


def test_photo_inventory_tracks_content_and_ignores_other_files(tmp_path):
    (tmp_path / "a.JPG").write_bytes(b"image")
    (tmp_path / "note.txt").write_text("notes")
    first = image_hashes(tmp_path)
    assert list(first) == ["a.JPG"]
    (tmp_path / "a.JPG").write_bytes(b"changed")
    assert image_hashes(tmp_path) != first


def test_offline_model_check_verifies_both_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(install_demo_models, "load_pinned_openclip_manifest", lambda: ({}, ""))
    checked = []
    monkeypatch.setattr(install_demo_models, "verify_pinned_openclip_cache", checked.append)
    monkeypatch.setattr(install_demo_models, "WEIGHTS", {"a": ("a.pth", "a"), "b": ("b.pth", "b")})
    monkeypatch.setattr(install_demo_models, "file_sha256", lambda path: Path(path).stem)
    install_demo_models.install(tmp_path, offline=True)
    assert checked == [tmp_path / "openclip"]
    monkeypatch.setattr(install_demo_models, "file_sha256", lambda path: "wrong")
    with pytest.raises(ValueError, match="checksum"):
        install_demo_models.install(tmp_path, offline=True)


def test_completed_job_needs_no_poll():
    job = {"status": "completed", "id": "one"}
    assert wait_job(None, job, "/jobs", 1) is job


def test_failed_job_is_not_reported_as_success():
    with pytest.raises(RuntimeError, match="model failed"):
        wait_job(None, {"status": "failed", "error": "model failed"}, "/jobs", 1)


def test_timeout_requests_job_cancellation():
    class Client:
        def post(self, path):
            self.path = path
    client = Client()
    with pytest.raises(TimeoutError):
        wait_job(client, {"status": "queued", "id": "one"}, "/jobs", -1)
    assert client.path == "/jobs/one/cancel"
