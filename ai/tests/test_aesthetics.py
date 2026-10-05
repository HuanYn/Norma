from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from ai.aesthetics.api import create_aesthetics_router
from ai.aesthetics.jobs import AestheticsJobManager
from ai.aesthetics.provider import (
    AssessmentScores,
    AestheticsProviderUnavailableError,
    PyiqaMusiqProvider,
    file_sha256,
)
from ai.aesthetics.service import (
    AestheticsCancelledError,
    AestheticsService,
    AestheticsSourceChangedError,
)
from ai.storage import Database


class FakeProvider:
    fingerprint = "test-only:fake-v1"

    def __init__(self):
        self.calls = []

    def status(self):
        return {"configured": True, "model_backed": False, "provider": self.fingerprint}

    def score(self, path):
        self.calls.append(path)
        return AssessmentScores(72.5, 6.5, "test-technical", "test-aesthetic")


@pytest.fixture
def fixture(tmp_path):
    database = Database(tmp_path / "data" / "norma.db")
    database.initialize()
    album = tmp_path / "photos"
    album.mkdir()
    paths = []
    with database.connect() as connection:
        connection.execute("INSERT INTO albums(id,name,source_path) VALUES ('album','Test',?)", (str(album),))
        for index in range(2):
            path = album / f"{index}.jpg"
            Image.new("RGB", (32, 32), (index * 100, 20, 30)).save(path)
            paths.append(path)
            stat = path.stat()
            connection.execute(
                """INSERT INTO photos(id,album_id,absolute_path,file_size,source_mtime_ns,metadata_json)
                   VALUES (?, 'album', ?, ?, ?, ?)""",
                (f"photo-{index}", str(path), stat.st_size, stat.st_mtime_ns,
                 json.dumps({"source_sha256": file_sha256(path)})),
            )
    provider = FakeProvider()
    service = AestheticsService(database, provider)
    service.initialize()
    return database, paths, provider, service


def test_import_is_optional_and_does_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import ai.aesthetics; assert 'torch' not in sys.modules; assert 'pyiqa' not in sys.modules"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_provider_reports_missing_models_without_downloading(tmp_path):
    provider = PyiqaMusiqProvider(tmp_path)
    state = provider.status()
    assert not state["configured"]
    assert state["automatic_download"] is False
    assert len(state["missing_weights"]) == 2
    with pytest.raises(AestheticsProviderUnavailableError, match="not configured"):
        provider.warmup()
    assert list(tmp_path.iterdir()) == []


def test_pinned_weight_integrity_and_explicit_ava_head(tmp_path, monkeypatch):
    from ai.aesthetics import provider as module
    weights = {}
    for metric in ("musiq", "musiq-ava"):
        filename = metric + ".pth"
        data = metric.encode()
        (tmp_path / filename).write_bytes(data)
        weights[metric] = filename, hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(module, "WEIGHTS", weights)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.1.16")
    calls = []
    monkeypatch.setitem(sys.modules, "pyiqa", SimpleNamespace(create_metric=lambda name, **kw: calls.append((name, kw)) or object()))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda v: v))
    provider = PyiqaMusiqProvider(tmp_path)
    provider.warmup()
    provider.warmup()
    assert len(calls) == 2
    assert calls[0][1]["num_class"] == 1
    assert calls[1][1]["num_class"] == 10
    assert all(call[1]["pretrained"] is False for call in calls)
    assert provider.status()["loaded"] is True
    (tmp_path / "musiq.pth").write_bytes(b"corrupted")
    with pytest.raises(AestheticsProviderUnavailableError, match="integrity"):
        PyiqaMusiqProvider(tmp_path).warmup()


def test_cache_progress_separate_scales_and_unchanged_originals(fixture):
    database, paths, provider, service = fixture
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    progress = []
    first = service.analyze("album", on_progress=lambda d, t: progress.append((d, t)))
    assert first["computed_count"] == 2
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert first["items"][0]["technical_quality"] == 72.5
    assert first["items"][0]["aesthetic_quality"] == 6.5
    second = service.analyze("album")
    assert second["reused_count"] == 2
    assert len(provider.calls) == 2
    assert service.cached("album")["scored_count"] == 2
    assert before == [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    assert service.analyze("album", force=True)["computed_count"] == 2
    assert database.current_version() == 15


def test_provider_change_and_corrupt_cache_recompute(fixture):
    database, _, provider, service = fixture
    service.analyze("album")
    provider.fingerprint = "test-only:v2"
    assert service.cached("album")["scored_count"] == 0
    assert service.analyze("album")["computed_count"] == 2
    with database.connect() as connection:
        connection.execute("UPDATE aesthetics_scores_v1 SET scores_json='not-json' WHERE photo_id='photo-0'")
    assert service.analyze("album")["computed_count"] == 1


def _mutate_preserving_stat(path):
    stat = path.stat()
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_same_size_mtime_content_drift_is_never_current(fixture):
    _, paths, _, service = fixture
    service.analyze("album")
    _mutate_preserving_stat(paths[0])
    current = service.cached("album")
    assert current["scored_count"] == 1
    assert current["stale_photo_ids"] == ["photo-0"]
    with pytest.raises(AestheticsSourceChangedError, match="changed"):
        service.analyze("album")


def test_mutation_during_inference_is_not_committed(fixture):
    database, paths, provider, service = fixture
    def mutate(path):
        _mutate_preserving_stat(path)
        return AssessmentScores(80, 6)
    provider.score = mutate
    with pytest.raises(AestheticsSourceChangedError):
        service.analyze("album")
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM aesthetics_scores_v1").fetchone()[0] == 0


def test_album_membership_change_is_not_reported_complete(fixture):
    database, _, _, service = fixture
    def progress(done, total):
        if done == total:
            with database.connect() as connection:
                connection.execute("DELETE FROM photos WHERE id='photo-0'")
    with pytest.raises(AestheticsSourceChangedError, match="membership"):
        service.analyze("album", on_progress=progress)


def test_cancellation_preserves_completed_photos(fixture):
    _, _, _, service = fixture
    done = []
    with pytest.raises(AestheticsCancelledError):
        service.analyze("album", on_progress=lambda n, t: done.append(n), should_cancel=lambda: done[-1] == 1)
    assert service.cached("album")["scored_count"] == 1
    assert service.analyze("album")["reused_count"] == 1


@pytest.mark.parametrize("scores", [(float("nan"), 4), (80, float("inf")), (80, 0)])
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        AssessmentScores(*scores)


def _wait(manager, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job.status in {"completed", "cancelled", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("assessment job did not finish")


def test_job_and_router_end_to_end_with_test_provider(fixture):
    _, _, _, service = fixture
    manager = AestheticsJobManager(service)
    manager.start()
    app = FastAPI()
    app.include_router(create_aesthetics_router(lambda: service, lambda: manager))
    try:
        with TestClient(app) as client:
            assert client.get("/aesthetics/status").status_code == 200
            assert client.get("/albums/missing/aesthetics").status_code == 404
            assert client.post("/albums/album/aesthetics/jobs", json={"force":"false"}).status_code == 422
            response = client.post("/albums/album/aesthetics/jobs", json={})
            assert response.status_code == 202, response.text
            job_id = response.json()["id"]
            job = _wait(manager, job_id)
            assert job.status == "completed", job.error
            assert job.progress == 1
            assert job.result["computed_count"] == 2
            assert client.get(f"/aesthetics/jobs/{job_id}").status_code == 200
            assert client.get("/albums/album/aesthetics").json()["scored_count"] == 2
    finally:
        manager.shutdown()


def test_job_cancel_and_duplicate_guard(fixture):
    _, _, provider, service = fixture
    entered, release = threading.Event(), threading.Event()
    def slow_score(path):
        entered.set()
        assert release.wait(5)
        return AssessmentScores(70, 5)
    provider.score = slow_score
    manager = AestheticsJobManager(service)
    manager.start()
    try:
        job = manager.submit("album")
        assert entered.wait(5)
        with pytest.raises(ValueError, match="already active"):
            manager.submit("album")
        manager.cancel(job.id)
        release.set()
        assert _wait(manager, job.id).status == "cancelled"
        assert service.cached("album")["scored_count"] == 0
    finally:
        release.set()
        manager.shutdown()


def test_interrupted_jobs_are_visible_and_not_auto_resumed(fixture):
    database, _, provider, service = fixture
    with database.connect() as connection:
        connection.execute("""INSERT INTO jobs(id,job_type,status,stage,payload_json)
            VALUES ('old','analyze_aesthetics','running','scoring_images:1/2','{}')""")
    manager = AestheticsJobManager(service)
    manager.start()
    try:
        job = manager.get("old")
        assert job.status == "failed"
        assert job.stage == "interrupted"
        assert provider.calls == []
    finally:
        manager.shutdown()
