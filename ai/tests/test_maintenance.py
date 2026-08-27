from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai import app as app_module
from ai.config import Settings, load_settings
from ai.index.embedding import EmbeddingProvider
from ai.maintenance import CacheMaintenanceService
from ai.provider_runtime import EmbeddingWarmupManager
from ai.schemas import CacheGcRequest, CacheQuotaRequest
from ai.storage import Database


def _file(path: Path, content: bytes = b"cache") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path.resolve()


def _old(path: Path) -> None:
    timestamp = time.time() - 7200
    os.utime(path, (timestamp, timestamp))


def _cache_fixture(tmp_path: Path) -> tuple[Database, Path, list[Path], Path]:
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    database.initialize()
    thumbnail = _file(data_dir / "thumbnails" / "album" / "photo.jpg")
    embedding = _file(data_dir / "embeddings" / "provider" / "album" / "photo.npy")
    descriptor = _file(
        data_dir / "faces" / "face-provider" / "album" / "descriptors" / "face.npy"
    )
    _file(data_dir / "faces" / "face-provider" / "album" / "thumbnails" / "face.jpg")
    orphans = [
        _file(data_dir / "thumbnails" / "album" / "orphan.jpg", b"one"),
        _file(data_dir / "embeddings" / "old" / "album" / "orphan.npy", b"two"),
        _file(data_dir / "faces" / "old" / "album" / "orphan.npy", b"three"),
    ]
    for orphan in orphans:
        _old(orphan)
    young = _file(data_dir / "embeddings" / "provider" / "album" / "young.npy")
    _file(data_dir / "models" / "model.bin", b"never scan models")
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO albums(id, name, source_path) VALUES ('album', 'Album', 'X')"
        )
        connection.execute(
            """
            INSERT INTO photos(
                id, album_id, absolute_path, thumbnail_path, embedding_path,
                face_provider, face_processed, face_count
            ) VALUES ('photo', 'album', 'X/photo.jpg', ?, ?, 'face-provider', 1, 1)
            """,
            (str(thumbnail), str(embedding)),
        )
        connection.execute(
            """
            INSERT INTO faces(id, photo_id, box_json, embedding_path)
            VALUES ('face', 'photo', '[0,0,10,10]', ?)
            """,
            (str(descriptor),),
        )
    return database, data_dir, orphans, young


def test_cache_gc_is_dry_run_by_default_and_never_scans_models(tmp_path: Path) -> None:
    database, data_dir, orphans, young = _cache_fixture(tmp_path)
    service = CacheMaintenanceService(database, data_dir)

    dry = service.collect(CacheGcRequest(min_age_seconds=3600))
    assert dry.dry_run is True
    assert dry.scanned_files == 8
    assert dry.referenced_files == 4
    assert dry.orphan_files == 3
    assert dry.young_orphan_files == 1
    assert dry.deleted_files == 0
    assert dry.orphan_bytes == sum(path.stat().st_size for path in orphans)
    assert all(path.exists() for path in orphans)

    applied = service.collect(CacheGcRequest(dry_run=False, min_age_seconds=3600))
    assert applied.deleted_files == 3
    assert applied.deleted_bytes == dry.orphan_bytes
    assert all(not path.exists() for path in orphans)
    assert young.exists()
    assert (data_dir / "models" / "model.bin").exists()
    history = service.list_runs(limit=10, offset=0)
    assert history.total == 2
    assert all(item.operation == "cache_gc" for item in history.items)
    assert all(item.status == "completed" for item in history.items)


def test_cache_gc_blocks_apply_while_a_job_is_active(tmp_path: Path) -> None:
    database, data_dir, _, _ = _cache_fixture(tmp_path)
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs(id, job_type, status, payload_json)
            VALUES ('job', 'prepare', 'queued', '{}')
            """
        )
    service = CacheMaintenanceService(database, data_dir)
    assert service.collect(CacheGcRequest()).dry_run is True
    with pytest.raises(ValueError, match="queued or running"):
        service.collect(CacheGcRequest(dry_run=False, min_age_seconds=0))
    history = service.list_runs(limit=10, offset=0)
    assert history.items[0].status == "failed"
    assert "queued or running" in (history.items[0].error or "")


def test_usage_and_quota_are_conservative_and_audited(tmp_path: Path) -> None:
    database, data_dir, orphans, _ = _cache_fixture(tmp_path)
    service = CacheMaintenanceService(database, data_dir)
    usage = service.usage()
    assert usage.categories["thumbnails"].files == 2
    assert usage.categories["embeddings"].files == 3
    assert usage.categories["faces"].files == 3
    assert usage.model_files == 1
    assert usage.database_bytes > 0
    assert usage.total_state_bytes == (
        usage.generated_bytes + usage.model_bytes + usage.database_bytes
    )

    impossible_budget = usage.model_bytes + usage.database_bytes
    dry = service.enforce_quota(
        CacheQuotaRequest(
            budget_bytes=impossible_budget,
            dry_run=True,
            min_age_seconds=0,
        )
    )
    assert dry.collection.orphan_files == 4
    assert dry.collection.deleted_files == 0
    assert dry.projected_total_state_bytes < dry.usage_before.total_state_bytes
    assert dry.satisfied is False
    assert any("cannot satisfy" in warning for warning in dry.warnings)
    assert all(path.exists() for path in orphans)

    applied = service.enforce_quota(
        CacheQuotaRequest(
            budget_bytes=dry.usage_before.total_state_bytes,
            dry_run=False,
            min_age_seconds=0,
        )
    )
    assert applied.collection.deleted_files == 4
    assert applied.usage_after.generated_bytes < applied.usage_before.generated_bytes
    assert service.list_runs(limit=10, offset=0).total == 2


def test_missing_quota_configuration_is_audited(tmp_path: Path) -> None:
    database = Database(tmp_path / "data" / "norma.db")
    service = CacheMaintenanceService(database, tmp_path / "data")
    with pytest.raises(ValueError, match="not configured"):
        service.enforce_quota(CacheQuotaRequest())
    run = service.list_runs(limit=1, offset=0).items[0]
    assert run.operation == "quota_enforce"
    assert run.status == "failed"
    assert "not configured" in (run.error or "")


def test_cache_gc_api_requires_explicit_non_dry_request(
    tmp_path: Path, monkeypatch
) -> None:
    database, data_dir, orphans, _ = _cache_fixture(tmp_path)
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(host="127.0.0.1", port=8765, data_dir=data_dir, log_level="INFO"),
    )

    with TestClient(app_module.app) as client:
        dry = client.post("/maintenance/cache/gc", json={"min_age_seconds": 3600})
        assert all(path.exists() for path in orphans)
        applied = client.post(
            "/maintenance/cache/gc",
            json={"dry_run": False, "min_age_seconds": 3600},
        )
        usage = client.get("/maintenance/cache/usage")
        enforced = client.post(
            "/maintenance/cache/enforce",
            json={"budget_bytes": 1, "dry_run": True, "min_age_seconds": 0},
        )
        history = client.get("/maintenance/runs")

    assert dry.status_code == 200, dry.text
    assert dry.json()["dry_run"] is True
    assert dry.json()["orphan_files"] == 3
    assert applied.status_code == 200, applied.text
    assert applied.json()["deleted_files"] == 3
    assert usage.status_code == 200
    assert usage.json()["generated_files"] == 5
    assert enforced.status_code == 200
    assert enforced.json()["satisfied"] is False
    assert history.status_code == 200
    assert history.json()["total"] == 3


class WarmableProvider(EmbeddingProvider):
    name = "warmable-v1"
    dimension = 2

    def __init__(self) -> None:
        self.loaded = False

    @property
    def model_backed(self) -> bool:
        return True

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @property
    def runtime_device(self) -> str | None:
        return "cpu" if self.loaded else None

    def embed_image(self, path: Path) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        self.loaded = True
        return np.asarray([1.0, 0.0], dtype=np.float32)


class BlockingWarmableProvider(WarmableProvider):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.calls = 0

    def warmup(self) -> None:
        self.calls += 1
        self.release.wait(timeout=2)
        self.loaded = True


def test_background_warmup_submission_is_idempotent() -> None:
    provider = BlockingWarmableProvider()
    manager = EmbeddingWarmupManager(lambda: provider)
    first = manager.submit()
    second = manager.submit()
    assert first.warmup_state == "loading"
    assert second.warmup_state == "loading"
    for _ in range(100):
        if provider.calls:
            break
        time.sleep(0.01)
    assert provider.calls == 1
    provider.release.set()
    for _ in range(100):
        status = manager.status()
        if status.warmup_state == "ready":
            break
        time.sleep(0.01)
    assert status.loaded is True
    assert status.finished_at is not None
    assert provider.calls == 1


def test_prewarm_environment_flag(monkeypatch) -> None:
    monkeypatch.setenv("NORMA_PREWARM_EMBEDDING", "yes")
    monkeypatch.setenv("NORMA_CACHE_BUDGET_GB", "1.5")
    settings = load_settings()
    assert settings.prewarm_embedding is True
    assert settings.cache_budget_bytes == round(1.5 * 1024**3)


def test_provider_status_and_warmup_endpoints(tmp_path: Path, monkeypatch) -> None:
    provider = WarmableProvider()
    data_dir = tmp_path / "data"
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(host="127.0.0.1", port=8765, data_dir=data_dir, log_level="INFO"),
    )
    monkeypatch.setattr(app_module, "embedding_provider", lambda: provider)

    with TestClient(app_module.app) as client:
        before = client.get("/providers/embedding/status")
        warmed = client.post("/providers/embedding/warmup")
        for _ in range(20):
            after = client.get("/providers/embedding/status")
            if after.json()["warmup_state"] == "ready":
                break
            time.sleep(0.01)

    assert before.status_code == 200
    assert before.json()["loaded"] is False
    assert before.json()["device"] is None
    assert before.json()["warmup_state"] == "idle"
    assert warmed.status_code == 202
    assert warmed.json()["warmup_state"] in {"loading", "ready"}
    assert after.json()["loaded"] is True
    assert after.json()["device"] == "cpu"
    assert after.json()["warmup_state"] == "ready"


def test_startup_prewarm_uses_background_manager(tmp_path: Path, monkeypatch) -> None:
    provider = WarmableProvider()
    data_dir = tmp_path / "data"
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(
            host="127.0.0.1",
            port=8765,
            data_dir=data_dir,
            log_level="INFO",
            prewarm_embedding=True,
        ),
    )
    monkeypatch.setattr(app_module, "embedding_provider", lambda: provider)

    with TestClient(app_module.app) as client:
        for _ in range(20):
            status = client.get("/providers/embedding/status").json()
            if status["warmup_state"] == "ready":
                break
            time.sleep(0.01)

    assert status["loaded"] is True
    assert status["warmup_state"] == "ready"
