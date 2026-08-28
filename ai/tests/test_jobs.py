from __future__ import annotations

import threading
import time
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from ai import app as app_module
from ai.config import Settings
from ai.index import AlbumIndexer
from ai.jobs import PrepareJobManager
from ai.people import PeopleIndexer
from ai.retrieval import RetrievalService
from ai.storage import Database


def _photo(path: Path, color: tuple[int, int, int], offset: int) -> None:
    image = Image.new("RGB", (560, 380), color)
    draw = ImageDraw.Draw(image)
    for x in range(25 + offset, 535, 60):
        draw.rectangle((x, 25, x + 13, 355), fill=(225, 175, 75))
    image.save(path, "JPEG", quality=93)


def _configure(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    folder = tmp_path / "album"
    folder.mkdir()
    _photo(folder / "night.jpg", (7, 15, 42), 0)
    _photo(folder / "green.jpg", (35, 105, 48), 5)
    _photo(folder / "gold.jpg", (205, 155, 62), 10)
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
            face_provider="opencv-haar",
        ),
    )
    return folder, data_dir


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 10) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/jobs/{job_id}").json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.03)
    raise AssertionError(f"job did not finish within {timeout} seconds")


def test_prepare_job_persists_progress_result_and_catalog(
    tmp_path: Path, monkeypatch
) -> None:
    folder, _ = _configure(tmp_path, monkeypatch)

    with TestClient(app_module.app) as client:
        created = client.post(
            "/jobs/prepare",
            json={
                "folder": str(folder),
                "name": "Background Trip",
                "include_people": False,
            },
        )
        assert created.status_code == 202, created.text
        job_id = created.json()["id"]
        completed = _wait_for_terminal(client, job_id)
        jobs = client.get("/jobs?status=completed")
        albums = client.get("/albums")

    assert completed["status"] == "completed", completed
    assert completed["stage"] == "completed"
    assert completed["progress"] == 1
    assert completed["result"]["album"]["total"] == 3
    assert completed["result"]["embedding"]["count"] == 3
    assert completed["result"]["embedding"]["computed_count"] == 3
    assert completed["result"]["embedding"]["reused_count"] == 0
    assert completed["result"]["people"] is None
    assert completed["started_at"] is not None
    assert completed["finished_at"] is not None
    assert jobs.json()["total"] == 1
    assert albums.json()["items"][0]["embedded_count"] == 3

    with TestClient(app_module.app) as client:
        repeated = client.post(
            "/jobs/prepare",
            json={"folder": str(folder), "include_people": False},
        )
        repeated_job = _wait_for_terminal(client, repeated.json()["id"])
    assert repeated_job["status"] == "completed"
    assert repeated_job["result"]["embedding"]["computed_count"] == 0
    assert repeated_job["result"]["embedding"]["reused_count"] == 3


def test_prepare_flags_run_base_quality_embeddings_and_people_on_demand(
    tmp_path: Path, monkeypatch
) -> None:
    folder, _ = _configure(tmp_path, monkeypatch)

    with TestClient(app_module.app) as client:
        imported = client.post(
            "/jobs/prepare",
            json={
                "folder": str(folder),
                "include_quality": False,
                "include_embeddings": False,
                "include_people": False,
            },
        )
        import_job = _wait_for_terminal(client, imported.json()["id"])
        album_id = import_job["result"]["album"]["album_id"]
        after_import = client.get(f"/albums/{album_id}").json()

        embedding = client.post(
            "/jobs/prepare",
            json={
                "folder": str(folder),
                "include_quality": False,
                "include_embeddings": True,
                "include_people": False,
            },
        )
        embedding_job = _wait_for_terminal(client, embedding.json()["id"])
        after_embedding = client.get(f"/albums/{album_id}").json()
        semantic_without_quality = client.post(
            "/albums/search",
            json={"album_id": album_id, "query": "night", "limit": 3},
        )

        people = client.post(
            "/jobs/prepare",
            json={
                "folder": str(folder),
                "include_quality": False,
                "include_embeddings": False,
                "include_people": True,
            },
        )
        people_job = _wait_for_terminal(client, people.json()["id"])
        after_people = client.get(f"/albums/{album_id}").json()

        quality = client.post(
            "/jobs/prepare",
            json={
                "folder": str(folder),
                "include_quality": True,
                "include_embeddings": False,
                "include_people": False,
            },
        )
        quality_job = _wait_for_terminal(client, quality.json()["id"])
        after_quality = client.get(f"/albums/{album_id}").json()

    assert import_job["status"] == "completed"
    assert import_job["progress"] == 1
    assert import_job["payload"]["include_quality"] is False
    assert import_job["payload"]["include_embeddings"] is False
    assert import_job["result"]["embedding"] is None
    assert import_job["result"]["people"] is None
    assert after_import["photo_count"] == 3
    assert after_import["quality_count"] == 0
    assert after_import["similar_group_count"] == 0
    assert after_import["embedded_count"] == 0
    assert after_import["people_processed_count"] == 0

    assert embedding_job["status"] == "completed"
    assert embedding_job["result"]["embedding"]["count"] == 3
    assert embedding_job["result"]["people"] is None
    assert after_embedding["quality_count"] == 0
    assert after_embedding["embedded_count"] == 3
    assert semantic_without_quality.status_code == 200
    assert all(
        match["quality_score"] is None
        for match in semantic_without_quality.json()["matches"]
    )

    assert people_job["status"] == "completed"
    assert people_job["result"]["embedding"] is None
    assert people_job["result"]["people"]["album_id"] == album_id
    assert after_people["quality_count"] == 0
    assert after_people["embedded_count"] == 3
    assert after_people["people_processed_count"] == 3

    assert quality_job["status"] == "completed"
    assert quality_job["result"]["embedding"] is None
    assert quality_job["result"]["people"] is None
    assert quality_job["result"]["album"]["computed_count"] == 3
    assert after_quality["quality_count"] == 3
    assert after_quality["embedded_count"] == 3
    assert after_quality["people_processed_count"] == 3


def test_cancel_after_index_commit_keeps_album_result_for_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    folder, _ = _configure(tmp_path, monkeypatch)
    committed = threading.Event()
    release = threading.Event()
    original_index = AlbumIndexer.index

    def index_then_pause(
        self,
        folder_path,
        album_name=None,
        *,
        analyze_quality=True,
        on_progress=None,
        should_cancel=None,
    ):
        result = original_index(
            self,
            folder_path,
            album_name,
            analyze_quality=analyze_quality,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        committed.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release committed index")
        return result

    monkeypatch.setattr(AlbumIndexer, "index", index_then_pause)
    try:
        with TestClient(app_module.app) as client:
            created = client.post(
                "/jobs/prepare",
                json={
                    "folder": str(folder),
                    "include_quality": False,
                    "include_embeddings": False,
                    "include_people": False,
                },
            )
            job_id = created.json()["id"]
            assert committed.wait(timeout=3)
            cancelled = client.post(f"/jobs/{job_id}/cancel")
            release.set()
            terminal = _wait_for_terminal(client, job_id)
    finally:
        release.set()

    assert cancelled.status_code == 200
    assert terminal["status"] == "cancelled"
    assert terminal["result"]["album"]["total"] == 3
    assert terminal["result"]["album"]["album_id"]


@pytest.mark.parametrize("stage", ["embedding", "people"])
def test_cancel_after_analysis_commit_keeps_final_stage_result(
    tmp_path: Path,
    monkeypatch,
    stage: str,
) -> None:
    folder, _ = _configure(tmp_path, monkeypatch)
    committed = threading.Event()
    release = threading.Event()
    target = RetrievalService if stage == "embedding" else PeopleIndexer
    original = target.embed_album if stage == "embedding" else target.index

    def analyze_then_pause(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        committed.set()
        if not release.wait(timeout=5):
            raise TimeoutError(f"test did not release committed {stage}")
        return result

    method = "embed_album" if stage == "embedding" else "index"
    monkeypatch.setattr(target, method, analyze_then_pause)
    payload = {
        "folder": str(folder),
        "include_quality": False,
        "include_embeddings": stage == "embedding",
        "include_people": stage == "people",
    }
    try:
        with TestClient(app_module.app) as client:
            created = client.post("/jobs/prepare", json=payload)
            job_id = created.json()["id"]
            assert committed.wait(timeout=5)
            client.post(f"/jobs/{job_id}/cancel")
            release.set()
            terminal = _wait_for_terminal(client, job_id)
    finally:
        release.set()

    assert terminal["status"] == "cancelled"
    assert terminal["result"]["album"]["total"] == 3
    assert terminal["result"][stage] is not None


def test_job_progress_writes_are_throttled_monotonic_and_finish_at_one(
    tmp_path: Path, monkeypatch
) -> None:
    manager = PrepareJobManager(
        Database(tmp_path / "data" / "norma.db"),
        tmp_path / "data",
        "lightweight",
        "opencv-haar",
    )
    writes: list[dict[str, object]] = []

    def record_stage(_job_id: str, **values) -> None:
        writes.append(values)

    monkeypatch.setattr(manager, "_set_stage", record_stage)
    try:
        manager._begin_progress("job", "indexing")
        for completed in range(1, 201):
            manager._indexing_progress(
                "job",
                completed,
                200,
                start=0.0,
                span=1.0,
            )
    finally:
        manager.shutdown()

    progresses = [float(write["progress"]) for write in writes]
    assert len(writes) < 40
    assert progresses == sorted(progresses)
    assert progresses[-1] == 1.0
    assert writes[-1]["result"] == {
        "indexing_progress": {"completed": 200, "total": 200}
    }


def test_prepare_job_rejects_duplicate_and_cancels_between_stages(
    tmp_path: Path, monkeypatch
) -> None:
    folder, _ = _configure(tmp_path, monkeypatch)
    started = threading.Event()
    release = threading.Event()
    original_index = AlbumIndexer.index

    def slow_index(
        self,
        folder_path,
        album_name=None,
        *,
        on_progress=None,
        should_cancel=None,
    ):
        if on_progress is not None:
            on_progress(1, 2)
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release index stage")
        return original_index(
            self,
            folder_path,
            album_name,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

    monkeypatch.setattr(AlbumIndexer, "index", slow_index)

    with TestClient(app_module.app) as client:
        created = client.post(
            "/jobs/prepare", json={"folder": str(folder), "include_people": False}
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        assert started.wait(timeout=2)
        duplicate = client.post(
            "/jobs/prepare", json={"folder": str(folder), "include_people": False}
        )
        cancel = client.post(f"/jobs/{job_id}/cancel")
        release.set()
        terminal = _wait_for_terminal(client, job_id)

    assert duplicate.status_code == 409
    assert cancel.status_code == 200
    assert cancel.json()["cancel_requested"] is True
    assert cancel.json()["stage"] == "indexing"
    assert cancel.json()["progress"] == 0.3
    assert cancel.json()["result"]["indexing_progress"] == {
        "completed": 1,
        "total": 2,
    }
    assert terminal["status"] == "cancelled"
    assert terminal["progress"] < 1


def test_job_endpoints_report_missing_and_invalid_requests(
    tmp_path: Path, monkeypatch
) -> None:
    _, _ = _configure(tmp_path, monkeypatch)
    with TestClient(app_module.app) as client:
        missing_folder = client.post(
            "/jobs/prepare", json={"folder": str(tmp_path / "missing")}
        )
        missing_job = client.get("/jobs/missing")
        invalid_filter = client.get("/jobs?status=unknown")

    assert missing_folder.status_code == 404
    assert missing_job.status_code == 404
    assert invalid_filter.status_code == 422


def test_manager_recovers_queued_job_and_marks_running_job_interrupted(
    tmp_path: Path, monkeypatch
) -> None:
    folder, data_dir = _configure(tmp_path, monkeypatch)
    database = Database(data_dir / "norma.db")
    database.initialize()
    payload = json.dumps(
        {"folder": str(folder.resolve()), "name": None, "include_people": False}
    )
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs(id, job_type, status, stage, progress, payload_json)
            VALUES ('queued-job', 'prepare_album', 'queued', 'queued', 0, ?)
            """,
            (payload,),
        )
        connection.execute(
            """
            INSERT INTO jobs(id, job_type, status, stage, progress, payload_json)
            VALUES ('running-job', 'prepare_album', 'running', 'embedding', 0.55, ?)
            """,
            (payload,),
        )

    manager = PrepareJobManager(database, data_dir, "lightweight", "opencv-haar")
    manager.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            recovered = manager.get("queued-job")
            if recovered.status in {"completed", "failed"}:
                break
            time.sleep(0.03)
        interrupted = manager.get("running-job")
    finally:
        manager.shutdown()

    assert recovered.status == "completed", recovered
    assert interrupted.status == "failed"
    assert interrupted.stage == "interrupted"
    assert "worker stopped" in interrupted.error
