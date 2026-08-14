from __future__ import annotations

import threading
import time
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ai import app as app_module
from ai.config import Settings
from ai.index import AlbumIndexer
from ai.jobs import PrepareJobManager
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
        Settings(host="127.0.0.1", port=8765, data_dir=data_dir, log_level="INFO"),
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


def test_prepare_job_rejects_duplicate_and_cancels_between_stages(
    tmp_path: Path, monkeypatch
) -> None:
    folder, _ = _configure(tmp_path, monkeypatch)
    started = threading.Event()
    release = threading.Event()
    original_index = AlbumIndexer.index

    def slow_index(self, folder_path, album_name=None):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release index stage")
        return original_index(self, folder_path, album_name)

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
