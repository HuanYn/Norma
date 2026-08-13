from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ai import app as app_module
from ai.storage import Database


def test_health_initializes_sqlite(tmp_path: Path, monkeypatch) -> None:
    database = Database(tmp_path / "norma.db")
    monkeypatch.setattr(app_module, "database", database)

    with TestClient(app_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "norma-ai",
        "status": "ok",
        "schema_version": 1,
    }
    assert database.path.exists()


def test_capabilities_are_explicit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "database", Database(tmp_path / "norma.db"))

    with TestClient(app_module.app) as client:
        response = client.get("/capabilities")

    payload = response.json()
    assert payload["image_types"] == [".jpg", ".jpeg"]
    assert payload["original_policy"] == "read-only"
    assert payload["milestones"]["video"] == "deferred"

