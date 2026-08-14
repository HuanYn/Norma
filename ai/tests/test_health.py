from __future__ import annotations

import sqlite3
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
        "schema_version": 2,
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


def test_migrates_existing_v1_database(tmp_path: Path) -> None:
    path = tmp_path / "norma.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO schema_migrations(version) VALUES (1);
        CREATE TABLE albums(id TEXT PRIMARY KEY, name TEXT, source_path TEXT);
        CREATE TABLE photos(
            id TEXT PRIMARY KEY,
            album_id TEXT,
            absolute_path TEXT,
            thumbnail_path TEXT,
            width INTEGER,
            height INTEGER,
            capture_time TEXT,
            quality_score REAL,
            blur_score REAL,
            similarity_group TEXT,
            embedding_path TEXT,
            file_size INTEGER,
            created_at TEXT
        );
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    with database.connect() as migrated:
        columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(photos)")
        }
        version = migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 2
    assert {"phash", "dhash", "auto_reject", "metadata_json"} <= columns
