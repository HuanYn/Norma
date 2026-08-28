from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai import app as app_module
from ai.people import canonical_face_provider_name
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
        "schema_version": 9,
        "face_provider": canonical_face_provider_name(
            app_module.settings.face_provider
        ),
    }
    assert database.path.exists()


def test_capabilities_are_explicit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "database", Database(tmp_path / "norma.db"))

    with TestClient(app_module.app) as client:
        response = client.get("/capabilities")
        providers = client.get("/providers/embedding")

    payload = response.json()
    assert payload["image_types"] == [".jpg", ".jpeg"]
    assert payload["original_policy"] == "read-only"
    assert payload["milestones"]["video"] == "deferred"
    assert (
        payload["milestones"]["library_lifecycle"] == "persistent-catalog-and-jobs-v1"
    )
    provider_items = providers.json()["items"]
    assert [item["id"] for item in provider_items] == [
        "lightweight",
        "openclip-multilingual",
    ]
    assert provider_items[0]["active"] is True


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
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(photos)")}
        version = migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 9
    assert {"phash", "dhash", "auto_reject", "metadata_json"} <= columns


def test_migrates_existing_v2_jobs_table(tmp_path: Path) -> None:
    path = tmp_path / "norma.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO schema_migrations(version) VALUES (1), (2);
        CREATE TABLE jobs(
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    with database.connect() as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(jobs)")}
        version = migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 9
    assert {
        "stage",
        "progress",
        "cancel_requested",
        "started_at",
        "finished_at",
    } <= columns


def test_migrates_existing_v3_photo_provider_column(tmp_path: Path) -> None:
    path = tmp_path / "norma.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO schema_migrations(version) VALUES (1), (2), (3);
        CREATE TABLE photos(
            id TEXT PRIMARY KEY,
            album_id TEXT,
            absolute_path TEXT,
            embedding_path TEXT
        );
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    with database.connect() as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(photos)")}
        version = migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 9
    assert "embedding_provider" in columns


def test_migrates_existing_v4_embedding_fingerprint_columns(tmp_path: Path) -> None:
    path = tmp_path / "norma.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO schema_migrations(version) VALUES (1), (2), (3), (4);
        CREATE TABLE photos(
            id TEXT PRIMARY KEY,
            album_id TEXT,
            absolute_path TEXT,
            embedding_path TEXT,
            embedding_provider TEXT
        );
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    with database.connect() as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(photos)")}
        version = migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 9
    assert {
        "source_mtime_ns",
        "embedding_source_size",
        "embedding_source_mtime_ns",
    } <= columns


def test_migrates_existing_v5_evaluation_tables(tmp_path: Path) -> None:
    path = tmp_path / "norma.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO schema_migrations(version) VALUES (1), (2), (3), (4), (5);
        CREATE TABLE albums(id TEXT PRIMARY KEY, name TEXT, source_path TEXT);
        CREATE TABLE photos(id TEXT PRIMARY KEY, album_id TEXT);
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    with database.connect() as migrated:
        tables = {
            row["name"]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 9
    assert {
        "evaluation_queries",
        "relevance_judgments",
        "evaluation_runs",
    } <= tables


def test_migrates_existing_v6_face_freshness_columns(tmp_path: Path) -> None:
    path = tmp_path / "norma.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO schema_migrations(version) VALUES (1), (2), (3), (4), (5), (6);
        CREATE TABLE photos(
            id TEXT PRIMARY KEY,
            album_id TEXT,
            absolute_path TEXT,
            file_size INTEGER,
            source_mtime_ns INTEGER
        );
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    with database.connect() as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(photos)")}
        version = migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 9
    assert {
        "face_provider",
        "face_source_size",
        "face_source_mtime_ns",
        "face_processed",
        "face_count",
    } <= columns


def test_migrates_existing_v7_maintenance_audit_table(tmp_path: Path) -> None:
    path = tmp_path / "norma.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO schema_migrations(version)
        VALUES (1), (2), (3), (4), (5), (6), (7);
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    with database.connect() as migrated:
        tables = {
            row["name"]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 9
    assert "maintenance_runs" in tables


def test_v9_photo_identity_migration_preserves_references_and_allows_overlap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "norma.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO schema_migrations(version)
        VALUES (1), (2), (3), (4), (5), (6), (7), (8);
        CREATE TABLE albums(
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source_path TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            indexed_at TEXT
        );
        CREATE TABLE photos(
            id TEXT PRIMARY KEY,
            album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
            absolute_path TEXT NOT NULL UNIQUE,
            thumbnail_path TEXT,
            width INTEGER,
            height INTEGER,
            capture_time TEXT,
            quality_score REAL,
            blur_score REAL,
            similarity_group TEXT,
            source_mtime_ns INTEGER,
            embedding_path TEXT,
            embedding_provider TEXT,
            embedding_source_size INTEGER,
            embedding_source_mtime_ns INTEGER,
            face_provider TEXT,
            face_source_size INTEGER,
            face_source_mtime_ns INTEGER,
            face_processed INTEGER NOT NULL DEFAULT 0,
            face_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            phash TEXT,
            dhash TEXT,
            auto_reject INTEGER NOT NULL DEFAULT 0,
            reject_reason TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE person_clusters(
            id TEXT PRIMARY KEY,
            album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
            label TEXT NOT NULL DEFAULT 'Unknown',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE faces(
            id TEXT PRIMARY KEY,
            photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            cluster_id TEXT REFERENCES person_clusters(id) ON DELETE SET NULL,
            box_json TEXT NOT NULL,
            embedding_path TEXT
        );
        CREATE TABLE evaluation_queries(
            id TEXT PRIMARY KEY,
            album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
            query_text TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(album_id, query_text)
        );
        CREATE TABLE relevance_judgments(
            query_id TEXT NOT NULL REFERENCES evaluation_queries(id) ON DELETE CASCADE,
            photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            relevance INTEGER NOT NULL CHECK(relevance BETWEEN 0 AND 3),
            annotator TEXT NOT NULL DEFAULT 'local',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(query_id, photo_id)
        );
        INSERT INTO albums(id, name, source_path) VALUES ('a', 'A', 'C:\\A');
        INSERT INTO photos(id, album_id, absolute_path)
        VALUES ('p1', 'a', 'C:\\shared.jpg');
        INSERT INTO faces(id, photo_id, box_json) VALUES ('f1', 'p1', '[0,0,1,1]');
        INSERT INTO evaluation_queries(id, album_id, query_text)
        VALUES ('q1', 'a', 'shared');
        INSERT INTO relevance_judgments(query_id, photo_id, relevance)
        VALUES ('q1', 'p1', 3);
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    with database.connect() as migrated:
        migrated.execute(
            "INSERT INTO albums(id, name, source_path) VALUES ('b', 'B', 'C:\\B')"
        )
        migrated.execute(
            """INSERT INTO photos(id, album_id, absolute_path)
               VALUES ('p2', 'b', 'C:\\shared.jpg')"""
        )
        with pytest.raises(sqlite3.IntegrityError):
            migrated.execute(
                """INSERT INTO photos(id, album_id, absolute_path)
                   VALUES ('p3', 'a', 'C:\\shared.jpg')"""
            )
        assert migrated.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 1
        assert (
            migrated.execute("SELECT COUNT(*) FROM relevance_judgments").fetchone()[0]
            == 1
        )
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
    assert database.current_version() == 9
