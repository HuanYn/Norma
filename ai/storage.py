from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 2

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS albums (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    indexed_at TEXT
);

CREATE TABLE IF NOT EXISTS photos (
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
    embedding_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_photos_album_id ON photos(album_id);

CREATE TABLE IF NOT EXISTS person_clusters (
    id TEXT PRIMARY KEY,
    album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    label TEXT NOT NULL DEFAULT 'Unknown',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS faces (
    id TEXT PRIMARY KEY,
    photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    cluster_id TEXT REFERENCES person_clusters(id) ON DELETE SET NULL,
    box_json TEXT NOT NULL,
    embedding_path TEXT
);

CREATE TABLE IF NOT EXISTS selections (
    id TEXT PRIMARY KEY,
    album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    raw_prompt TEXT NOT NULL,
    parse_json TEXT NOT NULL,
    result_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    album_id TEXT REFERENCES albums(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY,
    parameters_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

PHOTO_COLUMNS_V2: dict[str, str] = {
    "file_size": "INTEGER",
    "phash": "TEXT",
    "dhash": "TEXT",
    "auto_reject": "INTEGER NOT NULL DEFAULT 0",
    "reject_reason": "TEXT",
    "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
}


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (1,),
            )
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
            ).fetchone()
            current = int(row["version"] if row else 0)
            for version in range(current + 1, SCHEMA_VERSION + 1):
                if version <= current:
                    continue
                self._apply_migration(connection, version)
                connection.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)", (version,)
                )

    @staticmethod
    def _apply_migration(connection: sqlite3.Connection, version: int) -> None:
        if version != 2:
            raise RuntimeError(f"Missing database migration {version}")
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(photos)")
        }
        for name, declaration in PHOTO_COLUMNS_V2.items():
            if name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE photos ADD COLUMN {name} {declaration}"
                )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_photos_similarity_group
                ON photos(album_id, similarity_group)
            """
        )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def current_version(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
            ).fetchone()
        return int(row["version"] if row else 0)
