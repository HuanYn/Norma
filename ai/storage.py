from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 8

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
    stage TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_queries (
    id TEXT PRIMARY KEY,
    album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(album_id, query_text)
);

CREATE TABLE IF NOT EXISTS relevance_judgments (
    query_id TEXT NOT NULL REFERENCES evaluation_queries(id) ON DELETE CASCADE,
    photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    relevance INTEGER NOT NULL CHECK(relevance BETWEEN 0 AND 3),
    annotator TEXT NOT NULL DEFAULT 'local',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(query_id, photo_id)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    embedding_provider TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_evaluation_queries_album
    ON evaluation_queries(album_id, created_at);
CREATE INDEX IF NOT EXISTS idx_relevance_judgments_query
    ON relevance_judgments(query_id, relevance);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_album
    ON evaluation_runs(album_id, created_at DESC);

CREATE TABLE IF NOT EXISTS maintenance_runs (
    id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    dry_run INTEGER NOT NULL DEFAULT 1,
    request_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_maintenance_runs_created
    ON maintenance_runs(created_at DESC, id DESC);
"""

PHOTO_COLUMNS_V2: dict[str, str] = {
    "file_size": "INTEGER",
    "phash": "TEXT",
    "dhash": "TEXT",
    "auto_reject": "INTEGER NOT NULL DEFAULT 0",
    "reject_reason": "TEXT",
    "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
}

JOB_COLUMNS_V3: dict[str, str] = {
    "stage": "TEXT NOT NULL DEFAULT 'queued'",
    "progress": "REAL NOT NULL DEFAULT 0",
    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
    "started_at": "TEXT",
    "finished_at": "TEXT",
}

PHOTO_COLUMNS_V4: dict[str, str] = {
    "embedding_provider": "TEXT",
}

PHOTO_COLUMNS_V5: dict[str, str] = {
    "source_mtime_ns": "INTEGER",
    "embedding_source_size": "INTEGER",
    "embedding_source_mtime_ns": "INTEGER",
}

PHOTO_COLUMNS_V7: dict[str, str] = {
    "face_provider": "TEXT",
    "face_source_size": "INTEGER",
    "face_source_mtime_ns": "INTEGER",
    "face_processed": "INTEGER NOT NULL DEFAULT 0",
    "face_count": "INTEGER NOT NULL DEFAULT 0",
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
        if version == 2:
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
            return
        if version == 3:
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
            }
            for name, declaration in JOB_COLUMNS_V3.items():
                if name not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE jobs ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)"
            )
            return
        if version == 4:
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(photos)")
            }
            for name, declaration in PHOTO_COLUMNS_V4.items():
                if name not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE photos ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_photos_embedding_provider
                    ON photos(album_id, embedding_provider)
                """
            )
            return
        if version == 5:
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(photos)")
            }
            for name, declaration in PHOTO_COLUMNS_V5.items():
                if name not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE photos ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_photos_embedding_freshness
                    ON photos(
                        album_id, embedding_provider,
                        embedding_source_size, embedding_source_mtime_ns
                    )
                """
            )
            return
        if version == 6:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_queries (
                    id TEXT PRIMARY KEY,
                    album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
                    query_text TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(album_id, query_text)
                );
                CREATE TABLE IF NOT EXISTS relevance_judgments (
                    query_id TEXT NOT NULL
                        REFERENCES evaluation_queries(id) ON DELETE CASCADE,
                    photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                    relevance INTEGER NOT NULL CHECK(relevance BETWEEN 0 AND 3),
                    annotator TEXT NOT NULL DEFAULT 'local',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(query_id, photo_id)
                );
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id TEXT PRIMARY KEY,
                    album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
                    embedding_provider TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_evaluation_queries_album
                    ON evaluation_queries(album_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_relevance_judgments_query
                    ON relevance_judgments(query_id, relevance);
                CREATE INDEX IF NOT EXISTS idx_evaluation_runs_album
                    ON evaluation_runs(album_id, created_at DESC);
                """
            )
            return
        if version == 7:
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(photos)")
            }
            for name, declaration in PHOTO_COLUMNS_V7.items():
                if name not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE photos ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_photos_face_freshness
                    ON photos(
                        album_id, face_provider, face_processed,
                        face_source_size, face_source_mtime_ns
                    )
                """
            )
            return
        if version == 8:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS maintenance_runs (
                    id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dry_run INTEGER NOT NULL DEFAULT 1,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_maintenance_runs_created
                    ON maintenance_runs(created_at DESC, id DESC);
                """
            )
            return
        raise RuntimeError(f"Missing database migration {version}")

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
