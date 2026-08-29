from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 14


PREFERENCE_SCHEMA_V10_SQL = """
CREATE TABLE IF NOT EXISTS preference_events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    album_id TEXT NOT NULL,
    selection_id TEXT,
    suggestion_id TEXT,
    query_text TEXT NOT NULL,
    preferred_photo_id TEXT NOT NULL,
    rejected_photo_id TEXT NOT NULL,
    choice TEXT NOT NULL CHECK(
        choice IN ('preferred', 'tie', 'skip', 'both_bad')
    ),
    provider_fingerprint TEXT NOT NULL,
    feature_schema TEXT NOT NULL,
    preferred_features_json TEXT NOT NULL,
    rejected_features_json TEXT NOT NULL,
    base_margin REAL NOT NULL,
    context_json TEXT NOT NULL,
    model_id_at_display TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_preference_events_compatible
    ON preference_events(
        user_id, provider_fingerprint, feature_schema, created_at, id
    );

CREATE TRIGGER IF NOT EXISTS preference_events_no_update
BEFORE UPDATE ON preference_events
BEGIN
    SELECT RAISE(ABORT, 'preference_events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS preference_events_no_delete
BEFORE DELETE ON preference_events
BEGIN
    SELECT RAISE(ABORT, 'preference_events are immutable');
END;

CREATE TABLE IF NOT EXISTS preference_models (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    provider_fingerprint TEXT NOT NULL,
    feature_schema TEXT NOT NULL,
    projection_id TEXT,
    mean_json TEXT NOT NULL,
    covariance_json TEXT NOT NULL,
    training_pair_count INTEGER NOT NULL CHECK(training_pair_count >= 0),
    training_event_digest TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_preference_models_one_active
    ON preference_models(user_id, provider_fingerprint, feature_schema)
    WHERE active = 1;

CREATE INDEX IF NOT EXISTS idx_preference_models_history
    ON preference_models(
        user_id, provider_fingerprint, feature_schema, created_at, id
    );

CREATE TRIGGER IF NOT EXISTS preference_models_restrict_update
BEFORE UPDATE ON preference_models
WHEN NOT (
    OLD.active = 1 AND NEW.active = 0
    AND NEW.id IS OLD.id
    AND NEW.user_id IS OLD.user_id
    AND NEW.algorithm IS OLD.algorithm
    AND NEW.provider_fingerprint IS OLD.provider_fingerprint
    AND NEW.feature_schema IS OLD.feature_schema
    AND NEW.projection_id IS OLD.projection_id
    AND NEW.mean_json IS OLD.mean_json
    AND NEW.covariance_json IS OLD.covariance_json
    AND NEW.training_pair_count IS OLD.training_pair_count
    AND NEW.training_event_digest IS OLD.training_event_digest
    AND NEW.hyperparameters_json IS OLD.hyperparameters_json
    AND NEW.diagnostics_json IS OLD.diagnostics_json
    AND NEW.created_at IS OLD.created_at
)
BEGIN
    SELECT RAISE(ABORT, 'preference_models are immutable except deactivation');
END;

CREATE TRIGGER IF NOT EXISTS preference_models_no_delete
BEFORE DELETE ON preference_models
BEGIN
    SELECT RAISE(ABORT, 'preference_models are immutable');
END;
"""

PREFERENCE_SUGGESTION_SCHEMA_V11_SQL = """
CREATE TABLE IF NOT EXISTS preference_suggestions (
    id TEXT PRIMARY KEY,
    selection_id TEXT NOT NULL,
    album_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    left_photo_id TEXT NOT NULL,
    right_photo_id TEXT NOT NULL,
    provider_fingerprint TEXT NOT NULL,
    feature_schema TEXT NOT NULL,
    projection_id TEXT NOT NULL,
    model_id_at_display TEXT,
    acquisition_version TEXT NOT NULL,
    constraint_solver TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('shortlist', 'exhaustive')),
    candidate_digest TEXT NOT NULL,
    candidate_source_digest TEXT NOT NULL,
    candidate_ids_json TEXT NOT NULL,
    left_features_json TEXT NOT NULL,
    right_features_json TEXT NOT NULL,
    request_json TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(left_photo_id <> right_photo_id)
);

CREATE INDEX IF NOT EXISTS idx_preference_suggestions_context
    ON preference_suggestions(
        selection_id, user_id, provider_fingerprint, feature_schema,
        model_id_at_display, candidate_digest, created_at, id
    );

CREATE INDEX IF NOT EXISTS idx_preference_suggestions_exclusion
    ON preference_suggestions(
        selection_id, user_id, provider_fingerprint, feature_schema,
        candidate_digest, created_at, id
    );

CREATE TRIGGER IF NOT EXISTS preference_suggestions_no_update
BEFORE UPDATE ON preference_suggestions
BEGIN
    SELECT RAISE(ABORT, 'preference_suggestions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS preference_suggestions_no_delete
BEFORE DELETE ON preference_suggestions
BEGIN
    SELECT RAISE(ABORT, 'preference_suggestions are immutable');
END;
"""

PREFERENCE_SUGGESTION_EVENT_INDEX_V11_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_preference_events_one_per_suggestion
    ON preference_events(suggestion_id)
    WHERE suggestion_id IS NOT NULL;
"""

RAG_SCHEMA_V12_SQL = """
CREATE TABLE IF NOT EXISTS rag_runs (
    id TEXT PRIMARY KEY,
    album_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    retrieval_provider_fingerprint TEXT NOT NULL,
    generation_provider_fingerprint TEXT NOT NULL,
    candidate_digest TEXT NOT NULL,
    query_digest TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    request_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(length(candidate_digest) = 64
          AND candidate_digest NOT GLOB '*[^0-9a-f]*'),
    CHECK(length(query_digest) = 64
          AND query_digest NOT GLOB '*[^0-9a-f]*'),
    CHECK(length(evidence_digest) = 64
          AND evidence_digest NOT GLOB '*[^0-9a-f]*'),
    CHECK(json_valid(evidence_json)),
    CHECK(json_valid(result_json)),
    CHECK(json_valid(request_json))
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_rag_runs_context
    ON rag_runs(album_id, user_id, created_at DESC, id DESC);

CREATE TRIGGER IF NOT EXISTS rag_runs_no_update
BEFORE UPDATE ON rag_runs
BEGIN
    SELECT RAISE(ABORT, 'rag_runs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS rag_runs_no_replace
BEFORE INSERT ON rag_runs
WHEN EXISTS(SELECT 1 FROM rag_runs WHERE id = NEW.id)
BEGIN
    SELECT RAISE(ABORT, 'rag_runs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS rag_runs_no_delete
BEFORE DELETE ON rag_runs
BEGIN
    SELECT RAISE(ABORT, 'rag_runs are immutable');
END;
"""

SCHEMA_SQL = (
    """
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
    absolute_path TEXT NOT NULL,
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
    embedding_source_sha256 TEXT CHECK(
        embedding_source_sha256 IS NULL
        OR (length(embedding_source_sha256) = 64
            AND embedding_source_sha256 NOT GLOB '*[^0-9a-f]*')
    ),
    face_provider TEXT,
    face_source_size INTEGER,
    face_source_mtime_ns INTEGER,
    face_processed INTEGER NOT NULL DEFAULT 0,
    face_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(album_id, absolute_path)
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
    + PREFERENCE_SCHEMA_V10_SQL
    + PREFERENCE_SUGGESTION_SCHEMA_V11_SQL
    + RAG_SCHEMA_V12_SQL
)

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

PHOTO_COLUMNS_V13: dict[str, str] = {
    "embedding_source_sha256": (
        "TEXT CHECK(embedding_source_sha256 IS NULL "
        "OR (length(embedding_source_sha256) = 64 "
        "AND embedding_source_sha256 NOT GLOB '*[^0-9a-f]*'))"
    ),
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
            # Version 11 was developed in-place before release. Repair databases
            # initialized by an earlier development snapshot of v11 as well as
            # applying the tracked migration below for v10 databases.
            self._ensure_v11_suggestion_event_key(connection)

    @staticmethod
    def _ensure_v11_suggestion_event_key(
        connection: sqlite3.Connection,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(preference_events)")
        }
        if "suggestion_id" not in columns:
            connection.execute(
                "ALTER TABLE preference_events ADD COLUMN suggestion_id TEXT"
            )
        connection.executescript(PREFERENCE_SUGGESTION_EVENT_INDEX_V11_SQL)

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
        if version == 9:
            # SQLite cannot drop the legacy UNIQUE(absolute_path) constraint in
            # place. Rebuild photos transactionally so one source file can be
            # represented independently in overlapping parent/child albums.
            photo_columns = (
                "id",
                "album_id",
                "absolute_path",
                "thumbnail_path",
                "width",
                "height",
                "capture_time",
                "quality_score",
                "blur_score",
                "similarity_group",
                "source_mtime_ns",
                "embedding_path",
                "embedding_provider",
                "embedding_source_size",
                "embedding_source_mtime_ns",
                "face_provider",
                "face_source_size",
                "face_source_mtime_ns",
                "face_processed",
                "face_count",
                "created_at",
                "file_size",
                "phash",
                "dhash",
                "auto_reject",
                "reject_reason",
                "metadata_json",
            )
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(photos)")
            }
            defaults = {
                "album_id": "''",
                "absolute_path": "id",
                "face_processed": "0",
                "face_count": "0",
                "created_at": "CURRENT_TIMESTAMP",
                "auto_reject": "0",
                "metadata_json": "'{}'",
            }
            select_columns = [
                name
                if name in existing_columns
                else f"{defaults.get(name, 'NULL')} AS {name}"
                for name in photo_columns
            ]
            connection.commit()
            connection.execute("PRAGMA foreign_keys = OFF")
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DROP TABLE IF EXISTS photos_v9")
                connection.execute(
                    """
                    CREATE TABLE photos_v9 (
                        id TEXT PRIMARY KEY,
                        album_id TEXT NOT NULL
                            REFERENCES albums(id) ON DELETE CASCADE,
                        absolute_path TEXT NOT NULL,
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
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        UNIQUE(album_id, absolute_path)
                    )
                    """
                )
                column_sql = ", ".join(photo_columns)
                select_sql = ", ".join(select_columns)
                connection.execute(
                    f"INSERT INTO photos_v9({column_sql}) "
                    f"SELECT {select_sql} FROM photos"
                )
                connection.execute("DROP TABLE photos")
                connection.execute("ALTER TABLE photos_v9 RENAME TO photos")
                connection.execute(
                    "CREATE INDEX idx_photos_album_id ON photos(album_id)"
                )
                connection.execute(
                    """CREATE INDEX idx_photos_similarity_group
                       ON photos(album_id, similarity_group)"""
                )
                connection.execute(
                    """CREATE INDEX idx_photos_embedding_provider
                       ON photos(album_id, embedding_provider)"""
                )
                connection.execute(
                    """CREATE INDEX idx_photos_embedding_freshness
                       ON photos(
                           album_id, embedding_provider,
                           embedding_source_size, embedding_source_mtime_ns
                       )"""
                )
                connection.execute(
                    """CREATE INDEX idx_photos_face_freshness
                       ON photos(
                           album_id, face_provider, face_processed,
                           face_source_size, face_source_mtime_ns
                       )"""
                )
                violations = connection.execute("PRAGMA foreign_key_check").fetchall()
                if violations:
                    raise RuntimeError(
                        f"photo identity migration broke {len(violations)} foreign keys"
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.execute("PRAGMA foreign_keys = ON")
            return
        if version == 10:
            connection.executescript(PREFERENCE_SCHEMA_V10_SQL)
            return
        if version == 11:
            connection.executescript(PREFERENCE_SUGGESTION_SCHEMA_V11_SQL)
            Database._ensure_v11_suggestion_event_key(connection)
            return
        if version == 12:
            connection.executescript(RAG_SCHEMA_V12_SQL)
            return
        if version == 13:
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(photos)")
            }
            for name, declaration in PHOTO_COLUMNS_V13.items():
                if name not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE photos ADD COLUMN {name} {declaration}"
                    )
            # Existing vectors predate content binding.  They intentionally remain
            # NULL and are recomputed on their next use; current file hashes must
            # never be attached retroactively to an unverified legacy vector.
            return
        if version == 14:
            table = connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'rag_runs'"
            ).fetchone()
            if table is None:
                connection.executescript(RAG_SCHEMA_V12_SQL)
                return
            if "WITHOUT ROWID" in str(table["sql"]).upper():
                return

            columns = (
                "id",
                "album_id",
                "user_id",
                "query_text",
                "retrieval_provider_fingerprint",
                "generation_provider_fingerprint",
                "candidate_digest",
                "query_digest",
                "evidence_digest",
                "evidence_json",
                "result_json",
                "request_json",
                "created_at",
            )
            connection.execute("DROP TRIGGER IF EXISTS rag_runs_no_update")
            connection.execute("DROP TRIGGER IF EXISTS rag_runs_no_replace")
            connection.execute("DROP TRIGGER IF EXISTS rag_runs_no_delete")
            connection.execute("DROP INDEX IF EXISTS idx_rag_runs_context")
            connection.execute("ALTER TABLE rag_runs RENAME TO rag_runs_v13")
            connection.execute(
                """
                CREATE TABLE rag_runs (
                    id TEXT PRIMARY KEY,
                    album_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    retrieval_provider_fingerprint TEXT NOT NULL,
                    generation_provider_fingerprint TEXT NOT NULL,
                    candidate_digest TEXT NOT NULL,
                    query_digest TEXT NOT NULL,
                    evidence_digest TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK(length(candidate_digest) = 64
                          AND candidate_digest NOT GLOB '*[^0-9a-f]*'),
                    CHECK(length(query_digest) = 64
                          AND query_digest NOT GLOB '*[^0-9a-f]*'),
                    CHECK(length(evidence_digest) = 64
                          AND evidence_digest NOT GLOB '*[^0-9a-f]*'),
                    CHECK(json_valid(evidence_json)),
                    CHECK(json_valid(result_json)),
                    CHECK(json_valid(request_json))
                ) WITHOUT ROWID
                """
            )
            column_sql = ", ".join(columns)
            connection.execute(
                f"INSERT INTO rag_runs({column_sql}) "
                f"SELECT {column_sql} FROM rag_runs_v13"
            )
            source_count = connection.execute(
                "SELECT COUNT(*) FROM rag_runs_v13"
            ).fetchone()[0]
            target_count = connection.execute(
                "SELECT COUNT(*) FROM rag_runs"
            ).fetchone()[0]
            if source_count != target_count:
                raise RuntimeError("rag_runs v14 migration lost audit rows")
            connection.execute("DROP TABLE rag_runs_v13")
            connection.execute(
                """CREATE INDEX idx_rag_runs_context
                   ON rag_runs(album_id, user_id, created_at DESC, id DESC)"""
            )
            connection.execute(
                """CREATE TRIGGER rag_runs_no_update
                   BEFORE UPDATE ON rag_runs
                   BEGIN
                       SELECT RAISE(ABORT, 'rag_runs are immutable');
                   END"""
            )
            connection.execute(
                """CREATE TRIGGER rag_runs_no_replace
                   BEFORE INSERT ON rag_runs
                   WHEN EXISTS(SELECT 1 FROM rag_runs WHERE id = NEW.id)
                   BEGIN
                       SELECT RAISE(ABORT, 'rag_runs are immutable');
                   END"""
            )
            connection.execute(
                """CREATE TRIGGER rag_runs_no_delete
                   BEFORE DELETE ON rag_runs
                   BEGIN
                       SELECT RAISE(ABORT, 'rag_runs are immutable');
                   END"""
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
