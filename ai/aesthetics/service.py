from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from ai.aesthetics.provider import AssessmentProvider, AssessmentScores, file_sha256
from ai.storage import Database


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS aesthetics_scores_v1 (
    photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    provider_fingerprint TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_size INTEGER NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    scores_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(photo_id, provider_fingerprint)
);
"""


class AestheticsCancelledError(RuntimeError):
    pass


class AestheticsSourceChangedError(RuntimeError):
    pass


def initialize_aesthetics_schema(database: Database) -> None:
    """An additive, idempotent feature table; no core schema-version change."""
    with database.connect() as connection:
        connection.executescript(SCHEMA_SQL)


class AestheticsService:
    def __init__(self, database: Database, provider: AssessmentProvider) -> None:
        self.database = database
        self.provider = provider

    def initialize(self) -> None:
        initialize_aesthetics_schema(self.database)

    def _rows(self, album_id: str) -> list:
        with self.database.connect() as connection:
            if connection.execute("SELECT 1 FROM albums WHERE id=?", (album_id,)).fetchone() is None:
                raise KeyError("album not found")
            return connection.execute(
                """SELECT id, album_id, absolute_path, file_size, source_mtime_ns,
                   metadata_json, embedding_source_sha256 FROM photos
                   WHERE album_id=? ORDER BY id""", (album_id,)
            ).fetchall()

    def analyze(
        self,
        album_id: str,
        *,
        force: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        self.initialize()
        started = time.perf_counter()
        rows = self._rows(album_id)
        computed, reused = 0, 0
        items = []
        source_hashes = {}
        if on_progress:
            on_progress(0, len(rows))
        for row in rows:
            _check_cancel(should_cancel)
            source_hash = _current_source(row)
            source_hashes[row["id"]] = source_hash
            cached = None if force else self._cached(row["id"], source_hash)
            if cached is None:
                scores = self.provider.score(Path(row["absolute_path"]))
                _check_cancel(should_cancel)
                if _current_source(row) != source_hash:
                    raise AestheticsSourceChangedError("Photo changed during assessment; reindex the album.")
                self._save(row, source_hash, scores)
                computed += 1
            else:
                scores = cached
                reused += 1
            items.append({"photo_id": row["id"], "cached": cached is not None, **scores.as_dict()})
            if on_progress:
                on_progress(len(items), len(rows))
        _check_cancel(should_cancel)
        # Do not label an old snapshot as a completed current album if indexing
        # or source replacement raced with this multi-image job.
        current_rows = self._rows(album_id)
        if [row["id"] for row in current_rows] != [row["id"] for row in rows]:
            raise AestheticsSourceChangedError("Album membership changed during assessment; retry the current album.")
        for original, current in zip(rows, current_rows):
            _check_cancel(should_cancel)
            if (current["absolute_path"] != original["absolute_path"]
                    or _current_source(current) != source_hashes[current["id"]]):
                raise AestheticsSourceChangedError("Album contents changed during assessment; reindex and retry.")
        return {
            "album_id": album_id,
            "provider": self.provider.fingerprint,
            "total": len(rows),
            "computed_count": computed,
            "reused_count": reused,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "personalized": False,
            "items": items,
        }

    def cached(self, album_id: str) -> dict[str, object]:
        """Only current indexed source bytes may be presented as current scores."""
        self.initialize()
        rows = self._rows(album_id)
        items, stale = [], []
        for row in rows:
            try:
                source_hash = _current_source(row)
            except (OSError, AestheticsSourceChangedError):
                stale.append(row["id"])
                continue
            score = self._cached(row["id"], source_hash)
            if score is not None:
                items.append({"photo_id": row["id"], "source_sha256": source_hash, **score.as_dict()})
        return {
            "album_id": album_id,
            "provider": self.provider.fingerprint,
            "total": len(rows),
            "scored_count": len(items),
            "stale_photo_ids": stale,
            "personalized": False,
            "items": items,
        }

    def _cached(self, photo_id: str, source_hash: str) -> AssessmentScores | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT scores_json FROM aesthetics_scores_v1
                   WHERE photo_id=? AND provider_fingerprint=? AND source_sha256=?""",
                (photo_id, self.provider.fingerprint, source_hash),
            ).fetchone()
        if row is None:
            return None
        try:
            return AssessmentScores(**json.loads(row["scores_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save(self, row, source_hash: str, scores: AssessmentScores) -> None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM photos WHERE id=? AND album_id=?",
                (row["id"], row["album_id"]),
            ).fetchone()
            if (current is None or current["absolute_path"] != row["absolute_path"]
                    or _current_source(current) != source_hash):
                raise AestheticsSourceChangedError("Album changed during assessment; reindex and retry.")
            connection.execute(
                """INSERT INTO aesthetics_scores_v1
                   (photo_id, provider_fingerprint, source_sha256, source_size,
                    source_mtime_ns, scores_json)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(photo_id, provider_fingerprint) DO UPDATE SET
                    source_sha256=excluded.source_sha256,
                    source_size=excluded.source_size,
                    source_mtime_ns=excluded.source_mtime_ns,
                    scores_json=excluded.scores_json,
                    created_at=CURRENT_TIMESTAMP""",
                (row["id"], self.provider.fingerprint, source_hash,
                 row["file_size"], row["source_mtime_ns"], json.dumps(scores.as_dict())),
            )


def _current_source(row) -> str:
    try:
        metadata = json.loads(row["metadata_json"] or "{}")
    except (ValueError, TypeError):
        metadata = {}
    digest = metadata.get("source_sha256") if isinstance(metadata, dict) else None
    embedded = row["embedding_source_sha256"]
    if digest and embedded and digest != embedded:
        raise AestheticsSourceChangedError("Photo index hashes disagree; reindex the album.")
    expected = digest or embedded
    if not isinstance(expected, str) or len(expected) != 64:
        raise AestheticsSourceChangedError("Photo has no verified index digest; reindex the album.")
    try:
        path = Path(row["absolute_path"])
        before = path.stat()
        actual = file_sha256(path)
        after = path.stat()
        if (before.st_size != row["file_size"]
                or before.st_mtime_ns != row["source_mtime_ns"]
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or actual != expected):
            raise AestheticsSourceChangedError("Photo changed since indexing; reindex the album.")
    except OSError:
        raise AestheticsSourceChangedError("Indexed photo is no longer readable; reindex the album.") from None
    return actual


def _check_cancel(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel and should_cancel():
        raise AestheticsCancelledError("Aesthetics assessment cancelled between images.")
