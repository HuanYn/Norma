from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ai.index import AlbumIndexer, IndexingCancelledError
from ai.index.embedding import create_embedding_provider
from ai.people import PeopleCancelledError, PeopleIndexer, create_face_provider
from ai.retrieval import EmbeddingCancelledError, RetrievalService
from ai.schemas import JobListResponse, JobResponse, PrepareJobRequest
from ai.storage import Database


logger = logging.getLogger("norma.ai.jobs")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
PROGRESS_MIN_FRACTION = 0.01
PROGRESS_LARGE_FRACTION = 0.05
PROGRESS_MIN_INTERVAL_SECONDS = 0.05
PROGRESS_HEARTBEAT_SECONDS = 0.25


class PrepareJobManager:
    """Single-worker, SQLite-backed orchestration for album preparation."""

    def __init__(
        self,
        database: Database,
        data_dir: Path,
        embedding_provider: str,
        face_provider: str,
        embedding_device: str = "auto",
        embedding_batch_size: int = 8,
        model_cache_dir: Path | None = None,
    ) -> None:
        self.database = database
        self.data_dir = data_dir
        self.embedding_provider = embedding_provider
        self.face_provider = face_provider
        self.embedding_device = embedding_device
        self.embedding_batch_size = embedding_batch_size
        self.model_cache_dir = model_cache_dir or (data_dir / "models")
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="norma-prepare"
        )
        self.futures: dict[str, Future[None]] = {}
        self.lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.progress_checkpoints: dict[tuple[str, str], tuple[float, float]] = {}

    def start(self) -> None:
        self.database.initialize()
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status = 'failed', stage = 'interrupted',
                    error = 'worker stopped before this job completed',
                    finished_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE job_type = 'prepare_album' AND status = 'running'
                """
            )
            queued = connection.execute(
                """SELECT id FROM jobs
                   WHERE job_type = 'prepare_album' AND status = 'queued'
                   ORDER BY created_at, id"""
            ).fetchall()
        for row in queued:
            self._schedule(row["id"])

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def submit(self, request: PrepareJobRequest) -> JobResponse:
        folder = Path(request.folder).expanduser().resolve(strict=True)
        if not folder.is_dir():
            raise NotADirectoryError(str(folder))
        payload = {
            "folder": str(folder),
            "name": request.name,
            "include_quality": request.include_quality,
            "include_embeddings": request.include_embeddings,
            "include_people": request.include_people,
        }
        self.database.initialize()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """SELECT payload_json FROM jobs
                   WHERE job_type = 'prepare_album' AND status IN ('queued', 'running')"""
            ).fetchall()
            if any(
                json.loads(row["payload_json"])["folder"] == str(folder)
                for row in active
            ):
                raise ValueError("an active prepare job already exists for this folder")
            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO jobs(id, job_type, status, stage, progress, payload_json)
                VALUES (?, 'prepare_album', 'queued', 'queued', 0, ?)
                """,
                (job_id, json.dumps(payload, ensure_ascii=False)),
            )
        self._schedule(job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> JobResponse:
        return get_persisted_job(self.database, job_id)

    def list(self, *, limit: int, offset: int, status: str | None) -> JobListResponse:
        return list_persisted_jobs(
            self.database, limit=limit, offset=offset, status=status
        )

    def cancel(self, job_id: str) -> JobResponse:
        current = self.get(job_id)
        if current.job_type != "prepare_album":
            raise ValueError(f"unsupported job type: {current.job_type}")
        if current.status in TERMINAL_STATUSES:
            raise ValueError(f"job is already {current.status}")
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET cancel_requested = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (job_id,),
            )
        with self.lock:
            future = self.futures.get(job_id)
            cancelled_before_start = bool(future and future.cancel())
        if cancelled_before_start:
            self._mark_cancelled(job_id)
            with self.lock:
                self.futures.pop(job_id, None)
        return self.get(job_id)

    def _schedule(self, job_id: str) -> None:
        with self.lock:
            self.futures[job_id] = self.executor.submit(self._run, job_id)

    def _run(self, job_id: str) -> None:
        try:
            job = self.get(job_id)
            if job.cancel_requested:
                self._mark_cancelled(job_id)
                return
            request = PrepareJobRequest.model_validate(job.payload)
            ranges = _progress_ranges(request)
            index_start, index_span = ranges["indexing"]
            self._begin_progress(job_id, "indexing")
            self._set_stage(
                job_id,
                status="running",
                stage="indexing",
                progress=index_start,
            )
            index_kwargs: dict[str, object] = {
                "on_progress": lambda completed, total: self._indexing_progress(
                    job_id,
                    completed,
                    total,
                    start=index_start,
                    span=index_span,
                ),
                "should_cancel": lambda: self._is_cancel_requested(job_id),
            }
            # Omitting the keyword for the historical default keeps third-party
            # wrappers around AlbumIndexer.index compatible with the old signature.
            if not request.include_quality:
                index_kwargs["analyze_quality"] = False
            try:
                indexed = AlbumIndexer(self.database, self.data_dir).index(
                    Path(request.folder),
                    request.name,
                    **index_kwargs,
                )
            except IndexingCancelledError:
                self._mark_cancelled(job_id)
                return
            result: dict[str, object] = {
                "album": {
                    "album_id": indexed.album_id,
                    "name": indexed.name,
                    "source_path": indexed.source_path,
                    "total": indexed.total,
                    "computed_count": indexed.computed_count,
                    "reused_count": indexed.reused_count,
                    "rejected": indexed.rejected,
                    "similar_groups": indexed.similar_groups,
                    "duration_ms": indexed.duration_ms,
                    "provider": indexed.provider,
                    "errors": indexed.errors,
                },
                "embedding": None,
                "people": None,
            }
            self._set_stage(
                job_id,
                stage="indexing",
                progress=index_start + index_span,
                result=result,
            )
            if self._cancel_if_requested(job_id):
                return

            if request.include_embeddings:
                embedding_start, embedding_span = ranges["embedding"]
                self._begin_progress(job_id, "embedding")
                self._set_stage(
                    job_id,
                    stage="embedding",
                    progress=embedding_start,
                    result=result,
                )
                try:
                    embedded = RetrievalService(
                        self.database,
                        self.data_dir,
                        create_embedding_provider(
                            self.embedding_provider,
                            cache_dir=self.model_cache_dir,
                            device=self.embedding_device,
                            batch_size=self.embedding_batch_size,
                        ),
                    ).embed_album(
                        indexed.album_id,
                        on_progress=lambda completed, total: self._embedding_progress(
                            job_id,
                            result,
                            completed,
                            total,
                            start=embedding_start,
                            span=embedding_span,
                        ),
                        should_cancel=lambda: self._is_cancel_requested(job_id),
                    )
                except EmbeddingCancelledError:
                    self._mark_cancelled(job_id)
                    return
                result.pop("embedding_progress", None)
                result["embedding"] = embedded.model_dump()
                self._set_stage(
                    job_id,
                    stage="embedding",
                    progress=embedding_start + embedding_span,
                    result=result,
                )
                if self._cancel_if_requested(job_id):
                    return

            if request.include_people:
                people_start, people_span = ranges["people"]
                self._begin_progress(job_id, "people")
                self._set_stage(
                    job_id,
                    stage="people",
                    progress=people_start,
                    result=result,
                )
                try:
                    people = PeopleIndexer(
                        self.database,
                        self.data_dir,
                        create_face_provider(
                            self.face_provider, cache_dir=self.model_cache_dir
                        ),
                    ).index(
                        indexed.album_id,
                        on_progress=lambda completed, total: self._people_progress(
                            job_id,
                            result,
                            completed,
                            total,
                            start=people_start,
                            span=people_span,
                        ),
                        should_cancel=lambda: self._is_cancel_requested(job_id),
                    )
                except PeopleCancelledError:
                    self._mark_cancelled(job_id)
                    return
                result.pop("people_progress", None)
                result["people"] = {
                    "album_id": people.album_id,
                    "total_faces": people.total_faces,
                    "cluster_count": people.cluster_count,
                    "computed_count": people.computed_count,
                    "reused_count": people.reused_count,
                    "provider": people.provider,
                    "duration_ms": people.duration_ms,
                }
                self._set_stage(
                    job_id,
                    stage="people",
                    progress=people_start + people_span,
                    result=result,
                )
            if self._cancel_if_requested(job_id):
                return
            self._complete(job_id, result)
        except Exception as error:
            logger.exception("prepare job failed; job_id=%s", job_id)
            with self.database.connect() as connection:
                connection.execute(
                    """
                    UPDATE jobs SET status = 'failed', stage = 'failed', error = ?,
                        finished_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND status NOT IN ('completed', 'cancelled')
                    """,
                    (str(error), job_id),
                )
        finally:
            self._clear_progress(job_id)
            with self.lock:
                self.futures.pop(job_id, None)

    def _set_stage(
        self,
        job_id: str,
        *,
        stage: str,
        progress: float,
        status: str | None = None,
        result: dict[str, object] | None = None,
    ) -> None:
        result_json = (
            json.dumps(result, ensure_ascii=False) if result is not None else None
        )
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status = COALESCE(?, status), stage = ?, progress = ?,
                    result_json = COALESCE(?, result_json),
                    started_at = CASE WHEN ? = 'running' THEN COALESCE(started_at, CURRENT_TIMESTAMP)
                                      ELSE started_at END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, stage, progress, result_json, status, job_id),
            )

    def _cancel_if_requested(self, job_id: str) -> bool:
        if self._is_cancel_requested(job_id):
            self._mark_cancelled(job_id)
            return True
        return False

    def _is_cancel_requested(self, job_id: str) -> bool:
        """Read only the cancellation bit on hot per-item callback paths."""

        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return row is None or bool(row["cancel_requested"])

    def _begin_progress(self, job_id: str, stage: str) -> None:
        with self.progress_lock:
            self.progress_checkpoints[(job_id, stage)] = (0.0, time.monotonic())

    def _clear_progress(self, job_id: str) -> None:
        with self.progress_lock:
            stale = [key for key in self.progress_checkpoints if key[0] == job_id]
            for key in stale:
                self.progress_checkpoints.pop(key, None)

    def _should_persist_progress(
        self,
        job_id: str,
        stage: str,
        fraction: float,
        *,
        completed: int,
        total: int,
    ) -> bool:
        now = time.monotonic()
        key = (job_id, stage)
        with self.progress_lock:
            previous_fraction, previous_time = self.progress_checkpoints.get(
                key, (0.0, now)
            )
            elapsed = now - previous_time
            advanced = max(0.0, fraction - previous_fraction)
            persist = (
                completed >= total
                or advanced >= PROGRESS_LARGE_FRACTION
                or elapsed >= PROGRESS_HEARTBEAT_SECONDS
                or (
                    advanced >= PROGRESS_MIN_FRACTION
                    and elapsed >= PROGRESS_MIN_INTERVAL_SECONDS
                )
            )
            if persist:
                self.progress_checkpoints[key] = (fraction, now)
            elif key not in self.progress_checkpoints:
                self.progress_checkpoints[key] = (previous_fraction, previous_time)
        return persist

    def _embedding_progress(
        self,
        job_id: str,
        result: dict[str, object],
        completed: int,
        total: int,
        *,
        start: float,
        span: float,
    ) -> None:
        result["embedding_progress"] = {"completed": completed, "total": total}
        fraction = min(max(completed / max(total, 1), 0.0), 1.0)
        if not self._should_persist_progress(
            job_id,
            "embedding",
            fraction,
            completed=completed,
            total=total,
        ):
            return
        self._set_stage(
            job_id,
            stage="embedding",
            progress=start + span * fraction,
            result=result,
        )

    def _indexing_progress(
        self,
        job_id: str,
        completed: int,
        total: int,
        *,
        start: float,
        span: float,
    ) -> None:
        fraction = min(max(completed / max(total, 1), 0.0), 1.0)
        if not self._should_persist_progress(
            job_id,
            "indexing",
            fraction,
            completed=completed,
            total=total,
        ):
            return
        self._set_stage(
            job_id,
            stage="indexing",
            progress=start + span * fraction,
            result={
                "indexing_progress": {
                    "completed": completed,
                    "total": total,
                }
            },
        )

    def _people_progress(
        self,
        job_id: str,
        result: dict[str, object],
        completed: int,
        total: int,
        *,
        start: float,
        span: float,
    ) -> None:
        result["people_progress"] = {"completed": completed, "total": total}
        fraction = min(max(completed / max(total, 1), 0.0), 1.0)
        if not self._should_persist_progress(
            job_id,
            "people",
            fraction,
            completed=completed,
            total=total,
        ):
            return
        self._set_stage(
            job_id,
            stage="people",
            progress=start + span * fraction,
            result=result,
        )

    def _mark_cancelled(self, job_id: str) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status = 'cancelled', stage = 'cancelled',
                    finished_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status NOT IN ('completed', 'failed')
                """,
                (job_id,),
            )

    def _complete(self, job_id: str, result: dict[str, object]) -> None:
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET status = 'completed', stage = 'completed', progress = 1,
                    result_json = ?, error = NULL, finished_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND cancel_requested = 0
                """,
                (json.dumps(result, ensure_ascii=False), job_id),
            )
        if cursor.rowcount == 0:
            self._mark_cancelled(job_id)


def _progress_ranges(request: PrepareJobRequest) -> dict[str, tuple[float, float]]:
    """Allocate monotonic whole-job progress to only the requested stages."""

    if request.include_quality and request.include_embeddings:
        ranges = {"indexing": (0.05, 0.50)}
        if request.include_people:
            ranges["embedding"] = (0.55, 0.27)
            ranges["people"] = (0.82, 0.18)
        else:
            ranges["embedding"] = (0.55, 0.45)
        return ranges

    if request.include_embeddings and request.include_people:
        return {
            "indexing": (0.0, 0.10),
            "embedding": (0.10, 0.45),
            "people": (0.55, 0.45),
        }

    if request.include_embeddings:
        index_span = 0.50 if request.include_quality else 0.15
        return {
            "indexing": (0.0, index_span),
            "embedding": (index_span, 1.0 - index_span),
        }

    if request.include_people:
        index_span = 0.55 if request.include_quality else 0.15
        return {
            "indexing": (0.0, index_span),
            "people": (index_span, 1.0 - index_span),
        }

    # Base import and quality-only analysis both finish entirely in AlbumIndexer.
    return {"indexing": (0.0, 1.0)}


def _job_response(row: object) -> JobResponse:
    return JobResponse(
        id=row["id"],
        job_type=row["job_type"],
        status=row["status"],
        stage=row["stage"],
        progress=float(row["progress"]),
        payload=json.loads(row["payload_json"]),
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        error=row["error"],
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def get_persisted_job(database: Database, job_id: str) -> JobResponse:
    database.initialize()
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"job not found: {job_id}")
    return _job_response(row)


def list_persisted_jobs(
    database: Database,
    *,
    limit: int,
    offset: int,
    status: str | None,
) -> JobListResponse:
    database.initialize()
    where = ""
    parameters: list[object] = []
    if status is not None:
        where = "WHERE status = ?"
        parameters.append(status)
    with database.connect() as connection:
        total = int(
            connection.execute(
                f"SELECT COUNT(*) FROM jobs {where}", parameters
            ).fetchone()[0]
        )
        rows = connection.execute(
            f"SELECT * FROM jobs {where} ORDER BY created_at DESC, id DESC "
            "LIMIT ? OFFSET ?",
            [*parameters, limit, offset],
        ).fetchall()
    return JobListResponse(
        items=[_job_response(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )
