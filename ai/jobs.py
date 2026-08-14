from __future__ import annotations

import json
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ai.index import AlbumIndexer
from ai.index.embedding import create_embedding_provider
from ai.people import PeopleIndexer, create_face_provider
from ai.retrieval import RetrievalService
from ai.schemas import JobListResponse, JobResponse, PrepareJobRequest
from ai.storage import Database


logger = logging.getLogger("norma.ai.jobs")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class PrepareJobManager:
    """Single-worker, SQLite-backed orchestration for album preparation."""

    def __init__(
        self,
        database: Database,
        data_dir: Path,
        embedding_provider: str,
        face_provider: str,
    ) -> None:
        self.database = database
        self.data_dir = data_dir
        self.embedding_provider = embedding_provider
        self.face_provider = face_provider
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="norma-prepare"
        )
        self.futures: dict[str, Future[None]] = {}
        self.lock = threading.Lock()

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
            self._set_stage(job_id, status="running", stage="indexing", progress=0.05)
            request = PrepareJobRequest.model_validate(job.payload)
            indexed = AlbumIndexer(self.database, self.data_dir).index(
                Path(request.folder), request.name
            )
            result: dict[str, object] = {
                "album": {
                    "album_id": indexed.album_id,
                    "name": indexed.name,
                    "source_path": indexed.source_path,
                    "total": indexed.total,
                    "rejected": indexed.rejected,
                    "similar_groups": indexed.similar_groups,
                    "duration_ms": indexed.duration_ms,
                    "provider": indexed.provider,
                    "errors": indexed.errors,
                }
            }
            self._set_stage(job_id, stage="embedding", progress=0.55, result=result)
            if self._cancel_if_requested(job_id):
                return
            embedded = RetrievalService(
                self.database,
                self.data_dir,
                create_embedding_provider(self.embedding_provider),
            ).embed_album(indexed.album_id)
            result["embedding"] = embedded.model_dump()
            self._set_stage(job_id, stage="people", progress=0.82, result=result)
            if self._cancel_if_requested(job_id):
                return
            if request.include_people:
                people = PeopleIndexer(
                    self.database,
                    self.data_dir,
                    create_face_provider(self.face_provider),
                ).index(indexed.album_id)
                result["people"] = {
                    "album_id": people.album_id,
                    "total_faces": people.total_faces,
                    "cluster_count": people.cluster_count,
                    "provider": people.provider,
                    "duration_ms": people.duration_ms,
                }
            else:
                result["people"] = None
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
        if self.get(job_id).cancel_requested:
            self._mark_cancelled(job_id)
            return True
        return False

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
