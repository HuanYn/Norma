from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from ai.aesthetics.provider import AestheticsProviderUnavailableError
from ai.aesthetics.service import (
    AestheticsCancelledError,
    AestheticsService,
    AestheticsSourceChangedError,
)
from ai.jobs import get_persisted_job
from ai.schemas import JobResponse


JOB_TYPE = "analyze_aesthetics"


class AestheticsJobManager:
    """Single-worker, restart-visible assessment jobs; no background auto-analysis."""

    def __init__(self, service: AestheticsService) -> None:
        self.service = service
        self.database = service.database
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="norma-aesthetics")
        self._lock = threading.RLock()
        self._closed = False

    def start(self) -> None:
        self.service.initialize()
        # Never silently restart expensive inference after process interruption.
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE jobs SET status='failed', stage='interrupted',
                   error='Assessment worker restarted; completed image scores are cached. Retry explicitly.',
                   finished_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                   WHERE job_type=? AND status IN ('running', 'queued')""",
                (JOB_TYPE,),
            )

    def submit(self, album_id: str, *, force: bool = False) -> JobResponse:
        if not self.service.provider.status().get("configured", False):
            raise AestheticsProviderUnavailableError("Learned assessment models are not configured; check /aesthetics/status.")
        rows = self.service._rows(album_id)
        if not rows:
            raise ValueError("The album is empty.")
        if len(rows) > 5000:
            raise ValueError("Demo assessment supports at most 5000 indexed photos per album.")
        with self._lock:
            if self._closed:
                raise RuntimeError("Assessment worker is stopping.")
            with self.database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                active = connection.execute(
                    "SELECT payload_json FROM jobs WHERE job_type=? AND status IN ('queued','running')",
                    (JOB_TYPE,),
                ).fetchall()
                if any(json.loads(row["payload_json"]).get("album_id") == album_id for row in active):
                    raise ValueError("An aesthetics job is already active for this album.")
                if len(active) >= 8:
                    raise ValueError("Assessment queue is full; wait for current jobs to finish.")
                job_id = uuid.uuid4().hex
                connection.execute(
                    """INSERT INTO jobs(id, job_type, status, stage, progress, payload_json)
                       VALUES (?, ?, 'queued', 'queued', 0, ?)""",
                    (job_id, JOB_TYPE, json.dumps({"album_id": album_id, "force": force})),
                )
            self.executor.submit(self._run, job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> JobResponse:
        response = get_persisted_job(self.database, job_id)
        if response.job_type != JOB_TYPE:
            raise KeyError("aesthetics job not found")
        return response

    def cancel(self, job_id: str) -> JobResponse:
        self.get(job_id)
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE jobs SET cancel_requested=1, updated_at=CURRENT_TIMESTAMP,
                   status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,
                   stage=CASE WHEN status='queued' THEN 'cancelled' ELSE stage END,
                   finished_at=CASE WHEN status='queued' THEN CURRENT_TIMESTAMP ELSE finished_at END
                   WHERE id=? AND job_type=? AND status IN ('queued','running')""",
                (job_id, JOB_TYPE),
            )
        return self.get(job_id)

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            with self.database.connect() as connection:
                connection.execute(
                    """UPDATE jobs SET cancel_requested=1, updated_at=CURRENT_TIMESTAMP
                       WHERE job_type=? AND status IN ('queued','running')""", (JOB_TYPE,)
                )
        self.executor.shutdown(wait=True, cancel_futures=False)

    def _cancelled(self, job_id: str) -> bool:
        with self.database.connect() as connection:
            row = connection.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row is None or bool(row["cancel_requested"])

    def _progress(self, job_id: str, done: int, total: int) -> None:
        # Percentage describes completed photos, not fabricated model-load ETA.
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE jobs SET progress=?, stage=?, updated_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status='running'""",
                (min(done / total, 0.999) if total else 0,
                 f"scoring_images:{done}/{total}", job_id),
            )

    def _run(self, job_id: str) -> None:
        try:
            job = self.get(job_id)
            if job.status == "cancelled" or job.cancel_requested:
                raise AestheticsCancelledError()
            with self.database.connect() as connection:
                changed = connection.execute(
                    """UPDATE jobs SET status='running', stage='validating_models_and_sources',
                       started_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                       WHERE id=? AND status='queued' AND cancel_requested=0""", (job_id,)
                ).rowcount
            if not changed:
                raise AestheticsCancelledError()
            result = self.service.analyze(
                str(job.payload["album_id"]),
                force=bool(job.payload.get("force", False)),
                on_progress=lambda done, total: self._progress(job_id, done, total),
                should_cancel=lambda: self._cancelled(job_id),
            )
            with self.database.connect() as connection:
                changed = connection.execute(
                    """UPDATE jobs SET status='completed', stage='completed', progress=1,
                       result_json=?, error=NULL, finished_at=CURRENT_TIMESTAMP,
                       updated_at=CURRENT_TIMESTAMP WHERE id=? AND cancel_requested=0""",
                    (json.dumps(result), job_id),
                ).rowcount
            if not changed:
                raise AestheticsCancelledError()
        except AestheticsCancelledError:
            self._finish_error(job_id, "cancelled", None)
        except (AestheticsProviderUnavailableError, AestheticsSourceChangedError) as error:
            self._finish_error(job_id, "failed", str(error))
        except Exception:
            self._finish_error(job_id, "failed", "Assessment failed; verify the photo index and model environment before retrying.")

    def _finish_error(self, job_id: str, status: str, message: str | None) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE jobs SET status=?, stage=?, error=?,
                   finished_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status!='completed'""",
                (status, status, message, job_id),
            )
