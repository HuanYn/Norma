from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ai.schemas import (
    CacheCategoryUsage,
    CacheGcRequest,
    CacheGcResponse,
    CacheQuotaRequest,
    CacheQuotaResponse,
    CacheUsageResponse,
    MaintenanceRunListResponse,
    MaintenanceRunSummary,
)
from ai.storage import Database


class CacheMaintenanceService:
    """Audit disposable state, enforce a safe budget, and persist maintenance runs."""

    CACHE_ROOT_NAMES = ("thumbnails", "embeddings", "faces")

    def __init__(
        self,
        database: Database,
        data_dir: Path,
        *,
        model_cache_dir: Path | None = None,
        budget_bytes: int | None = None,
    ) -> None:
        self.database = database
        self.data_dir = data_dir.resolve()
        self.model_cache_dir = (model_cache_dir or (self.data_dir / "models")).resolve()
        self.budget_bytes = budget_bytes

    def collect(self, request: CacheGcRequest) -> CacheGcResponse:
        return self._audited("cache_gc", request.dry_run, request, self._collect)

    def usage(self, *, budget_bytes: int | None = None) -> CacheUsageResponse:
        self.database.initialize()
        effective_budget = (
            budget_bytes if budget_bytes is not None else self.budget_bytes
        )
        categories = {
            name: self._tree_usage((self.data_dir / name).resolve())
            for name in self.CACHE_ROOT_NAMES
        }
        models = self._tree_usage(self.model_cache_dir)
        categories["models"] = models
        generated_files = sum(categories[name].files for name in self.CACHE_ROOT_NAMES)
        generated_bytes = sum(categories[name].bytes for name in self.CACHE_ROOT_NAMES)
        database_bytes = sum(
            path.stat().st_size
            for path in (
                self.database.path,
                Path(f"{self.database.path}-wal"),
                Path(f"{self.database.path}-shm"),
            )
            if path.is_file()
        )
        total = generated_bytes + models.bytes + database_bytes
        over = max(0, total - effective_budget) if effective_budget else 0
        return CacheUsageResponse(
            data_dir=str(self.data_dir),
            categories=categories,
            generated_files=generated_files,
            generated_bytes=generated_bytes,
            model_files=models.files,
            model_bytes=models.bytes,
            database_bytes=database_bytes,
            total_state_bytes=total,
            budget_bytes=effective_budget,
            over_budget=bool(over),
            over_budget_bytes=over,
        )

    def enforce_quota(self, request: CacheQuotaRequest) -> CacheQuotaResponse:
        def execute(_: CacheQuotaRequest) -> CacheQuotaResponse:
            budget = request.budget_bytes or self.budget_bytes
            if budget is None:
                raise ValueError(
                    "cache budget is not configured; provide budget_bytes or "
                    "NORMA_CACHE_BUDGET_GB"
                )
            before = self.usage(budget_bytes=budget)
            collection = self._collect(
                CacheGcRequest(
                    dry_run=request.dry_run,
                    min_age_seconds=request.min_age_seconds,
                )
            )
            after = self.usage(budget_bytes=budget)
            reclaimable = (
                collection.orphan_bytes if request.dry_run else collection.deleted_bytes
            )
            projected = max(0, before.total_state_bytes - reclaimable)
            warnings: list[str] = []
            if request.dry_run and collection.orphan_files:
                warnings.append("dry-run only; eligible orphan files were not deleted")
            if before.over_budget and projected > budget:
                warnings.append(
                    "eligible orphan cleanup cannot satisfy the budget; referenced "
                    "caches, models, or database state require an explicit policy change"
                )
            if not before.over_budget:
                warnings.append("state is already within the configured budget")
            return CacheQuotaResponse(
                dry_run=request.dry_run,
                budget_bytes=budget,
                usage_before=before,
                collection=collection,
                usage_after=after,
                projected_total_state_bytes=projected,
                projected_satisfied=projected <= budget,
                satisfied=after.total_state_bytes <= budget,
                warnings=warnings,
            )

        return self._audited("quota_enforce", request.dry_run, request, execute)

    def list_runs(self, *, limit: int, offset: int) -> MaintenanceRunListResponse:
        self.database.initialize()
        with self.database.connect() as connection:
            total = int(
                connection.execute("SELECT COUNT(*) FROM maintenance_runs").fetchone()[
                    0
                ]
            )
            rows = connection.execute(
                """
                SELECT * FROM maintenance_runs
                ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return MaintenanceRunListResponse(
            items=[
                MaintenanceRunSummary(
                    id=row["id"],
                    operation=row["operation"],
                    status=row["status"],
                    dry_run=bool(row["dry_run"]),
                    request=json.loads(row["request_json"]),
                    result=json.loads(row["result_json"])
                    if row["result_json"]
                    else None,
                    error=row["error"],
                    created_at=row["created_at"],
                    finished_at=row["finished_at"],
                )
                for row in rows
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    def _collect(self, request: CacheGcRequest) -> CacheGcResponse:
        self.database.initialize()
        if not request.dry_run:
            self._ensure_no_active_jobs()
        roots = [(self.data_dir / name).resolve() for name in self.CACHE_ROOT_NAMES]
        references = self._references()
        now_ns = time.time_ns()
        min_age_ns = request.min_age_seconds * 1_000_000_000
        scanned_files = referenced_files = orphan_files = orphan_bytes = 0
        deleted_files = deleted_bytes = young_orphan_files = 0
        skipped_unsafe_files = failed_files = 0
        samples: list[str] = []
        errors: list[str] = []

        for root in roots:
            if not root.is_dir():
                continue
            for candidate in sorted(path for path in root.rglob("*") if path.is_file()):
                scanned_files += 1
                try:
                    resolved = candidate.resolve(strict=True)
                    stat = resolved.stat()
                except OSError as error:
                    failed_files += 1
                    if len(errors) < 10:
                        errors.append(f"unable to inspect {candidate}: {error}")
                    continue
                if not resolved.is_relative_to(root):
                    skipped_unsafe_files += 1
                    continue
                if resolved in references:
                    referenced_files += 1
                    continue
                if max(0, now_ns - stat.st_mtime_ns) < min_age_ns:
                    young_orphan_files += 1
                    continue
                orphan_files += 1
                orphan_bytes += stat.st_size
                if len(samples) < 20:
                    samples.append(str(resolved.relative_to(self.data_dir)))
                if request.dry_run:
                    continue
                try:
                    resolved.unlink()
                except OSError as error:
                    failed_files += 1
                    if len(errors) < 10:
                        errors.append(f"unable to delete {resolved}: {error}")
                else:
                    deleted_files += 1
                    deleted_bytes += stat.st_size

        return CacheGcResponse(
            dry_run=request.dry_run,
            min_age_seconds=request.min_age_seconds,
            scanned_files=scanned_files,
            referenced_files=referenced_files,
            orphan_files=orphan_files,
            orphan_bytes=orphan_bytes,
            deleted_files=deleted_files,
            deleted_bytes=deleted_bytes,
            young_orphan_files=young_orphan_files,
            skipped_unsafe_files=skipped_unsafe_files,
            failed_files=failed_files,
            orphan_samples=samples,
            errors=errors,
        )

    def _audited(self, operation, dry_run, request, function):
        self.database.initialize()
        run_id = uuid.uuid4().hex
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO maintenance_runs(
                    id, operation, status, dry_run, request_json
                ) VALUES (?, ?, 'running', ?, ?)
                """,
                (run_id, operation, int(dry_run), request.model_dump_json()),
            )
        try:
            result = function(request)
        except Exception as error:
            with self.database.connect() as connection:
                connection.execute(
                    """
                    UPDATE maintenance_runs SET status = 'failed', error = ?,
                        finished_at = CURRENT_TIMESTAMP WHERE id = ?
                    """,
                    (str(error), run_id),
                )
            raise
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE maintenance_runs SET status = 'completed', result_json = ?,
                    finished_at = CURRENT_TIMESTAMP WHERE id = ?
                """,
                (result.model_dump_json(), run_id),
            )
        return result

    @staticmethod
    def _tree_usage(root: Path) -> CacheCategoryUsage:
        files = total_bytes = 0
        if root.is_dir():
            for candidate in root.rglob("*"):
                if not candidate.is_file():
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                    if not resolved.is_relative_to(root):
                        continue
                    size = resolved.stat().st_size
                except OSError:
                    continue
                files += 1
                total_bytes += size
        return CacheCategoryUsage(files=files, bytes=total_bytes)

    def _references(self) -> set[Path]:
        references: set[Path] = set()
        with self.database.connect() as connection:
            photos = connection.execute(
                "SELECT album_id, thumbnail_path, embedding_path FROM photos"
            ).fetchall()
            faces = connection.execute(
                """
                SELECT f.id, f.embedding_path, p.album_id, p.face_provider
                FROM faces f JOIN photos p ON p.id = f.photo_id
                """
            ).fetchall()
        for row in photos:
            for key in ("thumbnail_path", "embedding_path"):
                if row[key]:
                    references.add(Path(row[key]).resolve())
        for row in faces:
            if row["embedding_path"]:
                references.add(Path(row["embedding_path"]).resolve())
            if row["face_provider"]:
                references.add(
                    (
                        self.data_dir
                        / "faces"
                        / row["face_provider"]
                        / row["album_id"]
                        / "thumbnails"
                        / f"{row['id']}.jpg"
                    ).resolve()
                )
        return references

    def _ensure_no_active_jobs(self) -> None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM jobs
                WHERE status IN ('queued', 'running')
                """
            ).fetchone()
        if int(row["count"] if row else 0):
            raise ValueError(
                "cache collection is blocked while jobs are queued or running"
            )
