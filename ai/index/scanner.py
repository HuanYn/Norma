from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from ai.index.quality import analyze_quality
from ai.index.similarity import (
    assign_similarity_groups,
    difference_hash,
    perceptual_hash,
)
from ai.schemas import AlbumIndexResponse, PhotoSummary
from ai.storage import Database


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}
PROVIDER_NAME = "pillow-opencv-fallback-v1"
SCAN_WORKER_CAP = 4
DEFAULT_SCAN_WORKERS = max(1, min(SCAN_WORKER_CAP, os.cpu_count() or 1))
SCAN_IN_FLIGHT_MULTIPLIER = 2


class IndexingCancelledError(RuntimeError):
    """Raised when album indexing is cancelled before its atomic persistence."""

    pass


@dataclass(slots=True)
class ScannedPhoto:
    id: str
    path: Path
    thumbnail_path: Path
    width: int
    height: int
    file_size: int
    source_mtime_ns: int
    capture_time: str | None
    quality_score: float | None
    blur_score: float | None
    phash: str | None
    dhash: str | None
    auto_reject: bool
    reject_reason: str | None
    quality_flags: list[str]
    metadata: dict[str, object]
    similarity_group: str | None = None
    reused: bool = False


@dataclass(slots=True)
class _ScanOutcome:
    index: int
    photo: ScannedPhoto | None = None
    error: str | None = None


class AlbumIndexer:
    def __init__(
        self,
        database: Database,
        data_dir: Path,
        *,
        scan_workers: int | None = None,
    ) -> None:
        resolved_workers = (
            DEFAULT_SCAN_WORKERS if scan_workers is None else scan_workers
        )
        if resolved_workers < 1 or resolved_workers > SCAN_WORKER_CAP:
            raise ValueError(f"scan_workers must be between 1 and {SCAN_WORKER_CAP}")
        self.database = database
        self.data_dir = data_dir
        self.scan_workers = resolved_workers

    def index(
        self,
        folder: Path,
        album_name: str | None = None,
        *,
        analyze_quality: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AlbumIndexResponse:
        started = time.perf_counter()
        folder = folder.expanduser().resolve(strict=True)
        if not folder.is_dir():
            raise NotADirectoryError(str(folder))

        paths = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            raise ValueError("所选文件夹中没有 JPG/JPEG 图片")

        album_id = uuid.uuid5(
            uuid.NAMESPACE_URL, f"norma:album:{str(folder).casefold()}"
        ).hex
        name = (album_name or folder.name).strip() or "Untitled Album"
        thumbnail_dir = self.data_dir / "thumbnails" / album_id
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        existing = self._existing_photos(album_id)
        membership_changed = set(existing) != {
            str(path.resolve()).casefold() for path in paths
        }

        photos, errors = self._scan_paths(
            paths,
            album_id,
            thumbnail_dir,
            existing,
            analyze_quality=analyze_quality,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

        if should_cancel is not None and should_cancel():
            raise IndexingCancelledError("album indexing cancelled before commit")

        quality_complete = all(_photo_quality_complete(photo) for photo in photos)
        if quality_complete and (analyze_quality or membership_changed):
            groups = assign_similarity_groups(
                [(photo.phash, photo.dhash) for photo in photos]
            )
            for photo in photos:
                photo.similarity_group = None
            for index, group_id in groups.items():
                photos[index].similarity_group = group_id
        else:
            groups = {}
            if not quality_complete:
                # Similarity groups are an album-wide result. A partial hash set can
                # make a group look complete when it is not, so expose no groups
                # until every current photo has quality/hash analysis.
                for photo in photos:
                    photo.similarity_group = None

        if should_cancel is not None and should_cancel():
            raise IndexingCancelledError("album indexing cancelled before commit")
        self._persist(album_id, name, folder, photos)
        summaries = [self._to_summary(album_id, photo) for photo in photos]
        return AlbumIndexResponse(
            album_id=album_id,
            name=name,
            source_path=str(folder),
            total=len(summaries),
            computed_count=sum(not photo.reused for photo in photos),
            reused_count=sum(photo.reused for photo in photos),
            rejected=sum(photo.auto_reject for photo in summaries),
            similar_groups=len(
                {
                    photo.similarity_group
                    for photo in photos
                    if photo.similarity_group is not None
                }
            ),
            duration_ms=round((time.perf_counter() - started) * 1000),
            provider=PROVIDER_NAME,
            photos=summaries,
            errors=errors,
        )

    def _scan_paths(
        self,
        paths: list[Path],
        album_id: str,
        thumbnail_dir: Path,
        existing: dict[str, object],
        *,
        analyze_quality: bool,
        on_progress: Callable[[int, int], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> tuple[list[ScannedPhoto], list[str]]:
        if should_cancel is not None and should_cancel():
            raise IndexingCancelledError("album indexing cancelled between photos")

        total = len(paths)
        outcomes: list[_ScanOutcome | None] = [None] * total
        max_in_flight = self.scan_workers * SCAN_IN_FLIGHT_MULTIPLIER
        executor = ThreadPoolExecutor(
            max_workers=self.scan_workers,
            thread_name_prefix="norma-scan",
        )
        pending: dict[Future[_ScanOutcome], int] = {}
        next_index = 0

        def fill_pending() -> None:
            nonlocal next_index
            while next_index < total and len(pending) < max_in_flight:
                index = next_index
                path = paths[index]
                pending[
                    executor.submit(
                        self._scan_path,
                        index,
                        path,
                        album_id,
                        thumbnail_dir,
                        existing,
                        analyze_quality,
                    )
                ] = index
                next_index += 1

        try:
            fill_pending()
            completed = 0
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                cancelled = False
                for future in done:
                    index = pending.pop(future)
                    outcome = future.result()
                    if outcome.index != index:
                        raise RuntimeError("photo scan outcome index mismatch")
                    outcomes[index] = outcome
                    completed += 1
                    if on_progress is not None:
                        on_progress(completed, total)
                    if should_cancel is not None and should_cancel():
                        cancelled = True
                        break
                if cancelled:
                    raise IndexingCancelledError(
                        "album indexing cancelled between photos"
                    )
                fill_pending()
        except BaseException:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        if any(outcome is None for outcome in outcomes):
            raise RuntimeError("album indexing finished with missing photo outcomes")
        finished = [outcome for outcome in outcomes if outcome is not None]
        photos = [outcome.photo for outcome in finished if outcome.photo is not None]
        errors = [outcome.error for outcome in finished if outcome.error is not None]
        return photos, errors

    def _scan_path(
        self,
        index: int,
        path: Path,
        album_id: str,
        thumbnail_dir: Path,
        existing: dict[str, object],
        analyze_quality: bool,
    ) -> _ScanOutcome:
        cached: object | None = None
        try:
            resolved = path.resolve()
            cached = existing.get(str(resolved).casefold())
            source_matches = cached is not None and _cached_source_matches(
                cached, resolved
            )
            cached_photo: ScannedPhoto | None = None
            if source_matches:
                try:
                    cached_photo = self._cached_photo(cached)
                except (TypeError, ValueError, json.JSONDecodeError):
                    cached_photo = None
                if (
                    cached_photo is not None
                    and cached_photo.thumbnail_path.is_file()
                    and (not analyze_quality or _photo_quality_complete(cached_photo))
                ):
                    return _ScanOutcome(index=index, photo=cached_photo)

            photo_id = str(cached["id"]) if cached is not None else None
            if analyze_quality:
                if photo_id is None:
                    photo = self._scan_photo(resolved, album_id, thumbnail_dir)
                else:
                    photo = self._scan_photo(
                        resolved,
                        album_id,
                        thumbnail_dir,
                        photo_id=photo_id,
                    )
            else:
                if photo_id is None:
                    photo = self._scan_photo(
                        resolved,
                        album_id,
                        thumbnail_dir,
                        include_quality=False,
                    )
                else:
                    photo = self._scan_photo(
                        resolved,
                        album_id,
                        thumbnail_dir,
                        photo_id=photo_id,
                        include_quality=False,
                    )
                if (
                    source_matches
                    and cached_photo is not None
                    and _photo_quality_complete(cached_photo)
                ):
                    _copy_quality_analysis(cached_photo, photo)
            return _ScanOutcome(index=index, photo=photo)
        except Exception as error:  # surfaced in the API; never silently swallowed
            preserved: ScannedPhoto | None = None
            if cached is not None:
                try:
                    preserved = self._cached_photo(cached)
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            return _ScanOutcome(
                index=index,
                photo=preserved,
                error=f"{path.name}: {error}",
            )

    def _existing_photos(self, album_id: str) -> dict[str, object]:
        self.database.initialize()
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, absolute_path, thumbnail_path, width, height,
                       file_size, source_mtime_ns, capture_time, quality_score,
                       blur_score, phash, dhash, auto_reject, reject_reason,
                       metadata_json, similarity_group
                FROM photos WHERE album_id = ?
                """,
                (album_id,),
            ).fetchall()
        return {str(row["absolute_path"]).casefold(): row for row in rows}

    @staticmethod
    def _cached_photo(row: object) -> ScannedPhoto:
        metadata = json.loads(row["metadata_json"] or "{}")
        return ScannedPhoto(
            id=row["id"],
            path=Path(row["absolute_path"]),
            thumbnail_path=Path(row["thumbnail_path"]),
            width=int(row["width"]),
            height=int(row["height"]),
            file_size=int(row["file_size"]),
            source_mtime_ns=int(row["source_mtime_ns"]),
            capture_time=row["capture_time"],
            quality_score=(
                float(row["quality_score"])
                if row["quality_score"] is not None
                else None
            ),
            blur_score=(
                float(row["blur_score"]) if row["blur_score"] is not None else None
            ),
            phash=row["phash"],
            dhash=row["dhash"],
            auto_reject=bool(row["auto_reject"]),
            reject_reason=row["reject_reason"],
            quality_flags=list(metadata.get("quality_flags", [])),
            metadata=metadata,
            similarity_group=row["similarity_group"],
            reused=True,
        )

    def _scan_photo(
        self,
        path: Path,
        album_id: str,
        thumbnail_dir: Path,
        *,
        photo_id: str | None = None,
        include_quality: bool = True,
    ) -> ScannedPhoto:
        before = path.stat()
        photo_id = (
            photo_id
            or uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"norma:photo:{album_id}:{str(path.resolve()).casefold()}",
            ).hex
        )
        thumbnail_path = thumbnail_dir / f"{photo_id}.jpg"

        with Image.open(path) as source:
            capture_time = _capture_time(source)
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            quality = analyze_quality(image) if include_quality else None
            phash = perceptual_hash(image) if include_quality else None
            dhash = difference_hash(image) if include_quality else None
            image.thumbnail((480, 480), Image.Resampling.LANCZOS)
            image.save(thumbnail_path, "JPEG", quality=84, optimize=True)

        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("source image changed during read-only indexing")

        metadata: dict[str, object] = {"provider": PROVIDER_NAME}
        if quality is not None:
            metadata.update(
                {
                    "brightness": quality.brightness,
                    "contrast": quality.contrast,
                    "overexposed_ratio": quality.overexposed_ratio,
                    "underexposed_ratio": quality.underexposed_ratio,
                    "entropy": quality.entropy,
                    "quality_flags": list(quality.flags),
                }
            )
        return ScannedPhoto(
            id=photo_id,
            path=path.resolve(),
            thumbnail_path=thumbnail_path.resolve(),
            width=width,
            height=height,
            file_size=before.st_size,
            source_mtime_ns=before.st_mtime_ns,
            capture_time=capture_time,
            quality_score=quality.quality_score if quality is not None else None,
            blur_score=quality.blur_score if quality is not None else None,
            phash=phash,
            dhash=dhash,
            auto_reject=quality.auto_reject if quality is not None else False,
            reject_reason=quality.reject_reason if quality is not None else None,
            quality_flags=list(quality.flags) if quality is not None else [],
            metadata=metadata,
        )

    def _persist(
        self,
        album_id: str,
        name: str,
        folder: Path,
        photos: list[ScannedPhoto],
    ) -> None:
        self.database.initialize()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO albums(id, name, source_path, indexed_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    source_path=excluded.source_path,
                    indexed_at=CURRENT_TIMESTAMP
                """,
                (album_id, name, str(folder)),
            )
            current_ids = {photo.id for photo in photos}
            existing = connection.execute(
                "SELECT id, file_size, source_mtime_ns FROM photos WHERE album_id = ?",
                (album_id,),
            ).fetchall()
            existing_by_id = {row["id"]: row for row in existing}
            stale = [row["id"] for row in existing if row["id"] not in current_ids]
            changed = [
                photo.id
                for photo in photos
                if photo.id in existing_by_id
                and (
                    existing_by_id[photo.id]["file_size"] != photo.file_size
                    or existing_by_id[photo.id]["source_mtime_ns"]
                    != photo.source_mtime_ns
                )
            ]
            added = [photo.id for photo in photos if photo.id not in existing_by_id]
            if stale or changed or added:
                # Clusters are album-wide. Once membership or any source changes,
                # retaining only the unaffected face rows would make the album look
                # ready while its clusters describe an obsolete snapshot.
                connection.execute(
                    """DELETE FROM faces WHERE photo_id IN
                       (SELECT id FROM photos WHERE album_id = ?)""",
                    (album_id,),
                )
                connection.execute(
                    "DELETE FROM person_clusters WHERE album_id = ?", (album_id,)
                )
                connection.execute(
                    """
                    UPDATE photos SET face_provider = NULL,
                        face_source_size = NULL, face_source_mtime_ns = NULL,
                        face_processed = 0, face_count = 0
                    WHERE album_id = ?
                    """,
                    (album_id,),
                )
            if stale:
                connection.executemany(
                    "DELETE FROM photos WHERE id = ?", [(id_,) for id_ in stale]
                )
            connection.executemany(
                """
                INSERT INTO photos(
                    id, album_id, absolute_path, thumbnail_path, width, height,
                    capture_time, quality_score, blur_score, similarity_group,
                    file_size, source_mtime_ns, phash, dhash, auto_reject,
                    reject_reason, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    absolute_path=excluded.absolute_path,
                    thumbnail_path=excluded.thumbnail_path,
                    width=excluded.width,
                    height=excluded.height,
                    capture_time=excluded.capture_time,
                    quality_score=excluded.quality_score,
                    blur_score=excluded.blur_score,
                    similarity_group=excluded.similarity_group,
                    file_size=excluded.file_size,
                    source_mtime_ns=excluded.source_mtime_ns,
                    phash=excluded.phash,
                    dhash=excluded.dhash,
                    auto_reject=excluded.auto_reject,
                    reject_reason=excluded.reject_reason,
                    metadata_json=excluded.metadata_json,
                    embedding_path=CASE
                        WHEN photos.file_size = excluded.file_size
                         AND photos.source_mtime_ns = excluded.source_mtime_ns
                        THEN photos.embedding_path ELSE NULL END,
                    embedding_provider=CASE
                        WHEN photos.file_size = excluded.file_size
                         AND photos.source_mtime_ns = excluded.source_mtime_ns
                        THEN photos.embedding_provider ELSE NULL END,
                    embedding_source_size=CASE
                        WHEN photos.file_size = excluded.file_size
                         AND photos.source_mtime_ns = excluded.source_mtime_ns
                        THEN photos.embedding_source_size ELSE NULL END,
                    embedding_source_mtime_ns=CASE
                        WHEN photos.file_size = excluded.file_size
                         AND photos.source_mtime_ns = excluded.source_mtime_ns
                        THEN photos.embedding_source_mtime_ns ELSE NULL END,
                    face_provider=CASE
                        WHEN photos.file_size = excluded.file_size
                         AND photos.source_mtime_ns = excluded.source_mtime_ns
                        THEN photos.face_provider ELSE NULL END,
                    face_source_size=CASE
                        WHEN photos.file_size = excluded.file_size
                         AND photos.source_mtime_ns = excluded.source_mtime_ns
                        THEN photos.face_source_size ELSE NULL END,
                    face_source_mtime_ns=CASE
                        WHEN photos.file_size = excluded.file_size
                         AND photos.source_mtime_ns = excluded.source_mtime_ns
                        THEN photos.face_source_mtime_ns ELSE NULL END,
                    face_processed=CASE
                        WHEN photos.file_size = excluded.file_size
                         AND photos.source_mtime_ns = excluded.source_mtime_ns
                        THEN photos.face_processed ELSE 0 END,
                    face_count=CASE
                        WHEN photos.file_size = excluded.file_size
                         AND photos.source_mtime_ns = excluded.source_mtime_ns
                        THEN photos.face_count ELSE 0 END
                """,
                [
                    (
                        photo.id,
                        album_id,
                        str(photo.path),
                        str(photo.thumbnail_path),
                        photo.width,
                        photo.height,
                        photo.capture_time,
                        photo.quality_score,
                        photo.blur_score,
                        photo.similarity_group,
                        photo.file_size,
                        photo.source_mtime_ns,
                        photo.phash,
                        photo.dhash,
                        int(photo.auto_reject),
                        photo.reject_reason,
                        json.dumps(photo.metadata, ensure_ascii=False),
                    )
                    for photo in photos
                ],
            )

    @staticmethod
    def _to_summary(album_id: str, photo: ScannedPhoto) -> PhotoSummary:
        return PhotoSummary(
            id=photo.id,
            filename=photo.path.name,
            absolute_path=str(photo.path),
            thumbnail_path=str(photo.thumbnail_path),
            thumbnail_url=f"/media/thumbnails/{album_id}/{photo.id}.jpg",
            width=photo.width,
            height=photo.height,
            file_size=photo.file_size,
            capture_time=photo.capture_time,
            quality_score=photo.quality_score,
            blur_score=photo.blur_score,
            similarity_group=photo.similarity_group,
            auto_reject=photo.auto_reject,
            reject_reason=photo.reject_reason,
            quality_flags=photo.quality_flags,
        )


def _capture_time(image: Image.Image) -> str | None:
    exif = image.getexif()
    value = exif.get(36867) or exif.get(306)
    return str(value) if value else None


def _cached_source_matches(row: object, path: Path) -> bool:
    required = (
        "id",
        "absolute_path",
        "file_size",
        "source_mtime_ns",
    )
    if any(row[key] is None for key in required):
        return False
    current = path.stat()
    return current.st_size == int(row["file_size"]) and current.st_mtime_ns == int(
        row["source_mtime_ns"]
    )


def _photo_quality_complete(photo: ScannedPhoto) -> bool:
    return all(
        value is not None
        for value in (
            photo.quality_score,
            photo.blur_score,
            photo.phash,
            photo.dhash,
        )
    )


def _copy_quality_analysis(source: ScannedPhoto, target: ScannedPhoto) -> None:
    """Keep valid analysis when only a derived base artifact needs rebuilding."""

    target.quality_score = source.quality_score
    target.blur_score = source.blur_score
    target.phash = source.phash
    target.dhash = source.dhash
    target.auto_reject = source.auto_reject
    target.reject_reason = source.reject_reason
    target.quality_flags = list(source.quality_flags)
    target.metadata = dict(source.metadata)
    target.similarity_group = source.similarity_group
