from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

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
    quality_score: float
    blur_score: float
    phash: str
    dhash: str
    auto_reject: bool
    reject_reason: str | None
    quality_flags: list[str]
    metadata: dict[str, object]
    similarity_group: str | None = None
    reused: bool = False


class AlbumIndexer:
    def __init__(self, database: Database, data_dir: Path) -> None:
        self.database = database
        self.data_dir = data_dir

    def index(self, folder: Path, album_name: str | None = None) -> AlbumIndexResponse:
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

        photos: list[ScannedPhoto] = []
        errors: list[str] = []
        for path in paths:
            try:
                resolved = path.resolve()
                cached = existing.get(str(resolved).casefold())
                if cached is not None and _cached_source_matches(cached, resolved):
                    try:
                        photos.append(self._cached_photo(cached))
                        continue
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                photos.append(self._scan_photo(resolved, album_id, thumbnail_dir))
            except Exception as error:  # surfaced in the API; never silently swallowed
                errors.append(f"{path.name}: {error}")

        groups = assign_similarity_groups(
            [(photo.phash, photo.dhash) for photo in photos]
        )
        for index, group_id in groups.items():
            photos[index].similarity_group = group_id

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
            similar_groups=len(set(groups.values())),
            duration_ms=round((time.perf_counter() - started) * 1000),
            provider=PROVIDER_NAME,
            photos=summaries,
            errors=errors,
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
            quality_score=float(row["quality_score"]),
            blur_score=float(row["blur_score"]),
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
        self, path: Path, album_id: str, thumbnail_dir: Path
    ) -> ScannedPhoto:
        before = path.stat()
        photo_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"norma:photo:{str(path.resolve()).casefold()}",
        ).hex
        thumbnail_path = thumbnail_dir / f"{photo_id}.jpg"

        with Image.open(path) as source:
            capture_time = _capture_time(source)
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            quality = analyze_quality(image)
            phash = perceptual_hash(image)
            dhash = difference_hash(image)
            thumbnail = image.copy()
            thumbnail.thumbnail((480, 480), Image.Resampling.LANCZOS)
            thumbnail.save(thumbnail_path, "JPEG", quality=84, optimize=True)

        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("source image changed during read-only indexing")

        metadata = {
            "provider": PROVIDER_NAME,
            "brightness": quality.brightness,
            "contrast": quality.contrast,
            "overexposed_ratio": quality.overexposed_ratio,
            "underexposed_ratio": quality.underexposed_ratio,
            "entropy": quality.entropy,
            "quality_flags": list(quality.flags),
        }
        return ScannedPhoto(
            id=photo_id,
            path=path.resolve(),
            thumbnail_path=thumbnail_path.resolve(),
            width=width,
            height=height,
            file_size=before.st_size,
            source_mtime_ns=before.st_mtime_ns,
            capture_time=capture_time,
            quality_score=quality.quality_score,
            blur_score=quality.blur_score,
            phash=phash,
            dhash=dhash,
            auto_reject=quality.auto_reject,
            reject_reason=quality.reject_reason,
            quality_flags=list(quality.flags),
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
                "SELECT id FROM photos WHERE album_id = ?", (album_id,)
            ).fetchall()
            stale = [row["id"] for row in existing if row["id"] not in current_ids]
            if stale or any(not photo.reused for photo in photos):
                connection.execute(
                    "DELETE FROM faces WHERE photo_id IN "
                    "(SELECT id FROM photos WHERE album_id = ?)",
                    (album_id,),
                )
                connection.execute(
                    "DELETE FROM person_clusters WHERE album_id = ?", (album_id,)
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
                        THEN photos.embedding_source_mtime_ns ELSE NULL END
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
        "thumbnail_path",
        "width",
        "height",
        "file_size",
        "source_mtime_ns",
        "quality_score",
        "blur_score",
        "phash",
        "dhash",
        "metadata_json",
    )
    if (
        any(row[key] is None for key in required)
        or not Path(row["thumbnail_path"]).is_file()
    ):
        return False
    current = path.stat()
    return current.st_size == int(row["file_size"]) and current.st_mtime_ns == int(
        row["source_mtime_ns"]
    )
