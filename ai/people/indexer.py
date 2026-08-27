from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ai.people.provider import FaceProvider
from ai.schemas import FaceSummary, PeopleIndexResponse, PersonClusterSummary
from ai.storage import Database


CLUSTER_THRESHOLD = 0.985


@dataclass(slots=True)
class IndexedFace:
    id: str
    photo_id: str
    box: tuple[int, int, int, int]
    descriptor_path: Path
    thumbnail_path: Path
    descriptor: np.ndarray
    cluster_id: str | None = None


class PeopleCancelledError(RuntimeError):
    pass


class PeopleIndexer:
    def __init__(
        self, database: Database, data_dir: Path, provider: FaceProvider
    ) -> None:
        self.database = database
        self.data_dir = data_dir
        self.provider = provider

    def index(
        self,
        album_id: str,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> PeopleIndexResponse:
        started = time.perf_counter()
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, absolute_path, file_size, source_mtime_ns,
                       face_provider, face_source_size, face_source_mtime_ns,
                       face_processed, face_count
                FROM photos WHERE album_id = ? ORDER BY id
                """,
                (album_id,),
            ).fetchall()
            stored_faces = connection.execute(
                """
                SELECT f.id, f.photo_id, f.box_json, f.embedding_path
                FROM faces f JOIN photos p ON p.id = f.photo_id
                WHERE p.album_id = ? ORDER BY f.photo_id, f.id
                """,
                (album_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"album not found or empty: {album_id}")

        base = self.data_dir / "faces" / self.provider.name / album_id
        descriptor_dir = base / "descriptors"
        thumbnail_dir = base / "thumbnails"
        descriptor_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        stored_by_photo: dict[str, list[object]] = {}
        for stored in stored_faces:
            stored_by_photo.setdefault(stored["photo_id"], []).append(stored)

        faces: list[IndexedFace] = []
        stale_rows = []
        reused_count = 0
        for row in rows:
            _ensure_source_matches_index(row)
            try:
                cached = self._load_cached_faces(
                    row,
                    stored_by_photo.get(row["id"], []),
                    thumbnail_dir,
                )
            except (OSError, ValueError, json.JSONDecodeError):
                cached = None
            if cached is None:
                stale_rows.append(row)
            else:
                faces.extend(cached)
                reused_count += 1
        if on_progress is not None:
            on_progress(reused_count, len(rows))
        if should_cancel is not None and should_cancel():
            raise PeopleCancelledError("people indexing cancelled between photos")

        computed_count = 0
        computed_faces: dict[str, list[IndexedFace]] = {}
        for row in stale_rows:
            if should_cancel is not None and should_cancel():
                raise PeopleCancelledError("people indexing cancelled between photos")
            detections = self.provider.detect(Path(row["absolute_path"]))
            photo_faces: list[IndexedFace] = []
            for face_index, detection in enumerate(detections):
                if detection.descriptor.shape != (
                    self.provider.dimension,
                ) or not np.all(np.isfinite(detection.descriptor)):
                    raise ValueError(
                        f"provider returned an invalid face descriptor for {row['absolute_path']}"
                    )
                descriptor_norm = float(np.linalg.norm(detection.descriptor))
                if descriptor_norm <= 1e-12:
                    raise ValueError(
                        f"provider returned a zero face descriptor for {row['absolute_path']}"
                    )
                descriptor = detection.descriptor.astype(np.float32) / descriptor_norm
                face_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"norma:face:{row['id']}:{face_index}:{detection.box}",
                ).hex
                descriptor_path = (descriptor_dir / f"{face_id}.npy").resolve()
                thumbnail_path = (thumbnail_dir / f"{face_id}.jpg").resolve()
                np.save(
                    descriptor_path,
                    descriptor,
                    allow_pickle=False,
                )
                preview = detection.crop.copy()
                preview.thumbnail((240, 240))
                preview.save(thumbnail_path, "JPEG", quality=86, optimize=True)
                indexed_face = IndexedFace(
                    id=face_id,
                    photo_id=row["id"],
                    box=detection.box,
                    descriptor_path=descriptor_path,
                    thumbnail_path=thumbnail_path,
                    descriptor=descriptor,
                )
                faces.append(indexed_face)
                photo_faces.append(indexed_face)
            computed_faces[row["id"]] = photo_faces
            computed_count += 1
            if on_progress is not None:
                on_progress(reused_count + computed_count, len(rows))

        for row in rows:
            _ensure_source_matches_index(row)
        if should_cancel is not None and should_cancel():
            raise PeopleCancelledError("people indexing cancelled before commit")

        components = _cluster([face.descriptor for face in faces])
        cluster_ids: list[str] = []
        for component in components:
            members = sorted(faces[index].id for index in component)
            cluster_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"norma:person:{album_id}:{':'.join(members)}",
            ).hex
            cluster_ids.append(cluster_id)
            for index in component:
                faces[index].cluster_id = cluster_id

        self._persist(album_id, rows, faces, cluster_ids, computed_faces)
        summaries_by_cluster: dict[str, list[FaceSummary]] = {
            cluster_id: [] for cluster_id in cluster_ids
        }
        for face in faces:
            if face.cluster_id is None:
                continue
            summaries_by_cluster[face.cluster_id].append(
                FaceSummary(
                    face_id=face.id,
                    photo_id=face.photo_id,
                    box=list(face.box),
                    thumbnail_url=(
                        f"/media/faces/{self.provider.name}/{album_id}/thumbnails/"
                        f"{face.id}.jpg"
                    ),
                )
            )
        clusters = [
            PersonClusterSummary(cluster_id=cluster_id, label="Unknown", faces=items)
            for cluster_id, items in summaries_by_cluster.items()
        ]
        clusters.sort(key=lambda cluster: (-len(cluster.faces), cluster.cluster_id))
        return PeopleIndexResponse(
            album_id=album_id,
            total_faces=len(faces),
            cluster_count=len(clusters),
            computed_count=computed_count,
            reused_count=reused_count,
            provider=self.provider.name,
            duration_ms=round((time.perf_counter() - started) * 1000),
            clusters=clusters,
        )

    def _load_cached_faces(
        self,
        photo: Mapping[str, object],
        stored_faces: list[object],
        thumbnail_dir: Path,
    ) -> list[IndexedFace] | None:
        if (
            not photo["face_processed"]
            or photo["face_provider"] != self.provider.name
            or photo["face_source_size"] != photo["file_size"]
            or photo["face_source_mtime_ns"] != photo["source_mtime_ns"]
            or int(photo["face_count"] or 0) != len(stored_faces)
        ):
            return None
        cached: list[IndexedFace] = []
        for stored in stored_faces:
            descriptor_path = Path(stored["embedding_path"]).resolve()
            thumbnail_path = (thumbnail_dir / f"{stored['id']}.jpg").resolve()
            descriptor = np.load(descriptor_path, allow_pickle=False).astype(np.float32)
            if (
                descriptor.shape != (self.provider.dimension,)
                or not np.all(np.isfinite(descriptor))
                or float(np.linalg.norm(descriptor)) <= 1e-12
                or not thumbnail_path.is_file()
            ):
                return None
            box_value = json.loads(stored["box_json"])
            if not isinstance(box_value, list) or len(box_value) != 4:
                return None
            cached.append(
                IndexedFace(
                    id=stored["id"],
                    photo_id=photo["id"],
                    box=tuple(int(value) for value in box_value),
                    descriptor_path=descriptor_path,
                    thumbnail_path=thumbnail_path,
                    descriptor=descriptor / float(np.linalg.norm(descriptor)),
                )
            )
        return cached

    def _persist(
        self,
        album_id: str,
        photos: list[object],
        faces: list[IndexedFace],
        cluster_ids: list[str],
        computed_faces: dict[str, list[IndexedFace]],
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM person_clusters WHERE album_id = ?", (album_id,)
            )
            connection.executemany(
                "DELETE FROM faces WHERE photo_id = ?",
                [(photo_id,) for photo_id in computed_faces],
            )
            connection.executemany(
                "INSERT INTO person_clusters(id, album_id, label) VALUES (?, ?, 'Unknown')",
                [(cluster_id, album_id) for cluster_id in cluster_ids],
            )
            connection.executemany(
                """
                INSERT INTO faces(id, photo_id, cluster_id, box_json, embedding_path)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    photo_id=excluded.photo_id,
                    cluster_id=excluded.cluster_id,
                    box_json=excluded.box_json,
                    embedding_path=excluded.embedding_path
                """,
                [
                    (
                        face.id,
                        face.photo_id,
                        face.cluster_id,
                        json.dumps(face.box),
                        str(face.descriptor_path),
                    )
                    for face in faces
                ],
            )
            photos_by_id = {photo["id"]: photo for photo in photos}
            for photo_id, photo_faces in computed_faces.items():
                photo = photos_by_id[photo_id]
                cursor = connection.execute(
                    """
                    UPDATE photos SET face_provider = ?, face_source_size = ?,
                        face_source_mtime_ns = ?, face_processed = 1, face_count = ?
                    WHERE id = ? AND file_size = ? AND source_mtime_ns = ?
                    """,
                    (
                        self.provider.name,
                        photo["file_size"],
                        photo["source_mtime_ns"],
                        len(photo_faces),
                        photo_id,
                        photo["file_size"],
                        photo["source_mtime_ns"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"photo changed while committing face cache: {photo_id}"
                    )


def _ensure_source_matches_index(row: Mapping[str, object]) -> None:
    if row["file_size"] is None or row["source_mtime_ns"] is None:
        raise ValueError("album needs re-indexing before incremental people indexing")
    path = Path(str(row["absolute_path"]))
    try:
        current = path.stat()
    except OSError as error:
        raise ValueError(f"indexed source is unavailable: {path}: {error}") from error
    if current.st_size != int(row["file_size"]) or current.st_mtime_ns != int(
        row["source_mtime_ns"]
    ):
        raise ValueError(f"source changed since indexing; re-index the album: {path}")


def _cluster(vectors: list[np.ndarray]) -> list[list[int]]:
    parents = list(range(len(vectors)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            if float(np.dot(vectors[left], vectors[right])) >= CLUSTER_THRESHOLD:
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(vectors)):
        components.setdefault(find(index), []).append(index)
    return list(components.values())
