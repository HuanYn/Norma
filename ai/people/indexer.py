from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ai.people.provider import FaceClusterPolicy, FaceProvider
from ai.schemas import FaceSummary, PeopleIndexResponse, PersonClusterSummary
from ai.storage import Database


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
            # A derived descriptor is disposable. Interrupted writes can make
            # ``np.load`` raise EOFError, while legacy or damaged rows may have
            # a NULL path and make ``Path`` raise TypeError. Treat both exactly
            # like a missing/invalid cache so only this photo is recomputed.
            except (EOFError, OSError, TypeError, ValueError):
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

        components = _cluster(faces, self.provider.cluster_policy)
        cluster_ids: list[str] = []
        for component in components:
            members = sorted(faces[index].id for index in component)
            cluster_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                (
                    f"norma:person:{self.provider.name}:"
                    f"{self.provider.cluster_policy.version}:{album_id}:"
                    f"{':'.join(members)}"
                ),
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

    def get(self, album_id: str) -> PeopleIndexResponse:
        """Read a complete persisted people result without running detection."""

        started = time.perf_counter()
        with self.database.connect() as connection:
            album = connection.execute(
                "SELECT id FROM albums WHERE id = ?", (album_id,)
            ).fetchone()
            photo_state = connection.execute(
                """
                SELECT COUNT(*) AS photo_count,
                       SUM(CASE WHEN face_processed = 1
                                 AND face_source_size = file_size
                                 AND face_source_mtime_ns = source_mtime_ns
                                THEN 1 ELSE 0 END) AS processed_count,
                       COUNT(DISTINCT CASE WHEN face_processed = 1
                                            AND face_source_size = file_size
                                            AND face_source_mtime_ns = source_mtime_ns
                                           THEN face_provider END) AS provider_count,
                       MAX(CASE WHEN face_processed = 1
                                 AND face_source_size = file_size
                                 AND face_source_mtime_ns = source_mtime_ns
                                THEN face_provider END) AS provider,
                       SUM(CASE WHEN face_processed = 1
                                 AND face_source_size = file_size
                                 AND face_source_mtime_ns = source_mtime_ns
                                THEN face_count ELSE 0 END) AS expected_face_count
                FROM photos WHERE album_id = ?
                """,
                (album_id,),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT pc.id AS cluster_id, pc.label, f.id AS face_id,
                       f.photo_id, f.box_json
                FROM person_clusters pc
                JOIN faces f ON f.cluster_id = pc.id
                JOIN photos p ON p.id = f.photo_id
                WHERE pc.album_id = ? AND p.album_id = ?
                  AND p.face_processed = 1
                  AND p.face_source_size = p.file_size
                  AND p.face_source_mtime_ns = p.source_mtime_ns
                ORDER BY pc.id, f.photo_id, f.id
                """,
                (album_id, album_id),
            ).fetchall()
        if album is None:
            raise KeyError(f"album not found: {album_id}")

        photo_count = int(photo_state["photo_count"] or 0)
        processed_count = int(photo_state["processed_count"] or 0)
        if photo_count == 0 or processed_count == 0:
            raise KeyError(
                "album has no persisted people analysis; run the people index first"
            )
        if (
            processed_count != photo_count
            or int(photo_state["provider_count"] or 0) != 1
            or not photo_state["provider"]
        ):
            raise ValueError(
                "persisted people analysis is incomplete; run the people index again"
            )
        expected_face_count = int(photo_state["expected_face_count"] or 0)
        if expected_face_count != len(rows):
            raise ValueError(
                "persisted people analysis is inconsistent; run the people index again"
            )

        provider = str(photo_state["provider"])
        by_cluster: dict[str, PersonClusterSummary] = {}
        for row in rows:
            cluster_id = str(row["cluster_id"])
            cluster = by_cluster.setdefault(
                cluster_id,
                PersonClusterSummary(
                    cluster_id=cluster_id,
                    label=str(row["label"]),
                    faces=[],
                ),
            )
            box_value = json.loads(row["box_json"])
            if not isinstance(box_value, list) or len(box_value) != 4:
                raise ValueError(f"invalid persisted face box: {row['face_id']}")
            cluster.faces.append(
                FaceSummary(
                    face_id=row["face_id"],
                    photo_id=row["photo_id"],
                    box=[int(value) for value in box_value],
                    thumbnail_url=(
                        f"/media/faces/{provider}/{album_id}/thumbnails/"
                        f"{row['face_id']}.jpg"
                    ),
                )
            )

        clusters = sorted(
            by_cluster.values(),
            key=lambda cluster: (-len(cluster.faces), cluster.cluster_id),
        )
        return PeopleIndexResponse(
            album_id=album_id,
            total_faces=sum(len(cluster.faces) for cluster in clusters),
            cluster_count=len(clusters),
            computed_count=0,
            reused_count=photo_count,
            provider=provider,
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


def _cluster(faces: list[IndexedFace], policy: FaceClusterPolicy) -> list[list[int]]:
    """Deterministic constrained agglomeration with guarded prototype attachment."""

    if not faces:
        return []

    vectors = np.vstack([face.descriptor for face in faces]).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.all(np.isfinite(vectors)):
        raise ValueError("cannot cluster invalid face descriptors")
    vectors /= norms
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    stable_keys = [(face.photo_id, face.id, index) for index, face in enumerate(faces)]
    parents = list(range(len(faces)))
    members = {index: [index] for index in range(len(faces))}
    photo_ids = {index: {faces[index].photo_id} for index in range(len(faces))}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    candidates = [
        (float(similarities[left, right]), left, right)
        for left in range(len(faces))
        for right in range(left + 1, len(faces))
        if faces[left].photo_id != faces[right].photo_id
        and float(similarities[left, right]) >= policy.minimum_similarity
    ]
    candidates.sort(
        key=lambda item: (
            -item[0],
            min(stable_keys[item[1]], stable_keys[item[2]]),
            max(stable_keys[item[1]], stable_keys[item[2]]),
        )
    )

    tolerance = 1e-7
    for _, left, right in candidates:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        if photo_ids[left_root] & photo_ids[right_root]:
            continue

        left_members = members[left_root]
        right_members = members[right_root]
        cross = similarities[np.ix_(left_members, right_members)]
        if float(np.min(cross)) + tolerance < policy.minimum_similarity:
            continue
        if float(np.mean(cross)) + tolerance < policy.mean_similarity:
            continue
        left_centroid = np.mean(vectors[left_members], axis=0)
        right_centroid = np.mean(vectors[right_members], axis=0)
        centroid_denominator = float(
            np.linalg.norm(left_centroid) * np.linalg.norm(right_centroid)
        )
        if centroid_denominator <= 1e-12:
            continue
        centroid_similarity = (
            float(np.dot(left_centroid, right_centroid)) / centroid_denominator
        )
        if centroid_similarity + tolerance < policy.centroid_similarity:
            continue

        if min(stable_keys[index] for index in left_members) <= min(
            stable_keys[index] for index in right_members
        ):
            winner, loser = left_root, right_root
        else:
            winner, loser = right_root, left_root
        parents[loser] = winner
        members[winner] = sorted(
            members[winner] + members.pop(loser), key=stable_keys.__getitem__
        )
        photo_ids[winner].update(photo_ids.pop(loser))

    components = [
        sorted(component, key=stable_keys.__getitem__)
        for root, component in members.items()
        if find(root) == root
    ]
    components.sort(key=lambda component: stable_keys[component[0]])
    if policy.prototype_attachment and len(components) > 1:
        components = _attach_mutual_prototypes(
            components,
            faces=faces,
            vectors=vectors,
            similarities=similarities,
            stable_keys=stable_keys,
            policy=policy,
        )
    return components


def _attach_mutual_prototypes(
    initial_components: list[list[int]],
    *,
    faces: list[IndexedFace],
    vectors: np.ndarray,
    similarities: np.ndarray,
    stable_keys: list[tuple[str, str, int]],
    policy: FaceClusterPolicy,
) -> list[list[int]]:
    """Join pose-fragmented seeds only when they are mutual best candidates.

    The first clustering pass deliberately requires every cross-cluster pair to
    clear a strict threshold. That protects precision but can split one identity
    by pose. This second pass compares cluster prototypes and cross-pair evidence,
    keeps the same-photo hard constraint, and recomputes evidence after each join.
    """

    components = [list(component) for component in initial_components]
    tolerance = 1e-7

    while len(components) > 1:
        centroids: list[np.ndarray] = []
        component_photo_ids: list[set[str]] = []
        for component in components:
            centroid = np.mean(vectors[component], axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm <= 1e-12:
                raise ValueError("cannot cluster invalid face descriptor centroids")
            centroids.append(centroid / norm)
            component_photo_ids.append({faces[index].photo_id for index in component})

        candidates: list[tuple[float, float, float, int, int]] = []
        for left in range(len(components)):
            for right in range(left + 1, len(components)):
                if component_photo_ids[left] & component_photo_ids[right]:
                    continue
                cross = similarities[np.ix_(components[left], components[right])]
                centroid_similarity = float(np.dot(centroids[left], centroids[right]))
                mean_similarity = float(np.mean(cross))
                max_similarity = float(np.max(cross))
                if min(len(components[left]), len(components[right])) >= 2:
                    centroid_threshold = policy.multi_cluster_centroid_similarity
                    mean_threshold = policy.multi_cluster_mean_similarity
                    max_threshold = policy.multi_cluster_max_similarity
                else:
                    centroid_threshold = policy.singleton_centroid_similarity
                    mean_threshold = policy.singleton_mean_similarity
                    max_threshold = policy.singleton_max_similarity
                if (
                    centroid_similarity + tolerance < centroid_threshold
                    or mean_similarity + tolerance < mean_threshold
                    or max_similarity + tolerance < max_threshold
                ):
                    continue
                candidates.append(
                    (
                        centroid_similarity,
                        mean_similarity,
                        max_similarity,
                        left,
                        right,
                    )
                )

        if not candidates:
            break
        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2],
                stable_keys[components[item[3]][0]],
                stable_keys[components[item[4]][0]],
            )
        )
        best_candidate: dict[int, tuple[float, float, float, int, int]] = {}
        for candidate in candidates:
            best_candidate.setdefault(candidate[3], candidate)
            best_candidate.setdefault(candidate[4], candidate)
        mutual = [
            candidate
            for candidate in candidates
            if best_candidate[candidate[3]] == candidate
            and best_candidate[candidate[4]] == candidate
        ]
        if not mutual:
            break

        _, _, _, left, right = mutual[0]
        merged = sorted(
            components[left] + components[right], key=stable_keys.__getitem__
        )
        components = [
            component
            for index, component in enumerate(components)
            if index not in {left, right}
        ]
        components.append(merged)
        components.sort(key=lambda component: stable_keys[component[0]])

    return components
