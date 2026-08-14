from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

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


class PeopleIndexer:
    def __init__(self, database: Database, data_dir: Path, provider: FaceProvider) -> None:
        self.database = database
        self.data_dir = data_dir
        self.provider = provider

    def index(self, album_id: str) -> PeopleIndexResponse:
        started = time.perf_counter()
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id, absolute_path FROM photos WHERE album_id = ? ORDER BY id",
                (album_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"album not found or empty: {album_id}")

        base = self.data_dir / "faces" / self.provider.name / album_id
        descriptor_dir = base / "descriptors"
        thumbnail_dir = base / "thumbnails"
        descriptor_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        faces: list[IndexedFace] = []
        for row in rows:
            detections = self.provider.detect(Path(row["absolute_path"]))
            for face_index, detection in enumerate(detections):
                if detection.descriptor.shape != (self.provider.dimension,) or not np.all(
                    np.isfinite(detection.descriptor)
                ):
                    raise ValueError(
                        f"provider returned an invalid face descriptor for {row['absolute_path']}"
                    )
                face_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"norma:face:{row['id']}:{face_index}:{detection.box}",
                ).hex
                descriptor_path = (descriptor_dir / f"{face_id}.npy").resolve()
                thumbnail_path = (thumbnail_dir / f"{face_id}.jpg").resolve()
                np.save(
                    descriptor_path,
                    detection.descriptor.astype(np.float32),
                    allow_pickle=False,
                )
                preview = detection.crop.copy()
                preview.thumbnail((240, 240))
                preview.save(thumbnail_path, "JPEG", quality=86, optimize=True)
                faces.append(
                    IndexedFace(
                        id=face_id,
                        photo_id=row["id"],
                        box=detection.box,
                        descriptor_path=descriptor_path,
                        thumbnail_path=thumbnail_path,
                        descriptor=detection.descriptor,
                    )
                )

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

        self._persist(album_id, faces, cluster_ids)
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
            provider=self.provider.name,
            duration_ms=round((time.perf_counter() - started) * 1000),
            clusters=clusters,
        )

    def _persist(
        self, album_id: str, faces: list[IndexedFace], cluster_ids: list[str]
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM faces WHERE photo_id IN "
                "(SELECT id FROM photos WHERE album_id = ?)",
                (album_id,),
            )
            connection.execute("DELETE FROM person_clusters WHERE album_id = ?", (album_id,))
            connection.executemany(
                "INSERT INTO person_clusters(id, album_id, label) VALUES (?, ?, 'Unknown')",
                [(cluster_id, album_id) for cluster_id in cluster_ids],
            )
            connection.executemany(
                """
                INSERT INTO faces(id, photo_id, cluster_id, box_json, embedding_path)
                VALUES (?, ?, ?, ?, ?)
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
