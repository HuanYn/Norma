from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ai.people.labels import canonical_json, hash_file, validate_cluster_evidence
from ai.selection.parser import person_key
from ai.storage import Database


@dataclass(frozen=True, slots=True)
class PeopleConstraintEvidence:
    labels_by_photo: dict[str, frozenset[str]]
    snapshot_sha256: str


def load_people_constraint_evidence(
    database: Database, album_id: str, minimums: dict[str, int]
) -> PeopleConstraintEvidence:
    """Use explicit, current names, not face similarity as a person's identity.

    An unfinished scan or unknown name is not evidence that someone is absent.
    Lower bounds count distinct photos, even if several clusters share a name.
    """
    with database.connect() as connection:
        connection.execute("BEGIN")
        photos = connection.execute(
            """SELECT id, absolute_path, file_size, source_mtime_ns,
                      face_provider, face_processed, face_source_size,
                      face_source_mtime_ns, face_source_sha256, face_count
               FROM photos WHERE album_id = ? ORDER BY id""",
            (album_id,),
        ).fetchall()
        if not photos or any(
            row["face_processed"] != 1
            or row["face_source_size"] != row["file_size"]
            or row["face_source_mtime_ns"] != row["source_mtime_ns"]
            or not row["face_provider"]
            or not row["face_source_sha256"]
            for row in photos
        ):
            raise ValueError("人物条件需要完整的人脸分析；请先运行人脸分组。")
        if len({row["face_provider"] for row in photos}) != 1:
            raise ValueError("人脸分析模型不一致；请重新运行人脸分组。")
        for row in photos:
            try:
                stat = Path(row["absolute_path"]).stat()
            except OSError as error:
                raise ValueError(
                    "人物证据的原照片不可用；请重新导入并分析。"
                ) from error
            if (stat.st_size, stat.st_mtime_ns) != (
                row["file_size"],
                row["source_mtime_ns"],
            ):
                raise ValueError("人物证据的原照片已变化；请重新导入并分析。")
            if hash_file(Path(row["absolute_path"])) != row["face_source_sha256"]:
                raise ValueError(
                    "person source evidence changed; run people indexing again"
                )
        clusters = connection.execute(
            "SELECT * FROM person_clusters WHERE album_id = ? ORDER BY id",
            (album_id,),
        ).fetchall()
        faces = connection.execute(
            """SELECT f.id, f.photo_id, f.cluster_id, pc.album_id AS cluster_album
               FROM faces f JOIN photos p ON p.id = f.photo_id
               LEFT JOIN person_clusters pc ON pc.id = f.cluster_id
               WHERE p.album_id = ? ORDER BY f.id""",
            (album_id,),
        ).fetchall()
        if any(face["cluster_album"] != album_id for face in faces) or any(
            sum(face["photo_id"] == row["id"] for face in faces)
            != int(row["face_count"] or 0)
            for row in photos
        ):
            raise ValueError("人物分组不完整；请重新运行人脸分组。")
        confirmed: dict[str, str] = {}
        for row in clusters:
            key = person_key(row["label"])
            if row["label_status"] == "confirmed" and key in minimums:
                validate_cluster_evidence(connection, album_id, row)
                confirmed[row["id"]] = key
        missing = sorted(set(minimums) - set(confirmed.values()))
        if missing:
            raise ValueError(
                "人物尚未命名或需要重新确认："
                + ", ".join(missing)
                + "；请在人物分组中保存名字，不能将未知视为不存在。"
            )
        labels: dict[str, set[str]] = {row["id"]: set() for row in photos}
        for face in faces:
            if face["cluster_id"] in confirmed:
                labels[face["photo_id"]].add(confirmed[face["cluster_id"]])
        snapshot = {
            "version": "confirmed-person-quotas-v1",
            "album_id": album_id,
            "photos": [
                [
                    row[key]
                    for key in (
                        "id",
                        "file_size",
                        "source_mtime_ns",
                        "face_provider",
                        "face_source_sha256",
                        "face_count",
                    )
                ]
                for row in photos
            ],
            "clusters": [
                [
                    row[key]
                    for key in (
                        "id",
                        "label",
                        "label_status",
                        "label_revision",
                        "identity_evidence_json",
                    )
                ]
                for row in clusters
            ],
            "faces": [[row["id"], row["photo_id"], row["cluster_id"]] for row in faces],
        }
    return PeopleConstraintEvidence(
        {key: frozenset(value) for key, value in labels.items()},
        hashlib.sha256(canonical_json(snapshot).encode()).hexdigest(),
    )
