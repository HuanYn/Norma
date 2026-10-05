from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ai.people.provider import FaceProvider
from ai.storage import Database

LabelStatus = Literal["unlabeled", "confirmed", "needs_review"]
MAX_LABEL_LENGTH = 80


@dataclass(frozen=True, slots=True)
class PersonLabelState:
    cluster_id: str
    label: str
    label_status: LabelStatus
    label_revision: int
    evidence_digest: str


def normalize_label(label: str) -> str:
    """Only explicit user labels identify a person; recognition never names them."""
    if not isinstance(label, str):
        raise ValueError("person label must be text")
    if any(unicodedata.category(char).startswith("C") for char in label):
        raise ValueError("person label cannot contain control or invisible characters")
    value = unicodedata.normalize("NFC", label).strip()
    if not value or len(value) > MAX_LABEL_LENGTH:
        raise ValueError(
            f"person label must contain 1 to {MAX_LABEL_LENGTH} characters"
        )
    if value.casefold() == "unknown":
        return "Unknown"
    if value.casefold() == "me":
        return "Me"
    return value


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provider_fingerprint(provider: FaceProvider) -> str:
    payload = {
        "name": provider.name,
        "dimension": provider.dimension,
        "policy": asdict(provider.cluster_policy),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_evidence(raw: str) -> dict[str, Any] | None:
    try:
        evidence = json.loads(raw)
        if (
            not isinstance(evidence, dict)
            or evidence.get("version") != 1
            or not isinstance(evidence.get("provider_fingerprint"), str)
            or not isinstance(evidence.get("provider_name"), str)
            or not isinstance(evidence.get("members"), list)
            or not evidence["members"]
        ):
            return None
        required = {
            "face_id",
            "photo_id",
            "box",
            "source_size",
            "source_mtime_ns",
            "source_sha256",
            "descriptor_sha256",
        }
        for member in evidence["members"]:
            if (
                not isinstance(member, dict)
                or not required.issubset(member)
                or not all(
                    isinstance(member[key], str) and member[key]
                    for key in ("face_id", "photo_id")
                )
                or not isinstance(member["box"], list)
                or len(member["box"]) != 4
                or not all(type(value) is int for value in member["box"])
                or type(member["source_size"]) is not int
                or member["source_size"] < 0
                or type(member["source_mtime_ns"]) is not int
                or not all(
                    isinstance(member[key], str)
                    and len(member[key]) == 64
                    and all(char in "0123456789abcdef" for char in member[key])
                    for key in ("source_sha256", "descriptor_sha256")
                )
            ):
                return None
        return evidence
    except (TypeError, ValueError):
        return None


def _state(row: sqlite3.Row) -> PersonLabelState:
    return PersonLabelState(
        cluster_id=str(row["id"]),
        label=str(row["label"]),
        label_status=row["label_status"],
        label_revision=int(row["label_revision"]),
        evidence_digest=hashlib.sha256(
            str(row["identity_evidence_json"]).encode()
        ).hexdigest(),
    )


def append_label_event(
    connection: sqlite3.Connection,
    *,
    album_id: str,
    cluster_id: str,
    label: str,
    status: LabelStatus,
    revision: int,
    reason: str,
    evidence_json: str,
) -> None:
    connection.execute(
        """
        INSERT INTO person_label_events
            (id, album_id, cluster_id, label, label_status, revision,
             reason, evidence_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            album_id,
            cluster_id,
            label,
            status,
            revision,
            reason,
            evidence_json,
        ),
    )


def validate_cluster_evidence(
    connection: sqlite3.Connection, album_id: str, row: sqlite3.Row
) -> dict[str, Any]:
    """Fail closed if a label no longer describes the exact stored face evidence."""
    evidence = parse_evidence(row["identity_evidence_json"])
    if evidence is None:
        raise ValueError(
            "person identity evidence is missing; run people indexing again"
        )
    stored = connection.execute(
        """
        SELECT f.id, f.photo_id, f.box_json, f.embedding_path,
               p.absolute_path, p.file_size, p.source_mtime_ns,
               p.face_provider, p.face_processed, p.face_source_size,
               p.face_source_mtime_ns, p.face_source_sha256
        FROM faces f JOIN photos p ON p.id = f.photo_id
        WHERE f.cluster_id = ? AND p.album_id = ?
        """,
        (row["id"], album_id),
    ).fetchall()
    by_id = {member["face_id"]: member for member in evidence["members"]}
    if len(by_id) != len(evidence["members"]) or set(by_id) != {
        face["id"] for face in stored
    }:
        raise ValueError("person membership changed; run people indexing again")
    for face in stored:
        member = by_id[face["id"]]
        try:
            source = Path(face["absolute_path"])
            stat = source.stat()
            valid = (
                face["face_processed"] == 1
                and face["face_provider"] == evidence["provider_name"]
                and face["photo_id"] == member["photo_id"]
                and json.loads(face["box_json"]) == member["box"]
                and face["file_size"] == member["source_size"]
                and face["source_mtime_ns"] == member["source_mtime_ns"]
                and face["face_source_size"] == member["source_size"]
                and face["face_source_mtime_ns"] == member["source_mtime_ns"]
                and face["face_source_sha256"] == member["source_sha256"]
                and stat.st_size == member["source_size"]
                and stat.st_mtime_ns == member["source_mtime_ns"]
                and hash_file(source) == member["source_sha256"]
                and hash_file(Path(face["embedding_path"]))
                == member["descriptor_sha256"]
            )
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "person source evidence is unavailable; run people indexing again"
            ) from error
        if not valid:
            raise ValueError(
                "person source evidence changed; run people indexing again"
            )
    return evidence


class PersonLabelService:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _row(
        connection: sqlite3.Connection, album_id: str, cluster_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM person_clusters WHERE album_id = ? AND id = ?",
            (album_id, cluster_id),
        ).fetchone()
        if row is None:
            raise KeyError("person cluster not found in this album")
        return row

    def get(self, album_id: str, cluster_id: str) -> PersonLabelState:
        with self.database.connect() as connection:
            row = self._row(connection, album_id, cluster_id)
            validate_cluster_evidence(connection, album_id, row)
            return _state(row)

    def set(
        self,
        album_id: str,
        cluster_id: str,
        label: str,
        *,
        expected_revision: int | None = None,
    ) -> PersonLabelState:
        label = normalize_label(label)
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, album_id, cluster_id)
            if (
                expected_revision is not None
                and expected_revision != row["label_revision"]
            ):
                raise ValueError("person label revision changed; refresh before saving")
            validate_cluster_evidence(connection, album_id, row)
            status: LabelStatus = "unlabeled" if label == "Unknown" else "confirmed"
            if row["label"] == label and row["label_status"] == status:
                return _state(row)
            revision = int(row["label_revision"]) + 1
            connection.execute(
                """
                UPDATE person_clusters
                SET label = ?, label_status = ?, label_revision = ?
                WHERE album_id = ? AND id = ?
                """,
                (label, status, revision, album_id, cluster_id),
            )
            append_label_event(
                connection,
                album_id=album_id,
                cluster_id=cluster_id,
                label=label,
                status=status,
                revision=revision,
                reason="explicit_reset" if label == "Unknown" else "explicit_label",
                evidence_json=row["identity_evidence_json"],
            )
            return _state(self._row(connection, album_id, cluster_id))


def reconcile_labels(
    connection: sqlite3.Connection,
    *,
    album_id: str,
    evidence_by_cluster: dict[str, str],
    computed_photo_ids: set[str],
) -> dict[str, tuple[str, LabelStatus, int]]:
    """Exact, unchanged correspondence only; no label propagation by similarity.

    Any growth, split, merge, model/policy switch, or recomputed member requires
    confirmation again. An unrelated new person does not erase existing labels.
    """
    old_rows = connection.execute(
        "SELECT * FROM person_clusters WHERE album_id = ?", (album_id,)
    ).fetchall()
    old_by_id = {row["id"]: row for row in old_rows}
    old_members: dict[str, set[str]] = {}
    for row in old_rows:
        evidence = parse_evidence(row["identity_evidence_json"])
        old_members[row["id"]] = (
            {member["photo_id"] for member in evidence["members"]}
            if evidence
            else {
                face["photo_id"]
                for face in connection.execute(
                    "SELECT photo_id FROM faces WHERE cluster_id = ?", (row["id"],)
                ).fetchall()
            }
        )
    preserved: set[str] = set()
    results: dict[str, tuple[str, LabelStatus, int]] = {}
    for cluster_id, raw in evidence_by_cluster.items():
        evidence = parse_evidence(raw)
        if evidence is None:
            raise ValueError("cannot persist invalid identity evidence")
        photo_ids = {member["photo_id"] for member in evidence["members"]}
        previous = old_by_id.get(cluster_id)
        safe = (
            previous is not None
            and previous["identity_evidence_json"] == raw
            and not photo_ids.intersection(computed_photo_ids)
        )
        if safe:
            results[cluster_id] = (
                previous["label"],
                previous["label_status"],
                int(previous["label_revision"]),
            )
            preserved.add(cluster_id)
            continue
        ancestors = [
            row
            for row in old_rows
            if old_members[row["id"]].intersection(photo_ids)
            and (row["label"] != "Unknown" or row["label_status"] == "needs_review")
        ]
        history = connection.execute(
            """
            SELECT COUNT(*) AS count, COALESCE(MAX(revision), 0) AS revision
            FROM person_label_events WHERE album_id = ? AND cluster_id = ?
            """,
            (album_id, cluster_id),
        ).fetchone()
        revision = max(
            int(history["revision"]),
            int(previous["label_revision"]) if previous else 0,
        )
        status: LabelStatus = (
            "needs_review" if ancestors or history["count"] else "unlabeled"
        )
        if previous is not None or history["count"] or ancestors:
            revision += 1
        results[cluster_id] = ("Unknown", status, revision)
        append_label_event(
            connection,
            album_id=album_id,
            cluster_id=cluster_id,
            label="Unknown",
            status=status,
            revision=revision,
            reason=(
                "identity_evidence_changed"
                if previous is not None or history["count"] or ancestors
                else "identity_indexed"
            ),
            evidence_json=raw,
        )
    for old in old_rows:
        if old["id"] not in preserved:
            append_label_event(
                connection,
                album_id=album_id,
                cluster_id=old["id"],
                label=old["label"],
                status="needs_review",
                revision=int(old["label_revision"]) + 1,
                reason="previous_identity_invalidated",
                evidence_json=old["identity_evidence_json"],
            )
    return results


def invalidate_photo_labels(
    connection: sqlite3.Connection, album_id: str, photo_ids: set[str]
) -> None:
    """Keep review history and advance generations before source rows disappear."""
    if not photo_ids:
        return
    rows = connection.execute(
        "SELECT * FROM person_clusters WHERE album_id = ?", (album_id,)
    ).fetchall()
    for row in rows:
        evidence = parse_evidence(row["identity_evidence_json"])
        members = (
            {member["photo_id"] for member in evidence["members"]}
            if evidence
            else {
                face["photo_id"]
                for face in connection.execute(
                    "SELECT photo_id FROM faces WHERE cluster_id = ?", (row["id"],)
                ).fetchall()
            }
        )
        if not members.intersection(photo_ids):
            continue
        revision = int(row["label_revision"]) + 1
        append_label_event(
            connection,
            album_id=album_id,
            cluster_id=row["id"],
            label=row["label"],
            status="needs_review",
            revision=revision,
            reason="album_source_invalidated",
            evidence_json=row["identity_evidence_json"],
        )
        connection.execute(
            """
            UPDATE person_clusters
            SET label = 'Unknown', label_status = 'needs_review', label_revision = ?
            WHERE album_id = ? AND id = ?
            """,
            (revision, album_id, row["id"]),
        )
