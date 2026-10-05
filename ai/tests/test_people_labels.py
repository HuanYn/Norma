from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ai.index import AlbumIndexer
from ai.people import indexer as indexer_module
from ai.people.indexer import PeopleIndexer
from ai.people.labels import PersonLabelService, normalize_label
from ai.people.provider import DetectedFace, FaceClusterPolicy, FaceProvider
from ai.storage import Database


class LabelFaceProvider(FaceProvider):
    name = "label-test-v1"
    dimension = 2
    cluster_policy = FaceClusterPolicy(
        version="label-policy-v1",
        minimum_similarity=0.99,
        mean_similarity=0.99,
        centroid_similarity=0.99,
    )

    def detect(self, path: Path) -> list[DetectedFace]:
        if path.stem.startswith("no-face"):
            return []
        descriptor = [0.0, 1.0] if path.stem.startswith("b") else [1.0, 0.0]
        return [
            DetectedFace(
                box=(1, 2, 10, 10),
                descriptor=np.asarray(descriptor, dtype=np.float32),
                crop=Image.new("RGB", (10, 10), "white"),
            )
        ]


def _image(path: Path, color: str = "blue") -> None:
    Image.new("RGB", (32, 32), color).save(path, "JPEG")


def _setup(tmp_path: Path, names: tuple[str, ...] = ("a1", "a2")):
    folder = tmp_path / "album"
    folder.mkdir()
    for name in names:
        _image(folder / f"{name}.jpg")
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    album_indexer = AlbumIndexer(database, data_dir)
    album_id = album_indexer.index(folder).album_id
    indexer = PeopleIndexer(database, data_dir, LabelFaceProvider())
    result = indexer.index(album_id)
    return folder, database, album_indexer, indexer, result


def _label_first(database: Database, result, label: str = "Me"):
    return PersonLabelService(database).set(
        result.album_id, result.clusters[0].cluster_id, label
    )


def test_label_round_trip_reset_idempotency_and_revision(tmp_path: Path) -> None:
    _, database, _, indexer, result = _setup(tmp_path)
    cluster_id = result.clusters[0].cluster_id
    labels = PersonLabelService(database)
    first = labels.set(result.album_id, cluster_id, "  me  ", expected_revision=0)
    assert (first.label, first.label_status, first.label_revision) == (
        "Me",
        "confirmed",
        1,
    )
    assert len(first.evidence_digest) == 64
    assert labels.get(result.album_id, cluster_id) == first
    assert labels.set(result.album_id, cluster_id, "Me", expected_revision=1) == first
    assert indexer.get(result.album_id).clusters[0].label == "Me"
    custom = labels.set(result.album_id, cluster_id, "同学张三", expected_revision=1)
    assert (custom.label, custom.label_revision) == ("同学张三", 2)
    reset = labels.set(result.album_id, cluster_id, "unknown", expected_revision=2)
    assert (reset.label, reset.label_status, reset.label_revision) == (
        "Unknown",
        "unlabeled",
        3,
    )
    with database.connect() as connection:
        events = connection.execute(
            "SELECT * FROM person_label_events ORDER BY revision"
        ).fetchall()
    assert [event["reason"] for event in events] == [
        "identity_indexed",
        "explicit_label",
        "explicit_label",
        "explicit_reset",
    ]


def test_label_unchanged_refresh_preserves_exact_evidence(tmp_path: Path) -> None:
    _, database, _, indexer, result = _setup(tmp_path)
    original = _label_first(database, result)
    refreshed = indexer.index(result.album_id)
    cluster = refreshed.clusters[0]
    assert refreshed.computed_count == 0
    assert refreshed.reused_count == 2
    assert (cluster.label, cluster.label_status, cluster.label_revision) == (
        "Me",
        "confirmed",
        1,
    )
    assert (
        PersonLabelService(database).get(result.album_id, cluster.cluster_id)
        == original
    )


def test_unrelated_addition_preserves_named_cluster(tmp_path: Path) -> None:
    folder, database, albums, indexer, result = _setup(tmp_path)
    original = _label_first(database, result)
    _image(folder / "b1.jpg", "green")
    albums.index(folder)
    refreshed = indexer.index(result.album_id)
    named = next(
        cluster
        for cluster in refreshed.clusters
        if cluster.cluster_id == original.cluster_id
    )
    assert (named.label, named.label_status, named.label_revision) == (
        "Me",
        "confirmed",
        1,
    )
    other = next(
        cluster
        for cluster in refreshed.clusters
        if cluster.cluster_id != original.cluster_id
    )
    assert (other.label, other.label_status) == ("Unknown", "unlabeled")


def test_cluster_growth_needs_confirmation(tmp_path: Path) -> None:
    folder, database, albums, indexer, result = _setup(tmp_path)
    old = _label_first(database, result)
    _image(folder / "a3.jpg", "green")
    albums.index(folder)
    refreshed = indexer.index(result.album_id)
    assert len(refreshed.clusters) == 1
    current = refreshed.clusters[0]
    assert current.cluster_id != old.cluster_id
    assert (current.label, current.label_status) == ("Unknown", "needs_review")
    with database.connect() as connection:
        events = connection.execute(
            "SELECT label, reason FROM person_label_events WHERE cluster_id = ?",
            (old.cluster_id,),
        ).fetchall()
    assert any(
        row["label"] == "Me" and row["reason"] == "previous_identity_invalidated"
        for row in events
    )
    with pytest.raises(KeyError):
        PersonLabelService(database).get(result.album_id, old.cluster_id)


def test_cluster_split_never_copies_identity(tmp_path: Path, monkeypatch) -> None:
    _, database, _, indexer, result = _setup(tmp_path)
    _label_first(database, result)
    monkeypatch.setattr(
        indexer_module,
        "_cluster",
        lambda faces, policy: [[i] for i in range(len(faces))],
    )
    refreshed = indexer.index(result.album_id)
    assert len(refreshed.clusters) == 2
    assert all(
        (cluster.label, cluster.label_status) == ("Unknown", "needs_review")
        for cluster in refreshed.clusters
    )


def test_cluster_merge_never_chooses_one_previous_name(
    tmp_path: Path, monkeypatch
) -> None:
    _, database, _, indexer, result = _setup(tmp_path, ("a1", "b1"))
    labels = PersonLabelService(database)
    for cluster, name in zip(result.clusters, ("Me", "Friend")):
        labels.set(result.album_id, cluster.cluster_id, name)
    monkeypatch.setattr(
        indexer_module, "_cluster", lambda faces, policy: [list(range(len(faces)))]
    )
    refreshed = indexer.index(result.album_id)
    assert len(refreshed.clusters) == 1
    assert (refreshed.clusters[0].label, refreshed.clusters[0].label_status) == (
        "Unknown",
        "needs_review",
    )


@pytest.mark.parametrize("change", ["provider", "policy_version", "policy_threshold"])
def test_provider_or_policy_change_invalidates_label(
    tmp_path: Path, change: str
) -> None:
    _, database, _, indexer, result = _setup(tmp_path)
    _label_first(database, result)
    if change == "provider":
        indexer.provider.name = "label-test-v2"
    elif change == "policy_version":
        indexer.provider.cluster_policy = replace(
            indexer.provider.cluster_policy, version="label-policy-v2"
        )
    else:
        indexer.provider.cluster_policy = replace(
            indexer.provider.cluster_policy, minimum_similarity=0.98
        )
    refreshed = indexer.index(result.album_id)
    assert (refreshed.clusters[0].label, refreshed.clusters[0].label_status) == (
        "Unknown",
        "needs_review",
    )


def test_source_changed_after_album_refresh_invalidates_name(tmp_path: Path) -> None:
    folder, database, albums, indexer, result = _setup(tmp_path)
    _label_first(database, result)
    _image(folder / "a1.jpg", "red")
    albums.index(folder)
    refreshed = indexer.index(result.album_id)
    assert refreshed.computed_count == 1
    assert (refreshed.clusters[0].label, refreshed.clusters[0].label_status) == (
        "Unknown",
        "needs_review",
    )


def test_same_size_mtime_source_mutation_is_not_hidden(tmp_path: Path) -> None:
    folder, database, _, indexer, result = _setup(tmp_path)
    label = _label_first(database, result)
    path = folder / "a1.jpg"
    stat = path.stat()
    content = bytearray(path.read_bytes())
    content[-10] ^= 1
    path.write_bytes(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    labels = PersonLabelService(database)
    with pytest.raises(ValueError, match="changed"):
        labels.get(result.album_id, label.cluster_id)
    with pytest.raises(ValueError, match="changed"):
        labels.set(result.album_id, label.cluster_id, "Friend")
    refreshed = indexer.index(result.album_id)
    assert refreshed.computed_count == 1
    assert refreshed.clusters[0].label_status == "needs_review"
    assert refreshed.clusters[0].label == "Unknown"


def test_descriptor_repair_requires_confirmation(tmp_path: Path) -> None:
    _, database, _, indexer, result = _setup(tmp_path)
    label = _label_first(database, result)
    with database.connect() as connection:
        path = Path(
            connection.execute("SELECT embedding_path FROM faces LIMIT 1").fetchone()[0]
        )
    path.write_bytes(b"broken disposable descriptor")
    with pytest.raises(ValueError, match="changed"):
        PersonLabelService(database).get(result.album_id, label.cluster_id)
    refreshed = indexer.index(result.album_id)
    assert refreshed.computed_count == 1
    assert refreshed.clusters[0].label_status == "needs_review"


def test_missing_or_cross_album_clusters_are_not_accessible(tmp_path: Path) -> None:
    _, database, albums, indexer, result = _setup(tmp_path)
    label = _label_first(database, result)
    second = tmp_path / "second"
    second.mkdir()
    _image(second / "a1.jpg")
    other_id = albums.index(second).album_id
    indexer.index(other_id)
    labels = PersonLabelService(database)
    for album_id, cluster_id in (
        (result.album_id, "missing"),
        (other_id, label.cluster_id),
        ("missing", label.cluster_id),
    ):
        with pytest.raises(KeyError):
            labels.get(album_id, cluster_id)
        with pytest.raises(KeyError):
            labels.set(album_id, cluster_id, "Friend")
    assert labels.get(result.album_id, label.cluster_id).label == "Me"


@pytest.mark.parametrize(
    "label",
    ["", "   ", "a" * 81, "a\nb", "a\tb", "a\x00b", "a\u200bb", "a\u202eb", None],
)
def test_invalid_labels_are_rejected(label) -> None:
    with pytest.raises(ValueError):
        normalize_label(label)


def test_unicode_labels_normalize_without_guessing_identity() -> None:
    assert normalize_label("  张三  ") == "张三"
    assert normalize_label("e\u0301") == "é"
    assert normalize_label("我") == "我"


def test_stale_revision_cannot_overwrite_new_label(tmp_path: Path) -> None:
    _, database, _, _, result = _setup(tmp_path)
    state = _label_first(database, result)
    labels = PersonLabelService(database)
    with pytest.raises(ValueError, match="revision changed"):
        labels.set(result.album_id, state.cluster_id, "Friend", expected_revision=0)
    assert labels.get(result.album_id, state.cluster_id) == state


def test_deleted_members_need_review_and_history_is_immutable(tmp_path: Path) -> None:
    folder, database, albums, indexer, result = _setup(tmp_path)
    _label_first(database, result)
    (folder / "a2.jpg").unlink()
    albums.index(folder)
    refreshed = indexer.index(result.album_id)
    assert refreshed.clusters[0].label_status == "needs_review"
    with database.connect() as connection:
        events = connection.execute("SELECT * FROM person_label_events").fetchall()
    assert len(events) >= 3
    assert json.loads(events[0]["evidence_json"])["version"] == 1
    for action in (
        "DELETE FROM person_label_events",
        "UPDATE person_label_events SET label = 'Fake'",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with database.connect() as connection:
                connection.execute(action)


def test_missing_evidence_requires_index_refresh_before_naming(tmp_path: Path) -> None:
    _, database, _, _, result = _setup(tmp_path)
    cluster_id = result.clusters[0].cluster_id
    with database.connect() as connection:
        connection.execute(
            "UPDATE person_clusters SET identity_evidence_json = '{}' WHERE id = ?",
            (cluster_id,),
        )
    with pytest.raises(ValueError, match="missing"):
        PersonLabelService(database).set(result.album_id, cluster_id, "Me")


def test_unlabeled_evidence_change_advances_revision(tmp_path: Path) -> None:
    folder, database, albums, indexer, result = _setup(tmp_path)
    cluster_id = result.clusters[0].cluster_id
    assert result.clusters[0].label_revision == 0
    _image(folder / "a1.jpg", "red")
    albums.index(folder)
    refreshed = indexer.index(result.album_id)
    assert refreshed.clusters[0].cluster_id == cluster_id
    assert refreshed.clusters[0].label_revision > 0
    with pytest.raises(ValueError, match="revision changed"):
        PersonLabelService(database).set(
            result.album_id, cluster_id, "Me", expected_revision=0
        )


def test_removed_and_reappeared_unlabeled_cluster_cannot_reuse_revision_zero(
    tmp_path: Path,
) -> None:
    folder, database, albums, indexer, result = _setup(tmp_path, ("a1", "b1"))
    with database.connect() as connection:
        removed_id = connection.execute(
            "SELECT f.cluster_id FROM faces f JOIN photos p ON p.id = f.photo_id WHERE p.absolute_path = ?",
            (str((folder / "a1.jpg").resolve()),),
        ).fetchone()[0]
    (folder / "a1.jpg").unlink()
    albums.index(folder)
    indexer.index(result.album_id)
    _image(folder / "a1.jpg")
    albums.index(folder)
    refreshed = indexer.index(result.album_id)
    restored = next(
        cluster for cluster in refreshed.clusters if cluster.cluster_id == removed_id
    )
    assert restored.label_revision > 0
    with pytest.raises(ValueError, match="revision changed"):
        PersonLabelService(database).set(
            result.album_id, removed_id, "Me", expected_revision=0
        )


def test_all_face_source_removal_also_invalidates_displayed_revision(
    tmp_path: Path,
) -> None:
    folder, database, albums, indexer, result = _setup(tmp_path, ("a1", "no-face"))
    original = result.clusters[0]
    path = folder / "a1.jpg"
    source = path.read_bytes()
    stamp = path.stat()
    path.unlink()
    albums.index(folder)
    path.write_bytes(source)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    albums.index(folder)
    refreshed = indexer.index(result.album_id)
    assert refreshed.clusters[0].cluster_id == original.cluster_id
    assert refreshed.clusters[0].label_revision > original.label_revision
    with pytest.raises(ValueError, match="revision changed"):
        PersonLabelService(database).set(
            result.album_id, original.cluster_id, "Me", expected_revision=0
        )


def test_no_face_cache_is_content_bound_even_if_size_and_time_unchanged(
    tmp_path: Path,
) -> None:
    folder, database, _, indexer, result = _setup(tmp_path, ("no-face",))
    assert result.total_faces == 0
    path = folder / "no-face.jpg"
    stat = path.stat()
    content = bytearray(path.read_bytes())
    content[-10] ^= 1
    path.write_bytes(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    refreshed = indexer.index(result.album_id)
    assert refreshed.computed_count == 1
    assert refreshed.reused_count == 0
    with database.connect() as connection:
        row = connection.execute("SELECT face_source_sha256 FROM photos").fetchone()
    assert row["face_source_sha256"] is not None
