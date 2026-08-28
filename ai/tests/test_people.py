from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from ai import app as app_module
from ai.config import Settings
from ai.index import AlbumIndexer
from ai.people.indexer import IndexedFace, PeopleCancelledError, PeopleIndexer, _cluster
from ai.people.provider import DetectedFace, FaceClusterPolicy, FaceProvider
from ai.storage import Database


class FakeFaceProvider(FaceProvider):
    name = "fake-face-v1"
    dimension = 3
    cluster_policy = FaceClusterPolicy(
        version="fake-cluster-v1",
        minimum_similarity=0.985,
        mean_similarity=0.985,
        centroid_similarity=0.985,
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def detect(self, path: Path) -> list[DetectedFace]:
        self.calls.append(path.name)
        if "no-face" in path.name:
            return []
        descriptor = (
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            if "person-b" in path.name
            else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        )
        return [
            DetectedFace(
                box=(30, 20, 90, 90),
                descriptor=descriptor,
                crop=Image.new("RGB", (120, 120), "tan"),
            )
        ]


def _photo(path: Path, color: str) -> None:
    image = Image.new("RGB", (480, 360), color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((130, 55, 350, 275), outline="white", width=14)
    draw.rectangle((200, 140, 225, 165), fill="black")
    draw.rectangle((270, 140, 295, 165), fill="black")
    image.save(path, "JPEG", quality=92)


def _indexed_face(face_id: str, photo_id: str, descriptor: np.ndarray) -> IndexedFace:
    return IndexedFace(
        id=face_id,
        photo_id=photo_id,
        box=(0, 0, 10, 10),
        descriptor_path=Path(f"{face_id}.npy"),
        thumbnail_path=Path(f"{face_id}.jpg"),
        descriptor=descriptor.astype(np.float32),
    )


def test_constrained_clustering_blocks_single_link_chains() -> None:
    radians = np.deg2rad([0.0, 50.0, 100.0])
    faces = [
        _indexed_face(
            f"face-{index}",
            f"photo-{index}",
            np.asarray([np.cos(angle), np.sin(angle)]),
        )
        for index, angle in enumerate(radians)
    ]
    policy = FaceClusterPolicy(
        version="test-v1",
        minimum_similarity=0.6,
        mean_similarity=0.6,
        centroid_similarity=0.6,
    )

    components = _cluster(faces, policy)

    assert sorted(len(component) for component in components) == [1, 2]


def test_constrained_clustering_never_merges_two_faces_from_one_photo() -> None:
    descriptor = np.asarray([1.0, 0.0], dtype=np.float32)
    faces = [
        _indexed_face("face-a", "photo-a", descriptor),
        _indexed_face("face-b", "photo-a", descriptor),
        _indexed_face("face-c", "photo-b", descriptor),
    ]
    policy = FaceClusterPolicy(
        version="test-v1",
        minimum_similarity=0.9,
        mean_similarity=0.9,
        centroid_similarity=0.9,
    )

    components = _cluster(faces, policy)

    assert sorted(len(component) for component in components) == [1, 2]
    for component in components:
        photo_ids = [faces[index].photo_id for index in component]
        assert len(photo_ids) == len(set(photo_ids))


def test_prototype_attachment_rejoins_pose_fragmented_seed_clusters() -> None:
    radians = np.deg2rad([0.0, 10.0, 55.0, 65.0])
    faces = [
        _indexed_face(
            f"face-{index}",
            f"photo-{index}",
            np.asarray([np.cos(angle), np.sin(angle)]),
        )
        for index, angle in enumerate(radians)
    ]
    policy = FaceClusterPolicy(
        version="prototype-test-v1",
        minimum_similarity=0.8,
        mean_similarity=0.8,
        centroid_similarity=0.8,
        prototype_attachment=True,
        multi_cluster_centroid_similarity=0.55,
        multi_cluster_mean_similarity=0.5,
        multi_cluster_max_similarity=0.7,
        singleton_centroid_similarity=0.9,
        singleton_mean_similarity=0.9,
        singleton_max_similarity=0.9,
    )

    components = _cluster(faces, policy)

    assert [len(component) for component in components] == [4]


def test_prototype_attachment_keeps_same_photo_cannot_link() -> None:
    radians = np.deg2rad([0.0, 5.0, 50.0, 55.0])
    photo_ids = ["shared", "photo-a", "shared", "photo-b"]
    faces = [
        _indexed_face(
            f"face-{index}",
            photo_ids[index],
            np.asarray([np.cos(angle), np.sin(angle)]),
        )
        for index, angle in enumerate(radians)
    ]
    policy = FaceClusterPolicy(
        version="prototype-test-v1",
        minimum_similarity=0.9,
        mean_similarity=0.9,
        centroid_similarity=0.9,
        prototype_attachment=True,
        multi_cluster_centroid_similarity=0.5,
        multi_cluster_mean_similarity=0.5,
        multi_cluster_max_similarity=0.5,
        singleton_centroid_similarity=0.5,
        singleton_mean_similarity=0.5,
        singleton_max_similarity=0.5,
    )

    components = _cluster(faces, policy)

    assert sorted(len(component) for component in components) == [2, 2]
    for component in components:
        component_photos = [faces[index].photo_id for index in component]
        assert len(component_photos) == len(set(component_photos))


def test_people_index_clusters_conservatively_and_persists(
    tmp_path: Path, monkeypatch
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _photo(album / "person-a-1.jpg", "navy")
    _photo(album / "person-a-2.jpg", "blue")
    _photo(album / "person-b.jpg", "green")
    _photo(album / "no-face.jpg", "gray")

    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexed = AlbumIndexer(database, data_dir).index(album)
    service = PeopleIndexer(database, data_dir, FakeFaceProvider())
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(
            host="127.0.0.1",
            port=8765,
            data_dir=data_dir,
            log_level="INFO",
        ),
    )
    monkeypatch.setattr(app_module, "people_indexer", lambda: service)

    with TestClient(app_module.app) as client:
        response = client.post(f"/albums/{indexed.album_id}/people/index")
        persisted = client.get(f"/albums/{indexed.album_id}/people")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["provider"] == "fake-face-v1"
    assert payload["total_faces"] == 3
    assert payload["cluster_count"] == 2
    assert payload["computed_count"] == 4
    assert payload["reused_count"] == 0
    assert sorted(len(cluster["faces"]) for cluster in payload["clusters"]) == [1, 2]
    assert all(
        Path(data_dir / match["thumbnail_url"].removeprefix("/media/")).exists()
        for cluster in payload["clusters"]
        for match in cluster["faces"]
    )
    assert persisted.status_code == 200, persisted.text
    persisted_payload = persisted.json()
    assert persisted_payload["provider"] == "fake-face-v1"
    assert persisted_payload["total_faces"] == 3
    assert persisted_payload["cluster_count"] == 2
    assert persisted_payload["computed_count"] == 0
    assert persisted_payload["reused_count"] == 4
    assert persisted_payload["clusters"] == payload["clusters"]
    assert len(service.provider.calls) == 4

    with database.connect() as connection:
        connection.execute(
            "UPDATE photos SET face_count = face_count + 1 WHERE id = ?",
            (indexed.photos[0].id,),
        )
    with TestClient(app_module.app) as client:
        inconsistent = client.get(f"/albums/{indexed.album_id}/people")
    assert inconsistent.status_code == 400
    assert "inconsistent" in inconsistent.json()["detail"]
    with database.connect() as connection:
        connection.execute(
            "UPDATE photos SET face_count = face_count - 1 WHERE id = ?",
            (indexed.photos[0].id,),
        )

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 3
        assert (
            connection.execute("SELECT COUNT(*) FROM person_clusters").fetchone()[0]
            == 2
        )

    AlbumIndexer(database, data_dir).index(album)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 3

    repeated = service.index(indexed.album_id)
    assert repeated.computed_count == 0
    assert repeated.reused_count == 4
    assert len(service.provider.calls) == 4

    _photo(album / "person-a-1.jpg", "red")
    AlbumIndexer(database, data_dir).index(album)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM person_clusters").fetchone()[0]
            == 0
        )

    refreshed = service.index(indexed.album_id)
    assert refreshed.total_faces == 3
    assert refreshed.cluster_count == 2
    assert refreshed.computed_count == 4
    assert refreshed.reused_count == 0
    assert len(service.provider.calls) == 8


@pytest.mark.parametrize("change", ["add", "delete", "modify"])
def test_album_change_invalidates_the_entire_people_snapshot(
    tmp_path: Path,
    change: str,
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    first = album / "person-a.jpg"
    second = album / "person-b.jpg"
    _photo(first, "navy")
    _photo(second, "green")
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexer = AlbumIndexer(database, data_dir)
    indexed = indexer.index(album)
    people = PeopleIndexer(database, data_dir, FakeFaceProvider()).index(
        indexed.album_id
    )
    assert people.total_faces == 2

    if change == "add":
        _photo(album / "person-c.jpg", "purple")
    elif change == "delete":
        second.unlink()
    else:
        _photo(first, "red")
    indexer.index(album, analyze_quality=False)

    with database.connect() as connection:
        states = connection.execute(
            """
            SELECT face_provider, face_source_size, face_source_mtime_ns,
                   face_processed, face_count
            FROM photos WHERE album_id = ?
            """,
            (indexed.album_id,),
        ).fetchall()
        face_count = connection.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        cluster_count = connection.execute(
            "SELECT COUNT(*) FROM person_clusters WHERE album_id = ?",
            (indexed.album_id,),
        ).fetchone()[0]
    assert states
    assert all(tuple(state) == (None, None, None, 0, 0) for state in states)
    assert face_count == 0
    assert cluster_count == 0


def test_people_index_rejects_missing_album(tmp_path: Path) -> None:
    database = Database(tmp_path / "data" / "norma.db")
    database.initialize()
    service = PeopleIndexer(database, tmp_path / "data", FakeFaceProvider())
    try:
        service.index("missing")
    except KeyError as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("missing album should raise KeyError")


@pytest.mark.parametrize(
    "corruption",
    ["missing", "empty", "truncated", "null-path"],
)
def test_people_index_recomputes_only_corrupt_face_cache(
    tmp_path: Path,
    corruption: str,
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _photo(album / "person-a.jpg", "navy")
    _photo(album / "person-b.jpg", "green")
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexed = AlbumIndexer(database, data_dir).index(album)
    provider = FakeFaceProvider()
    service = PeopleIndexer(database, data_dir, provider)
    first = service.index(indexed.album_id)
    assert first.computed_count == 2

    with database.connect() as connection:
        corrupt = connection.execute(
            """
            SELECT f.id, f.embedding_path FROM faces f
            JOIN photos p ON p.id = f.photo_id
            WHERE p.absolute_path LIKE '%person-b.jpg'
            """
        ).fetchone()
        descriptor_path = Path(corrupt["embedding_path"])
        if corruption == "null-path":
            connection.execute(
                "UPDATE faces SET embedding_path = NULL WHERE id = ?",
                (corrupt["id"],),
            )
    if corruption == "missing":
        descriptor_path.unlink()
    elif corruption == "empty":
        descriptor_path.write_bytes(b"")
    elif corruption == "truncated":
        payload = descriptor_path.read_bytes()
        descriptor_path.write_bytes(payload[: max(1, len(payload) // 2)])

    second = service.index(indexed.album_id)
    assert second.computed_count == 1
    assert second.reused_count == 1
    assert provider.calls.count("person-a.jpg") == 1
    assert provider.calls.count("person-b.jpg") == 2
    with database.connect() as connection:
        repaired = connection.execute(
            "SELECT embedding_path FROM faces WHERE id = ?", (corrupt["id"],)
        ).fetchone()
    assert repaired["embedding_path"]
    repaired_descriptor = np.load(repaired["embedding_path"], allow_pickle=False)
    assert repaired_descriptor.shape == (provider.dimension,)


def test_people_index_recomputes_when_provider_version_changes(tmp_path: Path) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _photo(album / "person-a.jpg", "navy")
    _photo(album / "person-b.jpg", "green")
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexed = AlbumIndexer(database, data_dir).index(album)
    first_provider = FakeFaceProvider()
    PeopleIndexer(database, data_dir, first_provider).index(indexed.album_id)

    second_provider = FakeFaceProvider()
    second_provider.name = "fake-face-v2"
    refreshed = PeopleIndexer(database, data_dir, second_provider).index(
        indexed.album_id
    )

    assert refreshed.computed_count == 2
    assert refreshed.reused_count == 0
    assert sorted(second_provider.calls) == ["person-a.jpg", "person-b.jpg"]
    with database.connect() as connection:
        providers = connection.execute(
            "SELECT DISTINCT face_provider FROM photos WHERE album_id = ?",
            (indexed.album_id,),
        ).fetchall()
    assert [row["face_provider"] for row in providers] == ["fake-face-v2"]


def test_people_index_checks_stale_sources_and_cancels_between_photos(
    tmp_path: Path,
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _photo(album / "person-a.jpg", "navy")
    _photo(album / "person-b.jpg", "green")
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexed = AlbumIndexer(database, data_dir).index(album)
    service = PeopleIndexer(database, data_dir, FakeFaceProvider())

    cancel = False

    def progress(completed: int, total: int) -> None:
        nonlocal cancel
        assert total == 2
        if completed == 1:
            cancel = True

    try:
        service.index(
            indexed.album_id,
            on_progress=progress,
            should_cancel=lambda: cancel,
        )
    except PeopleCancelledError as error:
        assert "between photos" in str(error)
    else:
        raise AssertionError("people indexing should honor cancellation")

    with database.connect() as connection:
        processed = connection.execute(
            "SELECT SUM(face_processed) FROM photos WHERE album_id = ?",
            (indexed.album_id,),
        ).fetchone()[0]
    assert processed == 0

    _photo(album / "person-a.jpg", "red")
    try:
        service.index(indexed.album_id)
    except ValueError as error:
        assert "source changed since indexing" in str(error)
    else:
        raise AssertionError("stale source should require re-indexing")
