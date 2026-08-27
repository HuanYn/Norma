from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ai import app as app_module
from ai.config import Settings
from ai.index import AlbumIndexer
from ai.people.indexer import PeopleCancelledError, PeopleIndexer
from ai.people.provider import DetectedFace, FaceProvider
from ai.storage import Database


class FakeFaceProvider(FaceProvider):
    name = "fake-face-v1"
    dimension = 3

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
        assert connection.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 2
        assert (
            connection.execute("SELECT COUNT(*) FROM person_clusters").fetchone()[0]
            == 0
        )

    refreshed = service.index(indexed.album_id)
    assert refreshed.total_faces == 3
    assert refreshed.cluster_count == 2
    assert refreshed.computed_count == 1
    assert refreshed.reused_count == 3
    assert service.provider.calls[-1] == "person-a-1.jpg"


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


def test_people_index_recomputes_only_corrupt_face_cache(tmp_path: Path) -> None:
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
            SELECT f.embedding_path FROM faces f JOIN photos p ON p.id = f.photo_id
            WHERE p.absolute_path LIKE '%person-b.jpg'
            """
        ).fetchone()
    Path(corrupt["embedding_path"]).unlink()

    second = service.index(indexed.album_id)
    assert second.computed_count == 1
    assert second.reused_count == 1
    assert provider.calls.count("person-a.jpg") == 1
    assert provider.calls.count("person-b.jpg") == 2


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
