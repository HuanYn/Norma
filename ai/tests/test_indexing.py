from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageFilter
import pytest

from ai import app as app_module
from ai.config import Settings
from ai.index import AlbumIndexer, IndexingCancelledError
from ai.index.scanner import ScannedPhoto
from ai.library import AlbumCatalogService
from ai.storage import Database


def _jpeg(path: Path, color: tuple[int, int, int], *, blurred: bool = False) -> None:
    image = Image.new("RGB", (800, 600), color)
    for offset in range(40, 760, 80):
        for y in range(70, 540):
            image.putpixel(
                (offset, y), (255 - color[0], 255 - color[1], 255 - color[2])
            )
    if blurred:
        image = image.filter(ImageFilter.GaussianBlur(radius=22))
    image.save(path, "JPEG", quality=92)


def _fake_scanned_photo(path: Path, thumbnail_dir: Path) -> ScannedPhoto:
    stat = path.stat()
    thumbnail_path = thumbnail_dir / f"{path.stem}.jpg"
    thumbnail_path.write_bytes(b"preview")
    ordinal = int(path.name[:2])
    return ScannedPhoto(
        id=f"photo-{ordinal}",
        path=path,
        thumbnail_path=thumbnail_path,
        width=800,
        height=600,
        file_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        capture_time=None,
        quality_score=75.0,
        blur_score=100.0,
        phash=f"{ordinal:016x}",
        dhash=f"{ordinal:016x}",
        auto_reject=False,
        reject_reason=None,
        quality_flags=[],
        metadata={},
    )


def test_indexes_jpgs_without_touching_originals(tmp_path: Path, monkeypatch) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _jpeg(album / "architecture.jpg", (60, 90, 130))
    _jpeg(album / "night.jpeg", (12, 16, 34))
    _jpeg(album / "blurred.jpg", (130, 80, 40), blurred=True)
    shutil.copyfile(album / "architecture.jpg", album / "architecture-copy.jpg")
    Image.new("RGB", (200, 200), "red").save(album / "ignored.png")
    before = {path: path.stat().st_mtime_ns for path in album.glob("*.jp*g")}

    data_dir = tmp_path / "data"
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(host="127.0.0.1", port=8765, data_dir=data_dir, log_level="INFO"),
    )

    with TestClient(app_module.app) as client:
        response = client.post("/albums/index", json={"folder": str(album)})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 4
    assert payload["computed_count"] == 4
    assert payload["reused_count"] == 0
    assert payload["provider"] == "pillow-opencv-fallback-v1"
    assert payload["similar_groups"] >= 1
    assert all(Path(photo["thumbnail_path"]).exists() for photo in payload["photos"])
    assert {path: path.stat().st_mtime_ns for path in before} == before

    with Database(data_dir / "norma.db").connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 4
        assert (
            connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[
                0
            ]
            == 14
        )


def test_rejects_folder_without_jpegs(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(host="127.0.0.1", port=8765, data_dir=data_dir, log_level="INFO"),
    )

    with TestClient(app_module.app) as client:
        response = client.post("/albums/index", json={"folder": str(empty)})

    assert response.status_code == 400
    assert "JPG/JPEG" in response.json()["detail"]


def test_incremental_index_reuses_unchanged_photos_and_refreshes_one_change(
    tmp_path: Path, monkeypatch
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    changed = album / "changed.jpg"
    stable = album / "stable.jpg"
    _jpeg(changed, (30, 70, 120))
    _jpeg(stable, (80, 120, 50))
    data_dir = tmp_path / "data"
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(host="127.0.0.1", port=8765, data_dir=data_dir, log_level="INFO"),
    )

    with TestClient(app_module.app) as client:
        first = client.post("/albums/index", json={"folder": str(album)}).json()
        thumbnail_mtimes = {
            photo["filename"]: Path(photo["thumbnail_path"]).stat().st_mtime_ns
            for photo in first["photos"]
        }
        second = client.post("/albums/index", json={"folder": str(album)}).json()
        second_thumbnail_mtimes = {
            photo["filename"]: Path(photo["thumbnail_path"]).stat().st_mtime_ns
            for photo in second["photos"]
        }
        _jpeg(changed, (190, 80, 35))
        third = client.post("/albums/index", json={"folder": str(album)}).json()

    assert second["computed_count"] == 0
    assert second["reused_count"] == 2
    assert second_thumbnail_mtimes == thumbnail_mtimes
    assert third["computed_count"] == 1
    assert third["reused_count"] == 1
    assert {photo["filename"]: photo["id"] for photo in third["photos"]} == {
        photo["filename"]: photo["id"] for photo in first["photos"]
    }


def test_base_index_defers_quality_and_quality_run_fills_the_same_rows(
    tmp_path: Path,
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    original = album / "original.jpg"
    duplicate = album / "duplicate.jpg"
    _jpeg(original, (45, 95, 145))
    shutil.copyfile(original, duplicate)
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexer = AlbumIndexer(database, data_dir)

    base = indexer.index(album, analyze_quality=False)
    base_ids = {photo.filename: photo.id for photo in base.photos}

    assert base.computed_count == 2
    assert base.reused_count == 0
    assert base.rejected == 0
    assert base.similar_groups == 0
    assert all(photo.quality_score is None for photo in base.photos)
    assert all(photo.blur_score is None for photo in base.photos)
    assert all(Path(photo.thumbnail_path).is_file() for photo in base.photos)
    with database.connect() as connection:
        base_rows = connection.execute(
            """SELECT quality_score, blur_score, phash, dhash, similarity_group
               FROM photos WHERE album_id = ?""",
            (base.album_id,),
        ).fetchall()
    assert all(all(value is None for value in row) for row in base_rows)

    analyzed = indexer.index(album, analyze_quality=True)
    repeated = indexer.index(album, analyze_quality=True)

    assert {photo.filename: photo.id for photo in analyzed.photos} == base_ids
    assert analyzed.computed_count == 2
    assert analyzed.reused_count == 0
    assert analyzed.similar_groups == 1
    assert all(photo.quality_score is not None for photo in analyzed.photos)
    assert repeated.computed_count == 0
    assert repeated.reused_count == 2


def test_partial_base_refresh_preserves_quality_but_clears_album_groups(
    tmp_path: Path,
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    first = album / "first.jpg"
    second = album / "second.jpg"
    _jpeg(first, (45, 95, 145))
    shutil.copyfile(first, second)
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexer = AlbumIndexer(database, data_dir)
    initial = indexer.index(album)
    assert initial.similar_groups == 1

    _jpeg(album / "new.jpg", (175, 105, 35))
    refreshed = indexer.index(album, analyze_quality=False)

    by_name = {photo.filename: photo for photo in refreshed.photos}
    assert by_name["first.jpg"].quality_score is not None
    assert by_name["second.jpg"].quality_score is not None
    assert by_name["new.jpg"].quality_score is None
    assert refreshed.similar_groups == 0
    assert all(photo.similarity_group is None for photo in refreshed.photos)

    completed = indexer.index(album)
    assert completed.computed_count == 1
    assert completed.reused_count == 2
    assert completed.similar_groups == 1


def test_base_refresh_clears_stale_group_when_a_member_is_deleted(
    tmp_path: Path,
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    first = album / "first.jpg"
    second = album / "second.jpg"
    _jpeg(first, (45, 95, 145))
    shutil.copyfile(first, second)
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexer = AlbumIndexer(database, data_dir)
    initial = indexer.index(album)
    assert initial.similar_groups == 1

    second.unlink()
    refreshed = indexer.index(album, analyze_quality=False)

    assert refreshed.total == 1
    assert refreshed.photos[0].quality_score is not None
    assert refreshed.photos[0].similarity_group is None
    assert refreshed.similar_groups == 0
    summary = AlbumCatalogService(database).get_album(initial.album_id)
    assert summary.quality_count == summary.photo_count == 1
    assert summary.similar_group_count == 0
    with database.connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*), MAX(similarity_group) FROM photos WHERE album_id = ?",
            (initial.album_id,),
        ).fetchone()
    assert tuple(row) == (1, None)


def test_failed_rescan_preserves_photo_but_invalidates_unverifiable_derived_rows(
    tmp_path: Path,
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    source = album / "photo.jpg"
    _jpeg(source, (45, 95, 145))
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexer = AlbumIndexer(database, data_dir)
    initial = indexer.index(album)
    photo = initial.photos[0]
    embedding_path = data_dir / "preserved.npy"
    embedding_path.write_bytes(b"derived")
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE photos SET embedding_path = ?, embedding_provider = 'test',
                embedding_source_size = file_size,
                embedding_source_mtime_ns = source_mtime_ns,
                embedding_source_sha256 = ?,
                face_provider = 'test-face', face_source_size = file_size,
                face_source_mtime_ns = source_mtime_ns,
                face_processed = 1, face_count = 1
            WHERE id = ?
            """,
            (
                str(embedding_path),
                hashlib.sha256(source.read_bytes()).hexdigest(),
                photo.id,
            ),
        )
        connection.execute(
            "INSERT INTO person_clusters(id, album_id, label) VALUES ('cluster', ?, 'Person')",
            (initial.album_id,),
        )
        connection.execute(
            """INSERT INTO faces(id, photo_id, cluster_id, box_json, embedding_path)
               VALUES ('face', ?, 'cluster', '[1, 2, 3, 4]', ?)""",
            (photo.id, str(data_dir / "face.npy")),
        )
        connection.execute(
            """INSERT INTO evaluation_queries(id, album_id, query_text)
               VALUES ('query', ?, 'portrait')""",
            (initial.album_id,),
        )
        connection.execute(
            """INSERT INTO relevance_judgments(query_id, photo_id, relevance)
               VALUES ('query', ?, 3)""",
            (photo.id,),
        )

    source.write_bytes(b"temporarily not a jpeg")
    base_retry = indexer.index(album, analyze_quality=False)
    quality_retry = indexer.index(album, analyze_quality=True)

    assert base_retry.total == quality_retry.total == 1
    assert base_retry.photos[0].id == quality_retry.photos[0].id == photo.id
    assert len(base_retry.errors) == len(quality_retry.errors) == 1
    assert base_retry.reused_count == quality_retry.reused_count == 1
    with database.connect() as connection:
        stored = connection.execute(
            """SELECT embedding_path, embedding_provider, face_processed, face_count
               FROM photos WHERE id = ?""",
            (photo.id,),
        ).fetchone()
        counts = (
            connection.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM person_clusters").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM relevance_judgments").fetchone()[
                0
            ],
        )
    assert stored["embedding_path"] is None
    assert stored["embedding_provider"] is None
    assert stored["face_processed"] == 0
    assert stored["face_count"] == 0
    assert counts == (0, 0, 1)


def test_same_size_restored_mtime_content_change_invalidates_embedding_cache(
    tmp_path: Path,
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    source = album / "photo.jpg"
    _jpeg(source, (45, 95, 145))
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexer = AlbumIndexer(database, data_dir)
    initial = indexer.index(album)
    photo = initial.photos[0]
    original = source.read_bytes()
    original_stat = source.stat()
    embedding_path = data_dir / "bound.npy"
    embedding_path.write_bytes(b"derived")
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE photos SET embedding_path = ?, embedding_provider = 'test',
                embedding_source_size = file_size,
                embedding_source_mtime_ns = source_mtime_ns,
                embedding_source_sha256 = ?
            WHERE id = ?
            """,
            (
                str(embedding_path),
                hashlib.sha256(original).hexdigest(),
                photo.id,
            ),
        )

    replacement = bytearray(original)
    replacement[100] ^= 1
    with Image.open(BytesIO(replacement)) as image:
        image.load()
    source.write_bytes(replacement)
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns

    indexer.index(album)

    with database.connect() as connection:
        row = connection.execute(
            """SELECT embedding_path, embedding_provider,
                      embedding_source_size, embedding_source_mtime_ns,
                      embedding_source_sha256
               FROM photos WHERE id = ?""",
            (photo.id,),
        ).fetchone()
    assert tuple(row) == (None, None, None, None, None)


def test_overlapping_parent_and_child_albums_keep_independent_photo_rows(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    photo = child / "shared.jpg"
    _jpeg(photo, (45, 95, 145))
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexer = AlbumIndexer(database, data_dir)

    child_album = indexer.index(child)
    parent_album = indexer.index(parent)
    repeated_child = indexer.index(child)
    repeated_parent = indexer.index(parent)

    assert child_album.total == parent_album.total == 1
    assert child_album.photos[0].id != parent_album.photos[0].id
    assert repeated_child.reused_count == repeated_parent.reused_count == 1
    assert repeated_child.photos[0].id == child_album.photos[0].id
    assert repeated_parent.photos[0].id == parent_album.photos[0].id
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT album_id, id, absolute_path FROM photos
               WHERE absolute_path = ? ORDER BY album_id""",
            (str(photo.resolve()),),
        ).fetchall()
    assert len(rows) == 2
    assert {row["album_id"] for row in rows} == {
        child_album.album_id,
        parent_album.album_id,
    }
    assert len({row["id"] for row in rows}) == 2


def test_index_reports_every_source_including_reused_and_failed_photos(
    tmp_path: Path,
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _jpeg(album / "first.jpg", (45, 95, 145))
    _jpeg(album / "second.jpeg", (145, 95, 45))
    (album / "unreadable.jpg").write_bytes(b"not a jpeg")
    database = Database(tmp_path / "data" / "norma.db")
    indexer = AlbumIndexer(database, tmp_path / "data")

    first_progress: list[tuple[int, int]] = []
    first = indexer.index(
        album, on_progress=lambda *value: first_progress.append(value)
    )
    second_progress: list[tuple[int, int]] = []
    second = indexer.index(
        album, on_progress=lambda *value: second_progress.append(value)
    )

    assert first_progress == [(1, 3), (2, 3), (3, 3)]
    assert second_progress == [(1, 3), (2, 3), (3, 3)]
    assert first.total == 2
    assert len(first.errors) == 1
    assert "unreadable.jpg" in first.errors[0]
    assert second.computed_count == 0
    assert second.reused_count == 2
    assert len(second.errors) == 1


def test_parallel_scan_preserves_path_order_progress_and_error_order(
    tmp_path: Path, monkeypatch
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    for filename in ("01.jpg", "02.jpg", "03-broken.jpg", "04-broken.jpg"):
        (album / filename).write_bytes(b"source")
    data_dir = tmp_path / "data"
    indexer = AlbumIndexer(Database(data_dir / "norma.db"), data_dir, scan_workers=4)
    barrier = threading.Barrier(4)
    completion_order: list[str] = []
    completion_lock = threading.Lock()
    delays = {
        "01.jpg": 0.30,
        "02.jpg": 0.20,
        "03-broken.jpg": 0.10,
        "04-broken.jpg": 0.00,
    }

    def fake_scan(path: Path, _album_id: str, thumbnail_dir: Path) -> ScannedPhoto:
        barrier.wait(timeout=2)
        time.sleep(delays[path.name])
        with completion_lock:
            completion_order.append(path.name)
        if "broken" in path.name:
            raise OSError("deliberate test failure")
        return _fake_scanned_photo(path, thumbnail_dir)

    monkeypatch.setattr(indexer, "_scan_photo", fake_scan)
    progress: list[tuple[int, int]] = []
    indexed = indexer.index(album, on_progress=lambda *value: progress.append(value))

    assert completion_order == [
        "04-broken.jpg",
        "03-broken.jpg",
        "02.jpg",
        "01.jpg",
    ]
    assert [photo.filename for photo in indexed.photos] == ["01.jpg", "02.jpg"]
    assert [error.split(":", 1)[0] for error in indexed.errors] == [
        "03-broken.jpg",
        "04-broken.jpg",
    ]
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_cancelled_parallel_scan_stops_refilling_and_never_persists(
    tmp_path: Path, monkeypatch
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    for ordinal in range(20):
        (album / f"{ordinal:02d}.jpg").write_bytes(b"source")
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexer = AlbumIndexer(database, data_dir, scan_workers=2)
    cancel_requested = threading.Event()
    state_lock = threading.Lock()
    started: list[str] = []
    active = 0
    max_active = 0

    def fake_scan(path: Path, _album_id: str, thumbnail_dir: Path) -> ScannedPhoto:
        nonlocal active, max_active
        with state_lock:
            started.append(path.name)
            active += 1
            max_active = max(max_active, active)
        try:
            if path.name == "00.jpg":
                time.sleep(0.05)
            else:
                assert cancel_requested.wait(timeout=2)
            return _fake_scanned_photo(path, thumbnail_dir)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(indexer, "_scan_photo", fake_scan)
    progress: list[tuple[int, int]] = []

    def request_cancel(completed: int, total: int) -> None:
        progress.append((completed, total))
        cancel_requested.set()

    with pytest.raises(IndexingCancelledError, match="between photos"):
        indexer.index(
            album,
            on_progress=request_cancel,
            should_cancel=cancel_requested.is_set,
        )

    assert progress == [(1, 20)]
    assert 1 <= len(started) <= 4
    assert max_active <= 2
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM albums").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 0


def test_cancelled_reindex_keeps_last_complete_album_snapshot(tmp_path: Path) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _jpeg(album / "first.jpg", (45, 95, 145))
    _jpeg(album / "second.jpg", (145, 95, 45))
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexer = AlbumIndexer(database, data_dir)
    indexed = indexer.index(album)
    with database.connect() as connection:
        before = [
            tuple(row)
            for row in connection.execute(
                """SELECT id, file_size, source_mtime_ns
                   FROM photos WHERE album_id = ? ORDER BY id""",
                (indexed.album_id,),
            ).fetchall()
        ]

    _jpeg(album / "third.jpg", (75, 125, 55))
    cancel = False
    progress: list[tuple[int, int]] = []

    def record_progress(completed: int, total: int) -> None:
        nonlocal cancel
        progress.append((completed, total))
        cancel = True

    with pytest.raises(IndexingCancelledError, match="between photos"):
        indexer.index(
            album,
            on_progress=record_progress,
            should_cancel=lambda: cancel,
        )

    with database.connect() as connection:
        after = [
            tuple(row)
            for row in connection.execute(
                """SELECT id, file_size, source_mtime_ns
                   FROM photos WHERE album_id = ? ORDER BY id""",
                (indexed.album_id,),
            ).fetchall()
        ]
    assert progress == [(1, 3)]
    assert after == before
