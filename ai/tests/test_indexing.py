from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

from ai import app as app_module
from ai.config import Settings
from ai.storage import Database


def _jpeg(path: Path, color: tuple[int, int, int], *, blurred: bool = False) -> None:
    image = Image.new("RGB", (800, 600), color)
    for offset in range(40, 760, 80):
        for y in range(70, 540):
            image.putpixel((offset, y), (255 - color[0], 255 - color[1], 255 - color[2]))
    if blurred:
        image = image.filter(ImageFilter.GaussianBlur(radius=22))
    image.save(path, "JPEG", quality=92)


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
    assert payload["provider"] == "pillow-opencv-fallback-v1"
    assert payload["similar_groups"] >= 1
    assert all(Path(photo["thumbnail_path"]).exists() for photo in payload["photos"])
    assert {path: path.stat().st_mtime_ns for path in before} == before

    with Database(data_dir / "norma.db").connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 4
        assert connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 2


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
