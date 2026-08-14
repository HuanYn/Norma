from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ai import app as app_module
from ai.config import Settings
from ai.storage import Database


def _photo(path: Path, color: tuple[int, int, int], offset: int) -> None:
    image = Image.new("RGB", (640, 420), color)
    draw = ImageDraw.Draw(image)
    for x in range(30 + offset, 610, 70):
        draw.line((x, 20, x, 400), fill=(240, 190, 90), width=12)
    image.save(path, "JPEG", quality=94)


def test_catalog_restores_album_counts_and_paginated_photos(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "trip"
    folder.mkdir()
    for index, color in enumerate(((8, 14, 38), (28, 80, 42), (210, 170, 75))):
        _photo(folder / f"photo-{index}.jpg", color, index * 7)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(host="127.0.0.1", port=8765, data_dir=data_dir, log_level="INFO"),
    )

    with TestClient(app_module.app) as client:
        indexed = client.post(
            "/albums/index", json={"folder": str(folder), "name": "Trip"}
        ).json()
        album_id = indexed["album_id"]
        assert client.post(f"/albums/{album_id}/embed").status_code == 200
        selected = client.post(
            "/selections",
            json={"album_id": album_id, "prompt": "Select 1 photo, include blurry"},
        )
        assert selected.status_code == 200, selected.text

        catalog = client.get("/albums?limit=1&offset=0")
        detail = client.get(f"/albums/{album_id}")
        first_page = client.get(
            f"/albums/{album_id}/photos?include_rejects=true&sort=quality&limit=2"
        )
        second_page = client.get(
            f"/albums/{album_id}/photos?include_rejects=true&sort=quality&limit=2&offset=2"
        )
        history = client.get(f"/albums/{album_id}/selections")
        missing = client.get("/albums/missing")

    assert catalog.status_code == 200
    assert catalog.json()["total"] == 1
    summary = detail.json()
    assert summary["name"] == "Trip"
    assert summary["photo_count"] == 3
    assert summary["embedded_count"] == 3
    assert summary["selection_count"] == 1
    assert first_page.json()["total"] == 3
    assert len(first_page.json()["items"]) == 2
    assert len(second_page.json()["items"]) == 1
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["selected_count"] == 1
    assert history.json()["items"][0]["feasible"] is True
    qualities = [photo["quality_score"] for photo in first_page.json()["items"]]
    assert qualities == sorted(qualities, reverse=True)
    assert missing.status_code == 404


def test_catalog_validates_photo_query_parameters(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "database", Database(tmp_path / "norma.db"))
    with TestClient(app_module.app) as client:
        assert client.get("/albums?limit=0").status_code == 422
        assert client.get("/albums/anything/photos?sort=unknown").status_code == 422
