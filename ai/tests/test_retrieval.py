from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ai import app as app_module
from ai.config import Settings
from ai.index.embedding import EmbeddingProviderUnavailableError
from ai.storage import Database


def _scene(
    path: Path, background: tuple[int, int, int], accent: tuple[int, int, int]
) -> None:
    image = Image.new("RGB", (640, 420), background)
    draw = ImageDraw.Draw(image)
    for x in range(30, 620, 70):
        draw.rectangle((x, 100, x + 34, 390), fill=accent)
    draw.ellipse((210, 35, 430, 255), outline=accent, width=18)
    image.save(path, "JPEG", quality=94)


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(
            host="127.0.0.1",
            port=8765,
            data_dir=data_dir,
            log_level="INFO",
            embedding_provider="lightweight",
        ),
    )
    return TestClient(app_module.app)


def test_text_image_and_subset_retrieval(tmp_path: Path, monkeypatch) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _scene(album / "dark-blue.jpg", (4, 9, 26), (15, 45, 100))
    shutil.copyfile(album / "dark-blue.jpg", album / "dark-blue-copy.jpg")
    _scene(album / "bright-yellow.jpg", (248, 224, 110), (235, 95, 25))
    _scene(album / "green-nature.jpg", (25, 92, 38), (85, 180, 62))

    with _client(tmp_path, monkeypatch) as client:
        indexed_response = client.post("/albums/index", json={"folder": str(album)})
        assert indexed_response.status_code == 200, indexed_response.text
        indexed = indexed_response.json()
        album_id = indexed["album_id"]
        ids = {photo["filename"]: photo["id"] for photo in indexed["photos"]}

        embedded_response = client.post(f"/albums/{album_id}/embed")
        assert embedded_response.status_code == 200, embedded_response.text
        embedded = embedded_response.json()
        assert embedded["count"] == 4
        assert embedded["provider"] == "lightweight-semantic-v1"
        assert embedded["dimension"] == 16

        text_response = client.post(
            "/albums/search",
            json={"album_id": album_id, "query": "夜景 dark night", "limit": 3},
        )
        assert text_response.status_code == 200, text_response.text
        text_result = text_response.json()
        assert text_result["mode"] == "text"
        assert text_result["matches"][0]["filename"] in {
            "dark-blue.jpg",
            "dark-blue-copy.jpg",
        }

        image_response = client.post(
            "/albums/search",
            json={
                "album_id": album_id,
                "reference_photo_id": ids["dark-blue.jpg"],
                "limit": 3,
            },
        )
        assert image_response.status_code == 200, image_response.text
        image_result = image_response.json()
        assert image_result["mode"] == "image"
        assert image_result["matches"][0]["filename"] == "dark-blue-copy.jpg"
        assert image_result["matches"][0]["score"] > 0.999

        subset_response = client.post(
            "/albums/search",
            json={
                "album_id": album_id,
                "reference_photo_id": ids["dark-blue.jpg"],
                "subset_photo_ids": [ids["green-nature.jpg"]],
            },
        )
        assert subset_response.status_code == 200, subset_response.text
        assert [match["filename"] for match in subset_response.json()["matches"]] == [
            "green-nature.jpg"
        ]


def test_retrieval_requires_one_query_and_cached_embeddings(
    tmp_path: Path, monkeypatch
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _scene(album / "photo.jpg", (20, 40, 80), (100, 130, 180))

    with _client(tmp_path, monkeypatch) as client:
        indexed = client.post("/albums/index", json={"folder": str(album)}).json()
        album_id = indexed["album_id"]
        photo_id = indexed["photos"][0]["id"]

        missing = client.post(
            "/albums/search", json={"album_id": album_id, "query": "night"}
        )
        assert missing.status_code == 404

        assert client.post(f"/albums/{album_id}/embed").status_code == 200
        unsupported = client.post(
            "/albums/search",
            json={"album_id": album_id, "query": "unrecognized-object-token"},
        )
        assert unsupported.status_code == 400
        assert "does not recognize" in unsupported.json()["detail"]

        invalid = client.post(
            "/albums/search",
            json={
                "album_id": album_id,
                "query": "night",
                "reference_photo_id": photo_id,
            },
        )
        assert invalid.status_code == 400


def test_reindex_invalidates_cached_embeddings(tmp_path: Path, monkeypatch) -> None:
    album = tmp_path / "album"
    album.mkdir()
    photo = album / "mutable.jpg"
    _scene(photo, (10, 20, 50), (80, 120, 190))

    with _client(tmp_path, monkeypatch) as client:
        indexed = client.post("/albums/index", json={"folder": str(album)}).json()
        album_id = indexed["album_id"]
        assert client.post(f"/albums/{album_id}/embed").status_code == 200
        assert (
            client.post(
                "/albums/search", json={"album_id": album_id, "query": "night"}
            ).status_code
            == 200
        )

        _scene(photo, (230, 210, 80), (250, 100, 30))
        assert (
            client.post("/albums/index", json={"folder": str(album)}).status_code == 200
        )
        stale_search = client.post(
            "/albums/search", json={"album_id": album_id, "query": "night"}
        )
        assert stale_search.status_code == 404


def test_retrieval_rejects_cache_from_another_provider(
    tmp_path: Path, monkeypatch
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _scene(album / "one.jpg", (10, 20, 50), (80, 120, 190))
    _scene(album / "two.jpg", (30, 80, 45), (120, 190, 90))

    with _client(tmp_path, monkeypatch) as client:
        indexed = client.post("/albums/index", json={"folder": str(album)}).json()
        album_id = indexed["album_id"]
        assert client.post(f"/albums/{album_id}/embed").status_code == 200
        with app_module.database.connect() as connection:
            connection.execute(
                """UPDATE photos SET embedding_provider = 'different-provider'
                   WHERE id = ?""",
                (indexed["photos"][0]["id"],),
            )
        response = client.post(
            "/albums/search", json={"album_id": album_id, "query": "night"}
        )

    assert response.status_code == 404
    assert "lightweight-semantic-v1" in response.json()["detail"]


def test_model_load_failure_is_reported_as_service_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    album = tmp_path / "album"
    album.mkdir()
    _scene(album / "photo.jpg", (10, 20, 50), (80, 120, 190))

    class UnavailableProvider:
        name = "unavailable-test-provider-v1"
        dimension = 512

        def embed_images(self, paths):
            raise EmbeddingProviderUnavailableError("model cache is unavailable")

    with _client(tmp_path, monkeypatch) as client:
        indexed = client.post("/albums/index", json={"folder": str(album)}).json()
        monkeypatch.setattr(
            app_module, "embedding_provider", lambda: UnavailableProvider()
        )
        response = client.post(f"/albums/{indexed['album_id']}/embed")

    assert response.status_code == 503
    assert response.json()["detail"] == "model cache is unavailable"
