from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from ai import app as app_module
from ai.storage import Database


def _write_jpeg(path: Path) -> bytes:
    content = b"\xff\xd8\xff\xe0derived-preview\xff\xd9"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def test_media_serves_only_generated_jpeg_preview_routes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    album_preview = data_dir / "thumbnails" / "album-1" / "photo-1.jpg"
    face_preview = (
        data_dir
        / "faces"
        / "opencv-yunet-sface"
        / "album-1"
        / "thumbnails"
        / "face-1.jpg"
    )
    album_content = _write_jpeg(album_preview)
    face_content = _write_jpeg(face_preview)
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, data_dir=data_dir),
    )
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))

    with TestClient(app_module.app) as client:
        album_response = client.get("/media/thumbnails/album-1/photo-1.jpg")
        face_response = client.get(
            "/media/faces/opencv-yunet-sface/album-1/thumbnails/face-1.jpg"
        )

    assert album_response.status_code == 200
    assert album_response.content == album_content
    assert album_response.headers["content-type"] == "image/jpeg"
    assert album_response.headers["x-content-type-options"] == "nosniff"
    assert face_response.status_code == 200
    assert face_response.content == face_content
    assert face_response.headers["content-type"] == "image/jpeg"


def test_media_rejects_database_embeddings_descriptors_and_traversal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    embedding = data_dir / "embeddings" / "provider" / "album-1" / "photo.npy"
    descriptor = (
        data_dir
        / "faces"
        / "opencv-yunet-sface"
        / "album-1"
        / "descriptors"
        / "face.npy"
    )
    thumbnail_npy = data_dir / "thumbnails" / "album-1" / "disguised.npy"
    for path in (embedding, descriptor, thumbnail_npy):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private-vector")
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, data_dir=data_dir),
    )
    monkeypatch.setattr(app_module, "database", Database(data_dir / "norma.db"))

    blocked_urls = (
        "/media/norma.db",
        "/media/embeddings/provider/album-1/photo.npy",
        ("/media/faces/opencv-yunet-sface/album-1/descriptors/face.npy"),
        "/media/thumbnails/album-1/disguised.npy",
        "/media/thumbnails/album-1/..%5C..%5Cnorma.db",
        "/media/thumbnails/album-1/%2e%2e/%2e%2e/norma.db",
        (
            "/media/faces/opencv-yunet-sface/album-1/thumbnails/"
            "..%5Cdescriptors%5Cface.npy"
        ),
    )
    with TestClient(app_module.app) as client:
        responses = [client.get(url, follow_redirects=False) for url in blocked_urls]

    assert (data_dir / "norma.db").is_file()
    assert [response.status_code for response in responses] == [404] * len(blocked_urls)
