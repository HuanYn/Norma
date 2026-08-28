from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ai.people import provider as provider_module
from ai.people.provider import (
    FaceProviderUnavailableError,
    OpenCvYuNetSFaceProvider,
    _ModelSpec,
    _ensure_model,
    canonical_face_provider_name,
    create_face_provider,
)


class _BytesResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def __enter__(self) -> _BytesResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_model_cache_replaces_corruption_atomically_and_reuses_valid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified model bytes"
    spec = _ModelSpec(
        filename="model.onnx",
        url="https://models.invalid/model.onnx",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    target = model_dir / spec.filename
    target.write_bytes(b"corrupt")
    requests: list[object] = []

    def fake_urlopen(request: object, *, timeout: int) -> _BytesResponse:
        requests.append((request, timeout))
        return _BytesResponse(payload)

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)

    assert _ensure_model(model_dir, spec) == target
    assert target.read_bytes() == payload
    assert _ensure_model(model_dir, spec) == target
    assert len(requests) == 1
    assert not list(model_dir.glob("*.tmp"))


def test_model_cache_rejects_bad_download_without_overwriting_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = b"expected"
    spec = _ModelSpec(
        filename="model.onnx",
        url="https://models.invalid/model.onnx",
        sha256=hashlib.sha256(expected).hexdigest(),
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    target = model_dir / spec.filename
    target.write_bytes(b"previous corrupt file")
    monkeypatch.setattr(
        provider_module,
        "urlopen",
        lambda *_args, **_kwargs: _BytesResponse(b"tampered"),
    )

    with pytest.raises(FaceProviderUnavailableError, match="SHA-256"):
        _ensure_model(model_dir, spec)

    assert target.read_bytes() == b"previous corrupt file"
    assert not list(model_dir.glob("*.tmp"))


class _FakeDetector:
    def __init__(self) -> None:
        self.input_size: tuple[int, int] | None = None

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_size = size

    def detect(self, _image: np.ndarray) -> tuple[int, np.ndarray]:
        detection = np.asarray(
            [
                160,
                80,
                320,
                400,
                240,
                200,
                400,
                200,
                320,
                280,
                260,
                380,
                380,
                380,
                0.99,
            ],
            dtype=np.float32,
        )
        return 1, detection.reshape(1, -1)


class _FakeRecognizer:
    def __init__(self) -> None:
        self.image_shape: tuple[int, ...] | None = None
        self.alignment_row: np.ndarray | None = None

    def alignCrop(self, image: np.ndarray, row: np.ndarray) -> np.ndarray:
        self.image_shape = image.shape
        self.alignment_row = row.copy()
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def feature(self, _aligned: np.ndarray) -> np.ndarray:
        return np.ones((1, 128), dtype=np.float32)


def test_yunet_sface_provider_restores_landmarks_aligns_and_normalizes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (2000, 1000), "navy").save(image_path)
    provider = OpenCvYuNetSFaceProvider(tmp_path / "models")
    detector = _FakeDetector()
    recognizer = _FakeRecognizer()
    provider._detector = detector
    provider._recognizer = recognizer

    faces = provider.detect(image_path)

    assert detector.input_size == (1600, 800)
    assert len(faces) == 1
    assert faces[0].box == (200, 100, 400, 500)
    assert faces[0].descriptor.shape == (128,)
    assert np.linalg.norm(faces[0].descriptor) == pytest.approx(1.0)
    assert recognizer.image_shape == (1000, 2000, 3)
    assert recognizer.alignment_row is not None
    assert recognizer.alignment_row[:4] == pytest.approx([200, 100, 400, 500])
    assert recognizer.alignment_row[4:14] == pytest.approx(
        [300, 250, 500, 250, 400, 350, 325, 475, 475, 475]
    )


def test_provider_factory_supports_canonical_alias_and_versioned_cache(
    tmp_path: Path,
) -> None:
    provider = create_face_provider("opencv-yunet-sface", cache_dir=tmp_path)

    assert isinstance(provider, OpenCvYuNetSFaceProvider)
    assert provider.model_cache_dir == (tmp_path / "opencv").resolve()
    assert "yunet-2023mar" in provider.name
    assert "sface-2021dec" in provider.name
    assert "8f2383e4" in provider.name
    assert "0ba9fbfa" in provider.name
    assert provider.cluster_policy.version in provider.name
    assert canonical_face_provider_name("opencv-yunet-sface") == provider.name
    assert provider.dimension == 128

    strict = create_face_provider("opencv-yunet-sface-strict", cache_dir=tmp_path)
    assert isinstance(strict, OpenCvYuNetSFaceProvider)
    assert strict.model_cache_dir == (tmp_path / "opencv").resolve()
    assert strict.cluster_policy.prototype_attachment is False
    assert strict.name.endswith("constrained-complete-link-v1")
    assert canonical_face_provider_name("opencv-yunet-sface-strict") == strict.name
