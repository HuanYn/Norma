from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(slots=True)
class DetectedFace:
    box: tuple[int, int, int, int]
    descriptor: np.ndarray
    crop: Image.Image


class FaceProvider(ABC):
    name: str
    dimension: int

    @abstractmethod
    def detect(self, path: Path) -> list[DetectedFace]:
        raise NotImplementedError


class OpenCvHaarDctProvider(FaceProvider):
    """Conservative CPU fallback; useful for plumbing, not biometric identity."""

    name = "opencv-haar-dct-v1"
    dimension = 79

    def __init__(self) -> None:
        cascade_path = (
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError(f"unable to load OpenCV face cascade: {cascade_path}")

    def detect(self, path: Path) -> list[DetectedFace]:
        before = path.stat()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        scale = min(1.0, 1000.0 / max(width, height))
        preview = image
        if scale < 1.0:
            preview = image.resize(
                (round(width * scale), round(height * scale)),
                Image.Resampling.LANCZOS,
            )
        preview_rgb = np.asarray(preview, dtype=np.uint8)
        gray = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2GRAY)
        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(40, 40),
        )

        faces: list[DetectedFace] = []
        for x, y, box_width, box_height in boxes:
            original = _original_box(x, y, box_width, box_height, scale, width, height)
            left, top, right, bottom = _expanded_box(original, width, height)
            crop = image.crop((left, top, right, bottom))
            faces.append(
                DetectedFace(
                    box=original,
                    descriptor=_descriptor(crop),
                    crop=crop,
                )
            )

        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source changed during face detection: {path}")
        faces.sort(
            key=lambda face: (face.box[1], face.box[0], face.box[2], face.box[3])
        )
        return faces


def create_face_provider(name: str) -> FaceProvider:
    if name.strip().casefold() in {"opencv-haar", "opencv-haar-dct-v1"}:
        return OpenCvHaarDctProvider()
    raise ValueError(f"Unknown face provider '{name}'. Available: opencv-haar")


def _descriptor(crop: Image.Image) -> np.ndarray:
    rgb = np.asarray(crop.resize((96, 96), Image.Resampling.LANCZOS), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray).astype(np.float32) / 255.0
    low_frequency = cv2.dct(gray)[:8, :8].reshape(-1)[1:]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = np.histogram(hsv[..., 0], bins=8, range=(0, 180), density=False)[0]
    saturation = np.histogram(hsv[..., 1], bins=4, range=(0, 256), density=False)[0]
    value = np.histogram(hsv[..., 2], bins=4, range=(0, 256), density=False)[0]
    color = np.concatenate((hue, saturation, value)).astype(np.float32)
    color /= max(float(color.sum()), 1.0)
    descriptor = np.concatenate((low_frequency, color)).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    return descriptor if norm <= 1e-12 else descriptor / norm


def _original_box(
    x: int,
    y: int,
    width: int,
    height: int,
    scale: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    inverse = 1.0 / scale
    left = max(0, round(x * inverse))
    top = max(0, round(y * inverse))
    right = min(image_width, round((x + width) * inverse))
    bottom = min(image_height, round((y + height) * inverse))
    return left, top, right - left, bottom - top


def _expanded_box(
    box: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x, y, width, height = box
    margin_x = round(width * 0.18)
    margin_top = round(height * 0.22)
    margin_bottom = round(height * 0.12)
    return (
        max(0, x - margin_x),
        max(0, y - margin_top),
        min(image_width, x + width + margin_x),
        min(image_height, y + height + margin_bottom),
    )
