from __future__ import annotations

import hashlib
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class FaceClusterPolicy:
    """Versioned constraints used when face descriptors are clustered."""

    version: str
    minimum_similarity: float
    mean_similarity: float
    centroid_similarity: float
    prototype_attachment: bool = False
    multi_cluster_centroid_similarity: float = 1.0
    multi_cluster_mean_similarity: float = 1.0
    multi_cluster_max_similarity: float = 1.0
    singleton_centroid_similarity: float = 1.0
    singleton_mean_similarity: float = 1.0
    singleton_max_similarity: float = 1.0


@dataclass(slots=True)
class DetectedFace:
    box: tuple[int, int, int, int]
    descriptor: np.ndarray
    crop: Image.Image


class FaceProvider(ABC):
    name: str
    dimension: int
    cluster_policy: FaceClusterPolicy

    @abstractmethod
    def detect(self, path: Path) -> list[DetectedFace]:
        raise NotImplementedError


class FaceProviderUnavailableError(RuntimeError):
    """Raised when a face provider cannot load its local model files."""


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    filename: str
    url: str
    sha256: str


YUNET_MODEL = _ModelSpec(
    filename="face_detection_yunet_2023mar.onnx",
    url=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
SFACE_MODEL = _ModelSpec(
    filename="face_recognition_sface_2021dec.onnx",
    url=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)

YUNET_MAX_SIDE = 1600
YUNET_SCORE_THRESHOLD = 0.8
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 120
MODEL_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

YUNET_SFACE_STRICT_CLUSTER_POLICY = FaceClusterPolicy(
    version="constrained-complete-link-v1",
    minimum_similarity=0.45,
    mean_similarity=0.45,
    centroid_similarity=0.45,
)
YUNET_SFACE_CLUSTER_POLICY = FaceClusterPolicy(
    version="constrained-prototype-attach-v2",
    minimum_similarity=0.45,
    mean_similarity=0.45,
    centroid_similarity=0.45,
    prototype_attachment=True,
    multi_cluster_centroid_similarity=0.40,
    multi_cluster_mean_similarity=0.30,
    multi_cluster_max_similarity=0.42,
    singleton_centroid_similarity=0.42,
    singleton_mean_similarity=0.36,
    singleton_max_similarity=0.48,
)
HAAR_DCT_CLUSTER_POLICY = FaceClusterPolicy(
    version="constrained-complete-link-v1",
    minimum_similarity=0.985,
    mean_similarity=0.985,
    centroid_similarity=0.985,
)


class OpenCvYuNetSFaceProvider(FaceProvider):
    """OpenCV Zoo YuNet detection plus five-point-aligned SFace embeddings."""

    name = (
        f"opencv-yunet-2023mar-{YUNET_MODEL.sha256[:8]}-"
        f"sface-2021dec-{SFACE_MODEL.sha256[:8]}-align112-v1-"
        f"{YUNET_SFACE_CLUSTER_POLICY.version}"
    )
    dimension = 128
    cluster_policy = YUNET_SFACE_CLUSTER_POLICY

    def __init__(
        self,
        model_cache_dir: Path,
        cluster_policy: FaceClusterPolicy = YUNET_SFACE_CLUSTER_POLICY,
    ) -> None:
        self.model_cache_dir = model_cache_dir.resolve()
        self.cluster_policy = cluster_policy
        self.name = _yunet_sface_provider_name(cluster_policy)
        self._detector = None
        self._recognizer = None
        self._model_lock = threading.RLock()

    def _ensure_loaded(self) -> None:
        if self._detector is not None and self._recognizer is not None:
            return
        with self._model_lock:
            if self._detector is not None and self._recognizer is not None:
                return
            detector_path = _ensure_model(self.model_cache_dir, YUNET_MODEL)
            recognizer_path = _ensure_model(self.model_cache_dir, SFACE_MODEL)
            try:
                detector = cv2.FaceDetectorYN.create(
                    str(detector_path),
                    "",
                    (320, 320),
                    YUNET_SCORE_THRESHOLD,
                    YUNET_NMS_THRESHOLD,
                    YUNET_TOP_K,
                )
                recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
            except (AttributeError, cv2.error) as error:
                raise FaceProviderUnavailableError(
                    "OpenCV YuNet/SFace support is unavailable; install "
                    "opencv-python-headless>=4.10"
                ) from error
            self._detector = detector
            self._recognizer = recognizer

    def detect(self, path: Path) -> list[DetectedFace]:
        self._ensure_loaded()
        before = path.stat()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        original_width, original_height = image.size
        original_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        preview_bgr, scale_x, scale_y = _detection_preview(original_bgr)

        with self._model_lock:
            detector = self._detector
            recognizer = self._recognizer
            if detector is None or recognizer is None:
                raise FaceProviderUnavailableError(
                    "OpenCV YuNet/SFace models did not initialize"
                )
            detector.setInputSize((preview_bgr.shape[1], preview_bgr.shape[0]))
            _, detections = detector.detect(preview_bgr)

            faces: list[DetectedFace] = []
            if detections is not None:
                for detection in np.asarray(detections, dtype=np.float32):
                    if detection.shape != (15,):
                        continue
                    restored = _restore_detection(detection, scale_x, scale_y)
                    box = _bounded_box(restored[:4], original_width, original_height)
                    if box[2] <= 0 or box[3] <= 0:
                        continue
                    try:
                        aligned = recognizer.alignCrop(original_bgr, restored)
                        feature = recognizer.feature(aligned)
                    except cv2.error:
                        continue
                    descriptor = _normalized_descriptor(feature, self.dimension)
                    if descriptor is None:
                        continue
                    left, top, right, bottom = _expanded_box(
                        box, original_width, original_height
                    )
                    faces.append(
                        DetectedFace(
                            box=box,
                            descriptor=descriptor,
                            crop=image.crop((left, top, right, bottom)),
                        )
                    )

        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source changed during face detection: {path}")
        faces.sort(
            key=lambda face: (face.box[1], face.box[0], face.box[2], face.box[3])
        )
        return faces


class OpenCvHaarDctProvider(FaceProvider):
    """Legacy CPU fallback for plumbing, not biometric identity."""

    name = f"opencv-haar-dct-v1-{HAAR_DCT_CLUSTER_POLICY.version}"
    dimension = 79
    cluster_policy = HAAR_DCT_CLUSTER_POLICY

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


def create_face_provider(name: str, *, cache_dir: Path | None = None) -> FaceProvider:
    canonical = canonical_face_provider_name(name)
    strict_name = _yunet_sface_provider_name(YUNET_SFACE_STRICT_CLUSTER_POLICY)
    if canonical in {OpenCvYuNetSFaceProvider.name, strict_name}:
        model_root = (cache_dir or Path(".norma/models")).resolve()
        policy = (
            YUNET_SFACE_STRICT_CLUSTER_POLICY
            if canonical == strict_name
            else YUNET_SFACE_CLUSTER_POLICY
        )
        return OpenCvYuNetSFaceProvider(model_root / "opencv", policy)
    return OpenCvHaarDctProvider()


def canonical_face_provider_name(name: str) -> str:
    """Resolve a configured alias to the cache/readiness provider fingerprint."""

    normalized = name.strip().casefold()
    if normalized in {
        "yunet-sface",
        "opencv-yunet-sface",
        OpenCvYuNetSFaceProvider.name,
    }:
        return OpenCvYuNetSFaceProvider.name
    strict_name = _yunet_sface_provider_name(YUNET_SFACE_STRICT_CLUSTER_POLICY)
    if normalized in {"opencv-yunet-sface-strict", strict_name}:
        return strict_name
    if normalized in {
        "opencv-haar",
        "opencv-haar-dct-v1",
        OpenCvHaarDctProvider.name,
    }:
        return OpenCvHaarDctProvider.name
    raise ValueError(
        f"Unknown face provider '{name}'. Available: opencv-yunet-sface, "
        "opencv-yunet-sface-strict, opencv-haar"
    )


def _yunet_sface_provider_name(policy: FaceClusterPolicy) -> str:
    return (
        f"opencv-yunet-2023mar-{YUNET_MODEL.sha256[:8]}-"
        f"sface-2021dec-{SFACE_MODEL.sha256[:8]}-align112-v1-{policy.version}"
    )


def _ensure_model(directory: Path, spec: _ModelSpec) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.filename
    if target.is_file() and _sha256(target) == spec.sha256:
        return target

    temporary = directory / f".{spec.filename}.{uuid.uuid4().hex}.tmp"
    digest = hashlib.sha256()
    request = Request(spec.url, headers={"User-Agent": "Norma/1.0"})
    try:
        with urlopen(request, timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with temporary.open("xb") as output:
                while chunk := response.read(MODEL_DOWNLOAD_CHUNK_BYTES):
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != spec.sha256:
            raise FaceProviderUnavailableError(
                f"Downloaded face model failed SHA-256 verification: {spec.filename}"
            )
        os.replace(temporary, target)
        return target
    except FaceProviderUnavailableError:
        raise
    except OSError as error:
        raise FaceProviderUnavailableError(
            f"Unable to download face model {spec.filename}: {error}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(MODEL_DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _detection_preview(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    scale = min(1.0, YUNET_MAX_SIDE / max(width, height))
    if scale >= 1.0:
        return image, 1.0, 1.0
    preview_width = max(1, round(width * scale))
    preview_height = max(1, round(height * scale))
    preview = cv2.resize(
        image,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )
    return preview, preview_width / width, preview_height / height


def _restore_detection(
    detection: np.ndarray, scale_x: float, scale_y: float
) -> np.ndarray:
    restored = detection.astype(np.float32, copy=True)
    restored[[0, 2, 4, 6, 8, 10, 12]] /= scale_x
    restored[[1, 3, 5, 7, 9, 11, 13]] /= scale_y
    return restored


def _bounded_box(
    values: np.ndarray, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x, y, width, height = (float(value) for value in values)
    left = max(0, min(image_width, round(x)))
    top = max(0, min(image_height, round(y)))
    right = max(left, min(image_width, round(x + width)))
    bottom = max(top, min(image_height, round(y + height)))
    return left, top, right - left, bottom - top


def _normalized_descriptor(feature: object, dimension: int) -> np.ndarray | None:
    descriptor = np.asarray(feature, dtype=np.float32).reshape(-1)
    if descriptor.shape != (dimension,) or not np.all(np.isfinite(descriptor)):
        return None
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1e-12:
        return None
    return descriptor / norm


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
