from __future__ import annotations

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


SEMANTIC_DIMENSIONS = (
    "night",
    "bright",
    "warm",
    "cool",
    "green",
    "colorful",
    "monochrome",
    "high_contrast",
    "soft",
    "architecture",
    "nature",
    "sky",
    "food",
    "landscape",
    "portrait",
    "cinematic",
)

CONCEPT_TERMS: dict[str, tuple[str, ...]] = {
    "night": ("night", "dark", "evening", "夜景", "夜晚", "暗"),
    "bright": ("bright", "daylight", "sunny", "明亮", "白天", "阳光"),
    "warm": ("warm", "orange", "red", "sunset", "暖色", "橙色", "红色", "日落"),
    "cool": ("cool", "blue", "cyan", "冷色", "蓝色", "青色"),
    "green": ("green", "forest", "grass", "绿色", "森林", "草地"),
    "colorful": ("colorful", "vivid", "多彩", "鲜艳"),
    "monochrome": ("monochrome", "black and white", "灰色", "黑白"),
    "high_contrast": ("contrast", "dramatic", "高对比", "戏剧性"),
    "soft": ("soft", "minimal", "柔和", "极简"),
    "architecture": ("architecture", "building", "city", "建筑", "城市"),
    "nature": ("nature", "outdoor", "mountain", "自然", "户外", "山"),
    "sky": ("sky", "cloud", "天空", "云"),
    "food": ("food", "cafe", "meal", "美食", "咖啡", "餐"),
    "landscape": ("landscape", "wide", "风景", "横幅"),
    "portrait": ("portrait", "person", "people", "人像", "人物"),
    "cinematic": ("cinematic", "film", "movie", "电影感", "胶片"),
}


class EmbeddingProvider(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed_image(self, path: Path) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_images(self, paths: Sequence[Path]) -> list[np.ndarray]:
        return [self.embed_image(path) for path in paths]


class EmbeddingProviderUnavailableError(RuntimeError):
    pass


class LightweightSemanticProvider(EmbeddingProvider):
    """Deterministic CPU baseline in a shared, interpretable semantic space."""

    name = "lightweight-semantic-v1"
    dimension = len(SEMANTIC_DIMENSIONS)

    def embed_image(self, path: Path) -> np.ndarray:
        before = path.stat()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            rgb = np.asarray(image, dtype=np.uint8)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source changed during embedding: {path}")

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hue = hsv[..., 0].astype(np.float32) * 2.0
        saturation = hsv[..., 1].astype(np.float32) / 255.0
        value = hsv[..., 2].astype(np.float32) / 255.0
        sat_mean = float(saturation.mean())
        value_mean = float(value.mean())
        contrast = min(1.0, float(gray.std()) / 82.0)

        saturated = saturation > 0.22
        warm_mask = ((hue < 70.0) | (hue >= 330.0)) & saturated
        cool_mask = (hue >= 175.0) & (hue <= 265.0) & saturated
        green_mask = (hue >= 70.0) & (hue < 170.0) & saturated
        top = max(1, rgb.shape[0] // 3)
        sky_mask = cool_mask[:top]

        edges = cv2.Canny(gray, 80, 180)
        edge_density = float(np.mean(edges > 0))
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 35, minLineLength=18, maxLineGap=6
        )
        straight_line_score = min(1.0, 0.0 if lines is None else len(lines) / 45.0)
        aspect = width / max(height, 1)
        warm = float(warm_mask.mean())
        cool = float(cool_mask.mean())
        green = float(green_mask.mean())

        features = np.asarray(
            [
                np.clip((0.48 - value_mean) / 0.48, 0.0, 1.0),
                value_mean,
                warm,
                cool,
                green,
                sat_mean,
                1.0 - sat_mean,
                contrast,
                1.0 - min(1.0, edge_density * 8.0),
                min(1.0, 0.55 * straight_line_score + edge_density * 4.0),
                min(1.0, green * 2.4 + edge_density * 0.6),
                min(1.0, float(sky_mask.mean()) * 2.5),
                min(1.0, warm * sat_mean * 3.2),
                np.clip((aspect - 1.0) / 0.8, 0.0, 1.0),
                np.clip((1.0 / aspect - 1.0) / 0.8, 0.0, 1.0),
                min(1.0, 0.35 * contrast + 0.35 * warm + 0.3 * (1.0 - value_mean)),
            ],
            dtype=np.float32,
        )
        return _normalize(features)

    def embed_text(self, text: str) -> np.ndarray:
        normalized_text = " ".join(text.casefold().split())
        features = np.zeros(self.dimension, dtype=np.float32)
        for index, concept in enumerate(SEMANTIC_DIMENSIONS):
            for term in CONCEPT_TERMS[concept]:
                normalized_term = term.casefold()
                if _contains_term(normalized_text, normalized_term):
                    features[index] += 1.0
        if not np.any(features):
            raise ValueError(
                "lightweight provider does not recognize this query; try concepts "
                "such as 夜景/night, 建筑/architecture, 自然/nature, 暖色/warm, "
                "蓝色/blue, 人像/portrait, or 电影感/cinematic"
            )
        return _normalize(features)


def create_embedding_provider(
    name: str,
    *,
    cache_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
) -> EmbeddingProvider:
    return _create_embedding_provider(
        name.strip().casefold(),
        str(cache_dir.resolve()) if cache_dir else "",
        device.strip().casefold(),
        batch_size,
    )


@lru_cache(maxsize=8)
def _create_embedding_provider(
    normalized: str,
    cache_dir: str,
    device: str,
    batch_size: int,
) -> EmbeddingProvider:
    if batch_size < 1 or batch_size > 256:
        raise ValueError("embedding batch size must be between 1 and 256")
    if normalized in {"lightweight", "lightweight-semantic-v1"}:
        return LightweightSemanticProvider()
    if normalized in {
        "openclip",
        "openclip-multilingual",
        "openclip-xlm-roberta-base-vit-b-32-laion5b-v1",
    }:
        from ai.index.openclip_provider import OpenClipMultilingualProvider

        return OpenClipMultilingualProvider(
            cache_dir=Path(cache_dir) if cache_dir else Path(".norma/models"),
            device=device,
            batch_size=batch_size,
        )
    raise ValueError(
        f"Unknown embedding provider '{normalized}'. "
        "Available: lightweight, openclip-multilingual."
    )


def embedding_provider_capabilities(active: str) -> list[dict[str, object]]:
    active_normalized = active.strip().casefold()
    return [
        {
            "id": "lightweight",
            "name": LightweightSemanticProvider.name,
            "dimension": LightweightSemanticProvider.dimension,
            "available": True,
            "model_backed": False,
            "multilingual": "bounded-dictionary",
            "active": active_normalized in {"lightweight", "lightweight-semantic-v1"},
            "install_extra": None,
        },
        {
            "id": "openclip-multilingual",
            "name": "openclip-xlm-roberta-base-vit-b-32-laion5b-v1",
            "dimension": 512,
            "available": all(
                find_spec(module) is not None
                for module in ("torch", "open_clip", "transformers")
            ),
            "model_backed": True,
            "multilingual": "xlm-roberta",
            "active": active_normalized
            in {
                "openclip",
                "openclip-multilingual",
                "openclip-xlm-roberta-base-vit-b-32-laion5b-v1",
            },
            "install_extra": "multimodal",
        },
    ]


def normalize_embedding(
    vector: np.ndarray, expected_dimension: int, *, label: str
) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (expected_dimension,):
        raise ValueError(
            f"{label} has shape {array.shape}, expected {(expected_dimension,)}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError(f"{label} is a zero embedding")
    return array / norm


def embedding_cache_is_current(row: Mapping[str, object], provider_name: str) -> bool:
    path = row["embedding_path"]
    if not path or row["embedding_provider"] != provider_name:
        return False
    source_size = row["file_size"]
    source_mtime_ns = row["source_mtime_ns"]
    if source_size is None or source_mtime_ns is None:
        return False
    if (
        row["embedding_source_size"] != source_size
        or row["embedding_source_mtime_ns"] != source_mtime_ns
    ):
        return False
    try:
        source_stat = Path(str(row["absolute_path"])).stat()
    except OSError:
        return False
    return (
        source_stat.st_size == int(source_size)
        and source_stat.st_mtime_ns == int(source_mtime_ns)
        and Path(str(path)).is_file()
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= 1e-12 else vector / norm


def _contains_term(text: str, term: str) -> bool:
    if any("\u4e00" <= character <= "\u9fff" for character in term):
        return term in text
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None
