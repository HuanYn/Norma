from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class QualityAnalysis:
    quality_score: float
    blur_score: float
    brightness: float
    contrast: float
    overexposed_ratio: float
    underexposed_ratio: float
    entropy: float
    auto_reject: bool
    reject_reason: str | None
    flags: tuple[str, ...]


def analyze_quality(image: Image.Image) -> QualityAnalysis:
    """Compute cheap cached signals without modifying the source image.

    This CPU fallback feeds the same domain fields as the vendored Rust fast
    core. Thresholds are intentionally conservative: the UI suggests folding,
    never deleting, an image.
    """

    rgb = image.convert("RGB")
    rgb.thumbnail((768, 768), Image.Resampling.LANCZOS)
    pixels = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    overexposed = float(np.mean(gray >= 250))
    underexposed = float(np.mean(gray <= 5))
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    flags: list[str] = []
    long_side = max(image.size)
    if blur_score < 18.0:
        flags.append("very_blurry")
    elif blur_score < 45.0:
        flags.append("blurry")
    if underexposed >= 0.72 or brightness < 18.0:
        flags.append("underexposed")
    if overexposed >= 0.72 or brightness > 240.0:
        flags.append("overexposed")
    if contrast < 12.0:
        flags.append("low_contrast")
    if entropy < 2.5:
        flags.append("low_information")
    if long_side < 480:
        flags.append("too_small")

    sharpness_component = min(1.0, math.log1p(blur_score) / math.log1p(900.0))
    exposure_component = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    contrast_component = min(1.0, contrast / 64.0)
    entropy_component = min(1.0, entropy / 7.5)
    resolution_component = min(1.0, long_side / 2200.0)
    quality_score = 100.0 * (
        0.36 * sharpness_component
        + 0.23 * exposure_component
        + 0.17 * contrast_component
        + 0.14 * entropy_component
        + 0.10 * resolution_component
    )

    rejecting_flags = {
        "very_blurry",
        "underexposed",
        "overexposed",
        "low_information",
    }
    auto_reject = any(flag in rejecting_flags for flag in flags)
    reason_map = {
        "very_blurry": "严重模糊或失焦",
        "underexposed": "画面严重欠曝",
        "overexposed": "画面严重过曝",
        "low_information": "画面有效信息过少",
    }
    reject_reason = next((reason_map[f] for f in flags if f in reason_map), None)
    return QualityAnalysis(
        quality_score=round(quality_score, 3),
        blur_score=round(blur_score, 3),
        brightness=round(brightness, 3),
        contrast=round(contrast, 3),
        overexposed_ratio=round(overexposed, 5),
        underexposed_ratio=round(underexposed, 5),
        entropy=round(entropy, 3),
        auto_reject=auto_reject,
        reject_reason=reject_reason,
        flags=tuple(flags),
    )

