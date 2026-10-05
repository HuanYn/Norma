"""Opt-in pretrained score fusion. Fixed weights are policy, NOT learned weights."""

from dataclasses import replace
import hashlib
import json

from ai.aesthetics.service import AestheticsService
from ai.selection.scoring import ScoreBreakdown

ALGORITHM = "openclip-musiq-fixed-fusion-v1"


def snapshot(service: AestheticsService | None, album_id: str) -> dict:
    if service is None:
        raise ValueError("学习型美学评分尚未配置，请先运行模型质量评估。")
    cached = service.cached(album_id)
    if (
        cached["stale_photo_ids"]
        or cached["scored_count"] != cached["total"]
        or not cached["total"]
    ):
        raise ValueError("学习型美学评分不完整或原图已变化，请重新运行模型质量评估。")
    items = {item["photo_id"]: item for item in cached["items"]}
    evidence = {"provider": cached["provider"], "items": items}
    digest = hashlib.sha256(
        json.dumps(
            evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "algorithm": ALGORITHM,
        "provider": cached["provider"],
        "snapshot_sha256": digest,
        "items": items,
        "weights": {"semantic": 0.72, "aesthetic": 0.18, "technical": 0.10},
        "quality_only_weights": {"aesthetic": 0.65, "technical": 0.35},
        "calibration": "fixed-demo-policy-not-optimized-on-heldout-data",
        "hard_quality_gate": "legacy technical quality; unchanged",
    }


def fuse(score: ScoreBreakdown, item: dict, semantic: bool) -> ScoreBreakdown:
    # Preserve raw predictions in evidence; clamp ONLY the soft-ranking inputs.
    technical = max(0.0, min(1.0, float(item["technical_quality"]) / 100))
    aesthetic = max(0.0, min(1.0, (float(item["aesthetic_quality"]) - 1) / 9))
    total = (
        0.72 * score.semantic + 0.18 * aesthetic + 0.10 * technical
        if semantic
        else 0.65 * aesthetic + 0.35 * technical
    )
    return replace(score, total=total)


def reasons(item: dict | None) -> list[str]:
    if item is None:
        return []
    return [
        f"MUSIQ technical {item['technical_quality']:.2f}/approximately 100; aesthetic {item['aesthetic_quality']:.2f}/10",
        "pretrained model scores with fixed fusion weights; no new training",
    ]


def verify(service, album_id, original):
    if (
        original
        and snapshot(service, album_id)["snapshot_sha256"]
        != original["snapshot_sha256"]
    ):
        raise ValueError("模型美学评分或原图在处理期间发生变化，请重新选片。")
