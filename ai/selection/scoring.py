from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ai.preferences.model import PreferenceModel, photo_features, preference_probability
from ai.preferences.runtime import PreferenceRuntime


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    total: float
    semantic: float
    quality: float
    preference: float
    preference_comparisons: int


def score_photo(
    row: Mapping[str, object],
    query_vector: np.ndarray | None,
    provider_dimension: int,
    preference_model: PreferenceModel,
) -> ScoreBreakdown:
    features = photo_features(row, query_vector, provider_dimension)
    semantic = features["semantic"]
    quality = float(row["quality_score"] or 0.0)
    quality_normalized = quality / 100.0
    preference = preference_probability(preference_model, features)
    if query_vector is not None:
        total = (
            0.62 * semantic + 0.25 * quality_normalized + 0.13 * preference
            if preference_model.comparisons > 0
            else 0.72 * semantic + 0.28 * quality_normalized
        )
    else:
        total = (
            0.87 * quality_normalized + 0.13 * preference
            if preference_model.comparisons > 0
            else quality_normalized
        )
    return ScoreBreakdown(
        total=total,
        semantic=semantic,
        quality=quality,
        preference=preference,
        preference_comparisons=preference_model.comparisons,
    )


def score_contextual_photo(
    row: Mapping[str, object],
    image_vector: np.ndarray,
    query_vector: np.ndarray,
    runtime: PreferenceRuntime,
    *,
    decision_features: np.ndarray | None = None,
) -> ScoreBreakdown:
    utility = (
        runtime.score_precomputed(decision_features)
        if decision_features is not None
        else runtime.score(
            image_vector,
            query_vector,
            auto_reject=bool(row["auto_reject"]),
            quality_missing=row["quality_score"] is None,
        )
    )
    return ScoreBreakdown(
        total=utility.total,
        semantic=utility.cosine,
        quality=float(row["quality_score"] or 0.0),
        preference=utility.preference_residual,
        preference_comparisons=runtime.comparisons,
    )


def grounded_reasons(
    score: ScoreBreakdown,
    semantic_enabled: bool,
    *,
    contextual_utility: bool = False,
) -> list[str]:
    reasons = [
        f"quality {score.quality:.1f}/100",
        "passes reject and collection constraints",
    ]
    if semantic_enabled:
        reasons.insert(0, f"semantic similarity {score.semantic:.3f}")
    if contextual_utility and score.preference_comparisons > 0:
        reasons.append(
            f"learned contextual utility residual {score.preference:+.3f} "
            f"from {score.preference_comparisons} comparisons"
        )
    elif score.preference_comparisons > 0:
        reasons.append(
            f"personal preference fit {score.preference:.3f} "
            f"from {score.preference_comparisons} comparisons"
        )
    return reasons
