from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ai.storage import Database


FEATURE_NAMES = (
    "semantic",
    "quality",
    "sharpness",
    "brightness",
    "contrast",
    "landscape",
    "portrait",
)


@dataclass(slots=True)
class PreferenceModel:
    user_id: str
    comparisons: int
    weights: dict[str, float]


def load_preference_model(database: Database, user_id: str = "local") -> PreferenceModel:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT parameters_json FROM user_preferences WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        return PreferenceModel(
            user_id=user_id,
            comparisons=0,
            weights={name: 0.0 for name in FEATURE_NAMES},
        )
    payload = json.loads(row["parameters_json"])
    stored_weights = payload.get("weights", {})
    return PreferenceModel(
        user_id=user_id,
        comparisons=int(payload.get("comparisons", 0)),
        weights={name: float(stored_weights.get(name, 0.0)) for name in FEATURE_NAMES},
    )


def save_preference_model(database: Database, model: PreferenceModel) -> None:
    payload = json.dumps(
        {
            "version": 1,
            "comparisons": model.comparisons,
            "weights": model.weights,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO user_preferences(user_id, parameters_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                parameters_json=excluded.parameters_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (model.user_id, payload),
        )


def photo_features(
    row: Mapping[str, object],
    query_vector: np.ndarray | None,
    expected_dimension: int,
) -> dict[str, float]:
    metadata = json.loads(str(row["metadata_json"] or "{}"))
    semantic = 0.0
    if query_vector is not None and row["embedding_path"]:
        vector = np.load(str(row["embedding_path"]), allow_pickle=False).astype(np.float32)
        if vector.shape != (expected_dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"invalid cached embedding at {row['embedding_path']}; re-run the embed endpoint"
            )
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            vector /= norm
        semantic = max(0.0, float(np.dot(query_vector, vector)))

    quality = float(row["quality_score"] or 0.0) / 100.0
    blur_score = max(0.0, float(row["blur_score"] or 0.0))
    sharpness = min(1.0, math.log1p(blur_score) / math.log1p(900.0))
    brightness = float(metadata.get("brightness", 127.5)) / 255.0
    contrast = min(1.0, float(metadata.get("contrast", 0.0)) / 64.0)
    width = max(1, int(row["width"] or 1))
    height = max(1, int(row["height"] or 1))
    return {
        "semantic": semantic,
        "quality": quality,
        "sharpness": sharpness,
        "brightness": min(1.0, max(0.0, brightness)),
        "contrast": contrast,
        "landscape": 1.0 if width > height * 1.08 else 0.0,
        "portrait": 1.0 if height > width * 1.08 else 0.0,
    }


def preference_probability(model: PreferenceModel, features: Mapping[str, float]) -> float:
    if model.comparisons <= 0:
        return 0.5
    score = sum(model.weights[name] * float(features[name]) for name in FEATURE_NAMES)
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score))))
