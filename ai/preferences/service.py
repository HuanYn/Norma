from __future__ import annotations

import json
import math
import uuid

import numpy as np

from ai.index.embedding import EmbeddingProvider
from ai.preferences.model import (
    FEATURE_NAMES,
    PreferenceModel,
    load_preference_model,
    photo_features,
    save_preference_model,
)
from ai.schemas import PairwiseFeedbackRequest, PreferenceModelResponse
from ai.storage import Database


class PreferenceService:
    def __init__(self, database: Database, provider: EmbeddingProvider) -> None:
        self.database = database
        self.provider = provider

    def record_pairwise(self, request: PairwiseFeedbackRequest) -> PreferenceModelResponse:
        if request.preferred_photo_id == request.rejected_photo_id:
            raise ValueError("preferred and rejected photos must be different")
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, width, height, quality_score, blur_score,
                       embedding_path, metadata_json
                FROM photos
                WHERE album_id = ? AND id IN (?, ?)
                """,
                (
                    request.album_id,
                    request.preferred_photo_id,
                    request.rejected_photo_id,
                ),
            ).fetchall()
            prompt = None
            if request.selection_id:
                selection = connection.execute(
                    "SELECT album_id, raw_prompt FROM selections WHERE id = ?",
                    (request.selection_id,),
                ).fetchone()
                if selection is None or selection["album_id"] != request.album_id:
                    raise KeyError(f"selection not found in album: {request.selection_id}")
                prompt = selection["raw_prompt"]
        by_id = {row["id"]: row for row in rows}
        if set(by_id) != {request.preferred_photo_id, request.rejected_photo_id}:
            raise KeyError("both feedback photos must exist in the requested album")

        query_vector: np.ndarray | None = None
        if prompt:
            try:
                query_vector = self.provider.embed_text(prompt)
            except ValueError:
                query_vector = None
        preferred_features = photo_features(
            by_id[request.preferred_photo_id], query_vector, self.provider.dimension
        )
        rejected_features = photo_features(
            by_id[request.rejected_photo_id], query_vector, self.provider.dimension
        )
        difference = {
            name: preferred_features[name] - rejected_features[name]
            for name in FEATURE_NAMES
        }

        model = load_preference_model(self.database, request.user_id)
        logit = sum(model.weights[name] * difference[name] for name in FEATURE_NAMES)
        probability_before = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
        learning_rate = 0.35 / math.sqrt(model.comparisons + 1.0)
        for name in FEATURE_NAMES:
            gradient = (1.0 - probability_before) * difference[name]
            regularized = model.weights[name] * 0.002
            model.weights[name] = round(
                max(-2.0, min(2.0, model.weights[name] + learning_rate * gradient - regularized)),
                8,
            )
        model.comparisons += 1
        save_preference_model(self.database, model)

        feedback_id = uuid.uuid4().hex
        payload = {
            "user_id": request.user_id,
            "selection_id": request.selection_id,
            "preferred_photo_id": request.preferred_photo_id,
            "rejected_photo_id": request.rejected_photo_id,
            "feature_difference": difference,
            "probability_before": probability_before,
        }
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO feedback(id, album_id, event_type, payload_json) VALUES (?, ?, ?, ?)",
                (
                    feedback_id,
                    request.album_id,
                    "pairwise_preference",
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
        return PreferenceModelResponse(
            feedback_id=feedback_id,
            user_id=model.user_id,
            comparisons=model.comparisons,
            probability_before=round(probability_before, 6),
            feature_difference={name: round(value, 6) for name, value in difference.items()},
            weights=model.weights,
        )
