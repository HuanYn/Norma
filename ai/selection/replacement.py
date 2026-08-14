from __future__ import annotations

import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from ai.index.embedding import EmbeddingProvider
from ai.preferences.model import load_preference_model
from ai.schemas import (
    SelectedPhoto,
    SelectionReplacementRequest,
    SelectionReplacementResponse,
    SelectionResponse,
)
from ai.selection.parser import has_semantic_content
from ai.selection.scoring import grounded_reasons, score_photo
from ai.storage import Database


class ReplacementService:
    def __init__(self, database: Database, provider: EmbeddingProvider) -> None:
        self.database = database
        self.provider = provider

    def replace(
        self, selection_id: str, request: SelectionReplacementRequest
    ) -> SelectionReplacementResponse:
        started = time.perf_counter()
        with self.database.connect() as connection:
            stored = connection.execute(
                "SELECT album_id, raw_prompt, result_json FROM selections WHERE id = ?",
                (selection_id,),
            ).fetchone()
        if stored is None or not stored["result_json"]:
            raise KeyError(f"selection not found: {selection_id}")
        original = SelectionResponse.model_validate_json(stored["result_json"])
        selected_by_id = {photo.photo_id: photo for photo in original.selected}
        if request.remove_photo_id not in selected_by_id:
            raise ValueError("remove_photo_id is not part of the selection")

        locked = [
            photo for photo in original.selected if photo.photo_id != request.remove_photo_id
        ]
        excluded_ids = set(selected_by_id)
        group_counts = Counter(
            photo.similarity_group for photo in locked if photo.similarity_group is not None
        )

        try:
            query_vector: np.ndarray | None = self.provider.embed_text(original.prompt)
        except ValueError as error:
            if has_semantic_content(original.prompt):
                raise ValueError(str(error)) from error
            query_vector = None
        preference_model = load_preference_model(self.database)
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, absolute_path, thumbnail_path, quality_score,
                       auto_reject, similarity_group, embedding_path,
                       width, height, blur_score, metadata_json
                FROM photos WHERE album_id = ? ORDER BY id
                """,
                (original.album_id,),
            ).fetchall()

        eligible: list[tuple[object, object]] = []
        for row in rows:
            if row["id"] in excluded_ids:
                continue
            quality = float(row["quality_score"] or 0.0)
            if original.constraints.exclude_rejects and bool(row["auto_reject"]):
                continue
            if quality < original.constraints.min_quality:
                continue
            group = row["similarity_group"]
            if group and group_counts[group] >= original.constraints.max_per_similarity_group:
                continue
            if query_vector is not None and not row["embedding_path"]:
                raise KeyError(
                    "album has no complete semantic cache; call the embed endpoint first"
                )
            score = score_photo(
                row,
                query_vector,
                self.provider.dimension,
                preference_model,
            )
            eligible.append((row, score))

        eligible.sort(key=lambda item: (-item[1].total, item[0]["id"]))
        if not eligible:
            return SelectionReplacementResponse(
                previous_selection_id=selection_id,
                replacement_selection_id=None,
                feasible=False,
                removed_photo_id=request.remove_photo_id,
                replacement=None,
                updated_selection=None,
                duration_ms=round((time.perf_counter() - started) * 1000),
                explanation=[
                    "No unselected photo can replace this item while preserving every original hard constraint."
                ],
            )

        row, score = eligible[0]
        replacement = SelectedPhoto(
            photo_id=row["id"],
            filename=Path(row["absolute_path"]).name,
            thumbnail_url=(
                f"/media/thumbnails/{original.album_id}/{Path(row['thumbnail_path']).name}"
            ),
            total_score=round(score.total, 6),
            semantic_score=round(score.semantic, 6),
            preference_score=round(score.preference, 6),
            quality_score=round(score.quality, 3),
            similarity_group=row["similarity_group"],
            reasons=grounded_reasons(score, query_vector is not None),
        )
        updated_photos = sorted(
            [*locked, replacement], key=lambda photo: (-photo.total_score, photo.photo_id)
        )
        replacement_selection_id = uuid.uuid4().hex
        updated = SelectionResponse(
            selection_id=replacement_selection_id,
            album_id=original.album_id,
            prompt=original.prompt,
            constraints=original.constraints,
            feasible=True,
            candidate_count=len(eligible),
            solver="deterministic-locked-replacement",
            solver_status="optimal",
            duration_ms=round((time.perf_counter() - started) * 1000),
            selected=updated_photos,
            warnings=[],
        )
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO selections(id, album_id, raw_prompt, parse_json, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    replacement_selection_id,
                    original.album_id,
                    original.prompt,
                    original.constraints.model_dump_json(),
                    updated.model_dump_json(),
                ),
            )
        return SelectionReplacementResponse(
            previous_selection_id=selection_id,
            replacement_selection_id=replacement_selection_id,
            feasible=True,
            removed_photo_id=request.remove_photo_id,
            replacement=replacement,
            updated_selection=updated,
            duration_ms=updated.duration_ms,
            explanation=[
                "Kept every other selected photo locked.",
                "Chose the highest-scoring unselected photo that preserves the original hard constraints.",
                *replacement.reasons,
            ],
        )
