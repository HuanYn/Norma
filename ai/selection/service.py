from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

from ai.index.embedding import (
    EmbeddingProvider,
    embedding_cache_is_current,
    normalize_embedding,
)
from ai.preferences.model import load_preference_model
from ai.schemas import (
    SelectedPhoto,
    SelectionConstraints,
    SelectionRequest,
    SelectionResponse,
)
from ai.selection.optimizer import OptimizationCandidate, optimize_collection
from ai.selection.parser import has_semantic_content, parse_selection_prompt
from ai.selection.scoring import grounded_reasons, score_photo
from ai.storage import Database


class SelectionService:
    def __init__(self, database: Database, provider: EmbeddingProvider) -> None:
        self.database = database
        self.provider = provider

    def select(self, request: SelectionRequest) -> SelectionResponse:
        started = time.perf_counter()
        intent = parse_selection_prompt(request.prompt)
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, absolute_path, thumbnail_path, quality_score,
                       auto_reject, similarity_group, embedding_path,
                       embedding_provider, file_size, source_mtime_ns,
                       embedding_source_size, embedding_source_mtime_ns,
                       width, height, blur_score, phash, dhash, metadata_json
                FROM photos WHERE album_id = ? ORDER BY id
                """,
                (request.album_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"album not found or empty: {request.album_id}")
        if any(
            value is None
            for row in rows
            for value in (
                row["quality_score"],
                row["blur_score"],
                row["phash"],
                row["dhash"],
            )
        ):
            raise ValueError(
                "album quality analysis is incomplete; run quality analysis first"
            )

        if request.subset_photo_ids is not None:
            allowed = set(request.subset_photo_ids)
            rows = [row for row in rows if row["id"] in allowed]

        warnings: list[str] = []
        try:
            query_vector: np.ndarray | None = self.provider.embed_text(request.prompt)
        except ValueError as error:
            if has_semantic_content(request.prompt):
                raise ValueError(str(error)) from error
            query_vector = None
            warnings.append(
                "No supported semantic concept was recognized; ranking uses quality only."
            )
        if query_vector is not None:
            query_vector = normalize_embedding(
                query_vector,
                self.provider.dimension,
                label="provider text embedding",
            )

        scored: list[dict[str, object]] = []
        preference_model = load_preference_model(self.database)
        for row in rows:
            quality = float(row["quality_score"] or 0.0)
            if intent.exclude_rejects and bool(row["auto_reject"]):
                continue
            if quality < intent.min_quality:
                continue
            if query_vector is not None:
                if not embedding_cache_is_current(row, self.provider.name):
                    raise KeyError(
                        "album has no complete semantic cache for provider "
                        f"{self.provider.name}; call the embed endpoint first"
                    )
            score = score_photo(
                row,
                query_vector,
                self.provider.dimension,
                preference_model,
            )
            scored.append(
                {
                    "row": row,
                    "score": score,
                    "total": score.total,
                }
            )

        optimization_candidates = []
        for index, item in enumerate(scored):
            row = item["row"]
            group = row["similarity_group"] or f"photo:{row['id']}"
            optimization_candidates.append(
                OptimizationCandidate(
                    index=index, score=float(item["total"]), group_key=group
                )
            )
        optimized = optimize_collection(
            optimization_candidates,
            intent.target_count,
            intent.max_per_similarity_group,
        )
        feasible = len(optimized.indices) == intent.target_count
        selected: list[SelectedPhoto] = []
        if feasible:
            for index in sorted(
                optimized.indices,
                key=lambda selected_index: (
                    -float(scored[selected_index]["total"]),
                    scored[selected_index]["row"]["id"],
                ),
            ):
                item = scored[index]
                row = item["row"]
                selected.append(
                    SelectedPhoto(
                        photo_id=row["id"],
                        filename=Path(row["absolute_path"]).name,
                        thumbnail_url=(
                            f"/media/thumbnails/{request.album_id}/"
                            f"{Path(row['thumbnail_path']).name}"
                        ),
                        total_score=round(float(item["total"]), 6),
                        semantic_score=round(item["score"].semantic, 6),
                        preference_score=round(item["score"].preference, 6),
                        quality_score=round(item["score"].quality, 3),
                        similarity_group=row["similarity_group"],
                        reasons=grounded_reasons(
                            item["score"], query_vector is not None
                        ),
                    )
                )
        else:
            warnings.append(
                f"Hard constraints allow fewer than {intent.target_count} photos; no partial selection returned."
            )

        constraints = SelectionConstraints(
            target_count=intent.target_count,
            min_quality=intent.min_quality,
            exclude_rejects=intent.exclude_rejects,
            max_per_similarity_group=intent.max_per_similarity_group,
        )
        selection_id = uuid.uuid4().hex
        response = SelectionResponse(
            selection_id=selection_id,
            album_id=request.album_id,
            prompt=request.prompt,
            constraints=constraints,
            feasible=feasible,
            candidate_count=len(scored),
            solver=optimized.solver,
            solver_status=optimized.status,
            duration_ms=round((time.perf_counter() - started) * 1000),
            selected=selected,
            warnings=warnings,
        )
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO selections(id, album_id, raw_prompt, parse_json, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    selection_id,
                    request.album_id,
                    request.prompt,
                    constraints.model_dump_json(),
                    response.model_dump_json(),
                ),
            )
        return response

    def get(self, selection_id: str) -> SelectionResponse:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM selections WHERE id = ?",
                (selection_id,),
            ).fetchone()
        if row is None or not row["result_json"]:
            raise KeyError(f"selection not found: {selection_id}")
        return SelectionResponse.model_validate_json(row["result_json"])
