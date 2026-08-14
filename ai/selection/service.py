from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

from ai.index.embedding import EmbeddingProvider
from ai.schemas import (
    SelectedPhoto,
    SelectionConstraints,
    SelectionRequest,
    SelectionResponse,
)
from ai.selection.optimizer import OptimizationCandidate, optimize_collection
from ai.selection.parser import has_semantic_content, parse_selection_prompt
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
                       auto_reject, similarity_group, embedding_path
                FROM photos WHERE album_id = ? ORDER BY id
                """,
                (request.album_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"album not found or empty: {request.album_id}")

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

        scored: list[dict[str, object]] = []
        for row in rows:
            quality = float(row["quality_score"] or 0.0)
            if intent.exclude_rejects and bool(row["auto_reject"]):
                continue
            if quality < intent.min_quality:
                continue
            semantic = 0.0
            if query_vector is not None:
                if not row["embedding_path"]:
                    raise KeyError(
                        "album has no complete semantic cache; call the embed endpoint first"
                    )
                vector = _load_vector(row["embedding_path"], self.provider.dimension)
                semantic = max(0.0, float(np.dot(query_vector, vector)))
            quality_normalized = quality / 100.0
            total = (
                0.72 * semantic + 0.28 * quality_normalized
                if query_vector is not None
                else quality_normalized
            )
            scored.append(
                {
                    "row": row,
                    "semantic": semantic,
                    "quality": quality,
                    "total": total,
                }
            )

        optimization_candidates = []
        for index, item in enumerate(scored):
            row = item["row"]
            group = row["similarity_group"] or f"photo:{row['id']}"
            optimization_candidates.append(
                OptimizationCandidate(index=index, score=float(item["total"]), group_key=group)
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
                        semantic_score=round(float(item["semantic"]), 6),
                        quality_score=round(float(item["quality"]), 3),
                        similarity_group=row["similarity_group"],
                        reasons=_reasons(item, query_vector is not None),
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


def _load_vector(path: str, expected_dimension: int) -> np.ndarray:
    vector = np.load(path, allow_pickle=False).astype(np.float32)
    if vector.shape != (expected_dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"invalid cached embedding at {path}; re-run the embed endpoint")
    norm = float(np.linalg.norm(vector))
    return vector if norm <= 1e-12 else vector / norm


def _reasons(item: dict[str, object], semantic_enabled: bool) -> list[str]:
    reasons = [f"quality {float(item['quality']):.1f}/100"]
    if semantic_enabled:
        reasons.insert(0, f"semantic similarity {float(item['semantic']):.3f}")
    return reasons
