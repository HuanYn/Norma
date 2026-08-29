from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ai.index.embedding import (
    EmbeddingProvider,
    embedding_cache_is_current,
    normalize_embedding,
)
from ai.preferences.model import load_preference_model
from ai.preferences.contextual import (
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    PROJECTION_ID,
    contextual_features,
    validate_features,
)
from ai.preferences.runtime import (
    IncompatiblePreferenceModelError,
    PreferenceRuntime,
    cosine_fallback_runtime,
    load_preference_runtime,
    supports_contextual_runtime,
)
from ai.schemas import (
    CandidateUniverseSummary,
    SelectedPhoto,
    SelectionConstraints,
    SelectionRequest,
    SelectionResponse,
)
from ai.selection.optimizer import OptimizationCandidate, optimize_collection
from ai.selection.parser import has_semantic_content, parse_selection_prompt
from ai.selection.scoring import (
    grounded_reasons,
    score_contextual_photo,
    score_photo,
)
from ai.storage import Database


DECISION_FEATURE_SNAPSHOT_VERSION = "capu-candidate-67d-group-v1"


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
                       embedding_source_sha256, width, height, blur_score,
                       phash, dhash, metadata_json
                FROM photos WHERE album_id = ? ORDER BY id
                """,
                (request.album_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"album not found or empty: {request.album_id}")
        album_photo_count = len(rows)
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
        subset_photo_count = len(rows)

        warnings: list[str] = []
        semantic_requested = has_semantic_content(request.prompt)
        if semantic_requested:
            query_vector: np.ndarray | None = self.provider.embed_text(request.prompt)
        else:
            query_vector = None
            warnings.append(
                "No semantic concept was requested; ranking uses the legacy quality only path."
            )
        if query_vector is not None:
            query_vector = normalize_embedding(
                query_vector,
                self.provider.dimension,
                label="provider text embedding",
            )

        runtime: PreferenceRuntime | None = None
        if query_vector is not None and supports_contextual_runtime(self.provider):
            try:
                runtime = load_preference_runtime(
                    self.database,
                    self.provider,
                    user_id=request.user_id,
                )
            except IncompatiblePreferenceModelError as error:
                runtime = cosine_fallback_runtime(
                    self.provider,
                    user_id=request.user_id,
                )
                warnings.append(
                    "The active contextual preference posterior is incompatible; "
                    f"this selection used cosine only. {error}"
                )

        scored: list[dict[str, object]] = []
        decision_features_by_id: dict[str, np.ndarray] | None = (
            {} if runtime is not None and query_vector is not None else None
        )
        preference_model = load_preference_model(self.database, request.user_id)
        excluded_reject_count = 0
        excluded_quality_count = 0
        for row in rows:
            quality = float(row["quality_score"] or 0.0)
            if intent.exclude_rejects and bool(row["auto_reject"]):
                excluded_reject_count += 1
                continue
            if quality < intent.min_quality:
                excluded_quality_count += 1
                continue
            if query_vector is not None:
                if not embedding_cache_is_current(row, self.provider.name):
                    raise KeyError(
                        "album has no complete semantic cache for provider "
                        f"{self.provider.name}; call the embed endpoint first"
                    )
            if runtime is not None and query_vector is not None:
                image_vector = _load_vector(
                    str(row["embedding_path"]), self.provider.dimension
                )
                decision_features = contextual_features(
                    image_vector,
                    query_vector,
                    auto_reject=bool(row["auto_reject"]),
                    quality_missing=row["quality_score"] is None,
                )
                if decision_features_by_id is not None:
                    decision_features_by_id[str(row["id"])] = decision_features
                score = score_contextual_photo(
                    row,
                    image_vector,
                    query_vector,
                    runtime,
                    decision_features=decision_features,
                )
            else:
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
                            item["score"],
                            query_vector is not None,
                            contextual_utility=runtime is not None,
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
        universe = _candidate_universe_summary(
            rows=[item["row"] for item in scored],
            album_photo_count=album_photo_count,
            subset_photo_count=subset_photo_count,
            excluded_reject_count=excluded_reject_count,
            excluded_quality_count=excluded_quality_count,
            decision_features_by_id=decision_features_by_id,
        )
        algorithm = (
            runtime.algorithm
            if runtime is not None
            else (
                "legacy-fixed-weight-selection-v1"
                if query_vector is not None
                else "legacy-quality-only-selection-v1"
            )
        )
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
            user_id=request.user_id,
            query_text=request.prompt if query_vector is not None else None,
            provider_fingerprint=self.provider.name,
            preference_model_id=runtime.model_id if runtime else None,
            preference_comparisons=(
                runtime.comparisons
                if runtime is not None
                else preference_model.comparisons
            ),
            algorithm=algorithm,
            feature_schema=runtime.feature_schema if runtime else None,
            projection_id=runtime.projection_id if runtime else None,
            candidate_universe=universe,
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


def _load_vector(path: str, expected_dimension: int) -> np.ndarray:
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid cached embedding at {path}; re-run the embed endpoint"
        ) from error
    return normalize_embedding(
        raw,
        expected_dimension,
        label=f"cached embedding at {path}",
    )


def _candidate_universe_summary(
    *,
    rows: list[object],
    album_photo_count: int,
    subset_photo_count: int,
    excluded_reject_count: int,
    excluded_quality_count: int,
    excluded_group_count: int = 0,
    decision_features_by_id: Mapping[str, np.ndarray] | None = None,
) -> CandidateUniverseSummary:
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    candidate_payload = [str(row["id"]) for row in ordered]
    source_payload = [
        [
            str(row["id"]),
            int(row["file_size"] or 0),
            int(row["source_mtime_ns"] or 0),
            str(row["embedding_provider"] or ""),
            int(row["embedding_source_size"] or 0),
            int(row["embedding_source_mtime_ns"] or 0),
            float(row["quality_score"] or 0.0),
            bool(row["auto_reject"]),
            str(row["similarity_group"] or ""),
        ]
        for row in ordered
    ]
    feature_digest = ""
    feature_version = None
    if decision_features_by_id is not None:
        feature_digest = _decision_feature_snapshot_sha256(
            ordered,
            decision_features_by_id,
        )
        feature_version = DECISION_FEATURE_SNAPSHOT_VERSION
    return CandidateUniverseSummary(
        album_photo_count=album_photo_count,
        subset_photo_count=subset_photo_count,
        eligible_photo_count=len(ordered),
        excluded_reject_count=excluded_reject_count,
        excluded_quality_count=excluded_quality_count,
        excluded_group_count=excluded_group_count,
        candidate_ids_sha256=_digest(candidate_payload),
        source_snapshot_sha256=_digest(source_payload),
        decision_feature_snapshot_version=feature_version,
        decision_feature_snapshot_sha256=feature_digest,
        candidate_photo_ids=candidate_payload,
    )


def _decision_feature_snapshot_sha256(
    rows: Sequence[Mapping[str, object]],
    features_by_id: Mapping[str, np.ndarray],
) -> str:
    """Digest the exact query-dependent candidate features and constraint groups."""

    ordered = sorted(rows, key=lambda row: str(row["id"]))
    expected_ids = [str(row["id"]) for row in ordered]
    if set(features_by_id) != set(expected_ids):
        raise ValueError(
            "decision feature snapshot must cover every candidate exactly once"
        )
    candidates: list[list[object]] = []
    for row in ordered:
        photo_id = str(row["id"])
        features = validate_features(
            features_by_id[photo_id],
            label=f"decision features for {photo_id}",
        )
        if features.shape != (FEATURE_DIMENSION,):  # pragma: no cover - validator
            raise ValueError("decision feature dimension mismatch")
        candidates.append(
            [
                photo_id,
                str(row["similarity_group"] or f"photo:{photo_id}"),
                [float(value).hex() for value in features],
            ]
        )
    return _digest(
        {
            "version": DECISION_FEATURE_SNAPSHOT_VERSION,
            "feature_schema": FEATURE_SCHEMA,
            "projection_id": PROJECTION_ID,
            "candidates": candidates,
        }
    )


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
