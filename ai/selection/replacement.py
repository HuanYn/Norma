from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import replace as replace_score
from pathlib import Path

import numpy as np

from ai.config import PreferenceMode, validate_preference_mode
from ai.index.embedding import (
    EmbeddingProvider,
    embedding_cache_is_current,
    normalize_embedding,
)
from ai.preferences.model import default_preference_model, load_preference_model
from ai.preferences.contextual import contextual_features
from ai.preferences.memory import MemoryReranker, MemoryRerankResult
from ai.preferences.runtime import (
    IncompatiblePreferenceModelError,
    PreferenceRuntime,
    cosine_fallback_runtime,
    load_preference_runtime,
    supports_contextual_runtime,
)
from ai.schemas import (
    SelectedPhoto,
    SelectionReplacementRequest,
    SelectionReplacementResponse,
    SelectionResponse,
)
from ai.selection.parser import has_semantic_content
from ai.selection.people_constraints import load_people_constraint_evidence
from ai.selection.scoring import (
    ScoreBreakdown,
    grounded_reasons,
    score_contextual_photo,
    score_photo,
)
from ai.selection.service import _candidate_universe_summary, _load_vector
from ai.storage import Database
from ai.selection import learned_quality


class ReplacementService:
    def __init__(
        self,
        database: Database,
        provider: EmbeddingProvider,
        *,
        preference_mode: PreferenceMode = "record-only",
        aesthetics_service=None,
    ) -> None:
        self.database = database
        self.provider = provider
        self.preference_mode = validate_preference_mode(preference_mode)
        self.aesthetics_service = aesthetics_service

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
        if original.constraints.category_counts or original.constraints.required_photo_ids or (original.intent_provenance or {}).get("collection_plan"):
            raise ValueError("整组选片包含类别或画面多样性约束，请重新选整组；旧版单张替换暂不支持这些约束。")
        assessment = None
        if original.learned_quality:
            if self.preference_mode != "record-only":
                raise ValueError("Learned quality fusion requires record-only mode")
            assessment = learned_quality.snapshot(self.aesthetics_service, original.album_id)
            if assessment["snapshot_sha256"] != original.learned_quality["snapshot_sha256"]:
                raise ValueError("模型美学证据已变化；请重新选片后再替换。")
        memory_enabled = bool(
            original.preference_memory
            and original.preference_memory.get("enabled") is True
        )
        if memory_enabled and self.preference_mode != "record-only":
            raise ValueError(
                "Case memory cannot be combined with the legacy adaptive training mode"
            )
        person_minimums = original.constraints.person_minimums
        people_evidence = (
            load_people_constraint_evidence(
                self.database, original.album_id, person_minimums
            )
            if person_minimums
            else None
        )
        if (
            people_evidence
            and people_evidence.snapshot_sha256 != original.people_snapshot_sha256
        ):
            raise ValueError("人物命名或证据已变化；请重新选片后再替换。")
        selected_by_id = {photo.photo_id: photo for photo in original.selected}
        if request.remove_photo_id not in selected_by_id:
            raise ValueError("remove_photo_id is not part of the selection")

        locked = [
            photo
            for photo in original.selected
            if photo.photo_id != request.remove_photo_id
        ]
        excluded_ids = set(selected_by_id)

        warnings: list[str] = []
        semantic_requested = original.query_text is not None or (
            original.intent_provenance is None and has_semantic_content(original.prompt)
        )
        query_text = original.query_text or (
            original.prompt if semantic_requested else None
        )
        if query_text is not None:
            query_vector: np.ndarray | None = self.provider.embed_text(query_text)
        else:
            query_vector = None
        if query_vector is not None:
            query_vector = normalize_embedding(
                query_vector,
                self.provider.dimension,
                label="provider text embedding",
            )
        user_id = original.user_id or "local"
        provider_unknown = original.provider_fingerprint is None
        provider_drift = bool(
            original.provider_fingerprint
            and original.provider_fingerprint != self.provider.name
        )
        if provider_unknown:
            warnings.append(
                "Legacy selection has no provider snapshot; replacement uses the current "
                f"provider {self.provider.name}, falls back to cosine for semantic "
                "ranking, and recomputes every score."
            )
        elif provider_drift:
            warnings.append(
                "Embedding provider drift detected: original selection used "
                f"{original.provider_fingerprint}, current provider is {self.provider.name}. "
                "Replacement falls back to current-provider cosine and recomputes every "
                "locked and candidate score without a preference posterior."
            )

        runtime: PreferenceRuntime | None = None
        provider_fallback = provider_drift or provider_unknown
        cosine_only = provider_fallback and query_vector is not None
        if query_vector is not None and supports_contextual_runtime(self.provider):
            if provider_fallback:
                runtime = cosine_fallback_runtime(
                    self.provider,
                    user_id=user_id,
                    algorithm="provider-drift-cosine-fallback-v1",
                )
            else:
                try:
                    runtime = load_preference_runtime(
                        self.database,
                        self.provider,
                        user_id=user_id,
                        preference_mode=self.preference_mode,
                    )
                except IncompatiblePreferenceModelError as error:
                    runtime = cosine_fallback_runtime(
                        self.provider,
                        user_id=user_id,
                    )
                    warnings.append(
                        "The active contextual preference posterior is incompatible; "
                        f"replacement used cosine only and recomputed all scores. {error}"
                    )
        if (
            not provider_fallback
            and runtime is not None
            and original.preference_model_id != runtime.model_id
        ):
            warnings.append(
                "Preference posterior changed since the original selection; every locked "
                "photo and replacement candidate was recomputed with one current model "
                f"snapshot ({runtime.model_id or 'zero-feedback cosine'})."
            )
        preference_model = (
            load_preference_model(self.database, user_id)
            if self.preference_mode == "adaptive"
            else default_preference_model(user_id)
        )
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
                (original.album_id,),
            ).fetchall()
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

        album_photo_count = len(rows)
        by_id = {str(row["id"]): row for row in rows}
        missing_locked = [
            photo.photo_id for photo in locked if photo.photo_id not in by_id
        ]
        if missing_locked:
            raise KeyError(
                "locked selection photos no longer exist in the album: "
                + ", ".join(missing_locked)
            )
        if len(locked) + 1 != original.constraints.target_count:
            raise ValueError("selection count is inconsistent; create a new selection")
        for photo in locked:
            row = by_id[photo.photo_id]
            if (
                original.constraints.exclude_rejects and bool(row["auto_reject"])
            ) or float(row["quality_score"] or 0.0) < original.constraints.min_quality:
                raise ValueError("locked photo quality changed; create a new selection")
        group_counts = Counter(
            by_id[photo.photo_id]["similarity_group"]
            for photo in locked
            if by_id[photo.photo_id]["similarity_group"] is not None
        )
        if any(
            count > original.constraints.max_per_similarity_group
            for count in group_counts.values()
        ):
            raise ValueError(
                "locked photo similarity groups changed; create a new selection"
            )
        if original.subset_photo_ids is not None:
            allowed = set(original.subset_photo_ids)
            if any(photo.photo_id not in allowed for photo in locked):
                raise ValueError("locked photos are outside the original subset")
            rows = [row for row in rows if row["id"] in allowed]

        decision_features_by_id: dict[str, np.ndarray] | None = (
            {} if runtime is not None and query_vector is not None else None
        )
        decision_vectors_by_id: dict[str, np.ndarray] = {}

        def load_contextual_decision(
            row: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            if query_vector is None or decision_features_by_id is None:
                raise RuntimeError("contextual decision snapshot is unavailable")
            photo_id = str(row["id"])
            if photo_id not in decision_features_by_id:
                image_vector = _load_vector(
                    str(row["embedding_path"]), self.provider.dimension
                )
                decision_vectors_by_id[photo_id] = image_vector
                decision_features_by_id[photo_id] = contextual_features(
                    image_vector,
                    query_vector,
                    auto_reject=bool(row["auto_reject"]),
                    quality_missing=row["quality_score"] is None,
                )
            return (
                decision_vectors_by_id[photo_id],
                decision_features_by_id[photo_id],
            )

        def compute_base_score(row: object) -> ScoreBreakdown:
            if query_vector is not None:
                if not embedding_cache_is_current(
                    row, self.provider.name, strict_source_hash=memory_enabled
                ):
                    drift = " after provider drift" if provider_drift else ""
                    raise KeyError(
                        "album has no complete semantic cache for provider "
                        f"{self.provider.name}{drift}; call the embed endpoint first"
                    )
                if runtime is not None:
                    image_vector, decision_features = load_contextual_decision(row)
                    return score_contextual_photo(
                        row,
                        image_vector,
                        query_vector,
                        runtime,
                        decision_features=decision_features,
                    )
                if cosine_only:
                    image_vector = _load_vector(
                        str(row["embedding_path"]), self.provider.dimension
                    )
                    cosine = float(np.dot(query_vector, image_vector))
                    return ScoreBreakdown(
                        total=cosine,
                        semantic=cosine,
                        quality=float(row["quality_score"] or 0.0),
                        preference=0.0,
                        preference_comparisons=0,
                    )
            return score_photo(
                row,
                query_vector,
                self.provider.dimension,
                preference_model,
            )

        def compute_score(row):
            score = compute_base_score(row)
            return learned_quality.fuse(score, assessment["items"][row["id"]], query_vector is not None) if assessment else score

        eligible: list[tuple[object, ScoreBreakdown]] = []
        decision_universe_rows: list[object] = []
        excluded_reject_count = 0
        excluded_quality_count = 0
        for row in rows:
            quality = float(row["quality_score"] or 0.0)
            if original.constraints.exclude_rejects and bool(row["auto_reject"]):
                excluded_reject_count += 1
                continue
            if quality < original.constraints.min_quality:
                excluded_quality_count += 1
                continue
            decision_universe_rows.append(row)
            if row["id"] in excluded_ids:
                continue
            group = row["similarity_group"]
            if (
                group
                and group_counts[group] >= original.constraints.max_per_similarity_group
            ):
                continue
            if people_evidence and any(
                sum(
                    label in people_evidence.labels_by_photo.get(photo.photo_id, ())
                    for photo in locked
                )
                + int(label in people_evidence.labels_by_photo[row["id"]])
                < minimum
                for label, minimum in person_minimums.items()
            ):
                continue
            score = compute_score(row)
            eligible.append((row, score))

        locked_scored = [
            (by_id[photo.photo_id], compute_score(by_id[photo.photo_id]))
            for photo in locked
        ]

        if decision_features_by_id is not None:
            for candidate_row in decision_universe_rows:
                if not embedding_cache_is_current(
                    candidate_row,
                    self.provider.name,
                    strict_source_hash=memory_enabled,
                ):
                    raise KeyError(
                        "album has no complete semantic cache for provider "
                        f"{self.provider.name}; call the embed endpoint first"
                    )
                load_contextual_decision(candidate_row)

        memory: MemoryRerankResult | None = None
        if memory_enabled:
            # Recompute locked and genuinely feasible replacement candidates in
            # one call. Old deltas/citations must not survive a changed memory
            # snapshot, and memory cannot reintroduce a hard-excluded candidate.
            memory_rows = [row for row, _ in [*locked_scored, *eligible]]
            memory = MemoryRerankResult(
                enabled=True,
                deltas={str(row["id"]): 0.0 for row in memory_rows},
            )
            if query_vector is None:
                memory.warnings.append(
                    "Case memory needs a semantic query; quality-only replacement "
                    "remains unchanged."
                )
            elif provider_fallback:
                memory.warnings.append(
                    "Case memory was not applied because the original embedding "
                    "provider is unknown or changed; replacement memory deltas are zero."
                )
            else:
                vectors: dict[str, np.ndarray] = {}
                for memory_row in memory_rows:
                    photo_id = str(memory_row["id"])
                    if not embedding_cache_is_current(
                        memory_row, self.provider.name, strict_source_hash=True
                    ):
                        raise KeyError(
                            "album has no complete content-verified semantic cache "
                            f"for provider {self.provider.name}; call the embed endpoint first"
                        )
                    # Reuse the exact vectors already used in contextual scores.
                    vectors[photo_id] = (
                        decision_vectors_by_id[photo_id]
                        if photo_id in decision_vectors_by_id
                        else _load_vector(
                            str(memory_row["embedding_path"]), self.provider.dimension
                        )
                    )
                memory = MemoryReranker(
                    self.database, self.provider, enabled=True,
                    **({"allow_proxy": True} if original.preference_memory.get("allow_proxy") is True else {})
                ).rerank(user_id, query_vector, vectors)
            warnings.extend(memory.warnings)

            def with_memory(items):
                return [
                    (
                        item_row,
                        replace_score(
                            item_score,
                            total=item_score.total
                            + memory.deltas.get(str(item_row["id"]), 0.0),
                        ),
                    )
                    for item_row, item_score in items
                ]

            eligible = with_memory(eligible)
            locked_scored = with_memory(locked_scored)

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
                    "No unselected photo can replace this item while preserving every original hard constraint.",
                    *warnings,
                ],
            )

        row, score = eligible[0]
        replacement = _selected_photo(
            row,
            score,
            album_id=original.album_id,
            semantic_enabled=query_vector is not None,
            contextual_utility=runtime is not None,
            memory_delta=memory.deltas.get(str(row["id"]), 0.0) if memory else 0.0,
            memory_applied=bool(memory and memory.applied),
            assessment=assessment["items"][row["id"]] if assessment else None,
        )
        rescored_locked = [
            _selected_photo(
                locked_row,
                locked_score,
                album_id=original.album_id,
                semantic_enabled=query_vector is not None,
                contextual_utility=runtime is not None,
                memory_delta=memory.deltas.get(str(locked_row["id"]), 0.0)
                if memory
                else 0.0,
                memory_applied=bool(memory and memory.applied),
                assessment=assessment["items"][locked_row["id"]] if assessment else None,
            )
            for locked_row, locked_score in locked_scored
        ]
        updated_photos = sorted(
            [*rescored_locked, replacement],
            key=lambda photo: (-photo.total_score, photo.photo_id),
        )
        replacement_selection_id = uuid.uuid4().hex
        updated = SelectionResponse(
            intent_provenance=original.intent_provenance,
            preference_memory=memory.as_dict() if memory else None,
            learned_quality=assessment,
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
            warnings=warnings,
            user_id=user_id,
            query_text=query_text,
            provider_fingerprint=self.provider.name,
            preference_model_id=runtime.model_id if runtime else None,
            preference_comparisons=(
                runtime.comparisons
                if runtime is not None
                else preference_model.comparisons
            ),
            algorithm=(
                runtime.algorithm
                if runtime is not None
                else (
                    "provider-drift-cosine-fallback-v1"
                    if cosine_only
                    else (
                        "legacy-fixed-weight-selection-v1"
                        if query_vector is not None
                        else "legacy-quality-only-selection-v1"
                    )
                )
            ),
            feature_schema=runtime.feature_schema if runtime else None,
            projection_id=runtime.projection_id if runtime else None,
            people_snapshot_sha256=people_evidence.snapshot_sha256
            if people_evidence
            else None,
            subset_photo_ids=original.subset_photo_ids,
            candidate_universe=_candidate_universe_summary(
                rows=decision_universe_rows,
                album_photo_count=album_photo_count,
                subset_photo_count=len(rows),
                excluded_reject_count=excluded_reject_count,
                excluded_quality_count=excluded_quality_count,
                decision_features_by_id=decision_features_by_id,
            ),
        )
        if assessment:
            updated.algorithm = learned_quality.ALGORITHM + ("+case-memory-v1" if memory and memory.applied else "")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            learned_quality.verify(self.aesthetics_service, original.album_id, assessment)
            if (
                people_evidence
                and load_people_constraint_evidence(
                    self.database, original.album_id, person_minimums
                ).snapshot_sha256
                != people_evidence.snapshot_sha256
            ):
                raise ValueError("人物证据在替换过程中发生变化；请重新选片。")
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
                "Recomputed every locked and candidate score with one model snapshot.",
                "Chose the highest-scoring unselected photo that preserves the original hard constraints.",
                *warnings,
                *replacement.reasons,
            ],
        )


def _selected_photo(
    row: object,
    score: ScoreBreakdown,
    *,
    album_id: str,
    semantic_enabled: bool,
    contextual_utility: bool,
    memory_delta: float = 0.0,
    memory_applied: bool = False,
    assessment: dict | None = None,
) -> SelectedPhoto:
    return SelectedPhoto(
        photo_id=row["id"],
        filename=Path(row["absolute_path"]).name,
        thumbnail_url=(
            f"/media/thumbnails/{album_id}/{Path(row['thumbnail_path']).name}"
        ),
        total_score=round(score.total, 6),
        semantic_score=round(score.semantic, 6),
        preference_score=round(score.preference, 6),
        quality_score=round(score.quality, 3),
        similarity_group=row["similarity_group"],
        memory_delta=memory_delta,
        learned_quality=assessment,
        reasons=grounded_reasons(
            score,
            semantic_enabled,
            contextual_utility=contextual_utility,
        )
        + learned_quality.reasons(assessment)
        + (
            [f"case memory {memory_delta:+.3f}; no model training"]
            if memory_applied
            else []
        ),
    )
