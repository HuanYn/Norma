from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ai.config import PreferenceMode, validate_preference_mode
from ai.index.embedding import (
    EmbeddingProvider,
    embedding_cache_is_current,
    normalize_embedding,
)
from ai.preferences.model import default_preference_model, load_preference_model
from ai.preferences.memory import MemoryReranker, MemoryRerankResult
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
from ai.selection.parser import (
    SelectionIntent,
    has_semantic_content,
    parse_selection_prompt,
    person_key,
)
from ai.selection.people_constraints import load_people_constraint_evidence
from ai.selection.scoring import (
    grounded_reasons,
    score_contextual_photo,
    score_photo,
)
from ai.storage import Database
from ai.selection import learned_quality
from ai.selection import collection


DECISION_FEATURE_SNAPSHOT_VERSION = "capu-candidate-67d-group-v1"


class SelectionService:
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
        self._category_prototypes = None

    def select(
        self,
        request: SelectionRequest,
        *,
        intent: SelectionIntent | None = None,
        semantic_query: str | None = None,
        intent_provenance: dict[str, object] | None = None,
        collection_diversity: bool = False,
    ) -> SelectionResponse:
        started = time.perf_counter()
        if request.allow_proxy_memory and not request.use_preference_memory:
            raise ValueError("Proxy memory requires explicit case-memory opt-in")
        if request.use_learned_quality and self.preference_mode != "record-only":
            raise ValueError("Learned quality fusion requires record-only mode; legacy adaptive weights are not mixed")
        assessment = learned_quality.snapshot(self.aesthetics_service, request.album_id) if request.use_learned_quality else None
        if request.use_preference_memory and self.preference_mode != "record-only":
            raise ValueError(
                "Case memory cannot be combined with the legacy adaptive training mode"
            )
        intent = intent or parse_selection_prompt(request.prompt)
        person_minimums = dict(intent.person_minimums)
        for label, minimum in request.person_minimums.items():
            key = person_key(label)
            person_minimums[key] = max(person_minimums.get(key, 0), minimum)
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
        people_evidence = (
            load_people_constraint_evidence(
                self.database, request.album_id, person_minimums
            )
            if person_minimums
            else None
        )
        ranking_query = request.prompt if semantic_query is None else semantic_query
        semantic_requested = (
            has_semantic_content(request.prompt)
            if semantic_query is None
            else bool(semantic_query.strip())
        )
        if semantic_requested:
            query_vector: np.ndarray | None = self.provider.embed_text(ranking_query)
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
                    preference_mode=self.preference_mode,
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
        preference_model = (
            load_preference_model(self.database, request.user_id)
            if self.preference_mode == "adaptive"
            else default_preference_model(request.user_id)
        )
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
            if assessment:
                score = learned_quality.fuse(score, assessment["items"][row["id"]], query_vector is not None)
            scored.append(
                {
                    "row": row,
                    "score": score,
                    "total": score.total,
                }
            )

        memory = None
        if request.use_preference_memory:
            if query_vector is None:
                memory = MemoryRerankResult(
                    enabled=True,
                    deltas={str(item["row"]["id"]): 0.0 for item in scored},
                    warnings=[
                        "Case memory needs a semantic query; quality-only selection remains unchanged."
                    ],
                )
                warnings.extend(memory.warnings)
            else:
                if any(
                    not embedding_cache_is_current(
                        item["row"],
                        self.provider.name,
                        strict_source_hash=True,
                    )
                    for item in scored
                ):
                    raise ValueError(
                        "Case memory requires current content-verified embeddings; reindex and embed first"
                    )
                memory = MemoryReranker(
                    self.database, self.provider, enabled=True,
                    **({"allow_proxy": True} if request.allow_proxy_memory else {})
                ).rerank(
                    request.user_id,
                    query_vector,
                    {
                        str(item["row"]["id"]): _load_vector(
                            str(item["row"]["embedding_path"]),
                            self.provider.dimension,
                        )
                        for item in scored
                    },
                )
                warnings.extend(memory.warnings)
                for item in scored:
                    item["total"] = float(item["total"]) + memory.deltas.get(
                        str(item["row"]["id"]), 0.0
                    )

        category_evidence = {}
        pair_conflicts, pair_penalties = [], []
        if intent.category_counts or collection_diversity:
            if not self.provider.model_backed:
                raise ValueError("整组选片需要预训练图文模型。")
            if any(not embedding_cache_is_current(item["row"], self.provider.name, strict_source_hash=True) for item in scored):
                raise ValueError("图片内容缓存已变化，请重新准备语义模型。")
            vectors = [_load_vector(str(item["row"]["embedding_path"]), self.provider.dimension) for item in scored]
            if intent.category_counts:
                if self._category_prototypes is None:
                    self._category_prototypes = collection.prototypes(self.provider)
                category_evidence = {
                    item["row"]["id"]: collection.classify(v, self._category_prototypes)
                    | {"source_sha256": item["row"]["embedding_source_sha256"], "provider": self.provider.name}
                    for item, v in zip(scored, vectors)
                }
            if collection_diversity:
                pair_conflicts, pair_penalties = collection.pair_constraints(vectors)
            intent_provenance = dict(intent_provenance or {})
            intent_provenance["collection_plan"] = {
                "version": collection.VERSION, "category_counts": intent.category_counts,
                "category_definition": "portrait=人物主体; landscape=自然景观主体; other/uncertain不充当上述类别",
                "evidence": category_evidence,
                "pair_conflicts": [[scored[a]["row"]["id"], scored[b]["row"]["id"]] for a, b in pair_conflicts],
                "pair_threshold": collection.PAIR_THRESHOLD,
                "penalty_start": collection.PENALTY_START,
                "penalty_weight": collection.PENALTY_WEIGHT,
                "calibration": "demo heuristics; no independent accuracy claim",
            }
        optimization_candidates = []
        for index, item in enumerate(scored):
            row = item["row"]
            group = row["similarity_group"] or f"photo:{row['id']}"
            optimization_candidates.append(
                OptimizationCandidate(
                    index=index,
                    score=float(item["total"]),
                    group_key=group,
                    category=category_evidence.get(row["id"], {}).get("label", "unknown"),
                    person_labels=(
                        people_evidence.labels_by_photo[row["id"]]
                        if people_evidence
                        else frozenset()
                    ),
                )
            )
        required_ids = set(request.required_photo_ids)
        indices_by_id = {item["row"]["id"]: index for index, item in enumerate(scored)}
        if not required_ids <= indices_by_id.keys():
            raise ValueError("指定保留照片不在当前可用候选中，请调整门槛或重新选择；不会默默丢弃指定照片。")
        if len(required_ids) > intent.target_count:
            raise ValueError("指定保留照片多于选片总数，请调整数量。")
        required_indices = {indices_by_id[photo_id] for photo_id in required_ids}
        optimized = optimize_collection(
            optimization_candidates,
            intent.target_count,
            intent.max_per_similarity_group,
            person_minimums,
            intent.category_counts,
            pair_conflicts,
            pair_penalties,
            required_indices,
        )
        feasible = len(optimized.indices) == intent.target_count
        if feasible:
            # Check the delivered set independently of the objective/solver status.
            chosen = set(optimized.indices)
            if not required_indices <= chosen:
                raise ValueError("结果缺少指定保留照片，未返回选片。")
            if any(sum(optimization_candidates[i].category == label for i in chosen) != count for label, count in intent.category_counts.items()) or any(a in chosen and b in chosen for a, b in pair_conflicts):
                raise ValueError("结果未通过数量、类别或相似照片校验；本次没有返回选片。")
        if intent.category_counts:
            warnings.append("人像/风景为OpenCLIP零样本主体判断，不是人工真值；不确定图片不用于凑类别配额。")
            if not feasible:
                warnings.append("符合类别判断且满足去重条件的照片不足，未放宽配额或用其他类别凑数。")
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
                        learned_quality=assessment["items"][row["id"]] if assessment else None,
                        category_evidence=category_evidence.get(row["id"]),
                        memory_delta=memory.deltas.get(str(row["id"]), 0.0)
                        if memory
                        else 0.0,
                        reasons=grounded_reasons(
                            item["score"],
                            query_vector is not None,
                            contextual_utility=runtime is not None,
                        )
                        + learned_quality.reasons(assessment["items"][row["id"]] if assessment else None)
                        + (
                            [
                                f"case memory {memory.deltas.get(str(row['id']), 0.0):+.3f}; no model training"
                            ]
                            if memory and memory.applied
                            else []
                        ),
                    )
                )
        else:
            warnings.append(
                f"Hard constraints could not be jointly satisfied or verified by the solver "
                f"({optimized.status}); no partial selection returned."
            )

        constraints = SelectionConstraints(
            target_count=intent.target_count,
            min_quality=intent.min_quality,
            exclude_rejects=intent.exclude_rejects,
            max_per_similarity_group=intent.max_per_similarity_group,
            person_minimums=person_minimums,
            category_counts=intent.category_counts,
            required_photo_ids=sorted(required_ids),
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
            query_text=ranking_query if query_vector is not None else None,
            intent_provenance=intent_provenance,
            preference_memory=memory.as_dict() if memory else None,
            learned_quality=assessment,
            provider_fingerprint=self.provider.name,
            preference_model_id=runtime.model_id if runtime else None,
            preference_comparisons=(
                runtime.comparisons
                if runtime is not None
                else preference_model.comparisons
            ),
            algorithm=(learned_quality.ALGORITHM if assessment else algorithm)
            + ("+case-memory-v1" if memory and memory.applied else "")
            + ("+collection-v1" if collection_diversity or intent.category_counts else ""),
            feature_schema=runtime.feature_schema if runtime else None,
            projection_id=runtime.projection_id if runtime else None,
            candidate_universe=universe,
            people_snapshot_sha256=people_evidence.snapshot_sha256
            if people_evidence
            else None,
            subset_photo_ids=request.subset_photo_ids,
        )
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (intent.category_counts or collection_diversity) and any(
                not embedding_cache_is_current(item["row"], self.provider.name, strict_source_hash=True)
                for item in scored
            ):
                raise ValueError("选片期间图片内容发生变化，请重新准备模型并选片。")
            learned_quality.verify(self.aesthetics_service, request.album_id, assessment)
            if (
                people_evidence
                and load_people_constraint_evidence(
                    self.database, request.album_id, person_minimums
                ).snapshot_sha256
                != people_evidence.snapshot_sha256
            ):
                raise ValueError("人物证据在选片过程中发生变化；请重新选片。")
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
