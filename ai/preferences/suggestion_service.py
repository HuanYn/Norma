from __future__ import annotations

import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ai.index.embedding import (
    EmbeddingProvider,
    embedding_cache_is_current,
    normalize_embedding,
)
from ai.preferences import acquisition
from ai.preferences.contextual import (
    FEATURE_SCHEMA,
    PROJECTION_ID,
    ContextualPreferencePosterior,
    contextual_features,
)
from ai.preferences.runtime import (
    IncompatiblePreferenceModelError,
    load_preference_runtime,
    posterior_for_acquisition,
    supports_contextual_runtime,
)
from ai.preferences.suggestion_repository import (
    PreferenceSuggestionRecord,
    PreferenceSuggestionRepository,
)
from ai.schemas import (
    CandidateUniverseSummary,
    PreferencePairPhoto,
    PreferencePairSuggestionRequest,
    PreferencePairSuggestionResponse,
    SelectionResponse,
)
from ai.selection.service import (
    DECISION_FEATURE_SNAPSHOT_VERSION,
    _candidate_universe_summary,
    _decision_feature_snapshot_sha256,
    _load_vector,
)
from ai.storage import Database


class PreferenceSuggestionConflictError(ValueError):
    """The persisted selection can no longer be reproduced safely."""


class PreferenceSuggestionNumericalError(RuntimeError):
    """PDRR-MC abstained after its one allowed higher-sample retry."""


class PreferenceSuggestionService:
    def __init__(self, database: Database, provider: EmbeddingProvider) -> None:
        self.database = database
        self.provider = provider
        self.repository = PreferenceSuggestionRepository(database)

    def suggest(
        self,
        selection_id: str,
        request: PreferencePairSuggestionRequest,
    ) -> PreferencePairSuggestionResponse:
        selection = self._load_selection(selection_id)
        if not selection.feasible:
            raise PreferenceSuggestionConflictError(
                "cannot suggest a preference pair for an infeasible selection"
            )
        if not selection.query_text:
            raise PreferenceSuggestionConflictError(
                "preference-pair acquisition requires a semantic selection query"
            )
        if not selection.provider_fingerprint:
            raise PreferenceSuggestionConflictError(
                "legacy selection has no provider fingerprint; create a new selection"
            )
        if selection.provider_fingerprint != self.provider.name:
            raise PreferenceSuggestionConflictError(
                "selection provider drift: display used "
                f"{selection.provider_fingerprint}, current provider is "
                f"{self.provider.name}; create a new selection"
            )
        if not supports_contextual_runtime(self.provider):
            raise PreferenceSuggestionConflictError(
                "preference-pair acquisition requires the 512D OpenCLIP provider"
            )

        try:
            runtime = load_preference_runtime(
                self.database,
                self.provider,
                user_id=selection.user_id,
            )
        except IncompatiblePreferenceModelError as error:
            raise PreferenceSuggestionConflictError(str(error)) from error
        if (
            runtime.feature_schema != FEATURE_SCHEMA
            or runtime.projection_id != PROJECTION_ID
        ):
            raise PreferenceSuggestionConflictError(
                "current preference runtime feature or projection version is incompatible"
            )
        posterior = posterior_for_acquisition(runtime)
        query_vector = normalize_embedding(
            self.provider.embed_text(selection.query_text),
            self.provider.dimension,
            label="preference acquisition query embedding",
        )

        all_rows = self._load_album_rows(selection.album_id)
        candidate_rows, current_universe = self._reproduce_candidate_universe(
            selection,
            all_rows,
        )
        candidates, features_by_id = self._build_candidates(
            candidate_rows,
            query_vector,
        )
        candidate_feature_digest = _decision_feature_snapshot_sha256(
            candidate_rows,
            features_by_id,
        )
        stored_universe = selection.candidate_universe
        if (
            stored_universe is None
            or candidate_feature_digest
            != stored_universe.decision_feature_snapshot_sha256
        ):
            raise PreferenceSuggestionConflictError(
                "selection candidate decision-feature snapshot drift; "
                "create a new selection"
            )
        excluded_pairs = (
            self.repository.list_excluded_pairs(
                selection_id=selection.selection_id,
                user_id=selection.user_id,
                provider_fingerprint=self.provider.name,
                feature_schema=FEATURE_SCHEMA,
                candidate_digest=current_universe.candidate_ids_sha256,
            )
            if request.exclude_previous
            else []
        )

        result, posterior_samples, retry_count = self._run_with_retry(
            posterior,
            candidates,
            selection,
            request,
            excluded_pairs,
        )
        suggestion_id = uuid.uuid4().hex
        score = result.suggested
        by_id = {str(row["id"]): row for row in candidate_rows}
        mode = "exhaustive" if result.exhaustive else "shortlist"
        response = PreferencePairSuggestionResponse(
            suggestion_id=suggestion_id,
            selection_id=selection.selection_id,
            album_id=selection.album_id,
            user_id=selection.user_id,
            query_text=selection.query_text,
            left=_photo_response(selection.album_id, by_id[score.left_photo_id]),
            right=_photo_response(selection.album_id, by_id[score.right_photo_id]),
            model_id_at_display=runtime.model_id,
            algorithm=runtime.algorithm,
            provider_fingerprint=self.provider.name,
            feature_schema=FEATURE_SCHEMA,
            projection_id=PROJECTION_ID,
            acquisition_version=result.version,
            constraint_solver=result.constraint_solver,
            constraint_violation_count=_constraint_violation_count(
                result.current_photo_ids,
                by_id,
                target_count=selection.constraints.target_count,
                max_per_group=selection.constraints.max_per_similarity_group,
            ),
            mode=mode,
            current_photo_ids=list(result.current_photo_ids),
            current_bayes_regret=result.current_bayes_regret,
            probability_left_preferred=score.probability_left_preferred,
            predictive_entropy=score.predictive_entropy,
            membership_variance=score.membership_variance,
            shortlist_score=score.shortlist_score,
            pdrr=score.pdrr,
            raw_pdrr_estimate=score.raw_pdrr_estimate,
            regret_if_left_preferred=score.regret_if_left_preferred,
            regret_if_right_preferred=score.regret_if_right_preferred,
            effective_sample_size_left=score.effective_sample_size_left,
            effective_sample_size_right=score.effective_sample_size_right,
            laplace_fallback_left=score.laplace_fallback_left,
            laplace_fallback_right=score.laplace_fallback_right,
            laplace_fallback_used=(
                score.laplace_fallback_left or score.laplace_fallback_right
            ),
            voi_invariant_ok=score.voi_invariant_ok,
            eligible_pair_count=result.eligible_pair_count,
            evaluated_pair_count=result.evaluated_pair_count,
            candidate_count=len(candidates),
            candidate_digest=current_universe.candidate_ids_sha256,
            candidate_source_digest=current_universe.source_snapshot_sha256,
            candidate_feature_digest=candidate_feature_digest,
            requested_posterior_samples=request.posterior_samples,
            posterior_samples=posterior_samples,
            retry_count=retry_count,
            shortlist_size=request.shortlist_size,
            seed=request.seed,
        )
        persisted = self.repository.insert(
            PreferenceSuggestionRecord(
                id=suggestion_id,
                selection_id=selection.selection_id,
                album_id=selection.album_id,
                user_id=selection.user_id,
                query_text=selection.query_text,
                left_photo_id=score.left_photo_id,
                right_photo_id=score.right_photo_id,
                provider_fingerprint=self.provider.name,
                feature_schema=FEATURE_SCHEMA,
                projection_id=PROJECTION_ID,
                model_id_at_display=runtime.model_id,
                acquisition_version=result.version,
                constraint_solver=result.constraint_solver,
                mode=mode,
                candidate_digest=current_universe.candidate_ids_sha256,
                candidate_source_digest=current_universe.source_snapshot_sha256,
                candidate_ids=tuple(current_universe.candidate_photo_ids),
                left_features=tuple(
                    float(value) for value in features_by_id[score.left_photo_id]
                ),
                right_features=tuple(
                    float(value) for value in features_by_id[score.right_photo_id]
                ),
                request=request.model_dump(mode="json"),
                diagnostics={
                    "retry_count": retry_count,
                    "requested_posterior_samples": request.posterior_samples,
                    "posterior_samples": posterior_samples,
                    "excluded_pair_count": len(excluded_pairs),
                    "candidate_feature_digest": candidate_feature_digest,
                    "constraint_violation_count": response.constraint_violation_count,
                    "voi_invariant_ok": score.voi_invariant_ok,
                    "raw_pdrr_estimate": score.raw_pdrr_estimate,
                    "effective_sample_size_left": score.effective_sample_size_left,
                    "effective_sample_size_right": score.effective_sample_size_right,
                    "laplace_fallback_left": score.laplace_fallback_left,
                    "laplace_fallback_right": score.laplace_fallback_right,
                },
                result=response.model_dump(mode="json"),
            )
        )
        return response.model_copy(update={"created_at": persisted.created_at})

    def _load_selection(self, selection_id: str) -> SelectionResponse:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM selections WHERE id = ?", (selection_id,)
            ).fetchone()
        if row is None or not row["result_json"]:
            raise KeyError(f"selection not found: {selection_id}")
        try:
            selection = SelectionResponse.model_validate_json(row["result_json"])
        except ValueError as error:
            raise PreferenceSuggestionConflictError(
                "selection audit cannot be validated; create a new selection"
            ) from error
        if selection.selection_id != selection_id:
            raise PreferenceSuggestionConflictError(
                "selection audit id does not match the requested selection"
            )
        return selection

    def _load_album_rows(self, album_id: str) -> list[Mapping[str, object]]:
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
                (album_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"album not found or empty: {album_id}")
        return list(rows)

    def _reproduce_candidate_universe(
        self,
        selection: SelectionResponse,
        all_rows: Sequence[Mapping[str, object]],
    ) -> tuple[list[Mapping[str, object]], CandidateUniverseSummary]:
        stored = selection.candidate_universe
        if stored is None or not stored.candidate_ids_sha256:
            raise PreferenceSuggestionConflictError(
                "selection has no candidate-universe audit; create a new selection"
            )
        if (
            stored.decision_feature_snapshot_version
            != DECISION_FEATURE_SNAPSHOT_VERSION
            or not stored.decision_feature_snapshot_sha256
        ):
            raise PreferenceSuggestionConflictError(
                "selection has no compatible decision-feature snapshot; "
                "create a new selection"
            )
        by_id = {str(row["id"]): row for row in all_rows}
        if stored.candidate_photo_ids:
            if len(set(stored.candidate_photo_ids)) != len(stored.candidate_photo_ids):
                raise PreferenceSuggestionConflictError(
                    "selection candidate audit contains duplicate photo ids"
                )
            missing = [
                photo_id
                for photo_id in stored.candidate_photo_ids
                if photo_id not in by_id
            ]
            if missing:
                raise PreferenceSuggestionConflictError(
                    "selection candidate photos no longer exist: " + ", ".join(missing)
                )
            rows = [by_id[photo_id] for photo_id in stored.candidate_photo_ids]
        else:
            rows = _filter_by_selection_constraints(selection, all_rows)
        for row in rows:
            if selection.constraints.exclude_rejects and bool(row["auto_reject"]):
                raise PreferenceSuggestionConflictError(
                    f"candidate {row['id']} now violates the reject constraint"
                )
            if float(row["quality_score"] or 0.0) < selection.constraints.min_quality:
                raise PreferenceSuggestionConflictError(
                    f"candidate {row['id']} now violates the quality constraint"
                )
        current = _candidate_universe_summary(
            rows=list(rows),
            album_photo_count=len(all_rows),
            subset_photo_count=len(rows),
            excluded_reject_count=0,
            excluded_quality_count=0,
        )
        if current.candidate_ids_sha256 != stored.candidate_ids_sha256:
            raise PreferenceSuggestionConflictError(
                "selection candidate digest drift; create a new selection"
            )
        if (
            stored.source_snapshot_sha256
            and current.source_snapshot_sha256 != stored.source_snapshot_sha256
        ):
            raise PreferenceSuggestionConflictError(
                "selection candidate source snapshot drift; create a new selection"
            )
        if len(rows) < 2:
            raise PreferenceSuggestionConflictError(
                "at least two reproducible candidates are required"
            )
        return list(rows), current

    def _build_candidates(
        self,
        rows: Sequence[Mapping[str, object]],
        query_vector: np.ndarray,
    ) -> tuple[list[acquisition.AcquisitionCandidate], dict[str, np.ndarray]]:
        candidates: list[acquisition.AcquisitionCandidate] = []
        features_by_id: dict[str, np.ndarray] = {}
        for row in rows:
            if not embedding_cache_is_current(row, self.provider.name):
                raise PreferenceSuggestionConflictError(
                    "selection candidate has no current semantic cache for provider "
                    f"{self.provider.name}: {row['id']}"
                )
            image_vector = _load_vector(
                str(row["embedding_path"]), self.provider.dimension
            )
            features = contextual_features(
                image_vector,
                query_vector,
                auto_reject=bool(row["auto_reject"]),
                quality_missing=row["quality_score"] is None,
            )
            photo_id = str(row["id"])
            features_by_id[photo_id] = features
            candidates.append(
                acquisition.AcquisitionCandidate(
                    photo_id=photo_id,
                    features=features,
                    group_key=str(row["similarity_group"] or f"photo:{photo_id}"),
                )
            )
        return candidates, features_by_id

    @staticmethod
    def _run_with_retry(
        posterior: ContextualPreferencePosterior,
        candidates: Sequence[acquisition.AcquisitionCandidate],
        selection: SelectionResponse,
        request: PreferencePairSuggestionRequest,
        excluded_pairs: Sequence[tuple[str, str]],
    ) -> tuple[acquisition.AcquisitionResult, int, int]:
        controls = {
            "target_count": selection.constraints.target_count,
            "max_per_group": selection.constraints.max_per_similarity_group,
            "seed": request.seed,
            "shortlist_size": request.shortlist_size,
            "exhaustive": request.exhaustive,
            "excluded_pairs": excluded_pairs,
        }
        try:
            result = acquisition.suggest_pair(
                posterior,
                candidates,
                posterior_samples=request.posterior_samples,
                **controls,
            )
            return result, request.posterior_samples, 0
        except acquisition.AcquisitionNumericalError as first_error:
            retry_samples = min(4096, request.posterior_samples * 2)
            if retry_samples <= request.posterior_samples:
                raise PreferenceSuggestionNumericalError(
                    "PDRR-MC violated its VOI numerical invariant and no larger "
                    f"sample retry is available: {first_error}"
                ) from first_error
            try:
                result = acquisition.suggest_pair(
                    posterior,
                    candidates,
                    posterior_samples=retry_samples,
                    **controls,
                )
                return result, retry_samples, 1
            except acquisition.AcquisitionNumericalError as second_error:
                raise PreferenceSuggestionNumericalError(
                    "PDRR-MC abstained after retrying with "
                    f"B={retry_samples}: {second_error}"
                ) from second_error
        except ValueError as error:
            raise PreferenceSuggestionConflictError(str(error)) from error


def _filter_by_selection_constraints(
    selection: SelectionResponse,
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if not (selection.constraints.exclude_rejects and bool(row["auto_reject"]))
        and float(row["quality_score"] or 0.0) >= selection.constraints.min_quality
    ]


def _photo_response(album_id: str, row: Mapping[str, object]) -> PreferencePairPhoto:
    return PreferencePairPhoto(
        photo_id=str(row["id"]),
        filename=Path(str(row["absolute_path"])).name,
        thumbnail_url=(
            f"/media/thumbnails/{album_id}/{Path(str(row['thumbnail_path'])).name}"
        ),
    )


def _constraint_violation_count(
    selected_ids: Sequence[str],
    by_id: Mapping[str, Mapping[str, object]],
    *,
    target_count: int,
    max_per_group: int,
) -> int:
    violations = 0 if len(selected_ids) == target_count else 1
    counts: dict[str, int] = {}
    for photo_id in selected_ids:
        row = by_id[photo_id]
        group = str(row["similarity_group"] or f"photo:{photo_id}")
        counts[group] = counts.get(group, 0) + 1
    violations += sum(max(0, count - max_per_group) for count in counts.values())
    return violations


__all__ = [
    "PreferenceSuggestionConflictError",
    "PreferenceSuggestionNumericalError",
    "PreferenceSuggestionService",
]
