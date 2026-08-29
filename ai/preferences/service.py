from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from ai.index.embedding import (
    EmbeddingProvider,
    embedding_cache_is_current,
    normalize_embedding,
)
from ai.preferences.contextual import (
    DEFAULT_PRIOR_LAMBDA,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    OPENCLIP_DIMENSION,
    PROJECTION_ID,
    PROJECTION_VERSION,
    BinaryPreferenceEvent,
    contextual_features,
    make_event,
    train,
    validate_covariance,
)
from ai.preferences.model import (
    FEATURE_NAMES,
    PreferenceModel,
    load_preference_model,
    photo_features,
    save_preference_model,
)
from ai.preferences.repository import (
    PreferenceEvent,
    PreferenceModelRecord,
    PreferenceRepository,
)
from ai.preferences.runtime import (
    CONTEXTUAL_MODEL_ALGORITHM,
    training_event_digest,
)
from ai.preferences.suggestion_repository import (
    PreferenceSuggestionRecord,
    PreferenceSuggestionRepository,
)
from ai.schemas import (
    PairwiseFeedbackRequest,
    PreferenceModelResponse,
    PreferenceStateResponse,
)
from ai.storage import Database


UPDATE_LOCK = threading.Lock()
CONTEXTUAL_ALGORITHM = CONTEXTUAL_MODEL_ALGORITHM
LOCAL_SELECTION_USER_ID = "local"
logger = logging.getLogger(__name__)


class PreferenceSuggestionAlreadyConsumedError(ValueError):
    """A displayed suggestion already has an immutable feedback event."""


@dataclass(frozen=True, slots=True)
class _ContextualPair:
    query_text: str
    preferred_features: np.ndarray
    rejected_features: np.ndarray
    base_margin: float
    context: Mapping[str, object]
    model_id_at_display: str | None


@dataclass(frozen=True, slots=True)
class _ContextualResult:
    event_id: str
    model: PreferenceModelRecord | None
    probability_before: float
    trained: bool


class PreferenceService:
    def __init__(self, database: Database, provider: EmbeddingProvider) -> None:
        self.database = database
        self.provider = provider
        self.repository = PreferenceRepository(database)
        self.suggestion_repository = PreferenceSuggestionRepository(database)

    def get_state(self, user_id: str) -> PreferenceStateResponse:
        legacy = load_preference_model(self.database, user_id)
        contextual = None
        if self._supports_contextual_preferences():
            contextual = self._load_compatible_active_model(user_id)
        return PreferenceStateResponse(
            user_id=legacy.user_id,
            comparisons=legacy.comparisons,
            weights=legacy.weights,
            algorithm=contextual.algorithm if contextual else None,
            contextual_model_id=contextual.id if contextual else None,
            provider_fingerprint=(
                contextual.provider_fingerprint if contextual else None
            ),
            feature_schema=contextual.feature_schema if contextual else None,
            contextual_comparisons=(
                contextual.training_pair_count if contextual else None
            ),
            contextual_diagnostics=(
                dict(contextual.diagnostics) if contextual else None
            ),
        )

    def record_pairwise(
        self, request: PairwiseFeedbackRequest
    ) -> PreferenceModelResponse:
        if request.preferred_photo_id == request.rejected_photo_id:
            raise ValueError("preferred and rejected photos must be different")
        rows, prompt, display_model_id, display_provider, suggestion = (
            self._load_feedback_context(request)
        )
        by_id = {str(row["id"]): row for row in rows}
        expected_ids = {
            request.preferred_photo_id,
            request.rejected_photo_id,
        }
        if set(by_id) != expected_ids:
            raise KeyError("both feedback photos must exist in the requested album")

        contextual_candidate = bool(
            request.selection_id
            and prompt
            and display_provider == self.provider.name
            and self._supports_contextual_preferences()
        )
        query_vector = self._embed_query(prompt, strict=contextual_candidate)
        if query_vector is not None:
            stale = [
                photo_id
                for photo_id, row in by_id.items()
                if not embedding_cache_is_current(row, self.provider.name)
            ]
            if stale:
                raise KeyError(
                    "feedback photos have no current semantic cache for provider "
                    f"{self.provider.name}; call the embed endpoint first"
                )

        preferred_row = by_id[request.preferred_photo_id]
        rejected_row = by_id[request.rejected_photo_id]
        preferred_legacy = photo_features(
            preferred_row, query_vector, self.provider.dimension
        )
        rejected_legacy = photo_features(
            rejected_row, query_vector, self.provider.dimension
        )
        difference = {
            name: preferred_legacy[name] - rejected_legacy[name]
            for name in FEATURE_NAMES
        }
        contextual_pair = None
        if contextual_candidate and query_vector is not None:
            contextual_pair = (
                self._build_suggestion_contextual_pair(suggestion, request)
                if suggestion is not None
                else self._build_contextual_pair(
                    prompt,
                    query_vector,
                    preferred_row,
                    rejected_row,
                    request,
                    display_model_id,
                )
            )

        feedback_id = uuid.uuid4().hex
        with UPDATE_LOCK:
            # The immutable contextual event is committed before either model
            # training or the mutable compatibility model is touched.
            contextual_result = (
                self._record_contextual_event(feedback_id, request, contextual_pair)
                if contextual_pair is not None
                else None
            )
            legacy, probability_before = self._update_legacy_model(
                request.user_id,
                difference,
                train_choice=request.choice == "preferred",
            )
            legacy_audit_persisted = self._insert_legacy_audit(
                feedback_id,
                request,
                difference,
                probability_before,
                contextual_result,
            )

        contextual_model = contextual_result.model if contextual_result else None
        return PreferenceModelResponse(
            feedback_id=feedback_id,
            user_id=legacy.user_id,
            comparisons=legacy.comparisons,
            probability_before=round(probability_before, 6),
            feature_difference={
                name: round(value, 6) for name, value in difference.items()
            },
            weights=legacy.weights,
            choice=request.choice,
            algorithm=contextual_model.algorithm if contextual_model else None,
            contextual_event_id=(
                contextual_result.event_id if contextual_result else None
            ),
            contextual_model_id=contextual_model.id if contextual_model else None,
            provider_fingerprint=(
                contextual_model.provider_fingerprint
                if contextual_model
                else (self.provider.name if contextual_result else None)
            ),
            feature_schema=(
                contextual_model.feature_schema
                if contextual_model
                else (FEATURE_SCHEMA if contextual_result else None)
            ),
            contextual_comparisons=(
                contextual_model.training_pair_count if contextual_model else 0
            )
            if contextual_result
            else None,
            contextual_probability_before=(
                round(contextual_result.probability_before, 6)
                if contextual_result
                else None
            ),
            contextual_trained=(
                contextual_result.trained if contextual_result else None
            ),
            contextual_diagnostics=(
                dict(contextual_model.diagnostics) if contextual_model else None
            ),
            legacy_audit_persisted=legacy_audit_persisted,
        )

    def _load_feedback_context(
        self, request: PairwiseFeedbackRequest
    ) -> tuple[
        Sequence[Mapping[str, object]],
        str | None,
        str | None,
        str | None,
        PreferenceSuggestionRecord | None,
    ]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, absolute_path, width, height, quality_score, blur_score,
                       auto_reject, file_size, source_mtime_ns, embedding_path,
                       embedding_provider, embedding_source_size,
                       embedding_source_mtime_ns, embedding_source_sha256,
                       metadata_json
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
            display_model_id = None
            display_provider = None
            suggestion = None
            if request.selection_id:
                selection = connection.execute(
                    "SELECT album_id, raw_prompt, result_json FROM selections WHERE id = ?",
                    (request.selection_id,),
                ).fetchone()
                if selection is None or selection["album_id"] != request.album_id:
                    raise KeyError(
                        f"selection not found in album: {request.selection_id}"
                    )
                try:
                    result = json.loads(selection["result_json"] or "{}")
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "selection result audit is invalid JSON"
                    ) from error
                selection_user_id = str(result.get("user_id", LOCAL_SELECTION_USER_ID))
                if request.user_id != selection_user_id:
                    raise ValueError(
                        f"persisted selections belong to the local user "
                        f"'{selection_user_id}'; feedback "
                        f"user_id '{request.user_id}' would pollute another model"
                    )
                raw_model_id = result.get("preference_model_id")
                display_model_id = str(raw_model_id) if raw_model_id else None
                raw_provider = result.get("provider_fingerprint")
                display_provider = str(raw_provider) if raw_provider else None
                if (
                    self._supports_contextual_preferences()
                    and display_provider is not None
                    and display_provider != self.provider.name
                ):
                    raise ValueError(
                        "selection provider drift prevents contextual feedback: "
                        f"display used {display_provider}, current provider is "
                        f"{self.provider.name}; create a new selection first"
                    )
                if "query_text" in result:
                    raw_query = result.get("query_text")
                    prompt = str(raw_query) if raw_query else None
                else:
                    prompt = str(selection["raw_prompt"])
                if (
                    self._supports_contextual_preferences()
                    and prompt is not None
                    and not prompt.strip()
                ):
                    raise ValueError("selection query must not be empty")
            if request.suggestion_id:
                if not request.selection_id:
                    raise ValueError(
                        "suggestion feedback must include its selection_id"
                    )
                suggestion = self.suggestion_repository.get(request.suggestion_id)
                if suggestion.selection_id != request.selection_id:
                    raise ValueError(
                        "suggestion does not belong to the requested selection"
                    )
                if suggestion.album_id != request.album_id:
                    raise ValueError(
                        "suggestion does not belong to the requested album"
                    )
                if suggestion.user_id != request.user_id:
                    raise ValueError(
                        "suggestion belongs to a different preference user"
                    )
                if {
                    suggestion.left_photo_id,
                    suggestion.right_photo_id,
                } != {
                    request.preferred_photo_id,
                    request.rejected_photo_id,
                }:
                    raise ValueError(
                        "feedback photos do not match the displayed suggestion pair"
                    )
                if suggestion.provider_fingerprint != self.provider.name:
                    raise ValueError(
                        "suggestion provider drift prevents feedback attribution: "
                        f"display used {suggestion.provider_fingerprint}, current "
                        f"provider is {self.provider.name}"
                    )
                if (
                    suggestion.feature_schema != FEATURE_SCHEMA
                    or suggestion.projection_id != PROJECTION_ID
                ):
                    raise ValueError(
                        "suggestion feature or projection version is incompatible"
                    )
                prompt = suggestion.query_text
                display_model_id = suggestion.model_id_at_display
                display_provider = suggestion.provider_fingerprint
        return rows, prompt, display_model_id, display_provider, suggestion

    def _embed_query(self, prompt: str | None, *, strict: bool) -> np.ndarray | None:
        if not prompt:
            return None
        try:
            raw = self.provider.embed_text(prompt)
        except ValueError:
            if strict:
                raise
            return None
        return normalize_embedding(
            raw,
            self.provider.dimension,
            label="provider text embedding",
        )

    def _build_contextual_pair(
        self,
        prompt: str | None,
        query_vector: np.ndarray,
        preferred_row: Mapping[str, object],
        rejected_row: Mapping[str, object],
        request: PairwiseFeedbackRequest,
        display_model_id: str | None,
    ) -> _ContextualPair:
        if not prompt:
            raise ValueError(
                "contextual preference feedback requires a selection query"
            )
        preferred_embedding = self._load_current_openclip_embedding(preferred_row)
        rejected_embedding = self._load_current_openclip_embedding(rejected_row)
        preferred_quality_missing = preferred_row["quality_score"] is None
        rejected_quality_missing = rejected_row["quality_score"] is None
        preferred_auto_reject = bool(preferred_row["auto_reject"])
        rejected_auto_reject = bool(rejected_row["auto_reject"])
        preferred_features = contextual_features(
            preferred_embedding,
            query_vector,
            auto_reject=preferred_auto_reject,
            quality_missing=preferred_quality_missing,
        )
        rejected_features = contextual_features(
            rejected_embedding,
            query_vector,
            auto_reject=rejected_auto_reject,
            quality_missing=rejected_quality_missing,
        )
        return _ContextualPair(
            query_text=prompt,
            preferred_features=preferred_features,
            rejected_features=rejected_features,
            base_margin=make_event(preferred_features, rejected_features).base_margin,
            context={
                "source": "selection-pairwise-feedback",
                "selection_id": request.selection_id,
                "provider_dimension": self.provider.dimension,
                "preferred_auto_reject": preferred_auto_reject,
                "rejected_auto_reject": rejected_auto_reject,
                "preferred_quality_missing": preferred_quality_missing,
                "rejected_quality_missing": rejected_quality_missing,
                "contextual_model_applied_to_display": display_model_id is not None,
            },
            model_id_at_display=display_model_id,
        )

    @staticmethod
    def _build_suggestion_contextual_pair(
        suggestion: PreferenceSuggestionRecord | None,
        request: PairwiseFeedbackRequest,
    ) -> _ContextualPair:
        if suggestion is None:  # pragma: no cover - guarded by caller
            raise ValueError("suggestion context is required")
        if request.preferred_photo_id == suggestion.left_photo_id:
            preferred = np.asarray(suggestion.left_features, dtype=np.float64)
            rejected = np.asarray(suggestion.right_features, dtype=np.float64)
        else:
            preferred = np.asarray(suggestion.right_features, dtype=np.float64)
            rejected = np.asarray(suggestion.left_features, dtype=np.float64)
        event = make_event(preferred, rejected)
        return _ContextualPair(
            query_text=suggestion.query_text,
            preferred_features=preferred,
            rejected_features=rejected,
            base_margin=event.base_margin,
            context={
                "source": "pdrr-suggestion-feedback",
                "suggestion_id": suggestion.id,
                "selection_id": suggestion.selection_id,
                "acquisition_version": suggestion.acquisition_version,
                "constraint_solver": suggestion.constraint_solver,
                "displayed_left_photo_id": suggestion.left_photo_id,
                "displayed_right_photo_id": suggestion.right_photo_id,
                "candidate_digest": suggestion.candidate_digest,
                "contextual_model_applied_to_display": (
                    suggestion.model_id_at_display is not None
                ),
            },
            model_id_at_display=suggestion.model_id_at_display,
        )

    def _load_current_openclip_embedding(self, row: Mapping[str, object]) -> np.ndarray:
        if not embedding_cache_is_current(row, self.provider.name):
            raise KeyError(
                "feedback photo has no current semantic cache for provider "
                f"{self.provider.name}; call the embed endpoint first"
            )
        path = str(row["embedding_path"])
        try:
            vector = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"invalid cached embedding at {path}; re-run the embed endpoint"
            ) from error
        if not embedding_cache_is_current(row, self.provider.name):
            raise KeyError(
                "feedback photo source or embedding cache changed while reading; "
                "call the embed endpoint first"
            )
        return normalize_embedding(
            vector,
            OPENCLIP_DIMENSION,
            label=f"cached OpenCLIP embedding at {path}",
        )

    def _record_contextual_event(
        self,
        event_id: str,
        request: PairwiseFeedbackRequest,
        pair: _ContextualPair,
    ) -> _ContextualResult:
        active_before = self._load_compatible_active_model(request.user_id)
        delta = pair.preferred_features - pair.rejected_features
        margin_before = pair.base_margin
        if active_before is not None and active_before.training_pair_count > 0:
            margin_before += float(np.dot(np.asarray(active_before.mean), delta))
        probability_before = _sigmoid(margin_before)

        event = PreferenceEvent(
            id=event_id,
            user_id=request.user_id,
            album_id=request.album_id,
            selection_id=request.selection_id,
            query_text=pair.query_text,
            preferred_photo_id=request.preferred_photo_id,
            rejected_photo_id=request.rejected_photo_id,
            choice=request.choice,
            provider_fingerprint=self.provider.name,
            feature_schema=FEATURE_SCHEMA,
            preferred_features=tuple(float(value) for value in pair.preferred_features),
            rejected_features=tuple(float(value) for value in pair.rejected_features),
            base_margin=pair.base_margin,
            context=pair.context,
            model_id_at_display=pair.model_id_at_display,
            suggestion_id=request.suggestion_id,
        )
        try:
            self.repository.insert_event(event)
        except sqlite3.IntegrityError as error:
            if request.suggestion_id and "suggestion_id" in str(error):
                raise PreferenceSuggestionAlreadyConsumedError(
                    "preference suggestion was already consumed: "
                    f"{request.suggestion_id}"
                ) from error
            raise

        if request.choice != "preferred":
            return _ContextualResult(
                event_id=event_id,
                model=active_before,
                probability_before=probability_before,
                trained=False,
            )

        events = self.repository.list_trainable_events(
            request.user_id, self.provider.name, FEATURE_SCHEMA
        )
        training_events = [
            BinaryPreferenceEvent(
                preferred_features=np.asarray(
                    stored.preferred_features, dtype=np.float64
                ),
                rejected_features=np.asarray(
                    stored.rejected_features, dtype=np.float64
                ),
                base_margin=stored.base_margin,
            )
            for stored in events
        ]
        posterior = train(training_events, prior_lambda=DEFAULT_PRIOR_LAMBDA)
        model = self.repository.activate_model(
            PreferenceModelRecord(
                id=uuid.uuid4().hex,
                user_id=request.user_id,
                algorithm=CONTEXTUAL_ALGORITHM,
                provider_fingerprint=self.provider.name,
                feature_schema=FEATURE_SCHEMA,
                projection_id=PROJECTION_ID,
                mean=tuple(float(value) for value in posterior.mean),
                covariance=tuple(
                    tuple(float(value) for value in row) for row in posterior.covariance
                ),
                training_pair_count=posterior.comparisons,
                training_event_digest=training_event_digest(events),
                hyperparameters={
                    "prior_lambda": posterior.prior_lambda,
                    "feature_dimension": FEATURE_DIMENSION,
                    "projection_version": PROJECTION_VERSION,
                    "optimizer": "damped-newton-armijo-full-batch",
                    "posterior": "laplace",
                },
                diagnostics=asdict(posterior.diagnostics),
            )
        )
        return _ContextualResult(
            event_id=event_id,
            model=model,
            probability_before=probability_before,
            trained=True,
        )

    def _update_legacy_model(
        self,
        user_id: str,
        difference: Mapping[str, float],
        *,
        train_choice: bool,
    ) -> tuple[PreferenceModel, float]:
        model = load_preference_model(self.database, user_id)
        logit = sum(model.weights[name] * difference[name] for name in FEATURE_NAMES)
        probability_before = _sigmoid(logit)
        if not train_choice:
            return model, probability_before

        learning_rate = 0.35 / math.sqrt(model.comparisons + 1.0)
        for name in FEATURE_NAMES:
            gradient = (1.0 - probability_before) * difference[name]
            regularized = model.weights[name] * 0.002
            model.weights[name] = round(
                max(
                    -2.0,
                    min(
                        2.0,
                        model.weights[name] + learning_rate * gradient - regularized,
                    ),
                ),
                8,
            )
        model.comparisons += 1
        save_preference_model(self.database, model)
        return model, probability_before

    def _insert_legacy_audit(
        self,
        feedback_id: str,
        request: PairwiseFeedbackRequest,
        difference: Mapping[str, float],
        probability_before: float,
        contextual: _ContextualResult | None,
    ) -> bool:
        payload = {
            "user_id": request.user_id,
            "selection_id": request.selection_id,
            "suggestion_id": request.suggestion_id,
            "preferred_photo_id": request.preferred_photo_id,
            "rejected_photo_id": request.rejected_photo_id,
            "choice": request.choice,
            "feature_difference": difference,
            "probability_before": probability_before,
            "contextual_event_id": contextual.event_id if contextual else None,
            "contextual_model_id": (
                contextual.model.id if contextual and contextual.model else None
            ),
        }
        try:
            with self.database.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO feedback(id, album_id, event_type, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        feedback_id,
                        request.album_id,
                        "pairwise_preference",
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    ),
                )
        except sqlite3.Error:
            if contextual is None:
                raise
            logger.exception(
                "legacy feedback audit failed after immutable contextual event %s",
                contextual.event_id,
            )
            return False
        return True

    def _supports_contextual_preferences(self) -> bool:
        return (
            self.provider.dimension == OPENCLIP_DIMENSION
            and self.provider.name.casefold().startswith("openclip")
        )

    def _load_compatible_active_model(
        self, user_id: str
    ) -> PreferenceModelRecord | None:
        model = self.repository.load_active_model(
            user_id, self.provider.name, FEATURE_SCHEMA
        )
        if model is None:
            return None
        if (
            model.algorithm != CONTEXTUAL_ALGORITHM
            or model.projection_id != PROJECTION_ID
            or len(model.mean) != FEATURE_DIMENSION
            or len(model.covariance) != FEATURE_DIMENSION
            or model.hyperparameters.get("feature_dimension") != FEATURE_DIMENSION
            or model.hyperparameters.get("projection_version") != PROJECTION_VERSION
            or model.hyperparameters.get("posterior") != "laplace"
        ):
            return None
        prior = model.hyperparameters.get("prior_lambda")
        if (
            isinstance(prior, bool)
            or not isinstance(prior, (int, float))
            or not math.isclose(
                float(prior), DEFAULT_PRIOR_LAMBDA, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            return None
        try:
            validate_covariance(np.asarray(model.covariance, dtype=np.float64))
        except ValueError:
            return None
        events = self.repository.list_trainable_events(
            user_id, self.provider.name, FEATURE_SCHEMA
        )
        if model.training_pair_count != len(
            events
        ) or model.training_event_digest != training_event_digest(events):
            return None
        return model


def _sigmoid(value: float) -> float:
    clipped = max(-20.0, min(20.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))
