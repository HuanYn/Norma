from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ai.index.embedding import EmbeddingProvider, normalize_embedding
from ai.preferences.contextual import (
    COSINE_FEATURE_INDEX,
    DEFAULT_PRIOR_LAMBDA,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    OPENCLIP_DIMENSION,
    PROJECTION_ID,
    PROJECTION_VERSION,
    ContextualPreferenceDiagnostics,
    ContextualPreferencePosterior,
    contextual_features,
    score_features,
    train,
    validate_covariance,
    validate_features,
)
from ai.preferences.repository import (
    PreferenceEvent,
    PreferenceModelRecord,
    PreferenceRepository,
)
from ai.storage import Database


CONTEXTUAL_MODEL_ALGORITHM = "bayesian-contextual-logistic-laplace-v1"
CONTEXTUAL_UTILITY_ALGORITHM = "openclip-contextual-posterior-utility-v1"
COSINE_FALLBACK_ALGORITHM = "openclip-cosine-zero-feedback-v1"
LEGACY_COSINE_ALGORITHM = "legacy-cosine-v1"


class IncompatiblePreferenceModelError(ValueError):
    """Raised when an active posterior cannot be used without mixing versions."""


@dataclass(frozen=True, slots=True)
class UtilityScore:
    total: float
    cosine: float
    preference_residual: float


@dataclass(frozen=True, slots=True)
class PreferenceRuntime:
    user_id: str
    provider_fingerprint: str
    algorithm: str
    model_id: str | None
    comparisons: int
    feature_schema: str | None
    projection_id: str | None
    posterior: ContextualPreferencePosterior | None
    training_event_digest: str | None = None

    @property
    def contextual(self) -> bool:
        return self.feature_schema == FEATURE_SCHEMA

    def score(
        self,
        image_embedding: np.ndarray,
        query_embedding: np.ndarray,
        *,
        auto_reject: bool = False,
        quality_missing: bool = False,
    ) -> UtilityScore:
        image = normalize_embedding(
            image_embedding,
            OPENCLIP_DIMENSION,
            label="runtime image embedding",
        ).astype(np.float64)
        query = normalize_embedding(
            query_embedding,
            OPENCLIP_DIMENSION,
            label="runtime query embedding",
        ).astype(np.float64)
        features = contextual_features(
            image,
            query,
            auto_reject=auto_reject,
            quality_missing=quality_missing,
        )
        return self.score_precomputed(features)

    def score_precomputed(self, features: np.ndarray) -> UtilityScore:
        """Score one validated 67D feature vector used by the decision audit."""

        vector = validate_features(features, label="runtime contextual features")
        cosine = float(vector[COSINE_FEATURE_INDEX])
        if self.posterior is None:
            # Keep the zero-feedback invariant literal: the exact value returned
            # by the decision runtime is the normalized embedding dot product.
            return UtilityScore(total=cosine, cosine=cosine, preference_residual=0.0)
        total = score_features(self.posterior, vector)
        return UtilityScore(
            total=total,
            cosine=cosine,
            preference_residual=total - cosine,
        )


def supports_contextual_runtime(provider: EmbeddingProvider) -> bool:
    return (
        provider.dimension == OPENCLIP_DIMENSION
        and provider.name.casefold().startswith("openclip")
    )


def load_preference_runtime(
    database: Database,
    provider: EmbeddingProvider,
    *,
    user_id: str = "local",
) -> PreferenceRuntime:
    """Load one immutable posterior snapshot for a complete decision operation.

    The lookup is deliberately keyed by the exact current provider fingerprint
    and feature schema.  Any active record with incompatible projection or
    posterior metadata is rejected instead of being partially consumed.
    """

    if not user_id.strip():
        raise ValueError("user_id must not be empty")
    if not supports_contextual_runtime(provider):
        return PreferenceRuntime(
            user_id=user_id,
            provider_fingerprint=provider.name,
            algorithm=LEGACY_COSINE_ALGORITHM,
            model_id=None,
            comparisons=0,
            feature_schema=None,
            projection_id=None,
            posterior=None,
        )

    repository = PreferenceRepository(database)
    events = repository.list_trainable_events(user_id, provider.name, FEATURE_SCHEMA)
    model = repository.load_active_model(user_id, provider.name, FEATURE_SCHEMA)
    if model is None:
        if events:
            raise IncompatiblePreferenceModelError(
                "contextual preference events exist but no compatible active posterior "
                f"is available for provider {provider.name}"
            )
        return PreferenceRuntime(
            user_id=user_id,
            provider_fingerprint=provider.name,
            algorithm=COSINE_FALLBACK_ALGORITHM,
            model_id=None,
            comparisons=0,
            feature_schema=FEATURE_SCHEMA,
            projection_id=PROJECTION_ID,
            posterior=None,
        )

    posterior = _posterior_from_record(model, events)
    return PreferenceRuntime(
        user_id=user_id,
        provider_fingerprint=provider.name,
        algorithm=CONTEXTUAL_UTILITY_ALGORITHM,
        model_id=model.id,
        comparisons=model.training_pair_count,
        feature_schema=FEATURE_SCHEMA,
        projection_id=PROJECTION_ID,
        posterior=posterior,
        training_event_digest=model.training_event_digest,
    )


def cosine_fallback_runtime(
    provider: EmbeddingProvider,
    *,
    user_id: str,
    algorithm: str = COSINE_FALLBACK_ALGORITHM,
) -> PreferenceRuntime:
    return PreferenceRuntime(
        user_id=user_id,
        provider_fingerprint=provider.name,
        algorithm=algorithm,
        model_id=None,
        comparisons=0,
        feature_schema=(
            FEATURE_SCHEMA if supports_contextual_runtime(provider) else None
        ),
        projection_id=(
            PROJECTION_ID if supports_contextual_runtime(provider) else None
        ),
        posterior=None,
    )


def posterior_for_acquisition(
    runtime: PreferenceRuntime,
) -> ContextualPreferencePosterior:
    if not runtime.contextual:
        raise ValueError("active embedding provider has no contextual preference space")
    return runtime.posterior if runtime.posterior is not None else train([])


def _posterior_from_record(
    model: PreferenceModelRecord,
    events: Sequence[PreferenceEvent],
) -> ContextualPreferencePosterior:
    if model.provider_fingerprint.strip() == "":
        raise IncompatiblePreferenceModelError(
            "posterior provider fingerprint is empty"
        )
    if model.algorithm != CONTEXTUAL_MODEL_ALGORITHM:
        raise IncompatiblePreferenceModelError(
            f"unsupported contextual posterior algorithm: {model.algorithm}"
        )
    if model.feature_schema != FEATURE_SCHEMA:
        raise IncompatiblePreferenceModelError(
            f"posterior feature schema drift: {model.feature_schema}"
        )
    if model.projection_id != PROJECTION_ID:
        raise IncompatiblePreferenceModelError(
            f"posterior projection drift: {model.projection_id}"
        )
    if len(model.mean) != FEATURE_DIMENSION:
        raise IncompatiblePreferenceModelError(
            f"posterior mean dimension is {len(model.mean)}, expected {FEATURE_DIMENSION}"
        )
    covariance = np.asarray(model.covariance, dtype=np.float64)
    try:
        validate_covariance(covariance)
    except ValueError as error:
        raise IncompatiblePreferenceModelError(
            f"posterior covariance is invalid: {error}"
        ) from error
    if model.training_pair_count != len(events):
        raise IncompatiblePreferenceModelError(
            "posterior training_pair_count does not match immutable event history"
        )
    if model.training_event_digest != training_event_digest(events):
        raise IncompatiblePreferenceModelError(
            "posterior training event digest does not match immutable event history"
        )

    hyperparameters = model.hyperparameters
    if hyperparameters.get("feature_dimension") != FEATURE_DIMENSION:
        raise IncompatiblePreferenceModelError(
            "posterior feature dimension metadata drift"
        )
    if hyperparameters.get("projection_version") != PROJECTION_VERSION:
        raise IncompatiblePreferenceModelError("posterior projection version drift")
    if hyperparameters.get("posterior") != "laplace":
        raise IncompatiblePreferenceModelError("posterior family must be laplace")
    prior = hyperparameters.get("prior_lambda")
    if (
        isinstance(prior, bool)
        or not isinstance(prior, (int, float))
        or not math.isclose(
            float(prior), DEFAULT_PRIOR_LAMBDA, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise IncompatiblePreferenceModelError("posterior prior_lambda drift")

    try:
        diagnostics = ContextualPreferenceDiagnostics(
            **{key: value for key, value in model.diagnostics.items()}
        )
    except TypeError as error:
        raise IncompatiblePreferenceModelError(
            "posterior diagnostics are incomplete or incompatible"
        ) from error
    return ContextualPreferencePosterior(
        mean=np.asarray(model.mean, dtype=np.float64),
        covariance=covariance,
        comparisons=model.training_pair_count,
        prior_lambda=float(prior),
        projection_id=PROJECTION_ID,
        projection_version=PROJECTION_VERSION,
        diagnostics=diagnostics,
    )


def training_event_digest(events: Sequence[PreferenceEvent]) -> str:
    payload: list[Mapping[str, object]] = [
        {
            "id": event.id,
            "preferred_features": event.preferred_features,
            "rejected_features": event.rejected_features,
            "base_margin": event.base_margin,
        }
        for event in events
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "CONTEXTUAL_MODEL_ALGORITHM",
    "CONTEXTUAL_UTILITY_ALGORITHM",
    "COSINE_FALLBACK_ALGORITHM",
    "LEGACY_COSINE_ALGORITHM",
    "IncompatiblePreferenceModelError",
    "PreferenceRuntime",
    "UtilityScore",
    "cosine_fallback_runtime",
    "load_preference_runtime",
    "posterior_for_acquisition",
    "supports_contextual_runtime",
    "training_event_digest",
]
