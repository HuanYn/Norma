from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np


OPENCLIP_DIMENSION = 512
PROJECTION_DIMENSION = 32
FEATURE_DIMENSION = 67
COSINE_FEATURE_INDEX = 64
AUTO_REJECT_FEATURE_INDEX = 65
QUALITY_MISSING_FEATURE_INDEX = 66
DEFAULT_PRIOR_LAMBDA = 1.0
PROJECTION_VERSION = "capu-contextual-shp-v1"
PROJECTION_ID = "openclip512-structured-signed-hadamard-r32-rows37x-plus11-signmix-v1"
FEATURE_SCHEMA = "capu-contextual-openclip512-67d-v1"


@dataclass(frozen=True, slots=True)
class BinaryPreferenceEvent:
    preferred_features: np.ndarray
    rejected_features: np.ndarray
    base_margin: float


@dataclass(frozen=True, slots=True)
class ContextualPreferenceDiagnostics:
    initial_objective: float
    final_objective: float
    initial_negative_log_likelihood: float
    final_negative_log_likelihood: float
    gradient_norm: float
    step_norm: float
    iterations: int
    converged: bool
    line_search_steps: int
    hessian_min_eigenvalue: float
    hessian_condition: float


@dataclass(frozen=True, slots=True)
class ContextualPreferencePosterior:
    mean: np.ndarray
    covariance: np.ndarray
    comparisons: int
    prior_lambda: float
    projection_id: str
    projection_version: str
    diagnostics: ContextualPreferenceDiagnostics


def projection_matrix() -> np.ndarray:
    """Return the fixed CAPU signed-Hadamard projection matrix."""

    return _projection_matrix_cached().copy()


def project_embedding(embedding: np.ndarray, *, label: str = "embedding") -> np.ndarray:
    vector = validate_openclip_embedding(embedding, label=label)
    return _projection_matrix_cached() @ vector


def contextual_features(
    image_embedding: np.ndarray,
    query_embedding: np.ndarray,
    *,
    auto_reject: bool | int | float = False,
    quality_missing: bool | int | float = False,
) -> np.ndarray:
    image = validate_openclip_embedding(image_embedding, label="image embedding")
    query = validate_openclip_embedding(query_embedding, label="query embedding")
    projected_image = _projection_matrix_cached() @ image
    projected_query = _projection_matrix_cached() @ query
    cosine = float(np.dot(image, query))
    return np.concatenate(
        (
            projected_image,
            projected_image * projected_query,
            np.asarray(
                [
                    cosine,
                    _binary_flag(auto_reject, label="auto_reject"),
                    _binary_flag(quality_missing, label="quality_missing"),
                ],
                dtype=np.float64,
            ),
        )
    )


def make_event(
    preferred_features: np.ndarray,
    rejected_features: np.ndarray,
) -> BinaryPreferenceEvent:
    preferred = validate_features(preferred_features, label="preferred_features")
    rejected = validate_features(rejected_features, label="rejected_features")
    return BinaryPreferenceEvent(
        preferred_features=preferred,
        rejected_features=rejected,
        base_margin=float(
            preferred[COSINE_FEATURE_INDEX] - rejected[COSINE_FEATURE_INDEX]
        ),
    )


def train(
    events: Sequence[BinaryPreferenceEvent],
    *,
    prior_lambda: float = DEFAULT_PRIOR_LAMBDA,
    max_iterations: int = 50,
    gradient_tolerance: float = 1e-8,
    step_tolerance: float = 1e-10,
    objective_tolerance: float = 1e-12,
) -> ContextualPreferencePosterior:
    prior = _validate_prior_lambda(prior_lambda)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if gradient_tolerance <= 0.0 or not math.isfinite(gradient_tolerance):
        raise ValueError("gradient_tolerance must be a finite positive value")
    if step_tolerance <= 0.0 or not math.isfinite(step_tolerance):
        raise ValueError("step_tolerance must be a finite positive value")
    if objective_tolerance <= 0.0 or not math.isfinite(objective_tolerance):
        raise ValueError("objective_tolerance must be a finite positive value")

    design, base_margins = _event_design(events)
    theta = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    initial_nll = _negative_log_likelihood_from_design(theta, design, base_margins)
    initial_objective = initial_nll

    if design.shape[0] == 0:
        covariance = np.eye(FEATURE_DIMENSION, dtype=np.float64) / prior
        hessian = np.eye(FEATURE_DIMENSION, dtype=np.float64) * prior
        diagnostics = _diagnostics(
            initial_objective=initial_objective,
            final_objective=initial_objective,
            initial_negative_log_likelihood=initial_nll,
            final_negative_log_likelihood=initial_nll,
            gradient=np.zeros(FEATURE_DIMENSION, dtype=np.float64),
            step_norm=0.0,
            iterations=0,
            converged=True,
            line_search_steps=0,
            hessian=hessian,
        )
        return ContextualPreferencePosterior(
            mean=theta,
            covariance=covariance,
            comparisons=0,
            prior_lambda=prior,
            projection_id=PROJECTION_ID,
            projection_version=PROJECTION_VERSION,
            diagnostics=diagnostics,
        )

    current_objective, gradient, hessian = _objective_gradient_hessian(
        theta, design, base_margins, prior
    )
    converged = False
    iterations = 0
    total_line_search_steps = 0
    step_norm = 0.0

    for iteration in range(1, max_iterations + 1):
        gradient_norm = float(np.linalg.norm(gradient, ord=2))
        if gradient_norm <= gradient_tolerance:
            converged = True
            break

        step = -np.linalg.solve(hessian, gradient)
        step_norm = float(np.linalg.norm(step, ord=2))
        if step_norm <= step_tolerance * (1.0 + float(np.linalg.norm(theta, ord=2))):
            converged = True
            break

        directional_derivative = float(np.dot(gradient, step))
        if directional_derivative >= 0.0:
            step = -gradient
            directional_derivative = -float(np.dot(gradient, gradient))

        step_size = 1.0
        accepted = False
        line_search_steps = 0
        for line_search_steps in range(41):
            candidate = theta + step_size * step
            candidate_objective = _objective_from_design(
                candidate, design, base_margins, prior
            )
            sufficient_decrease = (
                current_objective + 1e-4 * step_size * directional_derivative
            )
            if candidate_objective <= sufficient_decrease:
                accepted = True
                break
            step_size *= 0.5

        total_line_search_steps += line_search_steps
        if not accepted:
            break

        previous_objective = current_objective
        theta = candidate
        current_objective, gradient, hessian = _objective_gradient_hessian(
            theta, design, base_margins, prior
        )
        iterations = iteration
        objective_delta = abs(previous_objective - current_objective)
        if objective_delta <= objective_tolerance * (1.0 + abs(previous_objective)):
            converged = True
            break

    final_nll = _negative_log_likelihood_from_design(theta, design, base_margins)
    covariance = _laplace_covariance(hessian)
    diagnostics = _diagnostics(
        initial_objective=initial_objective,
        final_objective=current_objective,
        initial_negative_log_likelihood=initial_nll,
        final_negative_log_likelihood=final_nll,
        gradient=gradient,
        step_norm=step_norm,
        iterations=iterations,
        converged=converged,
        line_search_steps=total_line_search_steps,
        hessian=hessian,
    )
    posterior = ContextualPreferencePosterior(
        mean=theta.copy(),
        covariance=covariance,
        comparisons=design.shape[0],
        prior_lambda=prior,
        projection_id=PROJECTION_ID,
        projection_version=PROJECTION_VERSION,
        diagnostics=diagnostics,
    )
    validate_covariance(posterior.covariance)
    return posterior


def score(
    posterior: ContextualPreferencePosterior,
    image_embedding: np.ndarray,
    query_embedding: np.ndarray,
    *,
    auto_reject: bool | int | float = False,
    quality_missing: bool | int | float = False,
) -> float:
    features = contextual_features(
        image_embedding,
        query_embedding,
        auto_reject=auto_reject,
        quality_missing=quality_missing,
    )
    return score_features(posterior, features)


def score_features(
    posterior: ContextualPreferencePosterior,
    features: np.ndarray,
) -> float:
    _validate_posterior(posterior)
    vector = validate_features(features, label="features")
    cosine = float(vector[COSINE_FEATURE_INDEX])
    if posterior.comparisons == 0:
        return cosine
    return cosine + float(np.dot(posterior.mean, vector))


def margin(
    posterior: ContextualPreferencePosterior,
    preferred_features: np.ndarray,
    rejected_features: np.ndarray,
) -> float:
    event = make_event(preferred_features, rejected_features)
    return event_margin(posterior, event)


def event_margin(
    posterior: ContextualPreferencePosterior,
    event: BinaryPreferenceEvent,
) -> float:
    _validate_posterior(posterior)
    preferred, rejected, base_margin = _validate_event(event)
    if posterior.comparisons == 0:
        return base_margin
    return base_margin + float(np.dot(posterior.mean, preferred - rejected))


def sample(
    posterior: ContextualPreferencePosterior,
    *,
    n_samples: int = 1,
    seed: int = 0,
) -> np.ndarray:
    _validate_posterior(posterior)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    eigenvalues, eigenvectors = np.linalg.eigh(posterior.covariance)
    if float(eigenvalues[0]) < -1e-10:
        raise ValueError("posterior covariance must be positive semidefinite")
    scales = np.sqrt(np.clip(eigenvalues, 0.0, None))
    standard = rng.standard_normal((n_samples, FEATURE_DIMENSION))
    return posterior.mean + standard @ (eigenvectors * scales).T


def negative_log_likelihood(
    events: Sequence[BinaryPreferenceEvent],
    theta: np.ndarray,
) -> float:
    design, base_margins = _event_design(events)
    weights = validate_features(theta, label="theta")
    return _negative_log_likelihood_from_design(weights, design, base_margins)


def validate_openclip_embedding(embedding: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.shape != (OPENCLIP_DIMENSION,):
        raise ValueError(
            f"{label} has shape {vector.shape}, expected {(OPENCLIP_DIMENSION,)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(vector, ord=2))
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(f"{label} must be L2-normalized; got norm {norm:.8f}")
    return vector.copy()


def validate_features(features: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (FEATURE_DIMENSION,):
        raise ValueError(
            f"{label} has shape {vector.shape}, expected {(FEATURE_DIMENSION,)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    return vector.copy()


def validate_covariance(covariance: np.ndarray) -> None:
    matrix = np.asarray(covariance, dtype=np.float64)
    expected = (FEATURE_DIMENSION, FEATURE_DIMENSION)
    if matrix.shape != expected:
        raise ValueError(f"covariance has shape {matrix.shape}, expected {expected}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    if float(np.linalg.eigvalsh(matrix)[0]) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")


@lru_cache(maxsize=1)
def _projection_matrix_cached() -> np.ndarray:
    rows = (np.arange(PROJECTION_DIMENSION, dtype=np.int64) * 37 + 11) % (
        OPENCLIP_DIMENSION
    )
    columns = np.arange(OPENCLIP_DIMENSION, dtype=np.int64)
    hadamard_rows = np.empty(
        (PROJECTION_DIMENSION, OPENCLIP_DIMENSION), dtype=np.float64
    )
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            parity = int(row & column).bit_count() & 1
            hadamard_rows[row_index, column_index] = -1.0 if parity else 1.0
    signs = np.asarray(
        [_projection_sign(index) for index in range(OPENCLIP_DIMENSION)],
        dtype=np.float64,
    )
    return (hadamard_rows * signs) / math.sqrt(float(PROJECTION_DIMENSION))


def _projection_sign(index: int) -> float:
    value = (index + 1) * 0x9E3779B97F4A7C15
    value ^= value >> 30
    value *= 0xBF58476D1CE4E5B9
    value ^= value >> 27
    value *= 0x94D049BB133111EB
    value ^= value >> 31
    return -1.0 if value & 1 else 1.0


def _binary_flag(value: bool | int | float, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return float(value)
    if isinstance(value, (float, np.floating)) and value in {0.0, 1.0}:
        return float(value)
    raise ValueError(f"{label} must be a boolean or 0/1 flag")


def _event_design(
    events: Sequence[BinaryPreferenceEvent],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    base_margins: list[float] = []
    for event in events:
        preferred, rejected, base_margin = _validate_event(event)
        rows.append(preferred - rejected)
        base_margins.append(base_margin)
    if not rows:
        return (
            np.zeros((0, FEATURE_DIMENSION), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    return np.vstack(rows), np.asarray(base_margins, dtype=np.float64)


def _validate_event(
    event: BinaryPreferenceEvent,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not isinstance(event, BinaryPreferenceEvent):
        raise TypeError("events must be BinaryPreferenceEvent instances")
    preferred = validate_features(event.preferred_features, label="preferred_features")
    rejected = validate_features(event.rejected_features, label="rejected_features")
    base_margin = float(event.base_margin)
    if not math.isfinite(base_margin):
        raise ValueError("base_margin must be finite")
    expected = float(preferred[COSINE_FEATURE_INDEX] - rejected[COSINE_FEATURE_INDEX])
    if not math.isclose(base_margin, expected, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(
            "base_margin must equal preferred cosine minus rejected cosine"
        )
    return preferred, rejected, base_margin


def _validate_prior_lambda(value: float) -> float:
    prior = float(value)
    if not math.isfinite(prior) or prior <= 0.0:
        raise ValueError("prior_lambda must be a finite positive value")
    return prior


def _validate_posterior(posterior: ContextualPreferencePosterior) -> None:
    if not isinstance(posterior, ContextualPreferencePosterior):
        raise TypeError("posterior must be a ContextualPreferencePosterior")
    validate_features(posterior.mean, label="posterior.mean")
    validate_covariance(posterior.covariance)
    if posterior.comparisons < 0:
        raise ValueError("posterior.comparisons must be non-negative")
    _validate_prior_lambda(posterior.prior_lambda)


def _objective_from_design(
    theta: np.ndarray,
    design: np.ndarray,
    base_margins: np.ndarray,
    prior_lambda: float,
) -> float:
    margins = base_margins + design @ theta
    nll = float(np.sum(_softplus(-margins)))
    penalty = 0.5 * prior_lambda * float(np.dot(theta, theta))
    return nll + penalty


def _negative_log_likelihood_from_design(
    theta: np.ndarray,
    design: np.ndarray,
    base_margins: np.ndarray,
) -> float:
    margins = base_margins + design @ theta
    return float(np.sum(_softplus(-margins)))


def _objective_gradient_hessian(
    theta: np.ndarray,
    design: np.ndarray,
    base_margins: np.ndarray,
    prior_lambda: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    margins = base_margins + design @ theta
    probabilities = _sigmoid(margins)
    gradient = design.T @ (probabilities - 1.0) + prior_lambda * theta
    curvature = probabilities * (1.0 - probabilities)
    weighted_design = design * curvature[:, np.newaxis]
    hessian = design.T @ weighted_design
    hessian += np.eye(FEATURE_DIMENSION, dtype=np.float64) * prior_lambda
    objective = float(np.sum(_softplus(-margins)))
    objective += 0.5 * prior_lambda * float(np.dot(theta, theta))
    return objective, gradient, hessian


def _laplace_covariance(hessian: np.ndarray) -> np.ndarray:
    identity = np.eye(FEATURE_DIMENSION, dtype=np.float64)
    factor = np.linalg.cholesky(hessian)
    inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
    covariance = 0.5 * (inverse + inverse.T)
    validate_covariance(covariance)
    return covariance


def _diagnostics(
    *,
    initial_objective: float,
    final_objective: float,
    initial_negative_log_likelihood: float,
    final_negative_log_likelihood: float,
    gradient: np.ndarray,
    step_norm: float,
    iterations: int,
    converged: bool,
    line_search_steps: int,
    hessian: np.ndarray,
) -> ContextualPreferenceDiagnostics:
    eigenvalues = np.linalg.eigvalsh(hessian)
    min_eigenvalue = float(eigenvalues[0])
    max_eigenvalue = float(eigenvalues[-1])
    condition = math.inf if min_eigenvalue <= 0.0 else max_eigenvalue / min_eigenvalue
    return ContextualPreferenceDiagnostics(
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        initial_negative_log_likelihood=float(initial_negative_log_likelihood),
        final_negative_log_likelihood=float(final_negative_log_likelihood),
        gradient_norm=float(np.linalg.norm(gradient, ord=2)),
        step_norm=float(step_norm),
        iterations=int(iterations),
        converged=bool(converged),
        line_search_steps=int(line_search_steps),
        hessian_min_eigenvalue=min_eigenvalue,
        hessian_condition=float(condition),
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _softplus(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.logaddexp(0.0, values)


__all__ = [
    "AUTO_REJECT_FEATURE_INDEX",
    "COSINE_FEATURE_INDEX",
    "DEFAULT_PRIOR_LAMBDA",
    "FEATURE_DIMENSION",
    "OPENCLIP_DIMENSION",
    "PROJECTION_DIMENSION",
    "PROJECTION_ID",
    "PROJECTION_VERSION",
    "QUALITY_MISSING_FEATURE_INDEX",
    "BinaryPreferenceEvent",
    "ContextualPreferenceDiagnostics",
    "ContextualPreferencePosterior",
    "contextual_features",
    "event_margin",
    "make_event",
    "margin",
    "negative_log_likelihood",
    "project_embedding",
    "projection_matrix",
    "sample",
    "score",
    "score_features",
    "train",
    "validate_covariance",
    "validate_features",
    "validate_openclip_embedding",
]
