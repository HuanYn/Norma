from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from ai.preferences.contextual import (
    COSINE_FEATURE_INDEX,
    FEATURE_DIMENSION,
    ContextualPreferencePosterior,
    validate_covariance,
    validate_features,
)


DEFAULT_POSTERIOR_SAMPLES = 64
DEFAULT_SHORTLIST_SIZE = 16
ESS_FALLBACK_FRACTION = 0.25
ESS_FALLBACK_MAX = 16.0
NUMERICAL_STABILITY_TOLERANCE = 1e-8
ACQUISITION_VERSION = "capu-pdrr-v1"
CONSTRAINT_SOLVER = "exact-partition-matroid-v1"


class AcquisitionNumericalError(RuntimeError):
    """Raised when Monte Carlo diagnostics cannot support a pair suggestion."""


@dataclass(frozen=True, slots=True)
class AcquisitionCandidate:
    photo_id: str
    features: np.ndarray
    group_key: str


@dataclass(frozen=True, slots=True)
class PairAcquisitionScore:
    left_photo_id: str
    right_photo_id: str
    probability_left_preferred: float
    predictive_entropy: float
    membership_variance: float
    shortlist_score: float
    pdrr: float
    raw_pdrr_estimate: float
    regret_if_left_preferred: float
    regret_if_right_preferred: float
    effective_sample_size_left: float
    effective_sample_size_right: float
    laplace_fallback_left: bool
    laplace_fallback_right: bool
    voi_invariant_ok: bool


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    version: str
    constraint_solver: str
    current_photo_ids: tuple[str, ...]
    current_bayes_regret: float
    suggested: PairAcquisitionScore
    evaluated_pair_count: int
    eligible_pair_count: int
    exhaustive: bool
    posterior_samples: int
    seed: int


def suggest_pair(
    posterior: ContextualPreferencePosterior,
    candidates: Sequence[AcquisitionCandidate],
    *,
    target_count: int,
    max_per_group: int,
    posterior_samples: int = DEFAULT_POSTERIOR_SAMPLES,
    seed: int = 0,
    shortlist_size: int = DEFAULT_SHORTLIST_SIZE,
    exhaustive: bool = False,
    excluded_pairs: Iterable[tuple[str, str]] = (),
) -> AcquisitionResult:
    """Suggest the best shortlisted pair by posterior decision-regret reduction.

    The production path first shortlists pairs by predictive entropy times
    posterior set-membership variance, then estimates two-outcome PDRR by
    Monte Carlo while exactly re-solving Norma's current partition-constrained
    action. ``exhaustive=True`` evaluates every eligible pair; it does not make
    posterior integration exact.
    """

    _validate_posterior_input(posterior)
    normalized = _normalize_candidates(candidates)
    _validate_controls(
        normalized,
        target_count=target_count,
        max_per_group=max_per_group,
        posterior_samples=posterior_samples,
        shortlist_size=shortlist_size,
    )
    excluded = {_canonical_pair(*pair) for pair in excluded_pairs}
    eligible_pairs = [
        (left, right)
        for left, right in combinations(range(len(normalized)), 2)
        if _canonical_pair(normalized[left].photo_id, normalized[right].photo_id)
        not in excluded
    ]
    if not eligible_pairs:
        raise ValueError("at least one eligible candidate pair is required")

    feature_matrix = np.vstack([candidate.features for candidate in normalized])
    standard_draws = np.random.default_rng(seed).standard_normal(
        (posterior_samples, FEATURE_DIMENSION)
    )
    posterior_draws = _gaussian_draws(
        posterior.mean,
        posterior.covariance,
        standard_draws,
    )
    utilities = (
        posterior_draws @ feature_matrix.T
        + feature_matrix[:, COSINE_FEATURE_INDEX][np.newaxis, :]
    )
    mean_utility = (
        posterior.mean @ feature_matrix.T + feature_matrix[:, COSINE_FEATURE_INDEX]
    )
    groups = tuple(candidate.group_key for candidate in normalized)
    current_indices = _solve_partition_set(
        mean_utility,
        groups,
        target_count=target_count,
        max_per_group=max_per_group,
    )

    oracle_sets = np.zeros((posterior_samples, len(normalized)), dtype=np.float64)
    oracle_values = np.empty(posterior_samples, dtype=np.float64)
    for sample_index, utility in enumerate(utilities):
        selected = _solve_partition_set(
            utility,
            groups,
            target_count=target_count,
            max_per_group=max_per_group,
        )
        oracle_sets[sample_index, selected] = 1.0
        oracle_values[sample_index] = float(
            np.sum(_objective_values(utility[selected]))
        )

    current_values = np.sum(_objective_values(utilities[:, current_indices]), axis=1)
    current_regret = _nonnegative(float(np.mean(oracle_values - current_values)))

    heuristics: list[tuple[float, str, str, int, int, float, float, float]] = []
    for left, right in eligible_pairs:
        probabilities = _sigmoid(utilities[:, left] - utilities[:, right])
        probability_left = float(np.mean(probabilities))
        entropy = _binary_entropy(probability_left)
        membership_difference = oracle_sets[:, left] - oracle_sets[:, right]
        membership_variance = float(np.var(membership_difference))
        heuristic = entropy * membership_variance
        left_id = normalized[left].photo_id
        right_id = normalized[right].photo_id
        heuristics.append(
            (
                heuristic,
                left_id,
                right_id,
                left,
                right,
                probability_left,
                entropy,
                membership_variance,
            )
        )

    heuristics.sort(key=lambda item: (-item[0], item[1], item[2]))
    shortlisted = heuristics if exhaustive else heuristics[:shortlist_size]
    scored = [
        _evaluate_pair(
            item,
            utilities=utilities,
            oracle_values=oracle_values,
            groups=groups,
            target_count=target_count,
            max_per_group=max_per_group,
            current_regret=current_regret,
            posterior=posterior,
            feature_matrix=feature_matrix,
            standard_draws=standard_draws,
        )
        for item in shortlisted
    ]
    scored.sort(
        key=lambda item: (
            not item.voi_invariant_ok,
            -item.pdrr,
            -item.raw_pdrr_estimate,
            item.left_photo_id,
            item.right_photo_id,
        )
    )
    if not any(item.voi_invariant_ok for item in scored):
        raise AcquisitionNumericalError(
            "all evaluated pairs violated the non-negative value-of-information "
            "invariant; retry with more posterior samples"
        )
    current_ids = tuple(normalized[index].photo_id for index in current_indices)
    return AcquisitionResult(
        version=ACQUISITION_VERSION,
        constraint_solver=CONSTRAINT_SOLVER,
        current_photo_ids=current_ids,
        current_bayes_regret=current_regret,
        suggested=scored[0],
        evaluated_pair_count=len(scored),
        eligible_pair_count=len(eligible_pairs),
        exhaustive=exhaustive,
        posterior_samples=posterior_samples,
        seed=seed,
    )


def _evaluate_pair(
    heuristic: tuple[float, str, str, int, int, float, float, float],
    *,
    utilities: np.ndarray,
    oracle_values: np.ndarray,
    groups: tuple[str, ...],
    target_count: int,
    max_per_group: int,
    current_regret: float,
    posterior: ContextualPreferencePosterior,
    feature_matrix: np.ndarray,
    standard_draws: np.ndarray,
) -> PairAcquisitionScore:
    (
        shortlist_score,
        left_id,
        right_id,
        left,
        right,
        probability_left,
        entropy,
        membership_variance,
    ) = heuristic
    likelihood_left = _sigmoid(utilities[:, left] - utilities[:, right])
    likelihood_right = 1.0 - likelihood_left
    regret_left, ess_left, fallback_left = _outcome_regret(
        likelihood_left,
        utilities=utilities,
        oracle_values=oracle_values,
        groups=groups,
        target_count=target_count,
        max_per_group=max_per_group,
        posterior=posterior,
        feature_matrix=feature_matrix,
        left=left,
        right=right,
        outcome_left=True,
        standard_draws=standard_draws,
    )
    regret_right, ess_right, fallback_right = _outcome_regret(
        likelihood_right,
        utilities=utilities,
        oracle_values=oracle_values,
        groups=groups,
        target_count=target_count,
        max_per_group=max_per_group,
        posterior=posterior,
        feature_matrix=feature_matrix,
        left=left,
        right=right,
        outcome_left=False,
        standard_draws=standard_draws,
    )
    expected_regret = (
        probability_left * regret_left + (1.0 - probability_left) * regret_right
    )
    raw_pdrr = current_regret - expected_regret
    pdrr = max(0.0, raw_pdrr)
    voi_invariant_ok = raw_pdrr >= -NUMERICAL_STABILITY_TOLERANCE
    return PairAcquisitionScore(
        left_photo_id=left_id,
        right_photo_id=right_id,
        probability_left_preferred=probability_left,
        predictive_entropy=entropy,
        membership_variance=membership_variance,
        shortlist_score=shortlist_score,
        pdrr=pdrr,
        raw_pdrr_estimate=raw_pdrr,
        regret_if_left_preferred=regret_left,
        regret_if_right_preferred=regret_right,
        effective_sample_size_left=ess_left,
        effective_sample_size_right=ess_right,
        laplace_fallback_left=fallback_left,
        laplace_fallback_right=fallback_right,
        voi_invariant_ok=voi_invariant_ok,
    )


def _outcome_regret(
    likelihood: np.ndarray,
    *,
    utilities: np.ndarray,
    oracle_values: np.ndarray,
    groups: tuple[str, ...],
    target_count: int,
    max_per_group: int,
    posterior: ContextualPreferencePosterior,
    feature_matrix: np.ndarray,
    left: int,
    right: int,
    outcome_left: bool,
    standard_draws: np.ndarray,
) -> tuple[float, float, bool]:
    total = float(np.sum(likelihood))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("pair outcome has zero posterior probability")
    weights = likelihood / total
    effective_sample_size = float(1.0 / np.dot(weights, weights))
    fallback_threshold = min(
        utilities.shape[0] * ESS_FALLBACK_FRACTION,
        ESS_FALLBACK_MAX,
    )
    if effective_sample_size < fallback_threshold:
        updated_mean, updated_covariance = _rank_one_laplace_update(
            posterior,
            feature_matrix[left],
            feature_matrix[right],
            outcome_left=outcome_left,
        )
        updated_draws = _gaussian_draws(
            updated_mean,
            updated_covariance,
            standard_draws,
        )
        updated_utilities = (
            updated_draws @ feature_matrix.T
            + feature_matrix[:, COSINE_FEATURE_INDEX][np.newaxis, :]
        )
        updated_oracle = _oracle_values(
            updated_utilities,
            groups,
            target_count=target_count,
            max_per_group=max_per_group,
        )
        updated_mean = (
            updated_mean @ feature_matrix.T + feature_matrix[:, COSINE_FEATURE_INDEX]
        )
        selected = _solve_partition_set(
            updated_mean,
            groups,
            target_count=target_count,
            max_per_group=max_per_group,
        )
        selected_values = np.sum(
            _objective_values(updated_utilities[:, selected]), axis=1
        )
        regret = _nonnegative(float(np.mean(updated_oracle - selected_values)))
        return regret, effective_sample_size, True

    weighted_mean_utility = weights @ utilities
    selected = _solve_partition_set(
        weighted_mean_utility,
        groups,
        target_count=target_count,
        max_per_group=max_per_group,
    )
    selected_values = np.sum(_objective_values(utilities[:, selected]), axis=1)
    regret = _nonnegative(float(np.dot(weights, oracle_values - selected_values)))
    return regret, effective_sample_size, False


def _rank_one_laplace_update(
    posterior: ContextualPreferencePosterior,
    left_features: np.ndarray,
    right_features: np.ndarray,
    *,
    outcome_left: bool,
) -> tuple[np.ndarray, np.ndarray]:
    difference = left_features - right_features
    base_margin = float(
        left_features[COSINE_FEATURE_INDEX] - right_features[COSINE_FEATURE_INDEX]
    )
    covariance_direction = posterior.covariance @ difference
    variance = float(np.dot(difference, covariance_direction))
    if not math.isfinite(variance) or variance < -1e-10:
        raise ValueError("posterior covariance gives an invalid pair variance")
    variance = max(0.0, variance)
    prior_margin = base_margin + float(np.dot(posterior.mean, difference))
    target = 1.0 if outcome_left else 0.0
    lower, upper = (0.0, 1.0) if outcome_left else (-1.0, 0.0)
    alpha = 0.5 * (lower + upper)
    for _ in range(64):
        margin = prior_margin + variance * alpha
        probability_left = float(_sigmoid(np.asarray([margin]))[0])
        value = alpha + probability_left - target
        if abs(value) <= 1e-12:
            break
        if value > 0.0:
            upper = alpha
        else:
            lower = alpha
        derivative = 1.0 + variance * probability_left * (1.0 - probability_left)
        proposal = alpha - value / derivative
        alpha = proposal if lower < proposal < upper else 0.5 * (lower + upper)

    mode = posterior.mean + alpha * covariance_direction
    mode_margin = prior_margin + variance * alpha
    mode_probability = float(_sigmoid(np.asarray([mode_margin]))[0])
    curvature = mode_probability * (1.0 - mode_probability)
    denominator = 1.0 + curvature * variance
    covariance = posterior.covariance - (curvature / denominator) * np.outer(
        covariance_direction, covariance_direction
    )
    covariance = 0.5 * (covariance + covariance.T)
    return mode, covariance


def _gaussian_draws(
    mean: np.ndarray,
    covariance: np.ndarray,
    standard_draws: np.ndarray,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[0]) < -1e-10:
        raise ValueError("posterior covariance must be positive semidefinite")
    symmetric_root = (
        eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    ) @ eigenvectors.T
    return mean + standard_draws @ symmetric_root.T


def _oracle_values(
    utilities: np.ndarray,
    groups: tuple[str, ...],
    *,
    target_count: int,
    max_per_group: int,
) -> np.ndarray:
    values = np.empty(utilities.shape[0], dtype=np.float64)
    for sample_index, utility in enumerate(utilities):
        selected = _solve_partition_set(
            utility,
            groups,
            target_count=target_count,
            max_per_group=max_per_group,
        )
        values[sample_index] = float(np.sum(_objective_values(utility[selected])))
    return values


def _solve_partition_set(
    utilities: np.ndarray,
    groups: tuple[str, ...],
    *,
    target_count: int,
    max_per_group: int,
) -> np.ndarray:
    """Solve exact-K plus partition-capacity constraints deterministically.

    For Norma's current constraint family this ranked greedy algorithm is the
    exact optimizer for a weighted partition matroid, matching the feasible
    sets represented by the production CP-SAT model without per-pair solver
    startup overhead.
    """

    ranked = sorted(
        range(len(utilities)),
        key=lambda index: (
            -_production_objective_units(
                float(utilities[index]), index, len(utilities)
            ),
            index,
        ),
    )
    selected: list[int] = []
    counts: dict[str, int] = {}
    for index in ranked:
        group = groups[index]
        if counts.get(group, 0) >= max_per_group:
            continue
        selected.append(index)
        counts[group] = counts.get(group, 0) + 1
        if len(selected) == target_count:
            return np.asarray(selected, dtype=np.int64)
    raise ValueError("hard constraints are infeasible for the candidate pool")


def _normalize_candidates(
    candidates: Sequence[AcquisitionCandidate],
) -> tuple[AcquisitionCandidate, ...]:
    normalized: list[AcquisitionCandidate] = []
    ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, AcquisitionCandidate):
            raise TypeError("candidates must be AcquisitionCandidate instances")
        if not candidate.photo_id.strip():
            raise ValueError("candidate photo_id must be non-empty")
        if candidate.photo_id != candidate.photo_id.strip():
            raise ValueError("candidate photo_id must not contain edge whitespace")
        if candidate.photo_id in ids:
            raise ValueError(f"duplicate candidate photo_id: {candidate.photo_id}")
        if not candidate.group_key.strip():
            raise ValueError("candidate group_key must be non-empty")
        if candidate.group_key != candidate.group_key.strip():
            raise ValueError("candidate group_key must not contain edge whitespace")
        ids.add(candidate.photo_id)
        normalized.append(
            AcquisitionCandidate(
                photo_id=candidate.photo_id,
                features=validate_features(
                    candidate.features,
                    label=f"features for {candidate.photo_id}",
                ),
                group_key=candidate.group_key,
            )
        )
    normalized.sort(key=lambda candidate: candidate.photo_id)
    return tuple(normalized)


def _validate_posterior_input(posterior: ContextualPreferencePosterior) -> None:
    if not isinstance(posterior, ContextualPreferencePosterior):
        raise TypeError("posterior must be a ContextualPreferencePosterior")
    validate_features(posterior.mean, label="posterior.mean")
    validate_covariance(posterior.covariance)
    if (
        isinstance(posterior.comparisons, bool)
        or not isinstance(posterior.comparisons, int)
        or posterior.comparisons < 0
    ):
        raise ValueError("posterior.comparisons must be a non-negative integer")
    if not math.isfinite(posterior.prior_lambda) or posterior.prior_lambda <= 0.0:
        raise ValueError("posterior.prior_lambda must be finite and positive")


def _validate_controls(
    candidates: tuple[AcquisitionCandidate, ...],
    *,
    target_count: int,
    max_per_group: int,
    posterior_samples: int,
    shortlist_size: int,
) -> None:
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required")
    if not 1 <= target_count <= len(candidates):
        raise ValueError("target_count must fit the candidate pool")
    if max_per_group < 1:
        raise ValueError("max_per_group must be positive")
    if posterior_samples < 2:
        raise ValueError("posterior_samples must be at least two")
    if shortlist_size < 1:
        raise ValueError("shortlist_size must be positive")
    if any(
        candidate.features.shape != (FEATURE_DIMENSION,) for candidate in candidates
    ):
        raise ValueError("candidate feature dimension mismatch")


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError("excluded pair ids must be strings")
    if left == right:
        raise ValueError("a pair must contain two different photo ids")
    return (left, right) if left < right else (right, left)


def _binary_entropy(probability: float) -> float:
    clipped = min(1.0 - 1e-12, max(1e-12, probability))
    return -clipped * math.log(clipped) - (1.0 - clipped) * math.log(1.0 - clipped)


def _objective_values(values: np.ndarray) -> np.ndarray:
    """Mirror the production optimizer's six-decimal primary objective."""

    array = np.asarray(values, dtype=np.float64)
    rounded = np.fromiter(
        (round(float(value) * 1_000_000) / 1_000_000 for value in array.flat),
        dtype=np.float64,
        count=array.size,
    )
    return rounded.reshape(array.shape)


def _production_objective_units(value: float, index: int, count: int) -> int:
    primary = round(value * 1_000_000)
    deterministic_tie_break = count - index
    return primary * (count + 1) + deterministic_tie_break


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    result = np.empty_like(clipped)
    positive = clipped >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-clipped[positive]))
    exp_values = np.exp(clipped[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _nonnegative(value: float) -> float:
    if value >= 0.0:
        return value
    if value >= -1e-10:
        return 0.0
    raise RuntimeError(f"computed negative Bayes regret: {value}")


__all__ = [
    "ACQUISITION_VERSION",
    "CONSTRAINT_SOLVER",
    "DEFAULT_POSTERIOR_SAMPLES",
    "DEFAULT_SHORTLIST_SIZE",
    "ESS_FALLBACK_FRACTION",
    "AcquisitionCandidate",
    "AcquisitionNumericalError",
    "AcquisitionResult",
    "PairAcquisitionScore",
    "suggest_pair",
]
