from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from ai.preferences.acquisition import (
    ACQUISITION_VERSION,
    CONSTRAINT_SOLVER,
    AcquisitionCandidate,
    AcquisitionNumericalError,
    _rank_one_laplace_update,
    _solve_partition_set,
    suggest_pair,
)
from ai.preferences.contextual import (
    COSINE_FEATURE_INDEX,
    FEATURE_DIMENSION,
    sample,
    train,
)


def _features(cosine: float, *, signal: float = 0.0) -> np.ndarray:
    values = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    values[0] = signal
    values[COSINE_FEATURE_INDEX] = cosine
    return values


def _posterior(*, uncertainty: float = 0.2):
    baseline = train([])
    covariance = np.eye(FEATURE_DIMENSION, dtype=np.float64) * uncertainty
    return replace(baseline, covariance=covariance)


def test_suggest_pair_is_deterministic_and_respects_partition_constraints() -> None:
    candidates = [
        AcquisitionCandidate("a", _features(0.90, signal=0.3), "duplicate-a"),
        AcquisitionCandidate("b", _features(0.89, signal=-0.3), "duplicate-a"),
        AcquisitionCandidate("c", _features(0.75, signal=0.5), "unique-c"),
        AcquisitionCandidate("d", _features(0.74, signal=-0.5), "unique-d"),
    ]

    first = suggest_pair(
        _posterior(),
        candidates,
        target_count=2,
        max_per_group=1,
        posterior_samples=64,
        seed=11,
        exhaustive=True,
    )
    second = suggest_pair(
        _posterior(),
        candidates,
        target_count=2,
        max_per_group=1,
        posterior_samples=64,
        seed=11,
        exhaustive=True,
    )

    assert first == second
    assert first.version == ACQUISITION_VERSION
    assert first.constraint_solver == CONSTRAINT_SOLVER
    assert len(first.current_photo_ids) == 2
    assert not {"a", "b"}.issubset(first.current_photo_ids)
    assert first.suggested.pdrr >= 0.0
    assert first.evaluated_pair_count == 6
    assert first.eligible_pair_count == 6


def test_full_shortlist_matches_exhaustive_top_pair() -> None:
    candidates = [
        AcquisitionCandidate(
            str(index),
            _features(0.8 - index * 0.03, signal=(-1) ** index * 0.4),
            f"g-{index}",
        )
        for index in range(5)
    ]
    exhaustive = suggest_pair(
        _posterior(),
        candidates,
        target_count=2,
        max_per_group=1,
        posterior_samples=48,
        seed=7,
        exhaustive=True,
    )
    shortlisted = suggest_pair(
        _posterior(),
        candidates,
        target_count=2,
        max_per_group=1,
        posterior_samples=48,
        seed=7,
        shortlist_size=10,
    )

    assert shortlisted.suggested == exhaustive.suggested
    assert shortlisted.evaluated_pair_count == exhaustive.eligible_pair_count


def test_candidate_input_order_does_not_change_suggestion() -> None:
    candidates = [
        AcquisitionCandidate("c", _features(0.76, signal=0.4), "c"),
        AcquisitionCandidate("a", _features(0.81, signal=-0.5), "a"),
        AcquisitionCandidate("d", _features(0.73, signal=-0.3), "d"),
        AcquisitionCandidate("b", _features(0.79, signal=0.6), "b"),
    ]
    forward = suggest_pair(
        _posterior(),
        candidates,
        target_count=2,
        max_per_group=1,
        posterior_samples=48,
        seed=17,
        exhaustive=True,
    )
    reverse = suggest_pair(
        _posterior(),
        list(reversed(candidates)),
        target_count=2,
        max_per_group=1,
        posterior_samples=48,
        seed=17,
        exhaustive=True,
    )

    assert forward == reverse


def test_low_ess_outcome_uses_converged_rank_one_laplace_fallback() -> None:
    baseline = train([])
    mean = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    mean[0] = 2.0
    posterior = replace(
        baseline,
        mean=mean,
        covariance=np.eye(FEATURE_DIMENSION, dtype=np.float64) * 2.0,
        comparisons=1,
    )
    candidates = [
        AcquisitionCandidate("a", _features(0.8, signal=1.0), "a"),
        AcquisitionCandidate("b", _features(0.8, signal=-1.0), "b"),
        AcquisitionCandidate("c", _features(0.5), "c"),
    ]

    result = suggest_pair(
        posterior,
        candidates,
        target_count=1,
        max_per_group=1,
        posterior_samples=64,
        seed=3,
        exhaustive=True,
    )
    repeated = suggest_pair(
        posterior,
        candidates,
        target_count=1,
        max_per_group=1,
        posterior_samples=64,
        seed=3,
        exhaustive=True,
    )

    assert result == repeated
    assert result.suggested.laplace_fallback_left is False
    assert result.suggested.laplace_fallback_right is True
    assert result.suggested.effective_sample_size_right < 16.0
    assert result.suggested.pdrr >= 0.0
    mode, covariance = _rank_one_laplace_update(
        posterior,
        candidates[0].features,
        candidates[1].features,
        outcome_left=False,
    )
    assert mode[0] == pytest.approx(0.0, abs=1e-10)
    assert covariance[0, 0] == pytest.approx(2.0 / 3.0)


def test_partition_solver_matches_brute_force_and_never_violates_capacity() -> None:
    utilities = np.asarray([0.7, 0.95, 0.6, 0.8, 0.5], dtype=np.float64)
    groups = ("x", "x", "y", "z", "z")
    selected = _solve_partition_set(
        utilities,
        groups,
        target_count=3,
        max_per_group=1,
    )
    feasible = [
        indices
        for indices in combinations(range(len(utilities)), 3)
        if len({groups[index] for index in indices}) == 3
    ]
    brute_force = max(
        feasible,
        key=lambda indices: (
            sum(utilities[index] for index in indices),
            tuple(-i for i in indices),
        ),
    )

    assert set(selected) == set(brute_force)
    assert len({groups[index] for index in selected}) == len(selected)


def test_exhaustive_pdrr_matches_independent_small_pool_oracle() -> None:
    posterior = _posterior(uncertainty=0.05)
    candidates = [
        AcquisitionCandidate("c", _features(0.62, signal=0.1), "c"),
        AcquisitionCandidate("a", _features(0.72, signal=0.4), "a"),
        AcquisitionCandidate("b", _features(0.70, signal=-0.3), "b"),
    ]
    result = suggest_pair(
        posterior,
        candidates,
        target_count=1,
        max_per_group=1,
        posterior_samples=32,
        seed=23,
        exhaustive=True,
    )

    ordered = sorted(candidates, key=lambda candidate: candidate.photo_id)
    matrix = np.vstack([candidate.features for candidate in ordered])
    draws = sample(posterior, n_samples=32, seed=23)
    utilities = draws @ matrix.T + matrix[:, COSINE_FEATURE_INDEX]
    objective_values = np.round(utilities, decimals=6)
    oracle_values = np.max(objective_values, axis=1)
    mean_utility = posterior.mean @ matrix.T + matrix[:, COSINE_FEATURE_INDEX]
    current_values = objective_values[:, int(np.argmax(np.round(mean_utility, 6)))]
    current_regret = float(np.mean(oracle_values - current_values))
    by_id = {candidate.photo_id: index for index, candidate in enumerate(ordered)}
    left = by_id[result.suggested.left_photo_id]
    right = by_id[result.suggested.right_photo_id]
    margin = np.clip(utilities[:, left] - utilities[:, right], -40.0, 40.0)
    likelihood_left = 1.0 / (1.0 + np.exp(-margin))

    outcome_regrets = []
    for likelihood in (likelihood_left, 1.0 - likelihood_left):
        weights = likelihood / np.sum(likelihood)
        action = int(np.argmax(np.round(weights @ utilities, decimals=6)))
        outcome_regrets.append(
            float(np.dot(weights, oracle_values - objective_values[:, action]))
        )
    probability_left = float(np.mean(likelihood_left))
    raw_pdrr = current_regret - (
        probability_left * outcome_regrets[0]
        + (1.0 - probability_left) * outcome_regrets[1]
    )

    assert result.suggested.laplace_fallback_left is False
    assert result.suggested.laplace_fallback_right is False
    assert result.current_bayes_regret == pytest.approx(current_regret)
    assert result.suggested.raw_pdrr_estimate == pytest.approx(raw_pdrr)
    assert result.suggested.pdrr == pytest.approx(max(0.0, raw_pdrr))

    all_pair_scores = []
    for pair_left, pair_right in combinations(range(len(ordered)), 2):
        pair_margin = np.clip(
            utilities[:, pair_left] - utilities[:, pair_right], -40.0, 40.0
        )
        pair_likelihood_left = 1.0 / (1.0 + np.exp(-pair_margin))
        pair_regrets = []
        for likelihood in (pair_likelihood_left, 1.0 - pair_likelihood_left):
            weights = likelihood / np.sum(likelihood)
            action = int(np.argmax(np.round(weights @ utilities, decimals=6)))
            pair_regrets.append(
                float(np.dot(weights, oracle_values - objective_values[:, action]))
            )
        pair_probability_left = float(np.mean(pair_likelihood_left))
        pair_raw = current_regret - (
            pair_probability_left * pair_regrets[0]
            + (1.0 - pair_probability_left) * pair_regrets[1]
        )
        all_pair_scores.append(
            (
                max(0.0, pair_raw),
                pair_raw,
                ordered[pair_left].photo_id,
                ordered[pair_right].photo_id,
            )
        )
    expected = sorted(
        all_pair_scores,
        key=lambda item: (-item[0], -item[1], item[2], item[3]),
    )[0]
    assert (
        result.suggested.left_photo_id,
        result.suggested.right_photo_id,
    ) == expected[2:]


def test_excluded_pairs_are_never_suggested() -> None:
    candidates = [
        AcquisitionCandidate("a", _features(0.7, signal=0.6), "a"),
        AcquisitionCandidate("b", _features(0.7, signal=-0.6), "b"),
        AcquisitionCandidate("c", _features(0.6), "c"),
    ]
    initial = suggest_pair(
        _posterior(),
        candidates,
        target_count=1,
        max_per_group=1,
        posterior_samples=32,
        seed=3,
        exhaustive=True,
    )
    excluded = (initial.suggested.left_photo_id, initial.suggested.right_photo_id)
    replacement = suggest_pair(
        _posterior(),
        candidates,
        target_count=1,
        max_per_group=1,
        posterior_samples=32,
        seed=3,
        exhaustive=True,
        excluded_pairs=[excluded, tuple(reversed(excluded))],
    )

    assert {
        replacement.suggested.left_photo_id,
        replacement.suggested.right_photo_id,
    } != set(excluded)
    assert replacement.eligible_pair_count == 2


def test_infeasible_hard_constraints_fail_explicitly() -> None:
    candidates = [
        AcquisitionCandidate("a", _features(0.9), "same"),
        AcquisitionCandidate("b", _features(0.8), "same"),
        AcquisitionCandidate("c", _features(0.7), "same"),
    ]

    with pytest.raises(ValueError, match="infeasible"):
        suggest_pair(
            _posterior(),
            candidates,
            target_count=2,
            max_per_group=1,
            posterior_samples=16,
            exhaustive=True,
        )


def test_candidate_validation_rejects_duplicate_ids_and_bad_feature_shape() -> None:
    duplicate = [
        AcquisitionCandidate("a", _features(0.9), "a"),
        AcquisitionCandidate("a", _features(0.8), "b"),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        suggest_pair(
            _posterior(),
            duplicate,
            target_count=1,
            max_per_group=1,
        )

    invalid = [
        AcquisitionCandidate("a", np.zeros(2), "a"),
        AcquisitionCandidate("b", _features(0.8), "b"),
    ]
    with pytest.raises(ValueError, match="expected"):
        suggest_pair(
            _posterior(),
            invalid,
            target_count=1,
            max_per_group=1,
        )

    whitespace = [
        AcquisitionCandidate(" a", _features(0.9), "a"),
        AcquisitionCandidate("b", _features(0.8), " b"),
    ]
    with pytest.raises(ValueError, match="edge whitespace"):
        suggest_pair(
            _posterior(),
            whitespace,
            target_count=1,
            max_per_group=1,
        )

    broken_posterior = replace(
        _posterior(),
        covariance=np.full(
            (FEATURE_DIMENSION, FEATURE_DIMENSION), np.nan, dtype=np.float64
        ),
    )
    with pytest.raises(ValueError, match="non-finite"):
        suggest_pair(
            broken_posterior,
            [
                AcquisitionCandidate("a", _features(0.9), "a"),
                AcquisitionCandidate("b", _features(0.8), "b"),
            ],
            target_count=1,
            max_per_group=1,
        )


def test_partition_solver_matches_production_six_decimal_tie_break() -> None:
    utilities = np.asarray([0.5000000, 0.5000004], dtype=np.float64)
    selected = _solve_partition_set(
        utilities,
        ("a", "b"),
        target_count=1,
        max_per_group=1,
    )

    assert selected.tolist() == [0]


def test_all_negative_voi_estimates_abstain_instead_of_returning_pair() -> None:
    baseline = train([])
    mean = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    mean[0] = -13.365125950779468
    posterior = replace(
        baseline,
        mean=mean,
        covariance=np.eye(FEATURE_DIMENSION, dtype=np.float64) * 8.40314877809924,
        comparisons=1,
    )
    values = [
        ("a", 0.7009950623362629, 0.5937579405347071, "0"),
        ("b", -0.21946647974396488, 0.7845803945509804, "1"),
        ("c", -2.1692389099116145, -0.3605508311054697, "2"),
        ("d", 0.07457496609109689, 0.11790377266622021, "3"),
    ]
    candidates = [
        AcquisitionCandidate(photo_id, _features(cosine, signal=signal), group)
        for photo_id, signal, cosine, group in values
    ]

    with pytest.raises(AcquisitionNumericalError, match="value-of-information"):
        suggest_pair(
            posterior,
            candidates,
            target_count=2,
            max_per_group=1,
            posterior_samples=16,
            seed=269,
            exhaustive=True,
        )
