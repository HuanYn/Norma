from __future__ import annotations

import numpy as np
import pytest

from ai.preferences.contextual import (
    AUTO_REJECT_FEATURE_INDEX,
    COSINE_FEATURE_INDEX,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    OPENCLIP_DIMENSION,
    PROJECTION_DIMENSION,
    PROJECTION_ID,
    PROJECTION_VERSION,
    QUALITY_MISSING_FEATURE_INDEX,
    BinaryPreferenceEvent,
    contextual_features,
    make_event,
    margin,
    negative_log_likelihood,
    projection_matrix,
    sample,
    score,
    score_features,
    train,
)


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=OPENCLIP_DIMENSION)
    return vector / np.linalg.norm(vector)


def test_contextual_feature_dimensions_and_projection_are_deterministic() -> None:
    image = _unit(1)
    query = _unit(2)

    first_projection = projection_matrix()
    second_projection = projection_matrix()
    features = contextual_features(
        image,
        query,
        auto_reject=True,
        quality_missing=False,
    )
    repeated = contextual_features(
        image,
        query,
        auto_reject=True,
        quality_missing=False,
    )

    assert PROJECTION_ID == (
        "openclip512-structured-signed-hadamard-r32-rows37x-plus11-signmix-v1"
    )
    assert PROJECTION_VERSION == "capu-contextual-shp-v1"
    assert FEATURE_SCHEMA == "capu-contextual-openclip512-67d-v1"
    assert first_projection.shape == (PROJECTION_DIMENSION, OPENCLIP_DIMENSION)
    assert np.array_equal(first_projection, second_projection)
    assert features.shape == (FEATURE_DIMENSION,)
    assert np.array_equal(features, repeated)
    assert features[COSINE_FEATURE_INDEX] == pytest.approx(float(np.dot(image, query)))
    assert features[AUTO_REJECT_FEATURE_INDEX] == 1.0
    assert features[QUALITY_MISSING_FEATURE_INDEX] == 0.0


def test_zero_feedback_posterior_strictly_falls_back_to_cosine() -> None:
    image = _unit(3)
    query = _unit(4)
    posterior = train([])
    features = contextual_features(image, query)
    expected_cosine = float(np.dot(image, query))

    assert posterior.comparisons == 0
    assert np.array_equal(posterior.mean, np.zeros(FEATURE_DIMENSION))
    assert score(posterior, image, query) == pytest.approx(expected_cosine)
    assert score_features(posterior, features) == pytest.approx(expected_cosine)


def test_training_decreases_nll_and_raises_complete_item_margin() -> None:
    image = _unit(5)
    query = _unit(6)
    complete = contextual_features(image, query, quality_missing=False)
    missing = contextual_features(image, query, quality_missing=True)
    events = [make_event(complete, missing) for _ in range(6)]

    initial_margin = margin(train([]), complete, missing)
    posterior = train(events)
    trained_margin = margin(posterior, complete, missing)
    initial_nll = negative_log_likelihood(events, np.zeros(FEATURE_DIMENSION))
    trained_nll = negative_log_likelihood(events, posterior.mean)

    assert posterior.comparisons == len(events)
    assert posterior.diagnostics.converged
    assert (
        posterior.diagnostics.final_objective < posterior.diagnostics.initial_objective
    )
    assert trained_nll < initial_nll
    assert trained_margin > initial_margin
    assert score_features(posterior, complete) > score_features(posterior, missing)


def test_training_same_data_is_deterministic() -> None:
    query = _unit(7)
    preferred = contextual_features(_unit(8), query, auto_reject=False)
    rejected = contextual_features(_unit(9), query, auto_reject=True)
    events = [make_event(preferred, rejected) for _ in range(4)]

    first = train(events)
    second = train(events)

    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.covariance, second.covariance)
    assert first.diagnostics == second.diagnostics


def test_laplace_covariance_is_symmetric_positive_semidefinite() -> None:
    query = _unit(10)
    events = [
        make_event(
            contextual_features(_unit(11), query, quality_missing=False),
            contextual_features(_unit(12), query, quality_missing=True),
        ),
        make_event(
            contextual_features(_unit(13), query, auto_reject=False),
            contextual_features(_unit(14), query, auto_reject=True),
        ),
    ]

    posterior = train(events)
    eigenvalues = np.linalg.eigvalsh(posterior.covariance)

    assert posterior.covariance.shape == (FEATURE_DIMENSION, FEATURE_DIMENSION)
    assert np.allclose(posterior.covariance, posterior.covariance.T)
    assert float(eigenvalues[0]) >= -1e-10
    assert posterior.diagnostics.hessian_min_eigenvalue > 0.0


def test_posterior_sampling_is_seeded_and_reproducible() -> None:
    query = _unit(15)
    event = make_event(
        contextual_features(_unit(16), query),
        contextual_features(_unit(17), query, quality_missing=True),
    )
    posterior = train([event])

    first = sample(posterior, n_samples=3, seed=123)
    second = sample(posterior, n_samples=3, seed=123)
    different = sample(posterior, n_samples=3, seed=124)

    assert first.shape == (3, FEATURE_DIMENSION)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_invalid_inputs_are_rejected() -> None:
    image = _unit(18)
    query = _unit(19)
    features = contextual_features(image, query)
    bad_features = features.copy()
    bad_features[0] = np.nan

    with pytest.raises(ValueError, match="shape"):
        contextual_features(image[:3], query)
    with pytest.raises(ValueError, match="non-finite"):
        contextual_features(image, np.full(OPENCLIP_DIMENSION, np.nan))
    with pytest.raises(ValueError, match="L2-normalized"):
        contextual_features(image * 2.0, query)
    with pytest.raises(ValueError, match="auto_reject"):
        contextual_features(image, query, auto_reject=2)
    with pytest.raises(ValueError, match="non-finite"):
        make_event(bad_features, features)
    with pytest.raises(ValueError, match="base_margin"):
        train([BinaryPreferenceEvent(features, features, base_margin=1.0)])
    with pytest.raises(ValueError, match="prior_lambda"):
        train([], prior_lambda=0.0)
    with pytest.raises(ValueError, match="n_samples"):
        sample(train([]), n_samples=0)
