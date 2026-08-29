from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

# This benchmark is deliberately offline.  It must reuse the already cached model.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from ai.index.embedding import create_embedding_provider  # noqa: E402
import ai.preferences.contextual as contextual_module  # noqa: E402
from ai.preferences.contextual import (  # noqa: E402
    COSINE_FEATURE_INDEX,
    DEFAULT_PRIOR_LAMBDA,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    PROJECTION_ID,
    PROJECTION_VERSION,
    ContextualPreferencePosterior,
    contextual_features,
    make_event,
    train,
)


CATEGORIES = (
    "travel architecture",
    "city night photography",
    "mountain travel landscape",
)
BUDGETS = (0, 10, 30, 60)
DEFAULT_SEEDS = tuple(range(10))
QUERY_TEXT = "精选旅行摄影作品集"
CHOICE_TEMPERATURE = 0.55
TEST_FRACTION = 0.40
TARGET_COUNT = 6
MAX_PER_CATEGORY = 3
BOOTSTRAP_SEED = 20_260_828
ENTROPY_APPROXIMATION = "logistic-normal-sigmoid-moment-pi-over-8"
OPENCLIP_CACHE_MARKERS = (
    "models--laion--CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
    "models--xlm-roberta-base",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_PATH_BASE = {
    "id": "norma-repository-root",
    "anchor": ".",
    "path_style": "posix",
    "resolver": "benchmark-script-parents-1",
}

PROFILES: dict[str, dict[str, float]] = {
    "architecture_first": {
        "travel architecture": 0.90,
        "city night photography": 0.10,
        "mountain travel landscape": -0.65,
    },
    "city_night_first": {
        "travel architecture": 0.05,
        "city night photography": 0.90,
        "mountain travel landscape": -0.60,
    },
    "mountain_first": {
        "travel architecture": -0.30,
        "city night photography": -0.55,
        "mountain travel landscape": 0.90,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline controlled category-preference simulation for Norma's "
            "67D contextual OpenCLIP adapter"
        )
    )
    parser.add_argument("--album", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated non-negative integer replicate seeds",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--reuse-embeddings-from",
        type=Path,
        help="reuse the validated item/query embeddings in a previous result JSON",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path, *, label: str) -> str:
    """Return a public, clone-portable path anchored at ``PROJECT_ROOT``."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be inside the Norma repository for portable provenance: "
            f"{resolved}"
        ) from exc
    return relative.as_posix() or "."


def _validate_portable_path_base(payload: dict[str, Any], *, label: str) -> None:
    provenance = payload.get("provenance", {})
    if provenance.get("path_base") != PORTABLE_PATH_BASE:
        raise ValueError(
            f"{label} must declare the supported repository-relative POSIX path base"
        )
    for name, record in provenance.get("source_files", {}).items():
        raw = record.get("path")
        if not isinstance(raw, str):
            raise ValueError(f"{label} source file {name!r} has no portable path")
        portable = PurePosixPath(raw)
        if (
            portable.is_absolute()
            or "\\" in raw
            or ":" in raw
            or ".." in portable.parts
        ):
            raise ValueError(f"{label} source file {name!r} is not repo-relative")


def _stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_uniform(*parts: object) -> float:
    return (_stable_u64(*parts) + 0.5) / float(2**64)


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, -40.0, 40.0)
    result = np.empty_like(clipped)
    positive = clipped >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-clipped[positive]))
    exp_values = np.exp(clipped[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated integers") from exc
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("--seeds needs at least two distinct non-negative integers")
    return seeds


def _same_path(left: Path, right: Path) -> bool:
    """Compare paths safely even when the output does not exist yet."""

    left_resolved = left.resolve()
    right_resolved = right.resolve()
    if left_resolved.exists() and right_resolved.exists():
        try:
            return left_resolved.samefile(right_resolved)
        except OSError:
            pass
    return os.path.normcase(str(left_resolved)) == os.path.normcase(str(right_resolved))


def _validate_run_paths(
    *,
    cache_dir: Path,
    output: Path,
    reuse_embeddings_from: Path | None,
) -> None:
    """Reject the two path mistakes that previously invalidated pilot runs."""

    if reuse_embeddings_from is not None:
        if _same_path(reuse_embeddings_from, output):
            raise ValueError(
                "--reuse-embeddings-from and --output must be different files; "
                "overwriting the reuse source would make source_sha256 self-referential"
            )
        return

    # OpenClipMultilingualProvider appends its own ``openclip`` component.  The
    # CLI therefore accepts the parent model-cache root, not that provider
    # directory itself.  Because this benchmark is deliberately offline, both
    # pinned Hugging Face repositories must already be present before inference.
    provider_cache = cache_dir / "openclip"
    missing = [
        marker
        for marker in OPENCLIP_CACHE_MARKERS
        if not (provider_cache / marker).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "--cache-dir must be the parent Norma model-cache root; the provider "
            "appends 'openclip' itself. Missing offline cache entries under "
            f"{provider_cache}: {', '.join(missing)}"
        )


def _load_public_items(album: Path) -> tuple[list[dict[str, Any]], str]:
    attribution_path = album / "ATTRIBUTION.json"
    payload = json.loads(attribution_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for raw in payload["images"]:
        category = raw["search_term"]
        if category not in CATEGORIES:
            continue
        image_path = album / raw["file"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        selected.append(
            {
                "filename": raw["file"],
                "category": category,
                "source_page": raw["source_page"],
                "license_short_name": raw["license"].get("LicenseShortName", ""),
                "artist": raw["license"].get("Artist", ""),
                "file_sha256": _sha256_file(image_path),
                "path": image_path,
            }
        )
    selected.sort(key=lambda item: item["filename"])
    counts = {category: 0 for category in CATEGORIES}
    for item in selected:
        counts[item["category"]] += 1
    if any(counts[category] < 10 for category in CATEGORIES):
        raise ValueError(f"insufficient public images per category: {counts}")
    return selected, _sha256_file(attribution_path)


def _validate_embedding(vector: Any, *, label: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (512,) or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid OpenCLIP embedding for {label}: {array.shape}")
    norm = float(np.linalg.norm(array))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"non-unit OpenCLIP embedding for {label}: {norm}")
    return array


def _reuse_embeddings(
    source: Path,
    public_items: list[dict[str, Any]],
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate_portable_path_base(payload, label="reused result")
    embedded_items = payload["derived_data"]["items"]
    expected = {(item["filename"], item["file_sha256"]): item for item in public_items}
    observed = {
        (item["filename"], item["file_sha256"]): item for item in embedded_items
    }
    if set(expected) != set(observed):
        raise ValueError("reused embedding file does not match current public images")
    vectors = [
        _validate_embedding(
            observed[(item["filename"], item["file_sha256"])]["openclip_embedding"],
            label=item["filename"],
        )
        for item in public_items
    ]
    query_record = payload["derived_data"]["query"]
    if query_record["text"] != QUERY_TEXT:
        raise ValueError("reused result has a different contextual query")
    query = _validate_embedding(query_record["openclip_embedding"], label="query")
    return (
        vectors,
        query,
        {
            "mode": "validated-result-json-reuse",
            "source_path": _project_relative(source, label="--reuse-embeddings-from"),
            "source_sha256": _sha256_file(source),
            "seconds": 0.0,
        },
    )


def _embed_offline(
    public_items: list[dict[str, Any]],
    *,
    cache_dir: Path,
    device: str,
    batch_size: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    provider = create_embedding_provider(
        "openclip-multilingual",
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
    )
    started = time.perf_counter()
    image_vectors = provider.embed_images([item["path"] for item in public_items])
    query_vector = provider.embed_text(QUERY_TEXT)
    elapsed = time.perf_counter() - started
    vectors = [
        _validate_embedding(vector, label=item["filename"])
        for item, vector in zip(public_items, image_vectors, strict=True)
    ]
    query = _validate_embedding(query_vector, label="query")
    return (
        vectors,
        query,
        {
            "mode": "offline-cached-model-inference",
            "provider_name": provider.name,
            "device": device,
            "batch_size": batch_size,
            "seconds": elapsed,
        },
    )


def _split_indices(categories: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for category in CATEGORIES:
        category_indices = np.asarray(
            [index for index, value in enumerate(categories) if value == category],
            dtype=np.int64,
        )
        shuffled = rng.permutation(category_indices)
        test_count = int(round(len(category_indices) * TEST_FRACTION))
        test_indices.extend(int(index) for index in shuffled[:test_count])
        train_indices.extend(int(index) for index in shuffled[test_count:])
    train = np.asarray(sorted(train_indices), dtype=np.int64)
    test = np.asarray(sorted(test_indices), dtype=np.int64)
    if set(train) & set(test) or len(train) + len(test) != len(categories):
        raise RuntimeError("split is not image-disjoint and exhaustive")
    return train, test


def _all_pairs(indices: np.ndarray) -> np.ndarray:
    pairs = [
        (int(indices[left]), int(indices[right]))
        for left in range(len(indices))
        for right in range(left + 1, len(indices))
    ]
    return np.asarray(pairs, dtype=np.int64)


def _true_utilities(
    cosine_scores: np.ndarray,
    categories: list[str],
    profile_weights: dict[str, float],
) -> np.ndarray:
    return cosine_scores + np.asarray(
        [profile_weights[category] for category in categories], dtype=np.float64
    )


def _pair_outcome(
    *,
    seed: int,
    profile: str,
    left_filename: str,
    right_filename: str,
    probability_left: float,
) -> bool:
    canonical = tuple(sorted((left_filename, right_filename)))
    uniform = _stable_uniform("choice", seed, profile, *canonical)
    return uniform < probability_left


def _solve_partition_set(
    utilities: np.ndarray,
    indices: np.ndarray,
    categories: list[str],
) -> np.ndarray:
    ranked = sorted(
        (int(index) for index in indices),
        key=lambda index: (-float(utilities[index]), index),
    )
    selected: list[int] = []
    counts: dict[str, int] = {}
    for index in ranked:
        category = categories[index]
        if counts.get(category, 0) >= MAX_PER_CATEGORY:
            continue
        selected.append(index)
        counts[category] = counts.get(category, 0) + 1
        if len(selected) == TARGET_COUNT:
            return np.asarray(selected, dtype=np.int64)
    raise RuntimeError("controlled exact-K category-cap constraints are infeasible")


def _predict_scores(
    posterior: ContextualPreferencePosterior,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    cosine = feature_matrix[:, COSINE_FEATURE_INDEX]
    if posterior.comparisons == 0:
        return cosine.copy()
    return cosine + feature_matrix @ posterior.mean


def _evaluate_checkpoint(
    *,
    posterior: ContextualPreferencePosterior,
    test_indices: np.ndarray,
    test_pairs: np.ndarray,
    feature_matrix: np.ndarray,
    true_utilities: np.ndarray,
    categories: list[str],
    filenames: list[str],
) -> dict[str, Any]:
    predicted_scores = _predict_scores(posterior, feature_matrix)
    left, right = test_pairs[:, 0], test_pairs[:, 1]
    true_margins = true_utilities[left] - true_utilities[right]
    predicted_margins = predicted_scores[left] - predicted_scores[right]
    true_probabilities = _sigmoid(true_margins / CHOICE_TEMPERATURE)
    predicted_probabilities = np.clip(_sigmoid(predicted_margins), 1e-12, 1 - 1e-12)
    log_loss = float(
        np.mean(
            -true_probabilities * np.log(predicted_probabilities)
            - (1 - true_probabilities) * np.log(1 - predicted_probabilities)
        )
    )
    strict_correct = np.sign(predicted_margins) == np.sign(true_margins)
    tie_credit = np.where(predicted_margins == 0.0, 0.5, strict_correct.astype(float))
    pair_accuracy = float(np.mean(tie_credit))

    oracle_set = _solve_partition_set(true_utilities, test_indices, categories)
    predicted_set = _solve_partition_set(predicted_scores, test_indices, categories)
    raw_regret = float(
        np.sum(true_utilities[oracle_set]) - np.sum(true_utilities[predicted_set])
    )
    if raw_regret < -1e-10:
        raise RuntimeError(f"negative controlled set regret: {raw_regret}")
    raw_regret = max(0.0, raw_regret)
    diagnostics = asdict(posterior.diagnostics)
    diagnostics["posterior_mean_l2"] = float(np.linalg.norm(posterior.mean))
    return {
        "heldout_pair_count": int(len(test_pairs)),
        "heldout_pair_expected_log_loss": log_loss,
        "heldout_pair_order_accuracy": pair_accuracy,
        "constrained_set_regret": raw_regret,
        "constrained_set_regret_per_photo": raw_regret / TARGET_COUNT,
        "oracle_selection": [filenames[index] for index in oracle_set],
        "predicted_selection": [filenames[index] for index in predicted_set],
        "posterior": {
            "comparisons": posterior.comparisons,
            "prior_lambda": posterior.prior_lambda,
            "projection_id": posterior.projection_id,
            "projection_version": posterior.projection_version,
            "mean": posterior.mean.tolist(),
            "covariance_diagonal": np.diag(posterior.covariance).tolist(),
            "diagnostics": diagnostics,
        },
    }


def _select_random_pair(
    available_pairs: np.ndarray,
    *,
    seed: int,
    profile: str,
) -> Iterable[tuple[int, int]]:
    rng = np.random.default_rng(_stable_u64("random-pairs", seed, profile))
    order = rng.permutation(len(available_pairs))
    for pair_index in order:
        yield tuple(int(value) for value in available_pairs[pair_index])


def _select_entropy_pair(
    *,
    posterior: ContextualPreferencePosterior,
    available_pairs: np.ndarray,
    used_pairs: set[tuple[int, int]],
    feature_matrix: np.ndarray,
    filenames: list[str],
) -> tuple[int, int, float, float]:
    eligible = np.asarray(
        [
            pair
            for pair in available_pairs
            if tuple(int(value) for value in pair) not in used_pairs
        ],
        dtype=np.int64,
    )
    left, right = eligible[:, 0], eligible[:, 1]
    differences = feature_matrix[left] - feature_matrix[right]
    base_margins = (
        feature_matrix[left, COSINE_FEATURE_INDEX]
        - feature_matrix[right, COSINE_FEATURE_INDEX]
    )
    means = base_margins + differences @ posterior.mean
    variances = np.einsum(
        "ij,jk,ik->i", differences, posterior.covariance, differences, optimize=True
    )
    variances = np.clip(variances, 0.0, None)
    # Deterministic logistic-normal moment approximation; no Monte Carlo noise is
    # allowed to decide which pair is queried.
    adjusted = means / np.sqrt(1.0 + math.pi * variances / 8.0)
    probabilities = _sigmoid(adjusted)
    entropies = _binary_entropy(probabilities)
    ranked = sorted(
        range(len(eligible)),
        key=lambda index: (
            -float(entropies[index]),
            filenames[int(eligible[index, 0])],
            filenames[int(eligible[index, 1])],
        ),
    )
    selected_index = ranked[0]
    pair = tuple(int(value) for value in eligible[selected_index])
    return (
        pair[0],
        pair[1],
        float(probabilities[selected_index]),
        float(entropies[selected_index]),
    )


def _run_learning_method(
    *,
    method: str,
    seed: int,
    profile: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    feature_matrix: np.ndarray,
    true_utilities: np.ndarray,
    categories: list[str],
    filenames: list[str],
) -> dict[str, Any]:
    train_pairs = _all_pairs(train_indices)
    test_pairs = _all_pairs(test_indices)
    events = []
    trace: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    posterior = train(events, prior_lambda=DEFAULT_PRIOR_LAMBDA)
    checkpoints["0"] = _evaluate_checkpoint(
        posterior=posterior,
        test_indices=test_indices,
        test_pairs=test_pairs,
        feature_matrix=feature_matrix,
        true_utilities=true_utilities,
        categories=categories,
        filenames=filenames,
    )
    if method == "zero_feedback_cosine":
        for budget in BUDGETS[1:]:
            checkpoints[str(budget)] = checkpoints["0"]
        return {
            "seed": seed,
            "profile": profile,
            "method": method,
            "checkpoints": checkpoints,
            "feedback_trace": trace,
        }

    used_pairs: set[tuple[int, int]] = set()
    random_pairs = iter(_select_random_pair(train_pairs, seed=seed, profile=profile))
    for step in range(1, max(BUDGETS) + 1):
        acquisition_probability = None
        acquisition_entropy = None
        if method == "random_pair_contextual":
            left, right = next(random_pairs)
        elif method == "predictive_entropy_contextual":
            (
                left,
                right,
                acquisition_probability,
                acquisition_entropy,
            ) = _select_entropy_pair(
                posterior=posterior,
                available_pairs=train_pairs,
                used_pairs=used_pairs,
                feature_matrix=feature_matrix,
                filenames=filenames,
            )
        else:
            raise ValueError(f"unknown method: {method}")
        pair = (min(left, right), max(left, right))
        if pair in used_pairs:
            raise RuntimeError(f"duplicate feedback pair selected: {pair}")
        used_pairs.add(pair)
        probability_left = float(
            _sigmoid(
                (true_utilities[left] - true_utilities[right]) / CHOICE_TEMPERATURE
            )
        )
        left_preferred = _pair_outcome(
            seed=seed,
            profile=profile,
            left_filename=filenames[left],
            right_filename=filenames[right],
            probability_left=probability_left,
        )
        preferred, rejected = (left, right) if left_preferred else (right, left)
        events.append(make_event(feature_matrix[preferred], feature_matrix[rejected]))
        trace.append(
            {
                "step": step,
                "left": filenames[left],
                "right": filenames[right],
                "left_category": categories[left],
                "right_category": categories[right],
                "true_probability_left": probability_left,
                "observed_left_preferred": left_preferred,
                "feedback_uniform_draw": _stable_uniform(
                    "choice",
                    seed,
                    profile,
                    *sorted((filenames[left], filenames[right])),
                ),
                "acquisition_predictive_probability_left": acquisition_probability,
                "acquisition_predictive_entropy": acquisition_entropy,
            }
        )
        # Entropy acquisition consumes the posterior every step. Random acquisition
        # only needs it at reporting checkpoints.
        if method == "predictive_entropy_contextual" or step in BUDGETS:
            posterior = train(events, prior_lambda=DEFAULT_PRIOR_LAMBDA)
        if step in BUDGETS:
            checkpoints[str(step)] = _evaluate_checkpoint(
                posterior=posterior,
                test_indices=test_indices,
                test_pairs=test_pairs,
                feature_matrix=feature_matrix,
                true_utilities=true_utilities,
                categories=categories,
                filenames=filenames,
            )
    return {
        "seed": seed,
        "profile": profile,
        "method": method,
        "checkpoints": checkpoints,
        "feedback_trace": trace,
    }


def _bootstrap_summary(
    runs: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    resamples: int,
) -> dict[str, Any]:
    if resamples < 1_000:
        raise ValueError("bootstrap-resamples must be at least 1000")
    metrics = (
        "heldout_pair_expected_log_loss",
        "heldout_pair_order_accuracy",
        "constrained_set_regret_per_photo",
    )
    methods = (
        "zero_feedback_cosine",
        "random_pair_contextual",
        "predictive_entropy_contextual",
    )
    summary: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        summary[method] = {}
        for budget_index, budget in enumerate(BUDGETS):
            summary[method][str(budget)] = {}
            relevant = [run for run in runs if run["method"] == method]
            for metric_index, metric in enumerate(metrics):
                seed_means = []
                for seed in seeds:
                    values = [
                        float(run["checkpoints"][str(budget)][metric])
                        for run in relevant
                        if run["seed"] == seed
                    ]
                    if len(values) != len(PROFILES):
                        raise RuntimeError("each seed must contain every fixed profile")
                    seed_means.append(float(np.mean(values)))
                values = np.asarray(seed_means, dtype=np.float64)
                rng = np.random.default_rng(
                    BOOTSTRAP_SEED
                    + 10_000 * method_index
                    + 100 * budget_index
                    + metric_index
                )
                indices = rng.integers(
                    0, len(values), size=(resamples, len(values)), endpoint=False
                )
                bootstrap_means = np.mean(values[indices], axis=1)
                low, high = np.quantile(bootstrap_means, [0.025, 0.975])
                summary[method][str(budget)][metric] = {
                    "mean": float(np.mean(values)),
                    "sample_standard_deviation": float(np.std(values, ddof=1)),
                    "bootstrap_95_percentile_ci": [float(low), float(high)],
                    "raw_seed_means": seed_means,
                    "n_seed_replicates": len(seeds),
                    "profiles_averaged_within_seed": len(PROFILES),
                    "bootstrap_resamples": resamples,
                }
    return summary


def _paired_final_budget_comparisons(
    summary: dict[str, Any], *, resamples: int
) -> dict[str, Any]:
    budget = str(max(BUDGETS))
    specifications = (
        ("random_vs_zero", "zero_feedback_cosine", "random_pair_contextual"),
        (
            "entropy_vs_zero",
            "zero_feedback_cosine",
            "predictive_entropy_contextual",
        ),
        (
            "entropy_vs_random",
            "random_pair_contextual",
            "predictive_entropy_contextual",
        ),
    )
    metric_directions = {
        "heldout_pair_expected_log_loss": "lower",
        "heldout_pair_order_accuracy": "higher",
        "constrained_set_regret_per_photo": "lower",
    }
    comparisons: dict[str, Any] = {}
    for comparison_index, (name, reference, challenger) in enumerate(specifications):
        comparisons[name] = {
            "budget": int(budget),
            "reference": reference,
            "challenger": challenger,
            "metrics": {},
        }
        for metric_index, (metric, better) in enumerate(metric_directions.items()):
            reference_values = np.asarray(
                summary[reference][budget][metric]["raw_seed_means"],
                dtype=np.float64,
            )
            challenger_values = np.asarray(
                summary[challenger][budget][metric]["raw_seed_means"],
                dtype=np.float64,
            )
            if better == "lower":
                improvements = reference_values - challenger_values
            else:
                improvements = challenger_values - reference_values
            rng = np.random.default_rng(
                BOOTSTRAP_SEED + 1_000_000 + 100 * comparison_index + metric_index
            )
            indices = rng.integers(
                0,
                len(improvements),
                size=(resamples, len(improvements)),
                endpoint=False,
            )
            bootstrap_means = np.mean(improvements[indices], axis=1)
            low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            mean_improvement = float(np.mean(improvements))
            reference_mean = float(np.mean(reference_values))
            comparisons[name]["metrics"][metric] = {
                "better_direction": better,
                "positive_means_challenger_is_better": True,
                "paired_mean_improvement": mean_improvement,
                "paired_bootstrap_95_percentile_ci": [float(low), float(high)],
                "interval_excludes_zero": bool(low > 0.0 or high < 0.0),
                "raw_seed_improvements": improvements.tolist(),
                "relative_improvement_percent": (
                    100.0 * mean_improvement / reference_mean
                    if better == "lower" and reference_mean != 0.0
                    else None
                ),
                "percentage_point_improvement": (
                    100.0 * mean_improvement if better == "higher" else None
                ),
            }
    return comparisons


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_items(
    public_items: list[dict[str, Any]],
    image_vectors: list[np.ndarray],
    feature_matrix: np.ndarray,
    cosine_scores: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            **{key: value for key, value in item.items() if key != "path"},
            "openclip_embedding": vector.tolist(),
            "contextual_features": features.tolist(),
            "query_cosine": float(cosine),
            "simulated_latent_utilities": {
                profile: float(cosine + weights[item["category"]])
                for profile, weights in PROFILES.items()
            },
        }
        for item, vector, features, cosine in zip(
            public_items,
            image_vectors,
            feature_matrix,
            cosine_scores,
            strict=True,
        )
    ]


def main() -> None:
    args = _parse_args()
    seeds = _parse_seeds(args.seeds)
    album = args.album.resolve()
    cache_dir = args.cache_dir.resolve()
    output = args.output.resolve()
    reuse_source = (
        args.reuse_embeddings_from.resolve()
        if args.reuse_embeddings_from is not None
        else None
    )
    album_portable = _project_relative(album, label="--album")
    output_portable = _project_relative(output, label="--output")
    _validate_run_paths(
        cache_dir=cache_dir,
        output=output,
        reuse_embeddings_from=reuse_source,
    )
    public_items, attribution_sha256 = _load_public_items(album)
    if reuse_source is not None:
        image_vectors, query_vector, embedding_runtime = _reuse_embeddings(
            reuse_source, public_items
        )
    else:
        image_vectors, query_vector, embedding_runtime = _embed_offline(
            public_items,
            cache_dir=cache_dir,
            device=args.device,
            batch_size=args.batch_size,
        )

    filenames = [item["filename"] for item in public_items]
    categories = [item["category"] for item in public_items]
    image_matrix = np.vstack(image_vectors)
    cosine_scores = image_matrix @ query_vector
    feature_matrix = np.vstack(
        [contextual_features(vector, query_vector) for vector in image_vectors]
    )
    if feature_matrix.shape != (len(public_items), FEATURE_DIMENSION):
        raise RuntimeError(
            f"unexpected contextual feature shape: {feature_matrix.shape}"
        )

    splits: dict[str, Any] = {}
    runs: list[dict[str, Any]] = []
    started_simulation = time.perf_counter()
    for seed in seeds:
        train_indices, test_indices = _split_indices(categories, seed)
        splits[str(seed)] = {
            "train_filenames": [filenames[index] for index in train_indices],
            "test_filenames": [filenames[index] for index in test_indices],
            "train_count": int(len(train_indices)),
            "test_count": int(len(test_indices)),
            "train_category_counts": {
                category: sum(categories[index] == category for index in train_indices)
                for category in CATEGORIES
            },
            "test_category_counts": {
                category: sum(categories[index] == category for index in test_indices)
                for category in CATEGORIES
            },
            "exact_filename_overlap": [],
        }
        for profile, weights in PROFILES.items():
            true_utilities = _true_utilities(cosine_scores, categories, weights)
            for method in (
                "zero_feedback_cosine",
                "random_pair_contextual",
                "predictive_entropy_contextual",
            ):
                runs.append(
                    _run_learning_method(
                        method=method,
                        seed=seed,
                        profile=profile,
                        train_indices=train_indices,
                        test_indices=test_indices,
                        feature_matrix=feature_matrix,
                        true_utilities=true_utilities,
                        categories=categories,
                        filenames=filenames,
                    )
                )
    simulation_seconds = time.perf_counter() - started_simulation
    summary = _bootstrap_summary(runs, seeds=seeds, resamples=args.bootstrap_resamples)
    paired_comparisons = _paired_final_budget_comparisons(
        summary, resamples=args.bootstrap_resamples
    )
    category_counts = {category: categories.count(category) for category in CATEGORIES}
    result = {
        "schema_version": 1,
        "experiment_id": "contextual-preference-controlled-wikimedia-v1-20260828",
        "claim_boundary": {
            "evidence_type": "controlled semi-synthetic preference-learning test",
            "is_real_human_preference_data": False,
            "supports": (
                "whether the frozen-feature 67D adapter can recover three declared "
                "category-utility simulations on exact-file-disjoint public images"
            ),
            "does_not_support": [
                "real-user preference accuracy or satisfaction",
                "population generalization",
                "identity- or scene-disjoint generalization",
                "OpenCLIP pretraining generalization (the encoder is frozen and may have seen public images)",
                "DPO, SFT, LoRA, or end-to-end foundation-model fine-tuning",
                "statistical significance beyond sensitivity to the fixed split/choice seeds",
            ],
        },
        "assumptions": {
            "public_dataset": (
                f"{len(public_items)} Wikimedia Commons images selected from three "
                "download search terms"
            ),
            "category_labels": "Wikimedia acquisition search terms; proxy metadata, not blind human annotation",
            "category_counts": category_counts,
            "query_text": QUERY_TEXT,
            "user_profiles": PROFILES,
            "latent_utility_formula": "OpenCLIP cosine(image, query) + declared profile category bonus",
            "choice_probability_formula": (
                f"sigmoid((utility_left - utility_right) / {CHOICE_TEMPERATURE})"
            ),
            "choice_temperature": CHOICE_TEMPERATURE,
            "feedback_noise": (
                "one deterministic hash-seeded Bernoulli draw per seed/profile/unordered pair; "
                "the same pair outcome is shared across acquisition methods"
            ),
            "split": {
                "kind": "category-stratified exact-file-disjoint",
                "test_fraction": TEST_FRACTION,
                "seeds": list(seeds),
                "identity_or_scene_deduplication": False,
            },
            "feedback_budgets": list(BUDGETS),
            "training": {
                "model": "67D Bayesian contextual residual utility",
                "feature_schema": FEATURE_SCHEMA,
                "projection_id": PROJECTION_ID,
                "projection_version": PROJECTION_VERSION,
                "prior_lambda": DEFAULT_PRIOR_LAMBDA,
                "optimizer": "production deterministic damped Newton + Armijo MAP/Laplace",
                "frozen_encoder": True,
            },
            "acquisition_methods": {
                "zero_feedback_cosine": "no train feedback; frozen OpenCLIP cosine only",
                "random_pair_contextual": "uniform random unseen training-image pairs",
                "predictive_entropy_contextual": (
                    "maximum binary entropy of the Laplace posterior predictive over unseen "
                    f"pairs using {ENTROPY_APPROXIMATION}"
                ),
            },
            "heldout_pair_metrics": {
                "expected_log_loss": (
                    "cross-entropy between simulator probability and model probability over all "
                    "held-out unordered image pairs"
                ),
                "order_accuracy": (
                    "agreement between predicted and latent-utility pair order over all held-out pairs"
                ),
            },
            "constrained_set_metric": {
                "constraint": (
                    f"select exactly {TARGET_COUNT} held-out images, at most "
                    f"{MAX_PER_CATEGORY} per proxy category"
                ),
                "regret": (
                    "oracle latent-utility sum minus predicted-selection latent-utility sum, "
                    "divided by target count"
                ),
                "solver": "deterministic exact greedy optimizer for a partition matroid",
            },
            "uncertainty_summary": {
                "unit": "split/choice seed after averaging the three fixed user profiles",
                "interval": "nonparametric percentile bootstrap of the mean across seeds",
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": args.bootstrap_resamples,
                "caution": (
                    "overlapping finite-dataset splits are sensitivity replicates, not iid sampled users"
                ),
            },
            "protocol_history": (
                "A two-seed implementation pilot used these same profiles, budgets, "
                "temperature, prior, metrics, and acquisition definitions. No "
                "metric-driven hyperparameter changes were made before the ten-seed run."
            ),
        },
        "provenance": {
            "album_path": album_portable,
            "attribution_json_path": _project_relative(
                album / "ATTRIBUTION.json", label="ATTRIBUTION.json"
            ),
            "attribution_json_sha256": attribution_sha256,
            "path_base": PORTABLE_PATH_BASE,
            "source_files": {
                "benchmark_script": {
                    "path": _project_relative(Path(__file__), label="benchmark script"),
                    "sha256": _sha256_file(Path(__file__).resolve()),
                },
                "contextual_model": {
                    "path": _project_relative(
                        Path(contextual_module.__file__), label="contextual model"
                    ),
                    "sha256": _sha256_file(Path(contextual_module.__file__).resolve()),
                },
            },
            "offline_environment": {
                "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
                "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
                "HF_HUB_DISABLE_XET": os.environ["HF_HUB_DISABLE_XET"],
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "open_clip_torch": _package_version("open-clip-torch"),
                "torch": _package_version("torch"),
            },
            "embedding_runtime": embedding_runtime,
            "simulation_seconds": simulation_seconds,
        },
        "derived_data": {
            "items": _json_items(
                public_items, image_vectors, feature_matrix, cosine_scores
            ),
            "query": {
                "text": QUERY_TEXT,
                "openclip_embedding": query_vector.tolist(),
            },
            "contextual_feature_shape": list(feature_matrix.shape),
            "splits": splits,
        },
        "runs": runs,
        "summary": summary,
        "paired_final_budget_comparisons": paired_comparisons,
        "limitations": [
            "The only user signal is an explicitly programmed category bonus; no person supplied preferences.",
            "Search-term categories can be noisy and correlate with photographer/source artifacts.",
            "Train/test separation is by exact filename, not by photographer, scene, or semantic near-duplicate.",
            "Ten overlapping splits and three fixed profiles are a controlled stress test, not a population sample.",
            "The encoder remains frozen; this experiment evaluates only the small contextual preference adapter.",
            "Predictive-entropy acquisition is evaluated, not the production PDRR approximation.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    compact = {
        "output": output_portable,
        "images": len(public_items),
        "category_counts": category_counts,
        "seeds": len(seeds),
        "profiles": len(PROFILES),
        "runs": len(runs),
        "simulation_seconds": simulation_seconds,
        "budget_60": {
            method: {
                metric: values["mean"]
                for metric, values in budgets[str(max(BUDGETS))].items()
            }
            for method, budgets in summary.items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
