from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

# The matrices in this benchmark are small.  One BLAS thread per seed worker
# avoids oversubscription when the ten seeds run in separate processes.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import ai.preferences.acquisition as acquisition_module
import ai.preferences.contextual as contextual_module
from ai.preferences.acquisition import (
    ACQUISITION_VERSION,
    CONSTRAINT_SOLVER,
    DEFAULT_POSTERIOR_SAMPLES,
    DEFAULT_SHORTLIST_SIZE,
    AcquisitionCandidate,
    AcquisitionNumericalError,
    suggest_pair,
)
from ai.preferences.contextual import (
    COSINE_FEATURE_INDEX,
    DEFAULT_PRIOR_LAMBDA,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    PROJECTION_ID,
    PROJECTION_VERSION,
    ContextualPreferencePosterior,
    contextual_features,
    make_event,
    projection_matrix,
    train,
)


EXPECTED_SOURCE_SHA256 = (
    "16bfdde5c61fc6dca02d19676a441fd37b265effb9bd0631dc4947ad5bb2cdbc"
)
EXPECTED_SOURCE_EXPERIMENT = "contextual-preference-controlled-wikimedia-v1-20260828"
EXPECTED_PROVIDER = "openclip-xlm-roberta-base-vit-b-32-laion5b-raw-v2"
EXPECTED_QUERY = "精选旅行摄影作品集"
EXPECTED_IMAGE_COUNT = 70
EXPECTED_TRAIN_COUNT = 42
EXPECTED_TEST_COUNT = 28
EXPECTED_SEEDS = tuple(range(10))
CATEGORIES = (
    "travel architecture",
    "city night photography",
    "mountain travel landscape",
)
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
METHODS = (
    "zero_feedback_cosine",
    "random_pair_contextual",
    "predictive_entropy_contextual",
    "pdrr_mc_contextual",
)
FULL_BUDGETS = (0, 10, 30, 60)
CHOICE_TEMPERATURE = 0.55
TARGET_COUNT = 6
MAX_PER_CATEGORY = 3
BOOTSTRAP_SEED = 20_260_829
ENTROPY_APPROXIMATION = "logistic-normal-sigmoid-moment-pi-over-8"
PDRR_RETRY_SAMPLES = 128
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_PATH_BASE = {
    "id": "norma-repository-root",
    "anchor": ".",
    "path_style": "posix",
    "resolver": "benchmark-script-parents-1",
}


@dataclass(slots=True)
class ExperimentData:
    source_path: Path
    source_sha256: str
    filenames: tuple[str, ...]
    categories: tuple[str, ...]
    feature_matrix: np.ndarray
    cosine_scores: np.ndarray
    splits: dict[int, tuple[np.ndarray, np.ndarray]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled evaluation of production CAPU-PDRR-MC against random "
            "and predictive-entropy pair acquisition"
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("figures/contextual_preference_controlled_20260828.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("pilot", "full"), default="full")
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in EXPECTED_SEEDS)
    )
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--max-budget", type=int, choices=(10, 30, 60), default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--pilot-result",
        type=Path,
        help="optional completed pilot JSON recorded as immutable run provenance",
    )
    return parser.parse_args()


def _parse_ints(raw: str, *, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must contain distinct values")
    return values


def _parse_names(raw: str, *, allowed: Sequence[str], label: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or len(values) != len(set(values)) or unknown:
        raise ValueError(f"invalid {label}: duplicates or unknown names {unknown}")
    return values


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


def _resolve_project_path(raw: Any, *, label: str) -> Path:
    """Resolve a declared POSIX path without allowing escape from the clone."""

    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    portable = PurePosixPath(raw)
    if portable.is_absolute() or "\\" in raw or ":" in raw or ".." in portable.parts:
        raise ValueError(f"{label} must be a repository-relative POSIX path")
    resolved = PROJECT_ROOT.joinpath(*portable.parts).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{label} escapes the Norma repository")
    return resolved


def _validate_portable_path_base(payload: dict[str, Any], *, label: str) -> None:
    if payload.get("provenance", {}).get("path_base") != PORTABLE_PATH_BASE:
        raise ValueError(
            f"{label} must declare the supported repository-relative POSIX path base"
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _validate_vector(value: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != shape or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{label} must be finite with shape {shape}; got {vector.shape}"
        )
    return vector


def _same_path(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    if left.exists() and right.exists():
        try:
            return left.samefile(right)
        except OSError:
            pass
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _validate_source(
    source: Path, output: Path
) -> tuple[ExperimentData, dict[str, Any]]:
    source = source.resolve()
    output = output.resolve()
    if _same_path(source, output):
        raise ValueError(
            "--output must not overwrite the read-only embedding/split source"
        )
    source_sha_before = _sha256_file(source)
    if source_sha_before != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "source SHA-256 drifted: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha_before}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    source_sha_after_read = _sha256_file(source)
    if source_sha_after_read != source_sha_before:
        raise RuntimeError("source changed while it was being read")
    if payload.get("experiment_id") != EXPECTED_SOURCE_EXPERIMENT:
        raise ValueError("unexpected source experiment_id")
    _validate_portable_path_base(payload, label="source result")

    assumptions = payload.get("assumptions", {})
    training = assumptions.get("training", {})
    checks = {
        "query": assumptions.get("query_text") == EXPECTED_QUERY,
        "provider": payload.get("provenance", {})
        .get("embedding_runtime", {})
        .get("provider_name")
        == EXPECTED_PROVIDER,
        "feature_schema": training.get("feature_schema") == FEATURE_SCHEMA,
        "projection_id": training.get("projection_id") == PROJECTION_ID,
        "projection_version": training.get("projection_version") == PROJECTION_VERSION,
        "prior_lambda": math.isclose(
            float(training.get("prior_lambda", math.nan)),
            DEFAULT_PRIOR_LAMBDA,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"source semantic contract failed: {failed}")

    query_record = payload["derived_data"]["query"]
    if query_record.get("text") != EXPECTED_QUERY:
        raise ValueError("derived query text does not match the declared query")
    query = _validate_vector(
        query_record["openclip_embedding"], shape=(512,), label="query embedding"
    )
    query_norm = float(np.linalg.norm(query))
    if not math.isclose(query_norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"query embedding is not unit normalized: {query_norm}")

    items = payload["derived_data"]["items"]
    if len(items) != EXPECTED_IMAGE_COUNT:
        raise ValueError(f"expected 70 items, got {len(items)}")
    filenames = tuple(str(item["filename"]) for item in items)
    if len(filenames) != len(set(filenames)) or tuple(sorted(filenames)) != filenames:
        raise ValueError("source filenames must be unique and sorted")
    categories = tuple(str(item["category"]) for item in items)
    if set(categories) != set(CATEGORIES):
        raise ValueError("source proxy categories drifted")

    provenance = payload["provenance"]
    album = _resolve_project_path(
        provenance["album_path"], label="source provenance album_path"
    )
    attribution_path = _resolve_project_path(
        provenance["attribution_json_path"],
        label="source provenance attribution_json_path",
    )
    if attribution_path != (album / "ATTRIBUTION.json").resolve():
        raise ValueError("source attribution path is inconsistent with album_path")
    attribution_sha256 = _sha256_file(attribution_path)
    if attribution_sha256 != payload["provenance"]["attribution_json_sha256"]:
        raise ValueError("current ATTRIBUTION.json does not match the source manifest")

    features: list[np.ndarray] = []
    cosine_scores: list[float] = []
    file_checks: list[dict[str, Any]] = []
    max_feature_abs_error = 0.0
    max_cosine_abs_error = 0.0
    max_embedding_norm_error = 0.0
    manifest_digest = hashlib.sha256()
    for item in items:
        filename = str(item["filename"])
        expected_file_sha = str(item["file_sha256"])
        image_path = album / filename
        current_file_sha = _sha256_file(image_path)
        matched = current_file_sha == expected_file_sha
        file_checks.append(
            {
                "filename": filename,
                "expected_sha256": expected_file_sha,
                "current_sha256": current_file_sha,
                "matched": matched,
            }
        )
        if not matched:
            raise ValueError(f"public image SHA-256 mismatch: {filename}")
        manifest_digest.update(filename.encode("utf-8"))
        manifest_digest.update(b"\x00")
        manifest_digest.update(current_file_sha.encode("ascii"))
        manifest_digest.update(b"\n")

        embedding = _validate_vector(
            item["openclip_embedding"], shape=(512,), label=f"embedding {filename}"
        )
        norm_error = abs(float(np.linalg.norm(embedding)) - 1.0)
        max_embedding_norm_error = max(max_embedding_norm_error, norm_error)
        if norm_error > 1e-4:
            raise ValueError(f"embedding is not unit normalized: {filename}")
        recorded_features = _validate_vector(
            item["contextual_features"],
            shape=(FEATURE_DIMENSION,),
            label=f"features {filename}",
        )
        recomputed_features = contextual_features(embedding, query)
        feature_error = float(np.max(np.abs(recorded_features - recomputed_features)))
        max_feature_abs_error = max(max_feature_abs_error, feature_error)
        if feature_error > 1e-12:
            raise ValueError(
                f"67D contextual feature mismatch for {filename}: {feature_error}"
            )
        recorded_cosine = float(item["query_cosine"])
        recomputed_cosine = float(np.dot(embedding, query))
        cosine_error = abs(recorded_cosine - recomputed_cosine)
        max_cosine_abs_error = max(max_cosine_abs_error, cosine_error)
        if cosine_error > 1e-12:
            raise ValueError(f"query cosine mismatch for {filename}: {cosine_error}")
        for profile, weights in PROFILES.items():
            expected_utility = recomputed_cosine + weights[str(item["category"])]
            recorded_utility = float(item["simulated_latent_utilities"][profile])
            if not math.isclose(
                expected_utility, recorded_utility, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"source simulated utility mismatch for {filename}/{profile}"
                )
        features.append(recorded_features)
        cosine_scores.append(recorded_cosine)

    feature_matrix = np.vstack(features)
    cosine_array = np.asarray(cosine_scores, dtype=np.float64)
    if payload["derived_data"].get("contextual_feature_shape") != [
        EXPECTED_IMAGE_COUNT,
        FEATURE_DIMENSION,
    ]:
        raise ValueError("declared contextual feature shape drifted")

    index_by_filename = {filename: index for index, filename in enumerate(filenames)}
    raw_splits = payload["derived_data"]["splits"]
    if set(raw_splits) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("source must contain exactly the fixed ten splits")
    splits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    split_audit: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        raw = raw_splits[str(seed)]
        train_names = tuple(str(name) for name in raw["train_filenames"])
        test_names = tuple(str(name) for name in raw["test_filenames"])
        train_ids = set(train_names)
        test_ids = set(test_names)
        if (
            len(train_names) != EXPECTED_TRAIN_COUNT
            or len(test_names) != EXPECTED_TEST_COUNT
            or train_ids & test_ids
            or train_ids | test_ids != set(filenames)
        ):
            raise ValueError(f"invalid exact-file split for seed {seed}")
        train_indices = np.asarray(
            sorted(index_by_filename[name] for name in train_names), dtype=np.int64
        )
        test_indices = np.asarray(
            sorted(index_by_filename[name] for name in test_names), dtype=np.int64
        )
        splits[seed] = (train_indices, test_indices)
        split_audit[str(seed)] = {
            "train_count": len(train_names),
            "test_count": len(test_names),
            "exact_filename_overlap": sorted(train_ids & test_ids),
            "train_manifest_sha256": _sha256_bytes(
                "\n".join(sorted(train_names)).encode("utf-8")
            ),
            "test_manifest_sha256": _sha256_bytes(
                "\n".join(sorted(test_names)).encode("utf-8")
            ),
        }

    projection = projection_matrix()
    source_audit = {
        "source_path": _project_relative(source, label="--source"),
        "source_sha256_expected": EXPECTED_SOURCE_SHA256,
        "source_sha256_before_read": source_sha_before,
        "source_sha256_after_read": source_sha_after_read,
        "source_unchanged_during_read": source_sha_before == source_sha_after_read,
        "source_experiment_id": payload["experiment_id"],
        "semantic_contract": {
            **checks,
            "expected_query": EXPECTED_QUERY,
            "expected_provider": EXPECTED_PROVIDER,
            "production_feature_schema": FEATURE_SCHEMA,
            "production_projection_id": PROJECTION_ID,
            "production_projection_version": PROJECTION_VERSION,
            "production_feature_dimension": FEATURE_DIMENSION,
            "projection_matrix_sha256_float64_c_order": _sha256_bytes(
                np.asarray(projection, dtype=np.float64, order="C").tobytes()
            ),
        },
        "public_data": {
            "album_path": _project_relative(album, label="source album"),
            "attribution_json_path": _project_relative(
                attribution_path, label="source ATTRIBUTION.json"
            ),
            "attribution_json_sha256": attribution_sha256,
            "image_count": len(items),
            "all_file_sha256_match": all(row["matched"] for row in file_checks),
            "image_manifest_sha256": manifest_digest.hexdigest(),
            "file_sha256_verification": file_checks,
        },
        "numeric_recomputation": {
            "query_embedding_l2_norm": query_norm,
            "max_image_embedding_l2_norm_error": max_embedding_norm_error,
            "max_contextual_feature_abs_error": max_feature_abs_error,
            "max_query_cosine_abs_error": max_cosine_abs_error,
            "feature_shape": list(feature_matrix.shape),
        },
        "splits": split_audit,
        "leakage_guard": {
            "acquisition_candidate_source": "train_filenames only",
            "acquisition_candidate_count_per_seed": EXPECTED_TRAIN_COUNT,
            "test_candidate_count_supplied_to_acquisition": 0,
            "latent_utility_fields_supplied_to_acquisition": False,
            "note": (
                "The PDRR helper accepts only posterior, 42 train candidates, "
                "excluded train pairs, constraints, MC controls, and a seed. "
                "Simulator utility is consulted only after a pair is returned."
            ),
        },
    }
    return (
        ExperimentData(
            source_path=source,
            source_sha256=source_sha_before,
            filenames=filenames,
            categories=categories,
            feature_matrix=feature_matrix,
            cosine_scores=cosine_array,
            splits=splits,
        ),
        source_audit,
    )


def _all_pairs(indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (int(indices[left]), int(indices[right]))
            for left in range(len(indices))
            for right in range(left + 1, len(indices))
        ],
        dtype=np.int64,
    )


def _true_utilities(
    cosine_scores: np.ndarray,
    categories: Sequence[str],
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
) -> tuple[bool, float]:
    canonical = tuple(sorted((left_filename, right_filename)))
    uniform = _stable_uniform("choice", seed, profile, *canonical)
    return uniform < probability_left, uniform


def _solve_partition_set(
    utilities: np.ndarray,
    indices: np.ndarray,
    categories: Sequence[str],
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
    raise RuntimeError("exact-K plus per-category cap is infeasible")


def _predict_scores(
    posterior: ContextualPreferencePosterior, feature_matrix: np.ndarray
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
    categories: Sequence[str],
    filenames: Sequence[str],
    query_opportunities: int,
    abstentions: int,
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
            - (1.0 - true_probabilities) * np.log(1.0 - predicted_probabilities)
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
        "query_opportunities": query_opportunities,
        "observed_feedback_count": posterior.comparisons,
        "abstention_count": abstentions,
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


def _random_pair_order(
    available_pairs: np.ndarray, *, seed: int, profile: str
) -> Iterable[tuple[int, int]]:
    rng = np.random.default_rng(_stable_u64("random-pairs", seed, profile))
    for pair_index in rng.permutation(len(available_pairs)):
        yield tuple(int(value) for value in available_pairs[pair_index])


def _select_entropy_pair(
    *,
    posterior: ContextualPreferencePosterior,
    available_pairs: np.ndarray,
    used_pairs: set[tuple[int, int]],
    feature_matrix: np.ndarray,
    filenames: Sequence[str],
) -> tuple[int, int, dict[str, Any]]:
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
    selected = ranked[0]
    pair = tuple(int(value) for value in eligible[selected])
    return (
        pair[0],
        pair[1],
        {
            "predictive_probability_left": float(probabilities[selected]),
            "predictive_entropy": float(entropies[selected]),
            "predictive_variance": float(variances[selected]),
            "eligible_pair_count": int(len(eligible)),
            "approximation": ENTROPY_APPROXIMATION,
        },
    )


def _pdrr_candidates(
    *,
    train_indices: np.ndarray,
    feature_matrix: np.ndarray,
    filenames: Sequence[str],
    categories: Sequence[str],
) -> tuple[AcquisitionCandidate, ...]:
    return tuple(
        AcquisitionCandidate(
            photo_id=filenames[int(index)],
            features=feature_matrix[int(index)],
            group_key=categories[int(index)],
        )
        for index in train_indices
    )


def _select_pdrr_pair(
    *,
    posterior: ContextualPreferencePosterior,
    candidates: tuple[AcquisitionCandidate, ...],
    excluded_pairs: set[tuple[str, str]],
    seed: int,
    profile: str,
    step: int,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Call production acquisition without test indices or simulator utilities."""

    acquisition_seed = _stable_u64("capu-pdrr-mc", seed, profile, step)
    attempts: list[dict[str, Any]] = []
    result = None
    for posterior_samples in (DEFAULT_POSTERIOR_SAMPLES, PDRR_RETRY_SAMPLES):
        started = time.perf_counter()
        try:
            result = suggest_pair(
                posterior,
                candidates,
                target_count=TARGET_COUNT,
                max_per_group=MAX_PER_CATEGORY,
                posterior_samples=posterior_samples,
                seed=acquisition_seed,
                shortlist_size=DEFAULT_SHORTLIST_SIZE,
                exhaustive=False,
                excluded_pairs=excluded_pairs,
            )
        except AcquisitionNumericalError as exc:
            attempts.append(
                {
                    "posterior_samples": posterior_samples,
                    "seconds": time.perf_counter() - started,
                    "status": "all-evaluated-pairs-voi-invariant-failed",
                    "error": str(exc),
                }
            )
            continue
        attempts.append(
            {
                "posterior_samples": posterior_samples,
                "seconds": time.perf_counter() - started,
                "status": "success",
                "error": None,
            }
        )
        break

    base_diagnostics: dict[str, Any] = {
        "production_function": "ai.preferences.acquisition.suggest_pair",
        "acquisition_version": ACQUISITION_VERSION,
        "constraint_solver": CONSTRAINT_SOLVER,
        "mc_default_samples": DEFAULT_POSTERIOR_SAMPLES,
        "mc_retry_samples": PDRR_RETRY_SAMPLES,
        "shortlist_size": DEFAULT_SHORTLIST_SIZE,
        "exhaustive": False,
        "seed": acquisition_seed,
        "attempts": attempts,
        "retried_with_b128": len(attempts) == 2,
        "abstained": result is None,
        "test_candidate_count_supplied": 0,
        "latent_utility_supplied": False,
    }
    if result is None:
        return None, None, base_diagnostics
    if result.version != ACQUISITION_VERSION:
        raise RuntimeError("production acquisition version changed during the run")
    if result.constraint_solver != CONSTRAINT_SOLVER:
        raise RuntimeError("production constraint solver changed during the run")
    if result.suggested.voi_invariant_ok is not True:
        raise RuntimeError("production returned a selected pair with invalid VOI")
    candidate_ids = {candidate.photo_id for candidate in candidates}
    left_id = result.suggested.left_photo_id
    right_id = result.suggested.right_photo_id
    if left_id not in candidate_ids or right_id not in candidate_ids:
        raise RuntimeError("PDRR returned a non-training candidate")
    canonical = tuple(sorted((left_id, right_id)))
    if canonical in excluded_pairs:
        raise RuntimeError("PDRR returned an already queried pair")
    base_diagnostics["result"] = {
        "version": result.version,
        "constraint_solver": result.constraint_solver,
        "current_photo_ids": list(result.current_photo_ids),
        "current_bayes_regret": result.current_bayes_regret,
        "evaluated_pair_count": result.evaluated_pair_count,
        "eligible_pair_count": result.eligible_pair_count,
        "posterior_samples": result.posterior_samples,
        "seed": result.seed,
        "suggested": asdict(result.suggested),
    }
    return left_id, right_id, base_diagnostics


def _run_method(
    *,
    method: str,
    seed: int,
    profile: str,
    budgets: tuple[int, ...],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    data: ExperimentData,
) -> dict[str, Any]:
    started_method = time.perf_counter()
    train_pairs = _all_pairs(train_indices)
    test_pairs = _all_pairs(test_indices)
    true_utilities = _true_utilities(
        data.cosine_scores, data.categories, PROFILES[profile]
    )
    events = []
    trace: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    posterior = train(events, prior_lambda=DEFAULT_PRIOR_LAMBDA)
    checkpoints["0"] = _evaluate_checkpoint(
        posterior=posterior,
        test_indices=test_indices,
        test_pairs=test_pairs,
        feature_matrix=data.feature_matrix,
        true_utilities=true_utilities,
        categories=data.categories,
        filenames=data.filenames,
        query_opportunities=0,
        abstentions=0,
    )
    if method == "zero_feedback_cosine":
        for budget in budgets[1:]:
            checkpoints[str(budget)] = checkpoints["0"]
        return {
            "seed": seed,
            "profile": profile,
            "method": method,
            "train_candidate_count": int(len(train_indices)),
            "test_candidate_count": int(len(test_indices)),
            "checkpoints": checkpoints,
            "feedback_trace": trace,
            "runtime": {
                "total_seconds": time.perf_counter() - started_method,
                "selection_seconds": 0.0,
                "training_seconds": 0.0,
            },
        }

    used_index_pairs: set[tuple[int, int]] = set()
    used_id_pairs: set[tuple[str, str]] = set()
    random_pairs = iter(_random_pair_order(train_pairs, seed=seed, profile=profile))
    candidate_ids = {data.filenames[int(index)] for index in train_indices}
    train_manifest_sha = _sha256_bytes("\n".join(sorted(candidate_ids)).encode("utf-8"))
    candidates = _pdrr_candidates(
        train_indices=train_indices,
        feature_matrix=data.feature_matrix,
        filenames=data.filenames,
        categories=data.categories,
    )
    index_by_filename = {
        filename: index for index, filename in enumerate(data.filenames)
    }
    total_selection_seconds = 0.0
    total_training_seconds = 0.0
    abstentions = 0
    for step in range(1, max(budgets) + 1):
        acquisition: dict[str, Any]
        selection_started = time.perf_counter()
        if method == "random_pair_contextual":
            left, right = next(random_pairs)
            acquisition = {
                "kind": "uniform-random-unseen-train-pair",
                "eligible_pair_count": int(len(train_pairs) - len(used_index_pairs)),
            }
        elif method == "predictive_entropy_contextual":
            left, right, acquisition = _select_entropy_pair(
                posterior=posterior,
                available_pairs=train_pairs,
                used_pairs=used_index_pairs,
                feature_matrix=data.feature_matrix,
                filenames=data.filenames,
            )
            acquisition["kind"] = "predictive-entropy"
        elif method == "pdrr_mc_contextual":
            left_id, right_id, acquisition = _select_pdrr_pair(
                posterior=posterior,
                candidates=candidates,
                excluded_pairs=used_id_pairs,
                seed=seed,
                profile=profile,
                step=step,
            )
            acquisition["kind"] = "production-capu-pdrr-mc"
            acquisition["train_candidate_manifest_sha256"] = train_manifest_sha
            acquisition["train_candidate_count"] = len(candidate_ids)
            if left_id is None or right_id is None:
                selection_seconds = time.perf_counter() - selection_started
                total_selection_seconds += selection_seconds
                abstentions += 1
                trace.append(
                    {
                        "step": step,
                        "status": "abstained-after-b128",
                        "observed_feedback_count": len(events),
                        "selection_seconds": selection_seconds,
                        "training_seconds": 0.0,
                        "acquisition": acquisition,
                    }
                )
                if step in budgets:
                    checkpoints[str(step)] = _evaluate_checkpoint(
                        posterior=posterior,
                        test_indices=test_indices,
                        test_pairs=test_pairs,
                        feature_matrix=data.feature_matrix,
                        true_utilities=true_utilities,
                        categories=data.categories,
                        filenames=data.filenames,
                        query_opportunities=step,
                        abstentions=abstentions,
                    )
                continue
            left = index_by_filename[left_id]
            right = index_by_filename[right_id]
        else:
            raise ValueError(f"unknown method: {method}")
        selection_seconds = time.perf_counter() - selection_started
        total_selection_seconds += selection_seconds

        pair = (min(left, right), max(left, right))
        id_pair = tuple(sorted((data.filenames[left], data.filenames[right])))
        if pair in used_index_pairs or id_pair in used_id_pairs:
            raise RuntimeError(f"duplicate feedback pair selected: {id_pair}")
        if (
            data.filenames[left] not in candidate_ids
            or data.filenames[right] not in candidate_ids
        ):
            raise RuntimeError("an acquisition method selected a held-out image")
        used_index_pairs.add(pair)
        used_id_pairs.add(id_pair)

        probability_left = float(
            _sigmoid(
                (true_utilities[left] - true_utilities[right]) / CHOICE_TEMPERATURE
            )
        )
        left_preferred, uniform_draw = _pair_outcome(
            seed=seed,
            profile=profile,
            left_filename=data.filenames[left],
            right_filename=data.filenames[right],
            probability_left=probability_left,
        )
        preferred, rejected = (left, right) if left_preferred else (right, left)
        events.append(
            make_event(data.feature_matrix[preferred], data.feature_matrix[rejected])
        )
        training_started = time.perf_counter()
        if (
            method in ("predictive_entropy_contextual", "pdrr_mc_contextual")
            or step in budgets
        ):
            posterior = train(events, prior_lambda=DEFAULT_PRIOR_LAMBDA)
        training_seconds = time.perf_counter() - training_started
        total_training_seconds += training_seconds
        trace.append(
            {
                "step": step,
                "status": "feedback-recorded",
                "left": data.filenames[left],
                "right": data.filenames[right],
                "left_category": data.categories[left],
                "right_category": data.categories[right],
                "true_probability_left": probability_left,
                "observed_left_preferred": left_preferred,
                "feedback_uniform_draw": uniform_draw,
                "observed_feedback_count": len(events),
                "selection_seconds": selection_seconds,
                "training_seconds": training_seconds,
                "acquisition": acquisition,
            }
        )
        if step in budgets:
            checkpoints[str(step)] = _evaluate_checkpoint(
                posterior=posterior,
                test_indices=test_indices,
                test_pairs=test_pairs,
                feature_matrix=data.feature_matrix,
                true_utilities=true_utilities,
                categories=data.categories,
                filenames=data.filenames,
                query_opportunities=step,
                abstentions=abstentions,
            )

    return {
        "seed": seed,
        "profile": profile,
        "method": method,
        "train_candidate_count": int(len(train_indices)),
        "test_candidate_count": int(len(test_indices)),
        "train_candidate_manifest_sha256": train_manifest_sha,
        "checkpoints": checkpoints,
        "feedback_trace": trace,
        "runtime": {
            "total_seconds": time.perf_counter() - started_method,
            "selection_seconds": total_selection_seconds,
            "training_seconds": total_training_seconds,
        },
    }


def _run_seed(
    *,
    seed: int,
    profiles: tuple[str, ...],
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
    data: ExperimentData,
    acquisition_sha256: str,
    contextual_sha256: str,
) -> list[dict[str, Any]]:
    if _sha256_file(Path(acquisition_module.__file__).resolve()) != acquisition_sha256:
        raise RuntimeError("acquisition.py changed before a seed worker started")
    if _sha256_file(Path(contextual_module.__file__).resolve()) != contextual_sha256:
        raise RuntimeError("contextual.py changed before a seed worker started")
    train_indices, test_indices = data.splits[seed]
    results = []
    for profile in profiles:
        for method in methods:
            results.append(
                _run_method(
                    method=method,
                    seed=seed,
                    profile=profile,
                    budgets=budgets,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    data=data,
                )
            )
    return results


def _bootstrap_summary(
    runs: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    profiles: tuple[str, ...],
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
    resamples: int,
) -> dict[str, Any]:
    if resamples < 1_000:
        raise ValueError("bootstrap-resamples must be at least 1000")
    metrics = (
        "heldout_pair_expected_log_loss",
        "heldout_pair_order_accuracy",
        "constrained_set_regret_per_photo",
    )
    summary: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        summary[method] = {}
        relevant = [run for run in runs if run["method"] == method]
        for budget_index, budget in enumerate(budgets):
            summary[method][str(budget)] = {}
            for metric_index, metric in enumerate(metrics):
                seed_means = []
                for seed in seeds:
                    values = [
                        float(run["checkpoints"][str(budget)][metric])
                        for run in relevant
                        if run["seed"] == seed and run["profile"] in profiles
                    ]
                    if len(values) != len(profiles):
                        raise RuntimeError(
                            "each seed must contain every requested profile"
                        )
                    seed_means.append(float(np.mean(values)))
                values_array = np.asarray(seed_means, dtype=np.float64)
                if len(values_array) == 1:
                    low = high = float(values_array[0])
                    sample_sd = None
                else:
                    rng = np.random.default_rng(
                        BOOTSTRAP_SEED
                        + 10_000 * method_index
                        + 100 * budget_index
                        + metric_index
                    )
                    indices = rng.integers(
                        0,
                        len(values_array),
                        size=(resamples, len(values_array)),
                        endpoint=False,
                    )
                    bootstrap_means = np.mean(values_array[indices], axis=1)
                    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
                    sample_sd = float(np.std(values_array, ddof=1))
                summary[method][str(budget)][metric] = {
                    "mean": float(np.mean(values_array)),
                    "sample_standard_deviation": sample_sd,
                    "bootstrap_95_percentile_ci": [float(low), float(high)],
                    "raw_seed_means": seed_means,
                    "n_seed_replicates": len(seeds),
                    "profiles_averaged_within_seed": len(profiles),
                    "bootstrap_resamples": resamples if len(values_array) > 1 else 0,
                }
    return summary


def _paired_final_budget_comparisons(
    summary: dict[str, Any], *, methods: tuple[str, ...], budget: int, resamples: int
) -> dict[str, Any]:
    specifications = (
        ("pdrr_vs_zero", "zero_feedback_cosine", "pdrr_mc_contextual"),
        ("pdrr_vs_random", "random_pair_contextual", "pdrr_mc_contextual"),
        (
            "pdrr_vs_entropy",
            "predictive_entropy_contextual",
            "pdrr_mc_contextual",
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
        if reference not in methods or challenger not in methods:
            continue
        comparisons[name] = {
            "budget": budget,
            "reference": reference,
            "challenger": challenger,
            "metrics": {},
        }
        for metric_index, (metric, better) in enumerate(metric_directions.items()):
            reference_values = np.asarray(
                summary[reference][str(budget)][metric]["raw_seed_means"],
                dtype=np.float64,
            )
            challenger_values = np.asarray(
                summary[challenger][str(budget)][metric]["raw_seed_means"],
                dtype=np.float64,
            )
            improvements = (
                reference_values - challenger_values
                if better == "lower"
                else challenger_values - reference_values
            )
            mean_improvement = float(np.mean(improvements))
            if len(improvements) == 1:
                low = high = mean_improvement
                bootstrap_resamples = 0
            else:
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
                bootstrap_resamples = resamples
            reference_mean = float(np.mean(reference_values))
            comparisons[name]["metrics"][metric] = {
                "better_direction": better,
                "positive_means_challenger_is_better": True,
                "paired_mean_improvement": mean_improvement,
                "paired_bootstrap_95_percentile_ci": [float(low), float(high)],
                "supports_challenger_better": bool(low > 0.0),
                "supports_challenger_worse": bool(high < 0.0),
                "interval_crosses_or_touches_zero": bool(low <= 0.0 <= high),
                "raw_seed_improvements": improvements.tolist(),
                "bootstrap_resamples": bootstrap_resamples,
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


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _runtime_and_pdrr_diagnostics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_by_method: dict[str, Any] = {}
    for method in sorted({run["method"] for run in runs}):
        relevant = [run for run in runs if run["method"] == method]
        totals = [float(run["runtime"]["total_seconds"]) for run in relevant]
        selection = [float(run["runtime"]["selection_seconds"]) for run in relevant]
        training = [float(run["runtime"]["training_seconds"]) for run in relevant]
        runtime_by_method[method] = {
            "run_count": len(relevant),
            "total_seconds_sum": float(sum(totals)),
            "total_seconds_mean_per_seed_profile": float(np.mean(totals)),
            "selection_seconds_sum": float(sum(selection)),
            "training_seconds_sum": float(sum(training)),
        }

    pdrr_runs = [run for run in runs if run["method"] == "pdrr_mc_contextual"]
    pdrr_steps = [step for run in pdrr_runs for step in run["feedback_trace"]]
    successes = [step for step in pdrr_steps if step["status"] == "feedback-recorded"]
    abstentions = [step for step in pdrr_steps if step["status"] != "feedback-recorded"]
    retry_steps = [
        step
        for step in pdrr_steps
        if bool(step["acquisition"].get("retried_with_b128"))
    ]
    invariant_errors = sum(
        attempt["status"] == "all-evaluated-pairs-voi-invariant-failed"
        for step in pdrr_steps
        for attempt in step["acquisition"].get("attempts", [])
    )
    selected_scores = [step["acquisition"]["result"]["suggested"] for step in successes]
    ess_values = [
        float(score[key])
        for score in selected_scores
        for key in ("effective_sample_size_left", "effective_sample_size_right")
    ]
    fallback_count = sum(
        bool(score[key])
        for score in selected_scores
        for key in ("laplace_fallback_left", "laplace_fallback_right")
    )
    raw_pdrr = [float(score["raw_pdrr_estimate"]) for score in selected_scores]
    clipped_pdrr = [float(score["pdrr"]) for score in selected_scores]
    step_seconds = [float(step["selection_seconds"]) for step in pdrr_steps]
    selected_voi_violations = sum(
        not bool(score["voi_invariant_ok"]) for score in selected_scores
    )
    return {
        "runtime_by_method": runtime_by_method,
        "pdrr": {
            "query_opportunities": len(pdrr_steps),
            "successful_feedback": len(successes),
            "abstentions_after_b128": len(abstentions),
            "b128_retry_steps": len(retry_steps),
            "observable_all_evaluated_pair_voi_invariant_errors": int(invariant_errors),
            "selected_pair_voi_invariant_violations": int(selected_voi_violations),
            "laplace_fallback_outcomes": int(fallback_count),
            "laplace_fallback_fraction_of_selected_hypothetical_outcomes": (
                float(fallback_count / (2 * len(selected_scores)))
                if selected_scores
                else None
            ),
            "effective_sample_size": {
                "count": len(ess_values),
                "minimum": min(ess_values) if ess_values else None,
                "p05": _percentile(ess_values, 5.0),
                "median": statistics.median(ess_values) if ess_values else None,
                "p95": _percentile(ess_values, 95.0),
                "maximum": max(ess_values) if ess_values else None,
            },
            "raw_pdrr_estimate": {
                "minimum": min(raw_pdrr) if raw_pdrr else None,
                "mean": float(np.mean(raw_pdrr)) if raw_pdrr else None,
                "maximum": max(raw_pdrr) if raw_pdrr else None,
                "negative_within_tolerance_count": sum(
                    value < 0.0 for value in raw_pdrr
                ),
            },
            "clipped_pdrr_mean": float(np.mean(clipped_pdrr)) if clipped_pdrr else None,
            "selection_runtime_seconds": {
                "sum": sum(step_seconds),
                "mean": float(np.mean(step_seconds)) if step_seconds else None,
                "median": statistics.median(step_seconds) if step_seconds else None,
                "p95": _percentile(step_seconds, 95.0),
                "maximum": max(step_seconds) if step_seconds else None,
            },
            "diagnostic_scope": (
                "The public production result exposes the selected pair and raises "
                "only when all evaluated pairs violate the non-negative VOI invariant; "
                "counts of rejected unselected pairs are not observable here."
            ),
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pilot_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _validate_portable_path_base(payload, label="pilot result")
    diagnostics = payload["runtime_and_acquisition_diagnostics"]["pdrr"]
    seconds = float(diagnostics["selection_runtime_seconds"]["sum"])
    opportunities = int(diagnostics["query_opportunities"])
    return {
        "path": _project_relative(resolved, label="--pilot-result"),
        "sha256": _sha256_file(resolved),
        "run_kind": payload.get("run_kind"),
        "pdrr_query_opportunities": opportunities,
        "pdrr_selection_seconds": seconds,
        "seconds_per_pdrr_opportunity": seconds / opportunities,
        "estimated_serial_seconds_for_30_runs_x_60": (
            seconds / opportunities * 30 * 60
        ),
    }


def main() -> None:
    args = _parse_args()
    seeds = _parse_ints(args.seeds, label="--seeds")
    profiles = _parse_names(args.profiles, allowed=tuple(PROFILES), label="--profiles")
    methods = _parse_names(args.methods, allowed=METHODS, label="--methods")
    budgets = tuple(budget for budget in FULL_BUDGETS if budget <= args.max_budget)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if any(seed not in EXPECTED_SEEDS for seed in seeds):
        raise ValueError("only the source's fixed seeds 0..9 are permitted")
    if args.run_kind == "full" and (
        seeds != EXPECTED_SEEDS
        or profiles != tuple(PROFILES)
        or methods != METHODS
        or args.max_budget != 60
    ):
        raise ValueError(
            "full protocol is locked to seeds 0..9, all three profiles, all four "
            "methods, and budgets 0/10/30/60"
        )

    output = args.output.resolve()
    output_portable = _project_relative(output, label="--output")
    data, source_audit = _validate_source(args.source, output)
    acquisition_path = Path(acquisition_module.__file__).resolve()
    contextual_path = Path(contextual_module.__file__).resolve()
    script_path = Path(__file__).resolve()
    acquisition_sha256 = _sha256_file(acquisition_path)
    contextual_sha256 = _sha256_file(contextual_path)
    script_sha256 = _sha256_file(script_path)
    pilot = _pilot_provenance(args.pilot_result)

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    worker_count = min(args.workers, len(seeds))
    if worker_count == 1:
        for seed in seeds:
            seed_started = time.perf_counter()
            runs.extend(
                _run_seed(
                    seed=seed,
                    profiles=profiles,
                    methods=methods,
                    budgets=budgets,
                    data=data,
                    acquisition_sha256=acquisition_sha256,
                    contextual_sha256=contextual_sha256,
                )
            )
            print(
                json.dumps(
                    {
                        "completed_seed": seed,
                        "seconds": time.perf_counter() - seed_started,
                    }
                ),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_seed,
                    seed=seed,
                    profiles=profiles,
                    methods=methods,
                    budgets=budgets,
                    data=data,
                    acquisition_sha256=acquisition_sha256,
                    contextual_sha256=contextual_sha256,
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                seed_runs = future.result()
                runs.extend(seed_runs)
                print(
                    json.dumps({"completed_seed": seed, "run_count": len(seed_runs)}),
                    flush=True,
                )
    simulation_seconds = time.perf_counter() - started
    runs.sort(
        key=lambda run: (run["seed"], run["profile"], METHODS.index(run["method"]))
    )

    if _sha256_file(data.source_path) != data.source_sha256:
        raise RuntimeError("read-only source changed during simulation")
    if _sha256_file(acquisition_path) != acquisition_sha256:
        raise RuntimeError("acquisition.py changed during simulation")
    if _sha256_file(contextual_path) != contextual_sha256:
        raise RuntimeError("contextual.py changed during simulation")
    if _sha256_file(script_path) != script_sha256:
        raise RuntimeError("benchmark script changed during simulation")

    summary = _bootstrap_summary(
        runs,
        seeds=seeds,
        profiles=profiles,
        methods=methods,
        budgets=budgets,
        resamples=args.bootstrap_resamples,
    )
    paired_by_budget = {
        str(budget): _paired_final_budget_comparisons(
            summary,
            methods=methods,
            budget=budget,
            resamples=args.bootstrap_resamples,
        )
        for budget in budgets[1:]
    }
    paired = paired_by_budget[str(max(budgets))]
    diagnostics = _runtime_and_pdrr_diagnostics(runs)
    result = {
        "schema_version": 1,
        "experiment_id": "capu-pdrr-controlled-wikimedia-v1-20260829",
        "run_kind": args.run_kind,
        "claim_boundary": {
            "evidence_type": "controlled semi-synthetic acquisition-policy comparison",
            "supports": (
                "a narrow comparison of production CAPU-PDRR-MC, predictive entropy, "
                "and random unseen-pair acquisition under the fixed simulator and exact-file splits"
            ),
            "does_not_support": [
                "real-user preference accuracy, satisfaction, or population generalization",
                "identity-, photographer-, scene-, or near-duplicate-disjoint generalization",
                "DPO, SFT, LoRA, or end-to-end foundation-model fine-tuning",
                "an exact Bayesian PDRR computation; posterior integration uses Monte Carlo",
                "universal superiority over entropy or random acquisition",
                "iid-user statistical significance; the ten overlapping splits are sensitivity replicates",
            ],
        },
        "protocol": {
            "source": (
                "the fixed validated JSON is the only embedding/split source and remains read-only"
            ),
            "query_text": EXPECTED_QUERY,
            "provider": EXPECTED_PROVIDER,
            "feature_schema": FEATURE_SCHEMA,
            "projection_id": PROJECTION_ID,
            "projection_version": PROJECTION_VERSION,
            "frozen_encoder": True,
            "seeds": list(seeds),
            "profiles": {profile: PROFILES[profile] for profile in profiles},
            "feedback_budgets": list(budgets),
            "budget_semantics": (
                "pair-query opportunities; a PDRR B=128 numerical failure consumes one "
                "opportunity as an explicit abstention and contributes no fabricated feedback"
            ),
            "split": {
                "kind": "source-provided category-stratified exact-file-disjoint",
                "train_images": EXPECTED_TRAIN_COUNT,
                "heldout_images": EXPECTED_TEST_COUNT,
                "acquisition_pool": "train images only",
            },
            "latent_utility_formula": (
                "OpenCLIP query cosine + declared profile category bonus"
            ),
            "choice_probability_formula": (
                f"sigmoid((utility_left - utility_right) / {CHOICE_TEMPERATURE})"
            ),
            "feedback_noise": (
                "one deterministic SHA-256-seeded Bernoulli draw per seed/profile/"
                "unordered pair, shared whenever methods select the same pair"
            ),
            "learner": {
                "model": "67D Bayesian contextual residual utility",
                "prior_lambda": DEFAULT_PRIOR_LAMBDA,
                "optimizer": "production deterministic damped Newton + Armijo MAP/Laplace",
            },
            "acquisition_methods": {
                "zero_feedback_cosine": "frozen OpenCLIP cosine; no feedback",
                "random_pair_contextual": "uniform random unseen train pair",
                "predictive_entropy_contextual": (
                    "maximum binary entropy over unseen train pairs using "
                    f"{ENTROPY_APPROXIMATION}"
                ),
                "pdrr_mc_contextual": {
                    "production_function": "ai.preferences.acquisition.suggest_pair",
                    "description": (
                        "PDRR Monte Carlo estimate with common random numbers and an exact "
                        "partition-constrained re-solve for every hypothetical outcome"
                    ),
                    "posterior_samples_default": DEFAULT_POSTERIOR_SAMPLES,
                    "shortlist_size": DEFAULT_SHORTLIST_SIZE,
                    "exhaustive": False,
                    "numerical_policy": (
                        "on AcquisitionNumericalError retry once with B=128 using the same "
                        "acquisition seed; on a second failure abstain"
                    ),
                    "constraint_solver": CONSTRAINT_SOLVER,
                    "target_count": TARGET_COUNT,
                    "max_per_category": MAX_PER_CATEGORY,
                },
            },
            "evaluation": {
                "heldout_pair_expected_log_loss": (
                    "expected cross-entropy against simulator probabilities over all 378 "
                    "unordered pairs among the 28 held-out images"
                ),
                "heldout_pair_order_accuracy": (
                    "agreement with latent-utility order over all held-out pairs"
                ),
                "constrained_set_regret_per_photo": (
                    "held-out oracle utility minus predicted exact-K utility, divided by 6; "
                    "select exactly 6 with at most 3 per proxy category"
                ),
            },
            "uncertainty": {
                "unit": "split/choice seed after averaging the fixed profiles within seed",
                "interval": "nonparametric paired percentile bootstrap of the seed mean",
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": args.bootstrap_resamples,
            },
        },
        "provenance": {
            "path_base": PORTABLE_PATH_BASE,
            "source_audit": source_audit,
            "source_files": {
                "benchmark_script": {
                    "path": _project_relative(script_path, label="benchmark script"),
                    "sha256": script_sha256,
                },
                "production_acquisition": {
                    "path": _project_relative(
                        acquisition_path, label="production acquisition"
                    ),
                    "sha256": acquisition_sha256,
                    "version": ACQUISITION_VERSION,
                },
                "production_contextual_model": {
                    "path": _project_relative(
                        contextual_path, label="production contextual model"
                    ),
                    "sha256": contextual_sha256,
                },
            },
            "pilot": pilot,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": _package_version("scipy"),
                "workers": worker_count,
                "simulation_wall_seconds": simulation_seconds,
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            },
        },
        "data_manifest": {
            "filenames": list(data.filenames),
            "categories": list(data.categories),
            "query_cosines": data.cosine_scores.tolist(),
            "splits": {
                str(seed): {
                    "train_filenames": [
                        data.filenames[int(i)] for i in data.splits[seed][0]
                    ],
                    "test_filenames": [
                        data.filenames[int(i)] for i in data.splits[seed][1]
                    ],
                }
                for seed in seeds
            },
            "embeddings_and_features": (
                "not duplicated; pinned by source SHA-256 and revalidated numerically"
            ),
        },
        "runs": runs,
        "summary": summary,
        "paired_budget_comparisons": paired_by_budget,
        "paired_final_budget_comparisons": paired,
        "runtime_and_acquisition_diagnostics": diagnostics,
        "limitations": [
            "All preferences are generated by three declared category-bonus simulators, not people.",
            "Proxy categories are Wikimedia search terms and may be noisy.",
            "Train/test separation is by exact filename only; semantic and source overlap may remain.",
            "Ten overlapping splits and three fixed profiles are sensitivity replicates, not an iid population sample.",
            "PDRR uses B=64 Monte Carlo posterior draws and a 16-pair heuristic shortlist; exact re-solve refers only to the constrained action, not exact posterior integration or exhaustive pair search.",
            "The learner is a small 67D Bayesian adapter over frozen OpenCLIP features, not a multimodal foundation model fine-tune.",
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
        "run_kind": args.run_kind,
        "source_sha256": data.source_sha256,
        "seeds": list(seeds),
        "profiles": list(profiles),
        "methods": list(methods),
        "wall_seconds": simulation_seconds,
        "budget": max(budgets),
        "final_means": {
            method: {
                metric: values["mean"]
                for metric, values in summary[method][str(max(budgets))].items()
            }
            for method in methods
        },
        "pdrr_diagnostics": diagnostics["pdrr"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
