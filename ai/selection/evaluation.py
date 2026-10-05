"""Frozen pairwise holdout scoring, independent of selection/model internals.

Accepts externally supplied labels and score predictions. It never generates
labels from the model under test, edits labels or tunes ranking weights.
"""

from collections import defaultdict
import math

import numpy as np


def evaluate(document: dict, predictions: dict, *, seed=42, resamples=2000) -> dict:
    if document.get("schema") != "norma-independent-pairs-v1":
        raise ValueError("Unsupported evaluation schema")
    if document.get("labels_frozen_before_predictions") is not True:
        raise ValueError("Freeze labels before evaluating predictions")
    if not document.get("label_manifest_sha256") or not document.get("method_revision"):
        raise ValueError("Label digest and frozen method revision are required")
    cases = document["cases"]
    if not cases or not 100 <= resamples <= 10000:
        raise ValueError("No test cases or invalid bootstrap budget")
    if not all(
        isinstance(document.get(key), list)
        for key in ("development_and_memory_sha256", "development_and_memory_groups")
    ):
        raise ValueError(
            "Explicit development/memory inventory is required, even when empty"
        )
    memory_hashes = set(document["development_and_memory_sha256"])
    memory_groups = set(document["development_and_memory_groups"])
    image_groups = {}
    ids, groups = set(), defaultdict(list)
    for i, case in enumerate(cases):
        if case["id"] in ids or not case.get("group"):
            raise ValueError("Case IDs must be unique and source groups explicit")
        ids.add(case["id"])
        if case["winner"] not in {"left", "right", "tie"}:
            raise ValueError("Invalid independent preference label")
        digests = [case["left_sha256"], case["right_sha256"]]
        if (
            any(
                len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
                for h in digests
            )
            or digests[0] == digests[1]
        ):
            raise ValueError("Two distinct source SHA256s are required")
        if memory_hashes.intersection(digests) or case["group"] in memory_groups:
            raise ValueError(
                "Holdout overlaps development/memory by source hash or group"
            )
        groups[case["group"]].append(i)
        for digest in digests:
            if digest in image_groups and image_groups[digest] != case["group"]:
                raise ValueError(
                    "The same photo cannot be treated as independent source groups"
                )
            image_groups[digest] = case["group"]
    if "baseline" not in predictions or len(predictions) < 2:
        raise ValueError("Baseline and at least one method required")
    outcomes = {}
    for method, scores in predictions.items():
        if set(scores) != ids:
            raise ValueError("Every method must cover the exact same frozen case IDs")
        values = []
        for case in cases:
            pair = scores[case["id"]]
            if len(pair) != 2 or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in pair
            ):
                raise ValueError("Predictions must be two finite numeric scores")
            delta = pair[0] - pair[1]
            choice = "tie" if abs(delta) <= 1e-9 else "left" if delta > 0 else "right"
            # Exact three-way agreement. Ties are not silently counted as wins.
            values.append(float(choice == case["winner"]))
        outcomes[method] = np.array(values)
    rng = np.random.default_rng(seed)
    ordered_groups = sorted(groups)
    samples = [
        np.concatenate(
            [
                groups[ordered_groups[i]]
                for i in rng.integers(0, len(groups), len(groups))
            ]
        )
        for _ in range(resamples)
    ]
    baseline = float(outcomes["baseline"].mean())
    results = {}
    for method, values in outcomes.items():
        mean = float(values.mean())
        difference = values - outcomes["baseline"]
        ci = (
            None
            if len(groups) < 2
            else np.quantile(
                [difference[s].mean() for s in samples], [0.025, 0.975]
            ).tolist()
        )
        results[method] = {
            "agreement": mean,
            "delta_percentage_points": 100 * (mean - baseline),
            "relative_change_percent": 100 * (mean - baseline) / baseline
            if baseline
            else None,
            "paired_group_bootstrap_delta_ci95": ci,
        }
    human = (
        document.get("label_provenance") == "independent-human"
        and document.get("human_observed") is True
    )
    return {
        "schema": "norma-independent-pairs-result-v1",
        "cases": len(cases),
        "independent_groups": len(groups),
        "metric": "exact three-way pair agreement; source-group bootstrap",
        "labels": document.get("label_provenance", "unknown"),
        "human_observed": human,
        "claim_boundary": "independent human agreement on this fixed dataset only"
        if human
        else "proxy/controlled test only; NOT human preference improvement",
        "pilot": len(groups) < 30,
        "seed": seed,
        "bootstrap_resamples": resamples,
        "method_revision": document["method_revision"],
        "label_manifest_sha256": document["label_manifest_sha256"],
        "methods": results,
    }
