import copy
import pytest

from ai.selection.evaluation import evaluate


def fixture():
    document = dict(
        schema="norma-independent-pairs-v1",
        labels_frozen_before_predictions=True,
        label_manifest_sha256="fixture",
        method_revision="frozen-fixture-not-model",
        label_provenance="controlled-fixture",
        human_observed=False,
        development_and_memory_sha256=["f" * 64],
        development_and_memory_groups=["development"],
        cases=[
            dict(
                id="one",
                group="source-a",
                left_sha256="a" * 64,
                right_sha256="b" * 64,
                winner="left",
            ),
            dict(
                id="two",
                group="source-b",
                left_sha256="c" * 64,
                right_sha256="d" * 64,
                winner="right",
            ),
        ],
    )
    predictions = {
        "baseline": {"one": [1, 0], "two": [1, 0]},
        "fusion": {"one": [1, 0], "two": [0, 1]},
    }
    return document, predictions


def test_paired_report_is_reproducible_and_proxy_not_human():
    document, predictions = fixture()
    report = evaluate(document, predictions, resamples=100)
    assert report == evaluate(document, predictions, resamples=100)
    assert report["methods"]["fusion"]["delta_percentage_points"] == 50
    assert report["methods"]["fusion"]["relative_change_percent"] == 100
    assert not report["human_observed"] and report["pilot"]
    assert "NOT human" in report["claim_boundary"]


@pytest.mark.parametrize(
    "kind", ["hash", "group", "duplicate", "missing", "nonfinite", "unfrozen"]
)
def test_leakage_and_invalid_predictions_rejected(kind):
    document, predictions = fixture()
    if kind == "hash":
        document["cases"][0]["left_sha256"] = "f" * 64
    if kind == "group":
        document["cases"][0]["group"] = "development"
    if kind == "duplicate":
        document["cases"].append(copy.deepcopy(document["cases"][0]))
    if kind == "missing":
        predictions["fusion"].pop("two")
    if kind == "nonfinite":
        predictions["fusion"]["two"] = [float("nan"), 0]
    if kind == "unfrozen":
        document["labels_frozen_before_predictions"] = False
    with pytest.raises(ValueError):
        evaluate(document, predictions, resamples=100)


def test_ties_are_exact_and_zero_baseline_has_no_relative_ratio():
    document, predictions = fixture()
    for case in document["cases"]:
        case["winner"] = "tie"
    predictions["fusion"] = {"one": [1, 1], "two": [1, 1]}
    report = evaluate(document, predictions, resamples=100)
    assert report["methods"]["fusion"]["agreement"] == 1
    assert report["methods"]["fusion"]["relative_change_percent"] is None


def test_missing_inventory_or_same_photo_in_different_groups_fails():
    document, predictions = fixture()
    document.pop("development_and_memory_groups")
    with pytest.raises(ValueError, match="inventory"):
        evaluate(document, predictions, resamples=100)
    document, predictions = fixture()
    document["cases"][1]["left_sha256"] = document["cases"][0]["left_sha256"]
    with pytest.raises(ValueError, match="independent source groups"):
        evaluate(document, predictions, resamples=100)
