"""Deterministic contract tests, not pretrained classification benchmarks."""

import numpy as np
import pytest

from ai.workflow import normalize_query
from ai.selection.parser import parse_selection_prompt
from ai.selection.compact import CompactSelectionParser
from ai.selection.optimizer import OptimizationCandidate as C, optimize_collection
from ai.selection.collection import classify, pair_constraints
from ai.tests.test_structured_selection import FakeTextRuntime


@pytest.mark.parametrize(
    "prompt",
    [
        "挑出最有代表性的6张照片，一张人5张风景",
        "挑出最有代表性的6张照片，1张人像、五张风景",
        "选6张，一张人物和5张风景",
        "选六张，一张人物和五张风景",
        "一张人5张风景",
    ],
)
def test_compound_plan_preserves_counts(prompt):
    normalized, intent, defaulted = normalize_query(prompt, 9, 102)
    assert not defaulted and intent.target_count == 6
    assert intent.category_counts == {"portrait": 1, "landscape": 5}
    parsed = CompactSelectionParser(
        FakeTextRuntime(
            {
                "semantic_query": "旅行照片",
                "style_preferences": [],
                "unsupported_hard_requirements": [],
                "uncertain_requirements": [],
            }
        )
    ).parse(normalized)
    assert parsed.to_selection_intent() == intent


@pytest.mark.parametrize(
    "prompt",
    [
        "选6张，2张人5张风景",
        "选6张，1张人4张风景",
        "选6张，1张人2张人5张风景",
        "选6张，至少1张人5张风景",
        "选6张，选9张",
        "选6张，不要1张人5张风景",
        "挑出最有代表性的6张照片，选9张",
        "选6张，一张人5张建筑",
    ],
)
def test_conflicts_not_silently_defaulted(prompt):
    with pytest.raises(ValueError):
        normalize_query(prompt, 9, 102)


def test_simple_landscape_and_zero_portrait():
    assert normalize_query("挑4张风景", 9, 102)[1].target_count == 4
    assert parse_selection_prompt("选5张，0张人5张风景").category_counts == {
        "portrait": 0,
        "landscape": 5,
    }


def test_quota_and_pair_exclusion_are_joint_constraints():
    candidates = [
        C(i, 1 - i * 0.01, str(i), category=k)
        for i, k in enumerate(
            ["portrait", "portrait", "landscape", "landscape", "landscape", "uncertain"]
        )
    ]
    r = optimize_collection(
        candidates,
        3,
        1,
        category_counts={"portrait": 1, "landscape": 2},
        pair_conflicts=[(2, 3)],
    )
    assert len(r.indices) == 3
    assert sum(candidates[i].category == "portrait" for i in r.indices) == 1
    assert not {2, 3} <= set(r.indices)
    assert 5 not in r.indices
    r = optimize_collection(
        candidates, 5, 1, category_counts={"portrait": 1, "landscape": 4}
    )
    assert not r.indices


def test_soft_diversity_reduces_repeated_high_scoring_views():
    candidates = [C(i, score, str(i)) for i, score in enumerate([0.9, 0.89, 0.85])]
    r = optimize_collection(candidates, 2, 1, pair_penalties=[(0, 1, 0.2)])
    assert set(r.indices) == {0, 2}


def test_required_photos_count_toward_total_and_cannot_bypass_pair_exclusion():
    candidates = [C(i, 1 - .1*i, str(i)) for i in range(5)]
    result = optimize_collection(candidates, 3, 1, required_indices={3, 4})
    assert len(result.indices) == 3 and {3, 4} <= set(result.indices)
    assert not optimize_collection(candidates, 3, 1, pair_conflicts=[(3, 4)], required_indices={3, 4}).indices
    assert not optimize_collection(candidates, 1, 1, required_indices={3, 4}).indices
    assert not optimize_collection(candidates, 3, 1, required_indices={9}).indices


def test_uncertain_is_not_a_probability_and_pairs_are_not_transitive():
    evidence = classify(
        np.array([1.0, 0.0, 0.0]),
        {
            "portrait": np.array([0.5, 0.5, 0.0]),
            "landscape": np.array([0.501, 0.5, 0.0]),
            "other": np.array([0.0, 0.0, 1.0]),
        },
    )
    assert evidence["label"] == "uncertain"
    assert "probability" not in evidence
    angles = np.array([0.0, 0.2, 0.4])
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    conflicts, _ = pair_constraints(vectors)
    assert conflicts == [(0, 1), (1, 2)]  # A and C may coexist.
