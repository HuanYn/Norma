import copy
import pytest
from ai.selection.compact import CompactSelectionParser
from ai.selection.structured import StructuredIntentError, StructuredProviderError
from ai.tests.test_structured_selection import FakeTextRuntime

SOFT = dict(
    semantic_query="雪山",
    style_preferences=[],
    unsupported_hard_requirements=[],
    uncertain_requirements=[],
)


def parse(text, output=None):
    return CompactSelectionParser(
        FakeTextRuntime(copy.deepcopy(SOFT if output is None else output))
    ).parse(text)


def test_compiled_constraints_and_model_semantics():
    r = parse(
        "选9张，至少2张有我，质量至少40，相似组最多2张，雪山偏暖色",
        SOFT | {"style_preferences": ["暖色"]},
    )
    assert r.to_selection_intent().target_count == 9
    assert r.to_selection_intent().person_minimums == {"me": 2}
    assert r.to_selection_intent().min_quality == 40
    assert r.to_selection_intent().max_per_similarity_group == 2
    assert r.semantic_query == "雪山 暖色"
    assert r.provenance.contract_version == "norma-compact-semantics-v4"


@pytest.mark.parametrize(
    "text", ["选4张，不要自拍", "选4张，至少2张建筑", "挑四张雪山", "必须全是建筑"]
)
def test_model_cannot_swallow_unknown_hard_conditions(text):
    assert parse(text).status == "needs_clarification"


@pytest.mark.parametrize(
    "text", ["选3张，选4张", "允许废片，排除废片", "选99张", "至少2张有我，至少3张有我"]
)
def test_conflict_and_range_rejected_before_call(text):
    runtime = FakeTextRuntime(SOFT)
    with pytest.raises(StructuredIntentError):
        CompactSelectionParser(runtime).parse(text)
    assert not runtime.calls


def test_model_schema_and_issues_stay_strict():
    with pytest.raises(StructuredIntentError):
        parse("选3张", SOFT | {"hard_constraints": {"target_count": 4}})
    with pytest.raises(StructuredIntentError):
        parse(
            "选3张",
            SOFT
            | {"uncertain_requirements": [{"text": "不存在的条件", "reason": "不明"}]},
        )
    with pytest.raises(StructuredProviderError):
        parse("选3张", RuntimeError("secret"))


def test_duplicate_json_and_nonfinite_rejected():
    with pytest.raises(StructuredIntentError):
        parse("选3张", '{"semantic_query":"a","semantic_query":"b"}')
    with pytest.raises(StructuredIntentError):
        parse("选3张", '{"semantic_query":NaN}')


def test_unrequested_style_and_template_query_rejected():
    with pytest.raises(StructuredIntentError):
        parse("挑4张雪山照片", SOFT | {"style_preferences": ["暖色"]})
    with pytest.raises(StructuredIntentError):
        parse("选3张", SOFT | {"semantic_query": "想找的照片内容"})


def test_trip_request_rejects_copied_negative_examples():
    prompt = "选9张。挑出最适合这趟旅游展示的9张照片"
    for invented in ["自拍", "必须某类"]:
        with pytest.raises(StructuredIntentError, match="not grounded"):
            parse(
                prompt,
                SOFT
                | {
                    "unsupported_hard_requirements": [
                        {"text": invented, "reason": "model invention"}
                    ]
                },
            )
    result = parse(prompt, SOFT | {"semantic_query": "最适合这趟旅游展示的照片"})
    assert result.status == "ready"
    assert result.to_selection_intent().target_count == 9


def test_trip_use_does_not_require_new_hard_constraints():
    from ai.selection.compact import SYSTEM

    assert "不要自拍" not in SYSTEM and "必须某类" not in SYSTEM
    result = parse(
        "选4张，不要自拍",
        SOFT
        | {
            "unsupported_hard_requirements": [
                {"text": "不要自拍", "reason": "缺少可靠自拍证据"}
            ]
        },
    )
    assert result.status == "needs_clarification"
