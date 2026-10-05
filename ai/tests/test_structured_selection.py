from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import pytest

from ai.config import Settings
from ai.rag.models import VLMInputBudgetError
from ai.rag.transformers_runtime import TransformersQwen3VLRuntime
from ai.selection.parser import SelectionIntent
from ai.selection.structured import (
    CloudTextIntentRuntime,
    StructuredIntentError,
    StructuredProviderError,
    StructuredSelectionParser,
    UnsupportedSelectionRequirements,
    create_structured_parser,
)


def document(**changes):
    result = {
        "hard_constraints": {"target_count": 12, "min_quality": 0.0,
            "exclude_rejects": True, "max_per_similarity_group": 1,
            "person_minimums": {}},
        "semantic_query": "建筑", "style_preferences": ["暖色"],
        "unsupported_hard_requirements": [], "uncertain_requirements": [],
        "constraint_evidence": [],
    }
    result.update(changes)
    return result


class FakeTextRuntime:
    provider_fingerprint = "fake-text-test-v1"

    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate_text_json(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.output, Exception):
            raise self.output
        return copy.deepcopy(self.output)


def parse(output, prompt="暖色建筑"):
    return StructuredSelectionParser(FakeTextRuntime(output)).parse(prompt)


def test_default_values_semantic_text_and_provenance():
    runtime = FakeTextRuntime(document())
    result = StructuredSelectionParser(runtime).parse("暖色建筑")
    assert result.status == "ready"
    assert result.to_selection_intent() == SelectionIntent(12, 0.0, True, 1)
    assert result.semantic_query == "建筑 暖色"
    assert result.provenance.provider_fingerprint == runtime.provider_fingerprint
    assert len(result.provenance.prompt_sha256) == 64
    assert len(result.provenance.output_sha256) == 64
    assert result.provenance.max_new_tokens == 1024
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["temperature"] == 0.0
    assert "images" not in runtime.calls[0]


def test_known_supported_constraints_become_existing_intent():
    value = document()
    value["hard_constraints"] = {"target_count": 9, "min_quality": 30.0,
        "exclude_rejects": False, "max_per_similarity_group": 2,
        "person_minimums": {"我": 2}}
    pieces = [("target_count", "选9张"), ("min_quality", "质量至少30"),
              ("exclude_rejects", "允许废片"),
              ("max_per_similarity_group", "相似组最多2张"),
              ("person_minimums", "至少2张有我")]
    value["constraint_evidence"] = [{"field": field, "source_text": text} for field, text in pieces]
    result = parse(value, "，".join(text for _, text in pieces) + "，暖色建筑")
    assert result.to_selection_intent() == SelectionIntent(9, 30.0, False, 2, {"me": 2})


def test_chinese_numbers_and_simple_synonyms_have_bounded_verification():
    value = document()
    value["hard_constraints"]["target_count"] = 9
    value["hard_constraints"]["person_minimums"] = {"me": 2}
    value["constraint_evidence"] = [
        {"field": "target_count", "source_text": "挑选九张"},
        {"field": "person_minimums", "source_text": "最少两张有我"},
    ]
    assert parse(value, "挑选九张，最少两张有我").to_selection_intent().person_minimums == {"me": 2}


def test_named_person_is_label_only_not_inferred_identity():
    value = document()
    value["hard_constraints"]["person_minimums"] = {"Alex": 2}
    value["constraint_evidence"] = [{"field": "person_minimums", "source_text": '至少2张有“Alex”'}]
    assert parse(value, '至少2张有“Alex”').to_selection_intent().person_minimums == {"alex": 2}


def test_english_command_count_can_be_verified():
    value = document()
    value["hard_constraints"]["target_count"] = 9
    value["constraint_evidence"] = [{"field": "target_count", "source_text": "select 9 photos"}]
    assert parse(value, "select 9 photos").to_selection_intent().target_count == 9


def test_person_minimum_cannot_exceed_total_count():
    value = document()
    value["hard_constraints"]["person_minimums"] = {"me": 13}
    value["constraint_evidence"] = [{"field": "person_minimums", "source_text": "至少13张有我"}]
    assert parse(value, "至少13张有我").status == "needs_clarification"


def test_unsupported_requirement_retained_and_blocks_execution():
    value = document(unsupported_hard_requirements=[{"text": "不要自拍", "reason": "尚无自拍证据"}])
    result = parse(value, "暖色建筑，不要自拍")
    assert result.status == "needs_clarification"
    assert result.document.unsupported_hard_requirements[0].text == "不要自拍"
    with pytest.raises(UnsupportedSelectionRequirements):
        result.to_selection_intent()


@pytest.mark.parametrize("prompt", ["不要自拍", "至少3张建筑", "必须是去年拍摄", "选9张", "允许废片", "only landscape images"])
def test_silent_omission_of_known_hard_requirement_is_blocked(prompt):
    result = parse(document(), prompt)
    assert result.status == "needs_clarification"
    with pytest.raises(UnsupportedSelectionRequirements):
        result.to_selection_intent()


def test_broad_quote_cannot_hide_unrelated_requirement():
    value = document(constraint_evidence=[{"field": "target_count", "source_text": "选12张，不要自拍"}])
    result = parse(value, "选12张，不要自拍")
    assert result.status == "needs_clarification"


def test_observed_qwen_broad_quote_failure_is_still_rejected_after_prompt_fix():
    # Captured from the controlled local smoke run. The prompt improved; the
    # validation contract must not expand to accepting whole-request evidence.
    prompt = "选3张建筑，偏暖色电影感"
    value = document(semantic_query=prompt, style_preferences=["偏暖色电影感"])
    value["hard_constraints"]["target_count"] = 3
    value["constraint_evidence"] = [{"field": "target_count", "source_text": prompt}]
    value["unsupported_hard_requirements"] = [{"text": prompt, "reason": "ambiguous"}]
    with pytest.raises(StructuredIntentError, match="evidence"):
        parse(value, prompt)


def test_narrow_evidence_and_separate_soft_style_for_smoke_prompt():
    value = document(semantic_query="建筑", style_preferences=["暖色", "电影感"])
    value["hard_constraints"]["target_count"] = 3
    value["constraint_evidence"] = [{"field": "target_count", "source_text": "选3张"}]
    result = parse(value, "选3张建筑，偏暖色电影感")
    assert result.to_selection_intent().target_count == 3
    assert result.semantic_query == "建筑 暖色 电影感"


def test_second_real_qwen_output_with_soft_field_evidence_is_not_silently_repaired():
    value = document(semantic_query="建筑", style_preferences=["偏暖色", "电影感"])
    value["hard_constraints"]["target_count"] = 3
    value["constraint_evidence"] = [
        {"field": "target_count", "source_text": "选3张"},
        {"field": "semantic_query", "source_text": "建筑"},
    ]
    with pytest.raises(StructuredIntentError, match="strict intent validation"):
        parse(value, "选3张建筑，偏暖色电影感")


def test_novel_supported_wording_requires_review_not_fabricated_proof():
    value = document(constraint_evidence=[{"field": "target_count", "source_text": "凑够一打照片"}])
    result = parse(value, "凑够一打照片")
    assert result.status == "needs_clarification"


@pytest.mark.parametrize("field,value", [
    ("target_count", 51), ("target_count", True), ("target_count", "12"),
    ("min_quality", -1), ("min_quality", 101), ("min_quality", float("nan")),
    ("exclude_rejects", "false"), ("max_per_similarity_group", 11),
    ("person_minimums", {"me": 0}), ("person_minimums", {"Alex": 2, "alex": 3}),
])
def test_strict_hard_schema(field, value):
    payload = document()
    payload["hard_constraints"][field] = value
    with pytest.raises(StructuredIntentError):
        parse(payload)


def test_fabricated_or_changed_constraint_rejected():
    payload = document()
    payload["hard_constraints"]["target_count"] = 9
    with pytest.raises(StructuredIntentError, match="evidence"):
        parse(payload)
    payload["constraint_evidence"] = [{"field": "target_count", "source_text": "选8张"}]
    with pytest.raises(StructuredIntentError, match="evidence"):
        parse(payload, "选8张")


def test_source_must_be_exact_substring_and_issue_must_be_grounded():
    with pytest.raises(StructuredIntentError, match="grounded"):
        parse(document(constraint_evidence=[{"field": "target_count", "source_text": "选12张"}]))
    with pytest.raises(StructuredIntentError, match="grounded"):
        parse(document(uncertain_requirements=[{"text": "不在原文", "reason": "unknown"}]))


def test_conflicting_count_evidence_rejected():
    value = document(constraint_evidence=[
        {"field": "target_count", "source_text": "选12张"},
        {"field": "target_count", "source_text": "选9张"},
    ])
    with pytest.raises(StructuredIntentError, match="Conflicting"):
        parse(value, "选12张，选9张")


@pytest.mark.parametrize("wrapper", [lambda x: x, lambda x: "```json\n" + x + "\n```"])
def test_only_complete_json_or_complete_json_fence_accepted(wrapper):
    assert parse(wrapper(json.dumps(document(), ensure_ascii=False))).status == "ready"


@pytest.mark.parametrize("payload", [
    '{"semantic_query":"a","semantic_query":"b"}',
    'preface {"a":1}', '```json\n{}\n```\ntrailer', '[]', '{"x":NaN}',
    {"extra": "field"}, document(extra="untrusted"), document(semantic_query=1),
])
def test_malformed_unknown_or_duplicate_output_is_rejected(payload):
    with pytest.raises(StructuredIntentError):
        parse(payload)


def test_input_output_limits_and_path_rejection_happen_without_retry():
    runtime = FakeTextRuntime(document())
    parser = StructuredSelectionParser(runtime)
    for prompt in ["", " ", "x" * (16 * 1024 + 1), r"从 E:\private\photo.jpg 选图"]:
        with pytest.raises(StructuredIntentError):
            parser.parse(prompt)
    assert not runtime.calls
    with pytest.raises(StructuredIntentError):
        parse("x" * (32 * 1024 + 1))


def test_provider_failure_is_body_free_and_no_fallback():
    runtime = FakeTextRuntime(RuntimeError("private-token-response-body"))
    with pytest.raises(StructuredProviderError) as captured:
        StructuredSelectionParser(runtime).parse("暖色建筑")
    assert "private-token" not in str(captured.value)
    assert len(runtime.calls) == 1


def test_cloud_text_adapter_reuses_config_without_uploading_pixels():
    calls = []
    def transport(url, **kwargs):
        calls.append((url, kwargs))
        return json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(document(), ensure_ascii=False),
        }}]}).encode()
    runtime = CloudTextIntentRuntime(
        base_url="https://example.invalid/v1", model="test-model",
        api_key="unit-test-secret", timeout_seconds=4,
        json_response_format=True, thinking_mode="disabled", transport=transport,
    )
    result = StructuredSelectionParser(runtime).parse("暖色建筑")
    assert result.status == "ready"
    assert len(calls) == 1
    url, request = calls[0]
    assert url == "https://example.invalid/v1/chat/completions"
    body = json.loads(request["payload"])
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert "[REDACTED_LOCAL_PATH]" not in body["messages"][0]["content"]
    assert "$ref" not in body["messages"][0]["content"]
    assert isinstance(body["messages"][1]["content"], str)
    assert b"image_url" not in request["payload"]
    assert request["timeout_seconds"] == 4
    assert "unit-test-secret" not in result.provenance.model_dump_json()


@pytest.mark.parametrize("response", [
    b"bad", b"x" * (256 * 1024 + 1),
    json.dumps({"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "{}"}}]}).encode(),
], ids=["invalid-json", "oversized-envelope", "truncated-generation"])
def test_cloud_envelope_must_be_bounded_complete_valid_response(response):
    runtime = CloudTextIntentRuntime(
        base_url="https://example.invalid", model="test-model", api_key="unit-test-secret",
        transport=lambda *args, **kwargs: response,
    )
    with pytest.raises(StructuredProviderError):
        StructuredSelectionParser(runtime).parse("暖色建筑")


def test_cloud_rejects_json_escaped_secret_in_decoded_output():
    secret = "unit-test-secret"
    content = json.dumps(document(semantic_query=secret)).replace(
        secret, "".join("\\u" + format(ord(character), "04x") for character in secret),
    )
    response = json.dumps({"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content,
    }}]}).encode()
    runtime = CloudTextIntentRuntime(
        base_url="https://example.invalid", model="test-model", api_key=secret,
        transport=lambda *args, **kwargs: response,
    )
    with pytest.raises(StructuredProviderError) as captured:
        StructuredSelectionParser(runtime).parse("暖色建筑")
    assert secret not in str(captured.value)


def test_factory_can_inject_no_network_runtime_and_missing_config_fails():
    settings = Settings(host="127.0.0.1", port=8765, data_dir=Path("."), log_level="INFO")
    assert create_structured_parser(settings, runtime=FakeTextRuntime(document())).parse("建筑").status == "ready"
    with pytest.raises(StructuredProviderError, match="not configured"):
        create_structured_parser(settings)


def _loaded_local_runtime(input_length=3):
    calls = []
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(("tokenize", messages))
            return {"input_ids": [list(range(input_length))]}
        def batch_decode(self, value, **kwargs):
            calls.append(("decode", value))
            return ["  {}  "]
    class Model:
        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return [kwargs["input_ids"][0] + [100, 101]]
    runtime = object.__new__(TransformersQwen3VLRuntime)
    runtime._processor = Processor()
    runtime._model = Model()
    runtime._torch = None
    runtime._generation_lock = threading.Lock()
    runtime._assert_model_snapshot_unchanged = lambda: calls.append(("verify", None))
    runtime._ensure_loaded = lambda: calls.append(("load", None))
    return runtime, calls


def test_local_text_method_does_not_fake_images_and_retains_snapshot_checks():
    runtime, calls = _loaded_local_runtime()
    assert runtime.generate_text_json(system_prompt="schema", user_prompt="request", max_new_tokens=1024, temperature=0.0) == "{}"
    assert len([call for call in calls if call[0] == "verify"]) == 2
    messages = next(value for key, value in calls if key == "tokenize")
    assert all(item["type"] == "text" for message in messages for item in message["content"])
    generated = next(value for key, value in calls if key == "generate")
    assert generated["max_new_tokens"] == 1024
    assert generated["do_sample"] is False


def test_local_text_token_budget_stops_before_model_generation():
    runtime, calls = _loaded_local_runtime(input_length=4097)
    with pytest.raises(VLMInputBudgetError, match="4096"):
        runtime.generate_text_json(system_prompt="schema", user_prompt="request", max_new_tokens=1024, temperature=0.0)
    assert not any(key == "generate" for key, _ in calls)


@pytest.mark.parametrize("tokens", [True, 63, 1025, "256"])
def test_local_text_invalid_output_budget_does_not_load(tokens):
    runtime, calls = _loaded_local_runtime()
    with pytest.raises(ValueError):
        runtime.generate_text_json(system_prompt="schema", user_prompt="request", max_new_tokens=tokens, temperature=0.0)
    assert not calls


def test_local_existing_multimodal_method_still_requires_evidence():
    runtime, _ = _loaded_local_runtime()
    with pytest.raises(ValueError, match="at least one evidence image"):
        runtime.generate_json(system_prompt="schema", user_prompt="request", images=(), max_new_tokens=256, temperature=0.0)
