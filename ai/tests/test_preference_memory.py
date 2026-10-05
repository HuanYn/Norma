from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pytest

from ai.index.embedding import EmbeddingProvider
from ai.preferences.memory import (
    MEMORY_EVIDENCE_SCHEMA,
    MemoryEvidenceUnavailableError,
    MemoryReranker,
    build_memory_evidence,
)
from ai.preferences.repository import PreferenceEvent, PreferenceRepository
from ai.preferences.service import PreferenceService
from ai.schemas import PairwiseFeedbackRequest
from ai.storage import Database


def _unit(index):
    value = np.zeros(512, dtype=np.float32)
    value[index] = 1
    return value


class FrozenTestProvider(EmbeddingProvider):
    name = "openclip-test-memory-v1"
    dimension = 512

    def embed_image(self, path):
        raise AssertionError("memory must not embed images")

    def embed_text(self, text):
        return _unit(0)


@pytest.fixture
def fixture(tmp_path):
    database = Database(tmp_path / "norma.db")
    database.initialize()
    provider = FrozenTestProvider()
    with database.connect() as connection:
        connection.execute("INSERT INTO albums(id,name,source_path) VALUES ('album','Memory',?)", (str(tmp_path),))
        connection.execute(
            """INSERT INTO selections(id,album_id,raw_prompt,parse_json,result_json)
               VALUES ('selection','album','warm travel','{}',?)""",
            (json.dumps({"user_id":"local", "query_text":"warm travel", "provider_fingerprint":provider.name, "preference_model_id":None}),),
        )
        for photo_id, vector in (("preferred", _unit(1)), ("rejected", -_unit(1))):
            path = tmp_path / f"{photo_id}.jpg"
            path.write_bytes(photo_id.encode())
            vector_path = tmp_path / f"{photo_id}.npy"
            np.save(vector_path, vector, allow_pickle=False)
            stat = path.stat()
            connection.execute(
                """INSERT INTO photos(id,album_id,absolute_path,file_size,source_mtime_ns,
                   embedding_path,embedding_provider,embedding_source_size,
                   embedding_source_mtime_ns,embedding_source_sha256,quality_score,
                   width,height,auto_reject,metadata_json)
                   VALUES (?,'album',?,?,?,?,?,?,?,?,80,640,480,0,'{}')""",
                (photo_id,str(path),stat.st_size,stat.st_mtime_ns,str(vector_path),provider.name,
                 stat.st_size,stat.st_mtime_ns,hashlib.sha256(path.read_bytes()).hexdigest()),
            )
    with database.connect() as connection:
        preferred = connection.execute("SELECT * FROM photos WHERE id='preferred'").fetchone()
        rejected = connection.execute("SELECT * FROM photos WHERE id='rejected'").fetchone()
    evidence = build_memory_evidence(provider, _unit(0), preferred, rejected)
    base = PreferenceEvent(
        id="event-1", user_id="local", album_id="album", selection_id="selection",
        query_text="warm travel", preferred_photo_id="preferred", rejected_photo_id="rejected",
        choice="preferred", provider_fingerprint=provider.name, feature_schema="test-only",
        preferred_features=(1.0,), rejected_features=(-1.0,), base_margin=0.0,
        context={"source":"selection-pairwise-feedback", "memory_evidence":evidence},
        model_id_at_display=None,
    )
    return database, provider, base, tmp_path


def _insert(fixture, **changes):
    database, _, base, _ = fixture
    event = replace(base, **changes)
    PreferenceRepository(database).insert_event(event)
    return event


def _rerank(fixture, **options):
    database, provider, _, _ = fixture
    return MemoryReranker(database, provider, enabled=True, **options).rerank(
        "local", _unit(0), {"winner":_unit(1), "loser":-_unit(1), "neutral":_unit(2)}
    )


def test_disabled_and_no_history_are_exactly_neutral(fixture):
    database, provider, _, _ = fixture
    _insert(fixture)
    default = MemoryReranker(database, provider).rerank("local", _unit(0), {"x":_unit(1)})
    assert default.deltas == {"x":0.0}
    assert default.warnings == ["memory_disabled"]
    assert not default.applied and not default.enabled
    empty = MemoryReranker(database, provider, enabled=True).rerank("another-user", _unit(0), {"x":_unit(1)})
    assert empty.deltas == {"x":0.0} and empty.event_ids == []


def test_evidence_is_content_bound_and_reranking_is_read_only(fixture):
    database, _, base, _ = fixture
    _insert(fixture)
    with database.connect() as connection:
        before = list(connection.iterdump())
    result = _rerank(fixture)
    assert result.deltas["winner"] == pytest.approx(0.15)
    assert result.deltas["loser"] == pytest.approx(-0.15)
    assert result.deltas["neutral"] == 0
    assert result.applied
    assert result.event_ids == [base.id]
    assert result.citations["winner"][0].source == "interactive_feedback"
    assert result.citations["winner"][0].contribution == pytest.approx(0.15)
    assert json.loads(json.dumps(result.as_dict()))["algorithm"] == "frozen-openclip-case-memory-v1"
    with database.connect() as connection:
        assert list(connection.iterdump()) == before


def test_same_user_provider_choice_and_provenance_are_required(fixture):
    _insert(fixture, id="other-user", user_id="other")
    _insert(fixture, id="other-model", provider_fingerprint="openclip-other")
    for choice in ("tie", "skip", "both_bad"):
        _insert(fixture, id=choice, choice=choice)
    context = copy.deepcopy(fixture[2].context)
    context.update(source="assistant_proxy", authorized_by_user=True, human_observed=False)
    _insert(fixture, id="proxy", context=context)
    result = _rerank(fixture)
    assert not result.applied and not result.event_ids
    assert result.reviewed_event_count == 1
    assert result.skipped_counts == {"excluded_provenance":1}
    proxy = _rerank(fixture, allow_proxy=True)
    assert proxy.event_ids == ["proxy"]
    assert proxy.citations["winner"][0].source == "assistant_proxy"


def test_legacy_events_are_not_retroactively_authenticated(fixture):
    _insert(fixture, context={"source":"selection-pairwise-feedback"})
    result = _rerank(fixture)
    assert result.deltas["winner"] == 0
    assert result.skipped_counts["missing_immutable_evidence"] == 1


def test_irrelevant_query_does_not_influence_candidates(fixture):
    context = copy.deepcopy(fixture[2].context)
    context["memory_evidence"]["query_vector"] = _unit(5).tolist()
    _insert(fixture, context=context)
    result = _rerank(fixture)
    assert not result.applied
    assert result.skipped_counts == {"irrelevant_query":1}


@pytest.mark.parametrize("kind", ["source", "embedding", "reindexed", "removed"])
def test_changed_case_evidence_is_excluded(fixture, kind):
    database, _, _, root = fixture
    _insert(fixture)
    path = root / "preferred.jpg"
    stat = path.stat()
    if kind == "source":
        path.write_bytes(b"x" * stat.st_size)
        os.utime(path, ns=(stat.st_atime_ns,stat.st_mtime_ns))
    elif kind == "embedding":
        np.save(root / "preferred.npy", _unit(3), allow_pickle=False)
    elif kind == "reindexed":
        path.write_bytes(b"x" * stat.st_size)
        stat = path.stat()
        with database.connect() as connection:
            connection.execute(
                "UPDATE photos SET source_mtime_ns=?, embedding_source_mtime_ns=?, embedding_source_sha256=? WHERE id='preferred'",
                (stat.st_mtime_ns,stat.st_mtime_ns,hashlib.sha256(path.read_bytes()).hexdigest()),
            )
    else:
        path.unlink()
    result = _rerank(fixture)
    assert not result.applied
    assert result.skipped_counts["stale_or_invalid_source"] == 1


def test_duplicate_cases_do_not_amplify_and_budget_is_bounded(fixture):
    for index in range(10):
        _insert(fixture, id=f"event-{index}")
    result = _rerank(fixture, max_events=4, top_k=2)
    assert result.reviewed_event_count == 4
    assert len(result.event_ids) == 1
    assert result.skipped_counts["duplicate_case"] == 3
    assert "memory_window_truncated" in result.warnings
    assert all(abs(value) <= 0.15 for value in result.deltas.values())
    assert result.as_dict() == _rerank(fixture, max_events=4, top_k=2).as_dict()


def test_weak_single_case_is_attenuated_not_renormalized_to_full_strength(fixture):
    context = copy.deepcopy(fixture[2].context)
    context["memory_evidence"]["query_vector"] = (0.4 * _unit(0) + np.sqrt(0.84) * _unit(2)).tolist()
    _insert(fixture, context=context)
    result = _rerank(fixture)
    assert result.deltas["winner"] == pytest.approx(0.15 * 0.25**2, abs=1e-7)


def test_new_record_only_feedback_captures_evidence_without_training(fixture):
    database, provider, _, _ = fixture
    response = PreferenceService(database, provider).record_pairwise(PairwiseFeedbackRequest(
        album_id="album", selection_id="selection", preferred_photo_id="preferred", rejected_photo_id="rejected"
    ))
    assert response.trained is False and response.contextual_trained is False
    event = PreferenceRepository(database).get_event(response.contextual_event_id)
    assert event.context["memory_evidence"]["schema"] == MEMORY_EVIDENCE_SCHEMA
    assert len(event.context["memory_evidence"]["query_vector"]) == 512
    result = _rerank(fixture)
    assert result.event_ids == [event.id] and result.applied
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM preference_models").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM user_preferences").fetchone()[0] == 0


def test_optional_snapshot_unavailable_does_not_drop_feedback(fixture, monkeypatch):
    from ai.preferences import service as service_module
    database, provider, _, _ = fixture
    def unavailable(*args):
        raise MemoryEvidenceUnavailableError("unsupported_embedding_file_budget")
    monkeypatch.setattr(service_module, "build_memory_evidence", unavailable)
    response = PreferenceService(database, provider).record_pairwise(PairwiseFeedbackRequest(
        album_id="album", selection_id="selection", preferred_photo_id="preferred", rejected_photo_id="rejected"
    ))
    event = PreferenceRepository(database).get_event(response.contextual_event_id)
    assert event.context["memory_evidence_unavailable"] == "unsupported_embedding_file_budget"
    assert not _rerank(fixture).applied


def test_feedback_snapshots_structured_semantic_query_not_raw_prompt(fixture):
    database, provider, _, _ = fixture
    received = []
    def embed(text):
        received.append(text)
        return _unit(4)
    provider.embed_text = embed
    with database.connect() as connection:
        connection.execute("UPDATE selections SET raw_prompt='select 9, at least 2 me', result_json=? WHERE id='selection'", (
            json.dumps({"user_id":"local", "query_text":"warm city architecture", "provider_fingerprint":provider.name}),
        ))
    response = PreferenceService(database, provider).record_pairwise(PairwiseFeedbackRequest(
        album_id="album", selection_id="selection", preferred_photo_id="preferred", rejected_photo_id="rejected"
    ))
    event = PreferenceRepository(database).get_event(response.contextual_event_id)
    assert received == ["warm city architecture"]
    assert event.query_text == "warm city architecture"
    assert event.context["memory_evidence"]["query_vector"] == _unit(4).tolist()


def test_selection_feedback_memory_toggle_preserves_hard_constraints(tmp_path):
    from ai.selection import SelectionService
    from ai.schemas import SelectionRequest
    from ai.tests.test_contextual_decision_runtime import DecisionOpenClipProvider, _album

    database, album_id, ids = _album(tmp_path)
    provider = DecisionOpenClipProvider()
    with database.connect() as connection:
        connection.execute("UPDATE photos SET quality_score=80 WHERE id=?", (ids["a"],))
        connection.execute("UPDATE photos SET quality_score=50 WHERE id=?", (ids["d"],))
        connection.execute("UPDATE photos SET similarity_group='same-scene' WHERE id IN (?,?)", (ids["a"], ids["b"]))
    request = SelectionRequest(
        album_id=album_id,
        prompt="Select 2 photos of city, quality at least 60, max 1 per similarity group",
    )
    selector = SelectionService(database, provider)
    baseline = selector.select(request)
    assert {photo.photo_id for photo in baseline.selected} == {ids["a"], ids["c"]}
    feedback = PreferenceService(database, provider).record_pairwise(PairwiseFeedbackRequest(
        album_id=album_id, selection_id=baseline.selection_id,
        preferred_photo_id=ids["b"], rejected_photo_id=ids["a"],
    ))
    personalized = selector.select(request.model_copy(update={"use_preference_memory":True}))
    assert personalized.feasible
    assert {photo.photo_id for photo in personalized.selected} == {ids["b"], ids["c"]}
    assert personalized.preference_memory["applied"]
    assert personalized.preference_memory["event_ids"] == [feedback.contextual_event_id]
    assert any(photo.memory_delta > 0 for photo in personalized.selected)
    for selection in (baseline, personalized):
        assert len(selection.selected) == 2
        assert all(photo.quality_score >= 60 for photo in selection.selected)
        assert sum(photo.similarity_group == "same-scene" for photo in selection.selected) <= 1
        assert ids["d"] not in {photo.photo_id for photo in selection.selected}
    restored = selector.select(request)
    assert [photo.photo_id for photo in restored.selected] == [photo.photo_id for photo in baseline.selected]
    assert all(photo.memory_delta == 0 for photo in restored.selected)
    assert feedback.trained is False and feedback.contextual_trained is False
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM preference_models").fetchone()[0] == 0


@pytest.mark.parametrize("options", [
    {"max_delta":float("nan")}, {"max_delta":0.2}, {"top_k":0},
    {"max_events":True}, {"min_query_similarity":1.0}, {"enabled":"true"},
])
def test_invalid_budgets_fail_early(fixture, options):
    with pytest.raises(ValueError):
        MemoryReranker(fixture[0], fixture[1], **options)
