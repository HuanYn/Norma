from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ai.preferences.memory import MemoryCitation, MemoryRerankResult
from ai.schemas import SelectionReplacementRequest, SelectionRequest
from ai.selection import replacement as replacement_module
from ai.selection.replacement import ReplacementService
from ai.selection.service import SelectionService
from ai.tests.test_contextual_decision_runtime import DecisionOpenClipProvider, _album
from ai.tests.test_person_selection import _named_album
from ai.tests.test_selection import FakeSelectionProvider


def persist(database, response):
    with database.connect() as connection:
        connection.execute(
            "UPDATE selections SET result_json=? WHERE id=?",
            (response.model_dump_json(), response.selection_id),
        )


def original(database, album_id, *, prompt="pick 2 photos of city", **kwargs):
    result = SelectionService(database, DecisionOpenClipProvider()).select(
        SelectionRequest(album_id=album_id, prompt=prompt, user_id="demo-user", **kwargs),
        intent_provenance={"mode": "fixture-structured-intent", "hard_constraints_verified": True},
    )
    result.preference_memory = MemoryRerankResult(
        enabled=True, applied=True,
        deltas={photo.photo_id: 0.15 for photo in result.selected},
        event_ids=["old-memory-snapshot"],
    ).as_dict()
    for photo in result.selected:
        photo.total_score = 99.0
        photo.memory_delta = 0.15
    persist(database, result)
    return result


def mock_memory(monkeypatch, deltas):
    calls = []

    class FakeMemory:
        def __init__(self, database, provider, *, enabled):
            assert enabled is True

        def rerank(self, user_id, query_vector, candidate_vectors):
            calls.append((user_id, query_vector.copy(), dict(candidate_vectors)))
            values = {photo_id: deltas.get(photo_id, 0.0) for photo_id in candidate_vectors}
            return MemoryRerankResult(
                enabled=True, applied=any(values.values()), deltas=values,
                citations={photo_id: [MemoryCitation("new-event", "interactive_feedback", 0.9, value)]
                           for photo_id, value in values.items() if value},
                event_ids=["new-event"], warnings=["fixture-memory-snapshot"],
            )

    monkeypatch.setattr(replacement_module, "MemoryReranker", FakeMemory)
    return calls


def test_memory_replacement_reranks_once_and_recomputes_locked_snapshot(tmp_path, monkeypatch):
    database, album_id, ids = _album(tmp_path)
    initial = original(database, album_id)
    calls = mock_memory(monkeypatch, {ids["a"]: 0.10, ids["c"]: -0.15, ids["d"]: 0.15})
    result = ReplacementService(database, DecisionOpenClipProvider()).replace(
        initial.selection_id, SelectionReplacementRequest(remove_photo_id=ids["b"])
    )
    assert result.feasible
    assert result.replacement.photo_id == ids["d"]
    assert result.replacement.total_score == pytest.approx(0.55)
    assert result.replacement.memory_delta == 0.15
    locked = next(photo for photo in result.updated_selection.selected if photo.photo_id == ids["a"])
    assert locked.total_score == pytest.approx(0.92)
    assert locked.memory_delta == 0.10
    assert any("no model training" in reason for reason in locked.reasons)
    assert len(calls) == 1
    assert calls[0][0] == "demo-user"
    assert set(calls[0][2]) == {ids["a"], ids["c"], ids["d"]}
    assert np.isclose(calls[0][2][ids["d"]][0], 0.4)
    updated = result.updated_selection
    assert updated.intent_provenance == initial.intent_provenance
    assert updated.preference_memory["event_ids"] == ["new-event"]
    assert updated.preference_memory["citations"][ids["d"]][0]["event_id"] == "new-event"
    assert updated.preference_model_id is None
    assert updated.preference_comparisons == 0
    assert SelectionService(database, DecisionOpenClipProvider()).get(updated.selection_id) == updated


@pytest.mark.parametrize("exclusion", ["reject", "quality", "group", "subset"])
def test_memory_cannot_reintroduce_hard_excluded_candidates(tmp_path, monkeypatch, exclusion):
    database, album_id, ids = _album(tmp_path)
    with database.connect() as connection:
        connection.execute("UPDATE photos SET quality_score=90 WHERE id=?", (ids["a"],))
    kwargs = {"subset_photo_ids": [ids["a"], ids["b"], ids["d"]]} if exclusion == "subset" else {}
    initial = original(database, album_id, prompt="pick 2 photos of city quality at least 50", **kwargs)
    with database.connect() as connection:
        if exclusion == "reject":
            connection.execute("UPDATE photos SET auto_reject=1 WHERE id=?", (ids["c"],))
        elif exclusion == "quality":
            connection.execute("UPDATE photos SET quality_score=10 WHERE id=?", (ids["c"],))
        elif exclusion == "group":
            connection.execute("UPDATE photos SET similarity_group='locked-group' WHERE id IN (?,?)", (ids["a"], ids["c"]))
    calls = mock_memory(monkeypatch, {ids["c"]: 0.15, ids["d"]: -0.15})
    result = ReplacementService(database, DecisionOpenClipProvider()).replace(
        initial.selection_id, SelectionReplacementRequest(remove_photo_id=ids["b"])
    )
    assert result.feasible
    assert result.replacement.photo_id == ids["d"]
    assert set(calls[0][2]) == {ids["a"], ids["d"]}
    assert result.updated_selection.constraints == initial.constraints
    assert result.updated_selection.subset_photo_ids == initial.subset_photo_ids


def test_person_minimum_survives_memory_replacement(tmp_path, monkeypatch):
    database, album_id, ids, _, _ = _named_album(tmp_path)
    service = SelectionService(database, FakeSelectionProvider())
    initial = service.select(SelectionRequest(album_id=album_id, prompt="选2张夜景，至少2张有我"))
    initial.preference_memory = MemoryRerankResult(enabled=True, deltas={}).as_dict()
    persist(database, initial)
    chosen_ids = {photo.photo_id for photo in initial.selected}
    remove_id = next(photo_id for photo_id in chosen_ids if photo_id != ids["c"])
    calls = mock_memory(monkeypatch, {ids["a"]: 0.15, ids["b"]: 0.15})
    result = ReplacementService(database, FakeSelectionProvider()).replace(
        initial.selection_id, SelectionReplacementRequest(remove_photo_id=remove_id)
    )
    assert result.feasible
    assert result.updated_selection.constraints.person_minimums == {"me": 2}
    assert all(photo.photo_id in {ids["c"], ids["e"], ids["f"]} for photo in result.updated_selection.selected)
    assert ids["a"] not in calls[0][2] and ids["b"] not in calls[0][2]


@pytest.mark.parametrize("unknown", [False, True])
def test_provider_drift_preserves_enabled_trace_but_zeroes_memory(tmp_path, monkeypatch, unknown):
    database, album_id, ids = _album(tmp_path)
    initial = original(database, album_id)
    provider = DecisionOpenClipProvider()
    if unknown:
        initial.provider_fingerprint = None
        persist(database, initial)
    else:
        provider.name = "openclip-replacement-new-512d-v1"
        with database.connect() as connection:
            connection.execute("UPDATE photos SET embedding_provider=?", (provider.name,))
    calls = mock_memory(monkeypatch, {ids["d"]: 0.15})
    result = ReplacementService(database, provider).replace(
        initial.selection_id, SelectionReplacementRequest(remove_photo_id=ids["b"])
    )
    assert result.replacement.photo_id == ids["c"]
    assert calls == []
    memory = result.updated_selection.preference_memory
    assert memory["enabled"] is True and memory["applied"] is False
    assert memory["event_ids"] == [] and memory["citations"] == {}
    assert all(value == 0 for value in memory["deltas"].values())
    assert all(photo.memory_delta == 0 for photo in result.updated_selection.selected)
    assert any("not applied" in warning for warning in memory["warnings"])


def test_quality_only_memory_fallback_keeps_opt_in_trace(tmp_path, monkeypatch):
    database, album_id, ids = _album(tmp_path)
    initial = original(database, album_id, prompt="选2张")
    calls = mock_memory(monkeypatch, {})
    result = ReplacementService(database, DecisionOpenClipProvider()).replace(
        initial.selection_id, SelectionReplacementRequest(remove_photo_id=ids["b"])
    )
    assert result.feasible and result.replacement.photo_id == ids["d"]
    assert calls == []
    assert result.updated_selection.preference_memory["enabled"] is True
    assert result.updated_selection.preference_memory["applied"] is False
    assert any("semantic query" in warning for warning in result.updated_selection.warnings)


def test_memory_cannot_mix_with_adaptive_training(tmp_path, monkeypatch):
    database, album_id, ids = _album(tmp_path)
    initial = original(database, album_id)
    calls = mock_memory(monkeypatch, {})
    with pytest.raises(ValueError, match="legacy adaptive"):
        ReplacementService(database, DecisionOpenClipProvider(), preference_mode="adaptive").replace(
            initial.selection_id, SelectionReplacementRequest(remove_photo_id=ids["b"])
        )
    assert calls == []


def test_repeated_replacement_does_not_accumulate_old_memory_delta(tmp_path, monkeypatch):
    database, album_id, ids = _album(tmp_path)
    initial = original(database, album_id)
    mock_memory(monkeypatch, {ids["a"]: 0.1, ids["c"]: -0.15, ids["d"]: 0.15})
    service = ReplacementService(database, DecisionOpenClipProvider())
    first = service.replace(
        initial.selection_id, SelectionReplacementRequest(remove_photo_id=ids["b"])
    )
    assert first.replacement.photo_id == ids["d"]
    calls = mock_memory(monkeypatch, {ids["a"]: 0.1, ids["b"]: -0.15, ids["c"]: 0.15})
    second = service.replace(
        first.replacement_selection_id, SelectionReplacementRequest(remove_photo_id=ids["d"])
    )
    assert second.replacement.photo_id == ids["c"]
    locked = next(photo for photo in second.updated_selection.selected if photo.photo_id == ids["a"])
    assert locked.total_score == pytest.approx(0.92)
    assert locked.memory_delta == pytest.approx(0.1)
    assert len(calls) == 1


def test_unrequested_memory_never_runs(tmp_path, monkeypatch):
    database, album_id, ids = _album(tmp_path)
    initial = SelectionService(database, DecisionOpenClipProvider()).select(
        SelectionRequest(album_id=album_id, prompt="pick 2 photos of city")
    )
    calls = mock_memory(monkeypatch, {ids["d"]: 0.15})
    result = ReplacementService(database, DecisionOpenClipProvider()).replace(
        initial.selection_id, SelectionReplacementRequest(remove_photo_id=ids["b"])
    )
    assert result.replacement.photo_id == ids["c"]
    assert result.updated_selection.preference_memory is None
    assert result.replacement.memory_delta == 0
    assert calls == []


def test_same_stat_changed_pixels_cannot_drive_memory(tmp_path, monkeypatch):
    database, album_id, ids = _album(tmp_path)
    initial = original(database, album_id)
    with database.connect() as connection:
        row = connection.execute("SELECT absolute_path FROM photos WHERE id=?", (ids["d"],)).fetchone()
    source = Path(row["absolute_path"])
    before = source.stat()
    data = bytearray(source.read_bytes())
    data[-10] ^= 1
    source.write_bytes(data)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    calls = mock_memory(monkeypatch, {})
    with pytest.raises(KeyError, match="semantic cache"):
        ReplacementService(database, DecisionOpenClipProvider()).replace(
            initial.selection_id, SelectionReplacementRequest(remove_photo_id=ids["b"])
        )
    assert calls == []
