"""Controlled score fixtures verify wiring, NOT pretrained-model accuracy."""

from pathlib import Path

import pytest

from ai.aesthetics.provider import AssessmentScores
from ai.aesthetics.service import AestheticsService
from ai.preferences.service import PreferenceService
from ai.schemas import (
    PairwiseFeedbackRequest,
    SelectionRequest,
    SelectionReplacementRequest,
)
from ai.selection.service import SelectionService
from ai.selection.replacement import ReplacementService
from ai.tests.test_contextual_decision_runtime import _album, DecisionOpenClipProvider


class FixtureAssessment:
    fingerprint = "fixture-scores-not-real-musiq"

    def score(self, path):
        return AssessmentScores(
            80, {"a": 2.0, "b": 9.0, "c": 8.0, "d": 7.0}[Path(path).stem]
        )


def setup(tmp_path):
    database, album, ids = _album(tmp_path)
    assessment = AestheticsService(database, FixtureAssessment())
    assessment.analyze(album)
    return database, album, ids, assessment


def test_opt_in_changes_order_with_auditable_scores_and_no_training(tmp_path):
    db, album, ids, assessment = setup(tmp_path)
    service = SelectionService(
        db, DecisionOpenClipProvider(), aesthetics_service=assessment
    )
    before = service.select(
        SelectionRequest(album_id=album, prompt="pick 2 photos of city")
    )
    after = service.select(
        SelectionRequest(
            album_id=album, prompt="pick 2 photos of city", use_learned_quality=True
        )
    )
    assert before.selected[0].photo_id == ids["a"]
    assert after.selected[0].photo_id == ids["b"]
    assert before.learned_quality is None
    assert after.algorithm == "openclip-musiq-fixed-fusion-v1"
    assert after.selected[0].total_score == pytest.approx(
        0.72 * 0.8 + 0.18 * 8 / 9 + 0.10 * 0.8
    )
    assert after.selected[0].learned_quality["aesthetic_quality"] == 9
    assert len(after.learned_quality["snapshot_sha256"]) == 64
    assert all(
        len(i["source_sha256"]) == 64 for i in after.learned_quality["items"].values()
    )
    assert after.preference_comparisons == 0


def test_quality_only_and_hard_gates_are_not_replaced(tmp_path):
    db, album, ids, assessment = setup(tmp_path)
    service = SelectionService(
        db, DecisionOpenClipProvider(), aesthetics_service=assessment
    )
    result = service.select(
        SelectionRequest(
            album_id=album, prompt="选 2 张照片，质量至少 60", use_learned_quality=True
        )
    )
    assert ids["a"] not in [p.photo_id for p in result.selected]
    assert result.selected[0].total_score == pytest.approx(0.65 * 8 / 9 + 0.35 * 0.8)


def test_missing_or_stale_scores_fail_closed(tmp_path):
    db, album, ids, assessment = setup(tmp_path)
    request = SelectionRequest(
        album_id=album, prompt="pick 2 photos of city", use_learned_quality=True
    )
    with pytest.raises(ValueError, match="尚未配置"):
        SelectionService(db, DecisionOpenClipProvider()).select(request)
    with db.connect() as c:
        c.execute("DELETE FROM aesthetics_scores_v1 WHERE photo_id=?", (ids["a"],))
    with pytest.raises(ValueError, match="不完整"):
        SelectionService(
            db, DecisionOpenClipProvider(), aesthetics_service=assessment
        ).select(request)


def test_replacement_preserves_model_fusion_and_checks_snapshot(tmp_path):
    db, album, ids, assessment = setup(tmp_path)
    provider = DecisionOpenClipProvider()
    first = SelectionService(db, provider, aesthetics_service=assessment).select(
        SelectionRequest(
            album_id=album, prompt="pick 2 photos of city", use_learned_quality=True
        )
    )
    replacements = ReplacementService(db, provider, aesthetics_service=assessment)
    result = replacements.replace(
        first.selection_id, SelectionReplacementRequest(remove_photo_id=ids["a"])
    )
    assert result.feasible and result.replacement.photo_id == ids["c"]
    assert result.replacement.total_score == pytest.approx(
        0.72 * 0.62 + 0.18 * 7 / 9 + 0.10 * 0.8
    )
    assert result.updated_selection.learned_quality == first.learned_quality
    assert result.updated_selection.algorithm == "openclip-musiq-fixed-fusion-v1"
    with db.connect() as c:
        c.execute(
            "UPDATE aesthetics_scores_v1 SET scores_json=replace(scores_json, '9.0', '8.0') WHERE photo_id=?",
            (ids["b"],),
        )
    with pytest.raises(ValueError, match="证据已变化"):
        replacements.replace(
            first.selection_id, SelectionReplacementRequest(remove_photo_id=ids["a"])
        )


def test_feedback_then_reselect_is_auditable_opt_in_and_reversible(tmp_path):
    db, album, ids, assessment = setup(tmp_path)
    provider = DecisionOpenClipProvider()
    service = SelectionService(db, provider, aesthetics_service=assessment)
    kwargs = dict(
        album_id=album, prompt="pick 2 photos of city", use_learned_quality=True
    )
    first = service.select(SelectionRequest(**kwargs))
    # Controlled test of the user-interaction API; no production user is impersonated.
    feedback = PreferenceService(db, provider).record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album,
            preferred_photo_id=ids["a"],
            rejected_photo_id=ids["b"],
            selection_id=first.selection_id,
        )
    )
    assert not feedback.trained
    updated = service.select(SelectionRequest(**kwargs, use_preference_memory=True))
    assert updated.preference_memory["applied"]
    assert feedback.contextual_event_id in updated.preference_memory["event_ids"]
    assert any(p.memory_delta != 0 for p in updated.selected)
    assert updated.learned_quality == first.learned_quality
    reset = service.select(SelectionRequest(**kwargs))
    assert [p.model_dump() for p in reset.selected] == [
        p.model_dump() for p in first.selected
    ]


def test_adaptive_mode_cannot_silently_mix_with_demo_fusion(tmp_path):
    db, album, _, assessment = setup(tmp_path)
    with pytest.raises(ValueError, match="record-only"):
        SelectionService(
            db,
            DecisionOpenClipProvider(),
            aesthetics_service=assessment,
            preference_mode="adaptive",
        ).select(
            SelectionRequest(
                album_id=album, prompt="pick 2 city photos", use_learned_quality=True
            )
        )
