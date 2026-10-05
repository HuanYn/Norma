from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from ai import app as app_module
from ai import cli
from ai.config import Settings, load_settings
from ai.preferences import service as service_module
from ai.preferences.model import load_preference_model
from ai.preferences.repository import PreferenceRepository
from ai.preferences.runtime import RECORD_ONLY_ALGORITHM, load_preference_runtime
from ai.preferences.service import (
    PreferenceService,
    PreferenceSuggestionAlreadyConsumedError,
)
from ai.preferences.suggestion_service import PreferenceSuggestionService
from ai.rag.service import GroundedRAGService
from ai.retrieval import RetrievalService
from ai.schemas import (
    AlbumRAGRequest,
    AlbumSearchRequest,
    PairwiseFeedbackRequest,
    PreferencePairSuggestionRequest,
    SelectionReplacementRequest,
    SelectionRequest,
)
from ai.selection import ReplacementService, SelectionService
from ai.storage import Database
from ai.tests.test_contextual_decision_runtime import DecisionOpenClipProvider, _album
from ai.tests.test_preference_service_contextual import (
    FakeOpenClipProvider,
    _context,
    _request,
)
from ai.tests.test_rag_http_integration import _album as _rag_album, _valid_provider


def _model_rows(database: Database) -> tuple[list[tuple], list[tuple]]:
    with database.connect() as connection:
        return (
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT * FROM preference_models ORDER BY id"
                )
            ],
            [
                tuple(row)
                for row in connection.execute(
                    "SELECT * FROM user_preferences ORDER BY user_id"
                )
            ],
        )


@pytest.mark.parametrize("existing_model", [False, True])
def test_default_feedback_records_all_choices_without_training_or_model_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_model: bool
) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)
    provider = FakeOpenClipProvider()
    request = _request(album_id, selection_id, preferred_id, rejected_id)
    if existing_model:
        result = PreferenceService(
            database, provider, preference_mode="adaptive"
        ).record_pairwise(request)
        assert result.trained and result.contextual_trained
    before = _model_rows(database)
    train = Mock(side_effect=AssertionError("record-only must not fit a posterior"))
    legacy_update = Mock(
        side_effect=AssertionError("record-only must not update legacy weights")
    )
    activation = Mock(
        side_effect=AssertionError("record-only must not activate a model")
    )
    monkeypatch.setattr(service_module, "train", train)
    monkeypatch.setattr(PreferenceService, "_update_legacy_model", legacy_update)
    monkeypatch.setattr(PreferenceRepository, "activate_model", activation)
    service = PreferenceService(database, provider)
    for index, choice in enumerate(
        ("preferred", "preferred", "tie", "skip", "both_bad"), 1
    ):
        result = service.record_pairwise(request.model_copy(update={"choice": choice}))
        assert result.preference_mode == "record-only"
        assert result.trained is False
        assert result.contextual_trained is False
        assert result.contextual_model_id is None
        assert result.comparisons == result.contextual_comparisons == 0
        assert result.recorded_feedback_count == index + int(existing_model)
        assert set(result.weights.values()) == {0.0}
    state = service.get_state("local")
    assert state.preference_mode == "record-only" and state.trained is False
    assert state.recorded_feedback_count == 5 + int(existing_model)
    assert state.contextual_model_id is None and state.comparisons == 0
    assert _model_rows(database) == before
    train.assert_not_called()
    legacy_update.assert_not_called()
    activation.assert_not_called()
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM preference_events").fetchone()[
            0
        ] == 5 + int(existing_model)


def test_default_decisions_ignore_historical_posterior_and_legacy_weights(
    tmp_path: Path,
) -> None:
    database, album_id, ids = _album(tmp_path)
    provider = DecisionOpenClipProvider()
    selection_service = SelectionService(database, provider)
    selection = selection_service.select(
        SelectionRequest(album_id=album_id, prompt="Select 2 photos of city")
    )
    adaptive = PreferenceService(database, provider, preference_mode="adaptive")
    for _ in range(5):
        adaptive.record_pairwise(
            PairwiseFeedbackRequest(
                album_id=album_id,
                selection_id=selection.selection_id,
                preferred_photo_id=ids["b"],
                rejected_photo_id=ids["a"],
            )
        )
    before = _model_rows(database)
    assert load_preference_model(database).comparisons == 5
    assert load_preference_runtime(
        database, provider, preference_mode="adaptive"
    ).model_id
    runtime = load_preference_runtime(database, provider)
    assert runtime.algorithm == RECORD_ONLY_ALGORITHM and runtime.model_id is None
    search = RetrievalService(database, tmp_path / "data", provider).search(
        AlbumSearchRequest(album_id=album_id, query="city")
    )
    current = selection_service.select(
        SelectionRequest(album_id=album_id, prompt="Select 2 photos of city")
    )
    assert search.matches[0].photo_id == ids["a"]
    assert search.preference_comparisons == current.preference_comparisons == 0
    assert search.preference_model_id is current.preference_model_id is None
    assert search.algorithm == current.algorithm == RECORD_ONLY_ALGORITHM
    assert all(photo.preference_score == 0.0 for photo in current.selected)
    replaced = ReplacementService(database, provider).replace(
        current.selection_id, SelectionReplacementRequest(remove_photo_id=ids["a"])
    )
    assert replaced.updated_selection is not None
    assert replaced.updated_selection.preference_comparisons == 0
    assert replaced.updated_selection.algorithm == RECORD_ONLY_ALGORITHM
    quality_only = selection_service.select(
        SelectionRequest(album_id=album_id, prompt="Select 2 photos")
    )
    assert quality_only.preference_comparisons == 0
    assert all(photo.preference_score == 0.5 for photo in quality_only.selected)
    quality_replaced = ReplacementService(database, provider).replace(
        quality_only.selection_id,
        SelectionReplacementRequest(remove_photo_id=quality_only.selected[0].photo_id),
    )
    assert quality_replaced.updated_selection is not None
    assert quality_replaced.updated_selection.preference_comparisons == 0
    assert all(
        photo.preference_score == 0.5
        for photo in quality_replaced.updated_selection.selected
    )
    assert _model_rows(database) == before


def test_default_manual_feedback_without_selection_is_durably_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, album_id, _, preferred_id, rejected_id = _context(tmp_path)
    train = Mock(side_effect=AssertionError("manual recording must not train"))
    monkeypatch.setattr(service_module, "train", train)
    monkeypatch.setattr(PreferenceService, "_update_legacy_model", train)
    service = PreferenceService(database, FakeOpenClipProvider())
    result = service.record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album_id,
            preferred_photo_id=preferred_id,
            rejected_photo_id=rejected_id,
        )
    )
    assert result.trained is False
    assert result.contextual_event_id is None
    assert result.recorded_feedback_count == 1
    assert service.get_state("local").recorded_feedback_count == 1
    assert _model_rows(database) == ([], [])
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
    train.assert_not_called()


def test_record_only_can_consume_existing_suggestion_exactly_once(
    tmp_path: Path,
) -> None:
    database, album_id, _ = _album(tmp_path)
    provider = DecisionOpenClipProvider()
    selection = SelectionService(database, provider, preference_mode="adaptive").select(
        SelectionRequest(album_id=album_id, prompt="Select 1 photo of city")
    )
    suggestion = PreferenceSuggestionService(
        database, provider, preference_mode="adaptive"
    ).suggest(
        selection.selection_id, PreferencePairSuggestionRequest(posterior_samples=8)
    )
    request = PairwiseFeedbackRequest(
        album_id=album_id,
        selection_id=selection.selection_id,
        suggestion_id=suggestion.suggestion_id,
        preferred_photo_id=suggestion.left.photo_id,
        rejected_photo_id=suggestion.right.photo_id,
    )
    service = PreferenceService(database, provider)
    assert service.record_pairwise(request).trained is False
    with pytest.raises(PreferenceSuggestionAlreadyConsumedError):
        service.record_pairwise(request)
    assert service.get_state("local").recorded_feedback_count == 1
    assert _model_rows(database) == ([], [])


def test_default_rag_and_search_share_record_only_snapshot_with_old_model(
    tmp_path: Path,
) -> None:
    database, data_dir, album_id, ids, provider = _rag_album(tmp_path)
    selection = SelectionService(database, provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 1 photo of city")
    )
    PreferenceService(database, provider, preference_mode="adaptive").record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album_id,
            selection_id=selection.selection_id,
            preferred_photo_id=ids["b"],
            rejected_photo_id=ids["a"],
        )
    )
    result = GroundedRAGService(
        database, RetrievalService(database, data_dir, provider), _valid_provider
    ).run(album_id, AlbumRAGRequest(query="city", top_k=2))
    assert result.retrieval.algorithm == RECORD_ONLY_ALGORITHM
    assert result.retrieval.preference_model_id is None
    assert result.retrieval.preference_comparisons == 0
    assert result.retrieval.matches[0].photo_id == ids["a"]
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 1


def test_default_http_reports_recording_mode_and_disables_pdrr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "embedding_provider", FakeOpenClipProvider)
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(
            app_module.settings,
            preference_mode="record-only",
            data_dir=tmp_path / "data",
        ),
    )
    with TestClient(app_module.app) as client:
        for endpoint in ("/health", "/capabilities"):
            response = client.get(endpoint)
            assert response.status_code == 200
            assert response.json()["preference_mode"] == "record-only"
            assert response.json()["preference_training_enabled"] is False
        feedback = client.post(
            "/feedback/pairwise",
            json=_request(
                album_id, selection_id, preferred_id, rejected_id
            ).model_dump(),
        )
        assert feedback.status_code == 200
        assert feedback.json()["trained"] is False
        state = client.get("/preferences/local")
        assert state.json()["recorded_feedback_count"] == 1
        assert state.json()["trained"] is False
        suggestion = client.post(
            f"/selections/{selection_id}/preference-pairs/suggest", json={}
        )
        assert suggestion.status_code == 409
        assert "record-only" in suggestion.json()["detail"]


def test_preference_mode_is_strict_and_cli_data_dir_preserves_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NORMA_PREFERENCE_MODE", raising=False)
    assert load_settings().preference_mode == "record-only"
    monkeypatch.setenv("NORMA_PREFERENCE_MODE", "adaptive")
    assert cli._settings(tmp_path).preference_mode == "adaptive"
    assert cli._settings(tmp_path).data_dir == tmp_path.resolve()
    for value in ("", "train", "ADAPTIVE", "record-only "):
        monkeypatch.setenv("NORMA_PREFERENCE_MODE", value)
        with pytest.raises(ValueError, match="NORMA_PREFERENCE_MODE"):
            load_settings()
    with pytest.raises(ValueError, match="NORMA_PREFERENCE_MODE"):
        Settings("127.0.0.1", 8765, tmp_path, "INFO", preference_mode="invalid")


@pytest.mark.parametrize("mode", ["record-only", "adaptive"])
def test_cli_dispatch_passes_policy_to_feedback_and_search(
    tmp_path: Path, mode: str
) -> None:
    database, album_id, ids = _album(tmp_path)
    provider = DecisionOpenClipProvider()
    settings = Settings(
        "127.0.0.1", 8765, tmp_path / "data", "INFO", preference_mode=mode
    )
    parser = cli.build_parser()
    selection_args = parser.parse_args(["select", album_id, "Select 1 photo of city"])
    selection = cli._dispatch(selection_args, settings, database, provider)
    feedback_args = parser.parse_args(
        [
            "feedback",
            album_id,
            ids["b"],
            ids["a"],
            "--selection-id",
            selection["selection_id"],
        ]
    )
    feedback = cli._dispatch(feedback_args, settings, database, provider)
    assert feedback["preference_mode"] == mode
    assert feedback["trained"] is (mode == "adaptive")
    search_args = parser.parse_args(["search", album_id, "city"])
    search = cli._dispatch(search_args, settings, database, provider)
    assert (search["preference_model_id"] is not None) is (mode == "adaptive")
