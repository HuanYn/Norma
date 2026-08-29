from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai import app as app_module
from ai.index import AlbumIndexer
from ai.index.embedding import EmbeddingProvider
from ai.preferences import acquisition
from ai.preferences.contextual import FEATURE_DIMENSION, FEATURE_SCHEMA, PROJECTION_ID
from ai.preferences import service as preference_service_module
from ai.preferences.repository import PreferenceEvent, PreferenceRepository
from ai.preferences.service import (
    PreferenceService,
    PreferenceSuggestionAlreadyConsumedError,
)
from ai.preferences.suggestion_repository import PreferenceSuggestionRepository
from ai.preferences.suggestion_service import (
    PreferenceSuggestionConflictError,
    PreferenceSuggestionNumericalError,
    PreferenceSuggestionService,
)
from ai.schemas import (
    PairwiseFeedbackRequest,
    PreferencePairSuggestionRequest,
    SelectionRequest,
)
from ai.selection import SelectionService
from ai.storage import Database


class SuggestionOpenClipProvider(EmbeddingProvider):
    name = "openclip-suggestion-test-512d-v1"
    dimension = 512

    def embed_image(self, path: Path) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[0] = 1.0
        return vector


def _vector(cosine: float, axis: int) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = cosine
    vector[axis] = np.sqrt(1.0 - cosine**2)
    return vector


def _album(tmp_path: Path) -> tuple[Database, str, dict[str, str]]:
    folder = tmp_path / "album"
    folder.mkdir()
    for index, name in enumerate(("a", "b", "c", "d")):
        Image.new(
            "RGB",
            (480, 320),
            (35 + index * 25, 70 + index * 10, 120),
        ).save(folder / f"{name}.jpg", "JPEG")
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexed = AlbumIndexer(database, data_dir).index(folder)
    ids = {photo.filename[0]: photo.id for photo in indexed.photos}
    vectors = {
        "a": _vector(0.82, 1),
        "b": _vector(0.80, 2),
        "c": _vector(0.64, 3),
        "d": _vector(0.42, 4),
    }
    groups = {"a": "shared", "b": "shared", "c": None, "d": None}
    embedding_dir = data_dir / "suggestion-embeddings"
    embedding_dir.mkdir()
    with database.connect() as connection:
        for name, vector in vectors.items():
            path = embedding_dir / f"{name}.npy"
            np.save(path, vector, allow_pickle=False)
            source_sha256 = hashlib.sha256(
                (folder / f"{name}.jpg").read_bytes()
            ).hexdigest()
            connection.execute(
                """
                UPDATE photos SET quality_score = ?, auto_reject = 0,
                    similarity_group = ?, embedding_path = ?,
                    embedding_provider = ?, embedding_source_size = file_size,
                    embedding_source_mtime_ns = source_mtime_ns,
                    embedding_source_sha256 = ?
                WHERE id = ?
                """,
                (
                    70.0 + 5.0 * "abcd".index(name),
                    groups[name],
                    str(path.resolve()),
                    SuggestionOpenClipProvider.name,
                    source_sha256,
                    ids[name],
                ),
            )
    return database, indexed.album_id, ids


def _selection(
    database: Database,
    album_id: str,
    *,
    subset: list[str] | None = None,
) -> object:
    return SelectionService(database, SuggestionOpenClipProvider()).select(
        SelectionRequest(
            album_id=album_id,
            prompt="Select 2 photos of city, maximum 1 per similarity group",
            subset_photo_ids=subset,
        )
    )


def test_zero_feedback_suggestion_is_audited_and_constraint_feasible(
    tmp_path: Path,
) -> None:
    database, album_id, _ = _album(tmp_path)
    selection = _selection(database, album_id)
    response = PreferenceSuggestionService(
        database, SuggestionOpenClipProvider()
    ).suggest(selection.selection_id, PreferencePairSuggestionRequest())

    assert database.current_version() == 14
    assert response.model_id_at_display is None
    assert response.provider_fingerprint == SuggestionOpenClipProvider.name
    assert response.feature_schema == FEATURE_SCHEMA
    assert response.projection_id == PROJECTION_ID
    assert response.acquisition_version == acquisition.ACQUISITION_VERSION
    assert response.constraint_solver == acquisition.CONSTRAINT_SOLVER
    assert response.constraint_violation_count == 0
    assert response.requested_posterior_samples == 64
    assert response.posterior_samples in {64, 128}
    assert response.retry_count in {0, 1}
    assert response.candidate_count == 4
    assert response.eligible_pair_count == 6
    assert response.voi_invariant_ok
    assert response.pdrr >= 0.0
    assert response.left.photo_id != response.right.photo_id
    assert selection.candidate_universe is not None
    assert len(selection.candidate_universe.decision_feature_snapshot_sha256) == 64
    assert (
        response.candidate_feature_digest
        == selection.candidate_universe.decision_feature_snapshot_sha256
    )

    stored = PreferenceSuggestionRepository(database).get(response.suggestion_id)
    assert stored.left_photo_id == response.left.photo_id
    assert stored.right_photo_id == response.right.photo_id
    assert stored.model_id_at_display is None
    assert len(stored.left_features) == FEATURE_DIMENSION
    assert len(stored.right_features) == FEATURE_DIMENSION
    assert stored.result["constraint_violation_count"] == 0
    assert (
        stored.diagnostics["candidate_feature_digest"]
        == response.candidate_feature_digest
    )
    with database.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE preference_suggestions SET mode = 'exhaustive' WHERE id = ?",
                (response.suggestion_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "DELETE FROM preference_suggestions WHERE id = ?",
                (response.suggestion_id,),
            )


def test_trained_suggestion_feedback_preserves_display_model_and_nonbinary_audit(
    tmp_path: Path,
) -> None:
    database, album_id, ids = _album(tmp_path)
    selection = _selection(database, album_id)
    preference = PreferenceService(database, SuggestionOpenClipProvider())
    for _ in range(3):
        trained = preference.record_pairwise(
            PairwiseFeedbackRequest(
                album_id=album_id,
                selection_id=selection.selection_id,
                preferred_photo_id=ids["b"],
                rejected_photo_id=ids["a"],
            )
        )
    suggestion = PreferenceSuggestionService(
        database, SuggestionOpenClipProvider()
    ).suggest(selection.selection_id, PreferencePairSuggestionRequest(seed=9))

    assert suggestion.model_id_at_display == trained.contextual_model_id
    chosen = preference.record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album_id,
            selection_id=selection.selection_id,
            suggestion_id=suggestion.suggestion_id,
            preferred_photo_id=suggestion.left.photo_id,
            rejected_photo_id=suggestion.right.photo_id,
        )
    )
    assert chosen.contextual_comparisons == 4
    event = PreferenceRepository(database).get_event(chosen.contextual_event_id)
    assert event.model_id_at_display == suggestion.model_id_at_display
    assert event.suggestion_id == suggestion.suggestion_id
    assert event.context["source"] == "pdrr-suggestion-feedback"
    assert event.context["suggestion_id"] == suggestion.suggestion_id

    first_pair = {suggestion.left.photo_id, suggestion.right.photo_id}
    nonbinary_suggestions = []
    for offset, choice in enumerate(("tie", "skip", "both_bad"), start=1):
        next_suggestion = PreferenceSuggestionService(
            database, SuggestionOpenClipProvider()
        ).suggest(
            selection.selection_id,
            PreferencePairSuggestionRequest(seed=9 + offset),
        )
        nonbinary_suggestions.append(next_suggestion)
        if offset == 1:
            assert next_suggestion.model_id_at_display == chosen.contextual_model_id
            assert {
                next_suggestion.left.photo_id,
                next_suggestion.right.photo_id,
            } != first_pair
        audited = preference.record_pairwise(
            PairwiseFeedbackRequest(
                album_id=album_id,
                selection_id=selection.selection_id,
                suggestion_id=next_suggestion.suggestion_id,
                preferred_photo_id=next_suggestion.left.photo_id,
                rejected_photo_id=next_suggestion.right.photo_id,
                choice=choice,
            )
        )
        assert audited.contextual_trained is False
        assert audited.contextual_comparisons == 4
    events = PreferenceRepository(database).list_compatible_events(
        "local", SuggestionOpenClipProvider.name, FEATURE_SCHEMA
    )
    nonbinary_ids = {item.suggestion_id for item in nonbinary_suggestions}
    nonbinary_events = [
        event for event in events if event.suggestion_id in nonbinary_ids
    ]
    assert {event.choice for event in nonbinary_events} == {"tie", "skip", "both_bad"}
    assert {event.suggestion_id for event in nonbinary_events} == nonbinary_ids

    consumed = nonbinary_suggestions[0]
    with pytest.raises(
        PreferenceSuggestionAlreadyConsumedError, match="already consumed"
    ):
        preference.record_pairwise(
            PairwiseFeedbackRequest(
                album_id=album_id,
                selection_id=selection.selection_id,
                suggestion_id=consumed.suggestion_id,
                preferred_photo_id=consumed.left.photo_id,
                rejected_photo_id=consumed.right.photo_id,
            )
        )
    after_duplicate = PreferenceRepository(database).list_compatible_events(
        "local", SuggestionOpenClipProvider.name, FEATURE_SCHEMA
    )
    assert len(after_duplicate) == len(events)
    assert (
        PreferenceService(database, SuggestionOpenClipProvider())
        .get_state("local")
        .contextual_comparisons
        == 4
    )


def test_duplicate_suggestion_feedback_is_http_409_and_trains_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, album_id, _ = _album(tmp_path)
    selection = _selection(database, album_id)
    suggestion = PreferenceSuggestionService(
        database, SuggestionOpenClipProvider()
    ).suggest(selection.selection_id, PreferencePairSuggestionRequest(seed=17))
    payload = {
        "album_id": album_id,
        "selection_id": selection.selection_id,
        "suggestion_id": suggestion.suggestion_id,
        "preferred_photo_id": suggestion.left.photo_id,
        "rejected_photo_id": suggestion.right.photo_id,
    }
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(
        app_module, "embedding_provider", lambda: SuggestionOpenClipProvider()
    )
    with TestClient(app_module.app) as client:
        first = client.post("/feedback/pairwise", json=payload)
        duplicate = client.post("/feedback/pairwise", json=payload)

    assert first.status_code == 200
    assert first.json()["contextual_comparisons"] == 1
    assert duplicate.status_code == 409
    assert "already consumed" in duplicate.json()["detail"]
    with database.connect() as connection:
        events = connection.execute(
            "SELECT id, suggestion_id FROM preference_events WHERE suggestion_id = ?",
            (suggestion.suggestion_id,),
        ).fetchall()
        active = connection.execute(
            "SELECT training_pair_count FROM preference_models WHERE active = 1"
        ).fetchall()
    assert len(events) == 1
    assert [row["training_pair_count"] for row in active] == [1]


def test_concurrent_suggestion_feedback_is_one_shot_across_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, album_id, _ = _album(tmp_path)
    selection = _selection(database, album_id)
    suggestion = PreferenceSuggestionService(
        database, SuggestionOpenClipProvider()
    ).suggest(selection.selection_id, PreferencePairSuggestionRequest(seed=23))
    request = PairwiseFeedbackRequest(
        album_id=album_id,
        selection_id=selection.selection_id,
        suggestion_id=suggestion.suggestion_id,
        preferred_photo_id=suggestion.left.photo_id,
        rejected_photo_id=suggestion.right.photo_id,
    )
    services = [
        PreferenceService(Database(database.path), SuggestionOpenClipProvider()),
        PreferenceService(Database(database.path), SuggestionOpenClipProvider()),
    ]
    start_barrier = Barrier(2, timeout=10)
    insert_barrier = Barrier(2, timeout=10)
    original_insert = PreferenceRepository.insert_event

    class IndependentWorkerLock:
        """Model two worker processes that do not share UPDATE_LOCK."""

        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def insert_simultaneously(
        repository: PreferenceRepository,
        event: PreferenceEvent,
    ) -> PreferenceEvent:
        insert_barrier.wait()
        return original_insert(repository, event)

    monkeypatch.setattr(
        preference_service_module,
        "UPDATE_LOCK",
        IndependentWorkerLock(),
    )
    monkeypatch.setattr(
        PreferenceRepository,
        "insert_event",
        insert_simultaneously,
    )

    def submit(service: PreferenceService) -> tuple[str, object]:
        start_barrier.wait()
        try:
            return "success", service.record_pairwise(request)
        except PreferenceSuggestionAlreadyConsumedError as error:
            return "already_consumed", str(error)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(submit, services))

    assert sorted(status for status, _ in outcomes) == [
        "already_consumed",
        "success",
    ]
    failure = next(value for status, value in outcomes if status == "already_consumed")
    assert "already consumed" in str(failure)
    success = next(value for status, value in outcomes if status == "success")
    assert success.contextual_comparisons == 1

    with database.connect() as connection:
        events = connection.execute(
            "SELECT choice FROM preference_events WHERE suggestion_id = ?",
            (suggestion.suggestion_id,),
        ).fetchall()
        models = connection.execute(
            "SELECT active, training_pair_count FROM preference_models"
        ).fetchall()
    assert [row["choice"] for row in events] == ["preferred"]
    assert [(row["active"], row["training_pair_count"]) for row in models] == [(1, 1)]


def test_subset_universe_is_exact_and_source_digest_drift_is_rejected(
    tmp_path: Path,
) -> None:
    database, album_id, ids = _album(tmp_path)
    subset = [ids["a"], ids["b"], ids["c"]]
    selection = _selection(database, album_id, subset=subset)
    assert selection.candidate_universe is not None
    assert set(selection.candidate_universe.candidate_photo_ids) == set(subset)

    response = PreferenceSuggestionService(
        database, SuggestionOpenClipProvider()
    ).suggest(
        selection.selection_id,
        PreferencePairSuggestionRequest(exhaustive=True, seed=3),
    )
    stored = PreferenceSuggestionRepository(database).get(response.suggestion_id)
    assert response.candidate_count == 3
    assert set(stored.candidate_ids) == set(subset)
    assert (
        response.candidate_digest == selection.candidate_universe.candidate_ids_sha256
    )

    with database.connect() as connection:
        connection.execute(
            "UPDATE photos SET similarity_group = 'drifted' WHERE id = ?",
            (ids["c"],),
        )
    with pytest.raises(PreferenceSuggestionConflictError, match="snapshot drift"):
        PreferenceSuggestionService(database, SuggestionOpenClipProvider()).suggest(
            selection.selection_id,
            PreferencePairSuggestionRequest(exhaustive=True),
        )


def test_embedding_file_overwrite_breaks_decision_snapshot_and_returns_409(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, album_id, ids = _album(tmp_path)
    selection = _selection(database, album_id)
    assert selection.candidate_universe is not None
    assert selection.candidate_universe.decision_feature_snapshot_sha256
    with database.connect() as connection:
        row = connection.execute(
            "SELECT embedding_path, embedding_source_size, "
            "embedding_source_mtime_ns FROM photos WHERE id = ?",
            (ids["c"],),
        ).fetchone()
        source_metadata = (
            row["embedding_source_size"],
            row["embedding_source_mtime_ns"],
        )
    np.save(Path(row["embedding_path"]), _vector(0.15, 8), allow_pickle=False)

    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(
        app_module, "embedding_provider", lambda: SuggestionOpenClipProvider()
    )
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/selections/{selection.selection_id}/preference-pairs/suggest",
            json={},
        )

    assert response.status_code == 409
    assert "decision-feature snapshot drift" in response.json()["detail"]
    with database.connect() as connection:
        current_metadata = connection.execute(
            "SELECT embedding_source_size, embedding_source_mtime_ns "
            "FROM photos WHERE id = ?",
            (ids["c"],),
        ).fetchone()
        suggestion_count = connection.execute(
            "SELECT COUNT(*) FROM preference_suggestions"
        ).fetchone()[0]
    assert (
        current_metadata["embedding_source_size"],
        current_metadata["embedding_source_mtime_ns"],
    ) == source_metadata
    assert suggestion_count == 0


def test_legacy_selection_without_decision_snapshot_requires_rebuild(
    tmp_path: Path,
) -> None:
    database, album_id, _ = _album(tmp_path)
    selection = _selection(database, album_id)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT result_json FROM selections WHERE id = ?",
            (selection.selection_id,),
        ).fetchone()
        payload = json.loads(row["result_json"])
        payload["candidate_universe"].pop("decision_feature_snapshot_version")
        payload["candidate_universe"].pop("decision_feature_snapshot_sha256")
        connection.execute(
            "UPDATE selections SET result_json = ? WHERE id = ?",
            (json.dumps(payload), selection.selection_id),
        )

    with pytest.raises(PreferenceSuggestionConflictError, match="create a new"):
        PreferenceSuggestionService(database, SuggestionOpenClipProvider()).suggest(
            selection.selection_id,
            PreferencePairSuggestionRequest(),
        )
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM preference_suggestions"
            ).fetchone()[0]
            == 0
        )


def test_provider_drift_is_conflict_and_http_returns_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, album_id, _ = _album(tmp_path)
    selection = _selection(database, album_id)

    class DriftProvider(SuggestionOpenClipProvider):
        name = "openclip-suggestion-drift-v1"

    with pytest.raises(PreferenceSuggestionConflictError, match="provider drift"):
        PreferenceSuggestionService(database, DriftProvider()).suggest(
            selection.selection_id,
            PreferencePairSuggestionRequest(),
        )

    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "embedding_provider", lambda: DriftProvider())
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/selections/{selection.selection_id}/preference-pairs/suggest",
            json={},
        )
    assert response.status_code == 409
    assert "provider drift" in response.json()["detail"]


def test_numerical_failure_retries_b128_once_and_full_abstention_is_422(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, album_id, _ = _album(tmp_path)
    selection = _selection(database, album_id)
    original = acquisition.suggest_pair
    calls: list[int] = []

    def fail_once(*args: object, **kwargs: object):
        samples = int(kwargs["posterior_samples"])
        calls.append(samples)
        if len(calls) == 1:
            raise acquisition.AcquisitionNumericalError("synthetic B64 failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(acquisition, "suggest_pair", fail_once)
    retried = PreferenceSuggestionService(
        database, SuggestionOpenClipProvider()
    ).suggest(selection.selection_id, PreferencePairSuggestionRequest(seed=5))
    assert calls == [64, 128]
    assert retried.posterior_samples == 128
    assert retried.retry_count == 1

    def always_fail(*args: object, **kwargs: object):
        raise acquisition.AcquisitionNumericalError("synthetic persistent failure")

    monkeypatch.setattr(acquisition, "suggest_pair", always_fail)
    with pytest.raises(PreferenceSuggestionNumericalError, match="B=128"):
        PreferenceSuggestionService(database, SuggestionOpenClipProvider()).suggest(
            selection.selection_id,
            PreferencePairSuggestionRequest(exclude_previous=False),
        )
    with database.connect() as connection:
        count_before_http = connection.execute(
            "SELECT COUNT(*) FROM preference_suggestions"
        ).fetchone()[0]

    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(
        app_module, "embedding_provider", lambda: SuggestionOpenClipProvider()
    )
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/selections/{selection.selection_id}/preference-pairs/suggest",
            json={"exclude_previous": False},
        )
    assert response.status_code == 422
    assert "B=128" in response.json()["detail"]
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM preference_suggestions"
            ).fetchone()[0]
            == count_before_http
        )


def test_suggestion_feedback_rejects_wrong_pair_without_new_event(
    tmp_path: Path,
) -> None:
    database, album_id, ids = _album(tmp_path)
    selection = _selection(database, album_id)
    suggestion = PreferenceSuggestionService(
        database, SuggestionOpenClipProvider()
    ).suggest(selection.selection_id, PreferencePairSuggestionRequest(seed=2))
    wrong = next(
        photo_id
        for photo_id in ids.values()
        if photo_id not in {suggestion.left.photo_id, suggestion.right.photo_id}
    )
    with pytest.raises(ValueError, match="do not match"):
        PreferenceService(database, SuggestionOpenClipProvider()).record_pairwise(
            PairwiseFeedbackRequest(
                album_id=album_id,
                selection_id=selection.selection_id,
                suggestion_id=suggestion.suggestion_id,
                preferred_photo_id=suggestion.left.photo_id,
                rejected_photo_id=wrong,
            )
        )
    with database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM preference_events").fetchone()[0]
            == 0
        )


def test_v11_migration_keeps_suggestion_payload_json_valid(tmp_path: Path) -> None:
    database, album_id, _ = _album(tmp_path)
    selection = _selection(database, album_id)
    response = PreferenceSuggestionService(
        database, SuggestionOpenClipProvider()
    ).suggest(selection.selection_id, PreferencePairSuggestionRequest(seed=13))
    with database.connect() as connection:
        row = connection.execute(
            "SELECT request_json, diagnostics_json, result_json "
            "FROM preference_suggestions WHERE id = ?",
            (response.suggestion_id,),
        ).fetchone()
    assert json.loads(row["request_json"])["posterior_samples"] == 64
    assert json.loads(row["diagnostics_json"])["constraint_violation_count"] == 0
    assert json.loads(row["result_json"])["suggestion_id"] == response.suggestion_id
