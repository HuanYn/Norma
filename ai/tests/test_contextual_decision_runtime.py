from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai import app as app_module
from ai.index import AlbumIndexer
from ai.index.embedding import EmbeddingProvider, normalize_embedding
from ai.preferences.contextual import FEATURE_SCHEMA
from ai.preferences.repository import PreferenceRepository
from ai.preferences.runtime import (
    COSINE_FALLBACK_ALGORITHM,
    IncompatiblePreferenceModelError,
    load_preference_runtime,
)
from ai.preferences.service import PreferenceService
from ai.preferences.suggestion_service import PreferenceSuggestionService
from ai.retrieval import RetrievalService
from ai.schemas import (
    AlbumSearchRequest,
    PairwiseFeedbackRequest,
    PreferencePairSuggestionRequest,
    SelectionReplacementRequest,
    SelectionRequest,
)
from ai.selection import ReplacementService, SelectionService
from ai.storage import Database


class DecisionOpenClipProvider(EmbeddingProvider):
    name = "openclip-decision-test-512d-v1"
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
            (30 + index * 30, 60 + index * 10, 100),
        ).save(folder / f"{name}.jpg", "JPEG")
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexed = AlbumIndexer(database, data_dir).index(folder)
    photo_ids = {photo.filename[0]: photo.id for photo in indexed.photos}
    vectors = {
        "a": _vector(0.82, 1),
        "b": _vector(0.80, 2),
        "c": _vector(0.62, 3),
        "d": _vector(0.40, 4),
    }
    qualities = {"a": 10.0, "b": 100.0, "c": 80.0, "d": 70.0}
    embedding_dir = data_dir / "decision-embeddings"
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
                    similarity_group = NULL, embedding_path = ?,
                    embedding_provider = ?, embedding_source_size = file_size,
                    embedding_source_mtime_ns = source_mtime_ns,
                    embedding_source_sha256 = ?
                WHERE id = ?
                """,
                (
                    qualities[name],
                    str(path.resolve()),
                    DecisionOpenClipProvider.name,
                    source_sha256,
                    photo_ids[name],
                ),
            )
    return database, indexed.album_id, photo_ids


def test_zero_feedback_openclip_utility_is_exact_cosine_and_ignores_quality_weight(
    tmp_path: Path,
) -> None:
    database, album_id, ids = _album(tmp_path)
    provider = DecisionOpenClipProvider()

    with database.connect() as connection:
        row = connection.execute(
            "SELECT embedding_path FROM photos WHERE id = ?", (ids["a"],)
        ).fetchone()
    image_vector = normalize_embedding(
        np.load(row["embedding_path"], allow_pickle=False),
        provider.dimension,
        label="test image",
    )
    query_vector = normalize_embedding(
        provider.embed_text("city"), provider.dimension, label="test query"
    )
    runtime = load_preference_runtime(database, provider, user_id="alice")
    utility = runtime.score(image_vector, query_vector)
    expected_cosine = float(
        np.dot(image_vector.astype(np.float64), query_vector.astype(np.float64))
    )
    assert utility.total == expected_cosine
    assert utility.cosine == expected_cosine
    assert utility.preference_residual == 0.0

    search = RetrievalService(database, tmp_path / "data", provider).search(
        AlbumSearchRequest(album_id=album_id, query="city", user_id="alice")
    )
    selection = SelectionService(database, provider).select(
        SelectionRequest(
            album_id=album_id,
            prompt="Select 1 photo of city",
            user_id="alice",
        )
    )

    assert search.algorithm == COSINE_FALLBACK_ALGORITHM
    assert search.preference_comparisons == 0
    assert search.matches[0].photo_id == ids["a"]
    assert search.matches[0].score == search.matches[0].semantic_score
    assert search.matches[0].preference_residual == 0.0
    assert selection.selected[0].photo_id == ids["a"]
    assert selection.selected[0].total_score == selection.selected[0].semantic_score
    assert selection.selected[0].preference_score == 0.0
    assert selection.selected[0].quality_score == 10.0
    assert selection.user_id == "alice"
    assert selection.query_text == "Select 1 photo of city"
    assert selection.provider_fingerprint == provider.name
    assert selection.candidate_universe is not None
    assert selection.candidate_universe.eligible_photo_count == 4
    assert len(selection.candidate_universe.candidate_ids_sha256) == 64

    persisted = SelectionService(database, provider).get(selection.selection_id)
    assert persisted == selection


def test_learned_posterior_reorders_search_and_selection_per_user(
    tmp_path: Path,
) -> None:
    database, album_id, ids = _album(tmp_path)
    provider = DecisionOpenClipProvider()
    initial = SelectionService(database, provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 1 photo of city")
    )
    feedback = PreferenceService(database, provider)
    for _ in range(8):
        latest = feedback.record_pairwise(
            PairwiseFeedbackRequest(
                album_id=album_id,
                selection_id=initial.selection_id,
                preferred_photo_id=ids["b"],
                rejected_photo_id=ids["a"],
            )
        )

    local_search = RetrievalService(database, tmp_path / "data", provider).search(
        AlbumSearchRequest(album_id=album_id, query="city")
    )
    other_search = RetrievalService(database, tmp_path / "data", provider).search(
        AlbumSearchRequest(album_id=album_id, query="city", user_id="other")
    )
    learned_selection = SelectionService(database, provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 1 photo of city")
    )

    assert local_search.preference_model_id == latest.contextual_model_id
    assert local_search.preference_comparisons == 8
    assert local_search.matches[0].photo_id == ids["b"]
    assert local_search.matches[0].preference_residual != 0.0
    assert learned_selection.selected[0].photo_id == ids["b"]
    assert learned_selection.preference_model_id == latest.contextual_model_id
    assert learned_selection.preference_comparisons == 8
    assert learned_selection.feature_schema == FEATURE_SCHEMA
    assert other_search.preference_model_id is None
    assert other_search.preference_comparisons == 0
    assert other_search.matches[0].photo_id == ids["a"]


def test_replacement_uses_one_current_model_and_rescores_locked_photos(
    tmp_path: Path,
) -> None:
    database, album_id, ids = _album(tmp_path)
    provider = DecisionOpenClipProvider()
    seed_selection = SelectionService(database, provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 2 photos of city")
    )
    feedback = PreferenceService(database, provider)
    first_model = feedback.record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album_id,
            selection_id=seed_selection.selection_id,
            preferred_photo_id=ids["b"],
            rejected_photo_id=ids["a"],
        )
    )
    original = SelectionService(database, provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 2 photos of city")
    )
    original_by_id = {photo.photo_id: photo for photo in original.selected}
    assert original.preference_model_id == first_model.contextual_model_id

    for _ in range(5):
        latest = feedback.record_pairwise(
            PairwiseFeedbackRequest(
                album_id=album_id,
                selection_id=original.selection_id,
                preferred_photo_id=ids["c"],
                rejected_photo_id=ids["a"],
            )
        )
    removed = next(iter(original_by_id))
    result = ReplacementService(database, provider).replace(
        original.selection_id,
        SelectionReplacementRequest(remove_photo_id=removed),
    )

    assert result.feasible
    assert result.updated_selection is not None
    updated = result.updated_selection
    assert updated.preference_model_id == latest.contextual_model_id
    assert updated.preference_model_id != original.preference_model_id
    assert updated.preference_comparisons == 6
    assert updated.candidate_universe is not None
    assert len(updated.candidate_universe.decision_feature_snapshot_sha256) == 64
    suggestion = PreferenceSuggestionService(database, provider).suggest(
        updated.selection_id,
        PreferencePairSuggestionRequest(seed=31),
    )
    assert (
        suggestion.candidate_feature_digest
        == updated.candidate_universe.decision_feature_snapshot_sha256
    )
    assert any("one current model snapshot" in warning for warning in updated.warnings)
    locked_id = next(photo_id for photo_id in original_by_id if photo_id != removed)
    rescored_locked = next(
        photo for photo in updated.selected if photo.photo_id == locked_id
    )
    assert rescored_locked.total_score != original_by_id[locked_id].total_score
    assert any("Recomputed every locked" in line for line in result.explanation)


def test_provider_drift_falls_back_to_cosine_with_audited_warning(
    tmp_path: Path,
) -> None:
    database, album_id, _ = _album(tmp_path)
    original_provider = DecisionOpenClipProvider()
    original = SelectionService(database, original_provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 2 photos of city")
    )

    class DriftProvider(DecisionOpenClipProvider):
        name = "openclip-decision-drift-512d-v1"

    drift_provider = DriftProvider()
    with database.connect() as connection:
        connection.execute(
            "UPDATE photos SET embedding_provider = ? WHERE album_id = ?",
            (drift_provider.name, album_id),
        )
    result = ReplacementService(database, drift_provider).replace(
        original.selection_id,
        SelectionReplacementRequest(remove_photo_id=original.selected[0].photo_id),
    )

    assert result.updated_selection is not None
    assert result.updated_selection.algorithm == "provider-drift-cosine-fallback-v1"
    assert result.updated_selection.preference_model_id is None
    assert result.updated_selection.preference_comparisons == 0
    assert any(
        "provider drift" in warning.casefold()
        for warning in result.updated_selection.warnings
    )
    assert all(
        photo.total_score == photo.semantic_score
        for photo in result.updated_selection.selected
    )


def test_runtime_refuses_event_history_without_an_active_compatible_model(
    tmp_path: Path,
) -> None:
    database, album_id, ids = _album(tmp_path)
    provider = DecisionOpenClipProvider()
    selection = SelectionService(database, provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 1 photo of city")
    )
    PreferenceService(database, provider).record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album_id,
            selection_id=selection.selection_id,
            preferred_photo_id=ids["b"],
            rejected_photo_id=ids["a"],
        )
    )
    repository = PreferenceRepository(database)
    active = repository.load_active_model("local", provider.name, FEATURE_SCHEMA)
    assert active is not None
    with database.connect() as connection:
        connection.execute(
            "UPDATE preference_models SET active = 0 WHERE id = ?",
            (active.id,),
        )

    with pytest.raises(IncompatiblePreferenceModelError, match="no compatible active"):
        load_preference_runtime(database, provider)


def test_runtime_strictly_rejects_projection_drift(tmp_path: Path) -> None:
    database, album_id, ids = _album(tmp_path)
    provider = DecisionOpenClipProvider()
    selection = SelectionService(database, provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 1 photo of city")
    )
    PreferenceService(database, provider).record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album_id,
            selection_id=selection.selection_id,
            preferred_photo_id=ids["b"],
            rejected_photo_id=ids["a"],
        )
    )
    repository = PreferenceRepository(database)
    active = repository.load_active_model("local", provider.name, FEATURE_SCHEMA)
    assert active is not None
    repository.activate_model(
        replace(active, id="projection-drift-model", projection_id="wrong-projection")
    )

    with pytest.raises(IncompatiblePreferenceModelError, match="projection drift"):
        load_preference_runtime(database, provider)


def test_http_search_and_selection_propagate_user_and_runtime_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, album_id, _ = _album(tmp_path)
    provider = DecisionOpenClipProvider()
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "embedding_provider", lambda: provider)

    with TestClient(app_module.app) as client:
        search = client.post(
            "/albums/search",
            json={"album_id": album_id, "query": "city", "user_id": "portfolio"},
        )
        selection = client.post(
            "/selections",
            json={
                "album_id": album_id,
                "prompt": "Select 1 photo of city",
                "user_id": "portfolio",
            },
        )

    assert search.status_code == 200, search.text
    assert search.json()["user_id"] == "portfolio"
    assert search.json()["algorithm"] == COSINE_FALLBACK_ALGORITHM
    assert selection.status_code == 200, selection.text
    payload = selection.json()
    assert payload["user_id"] == "portfolio"
    assert payload["query_text"] == "Select 1 photo of city"
    assert payload["provider_fingerprint"] == provider.name
    assert payload["algorithm"] == COSINE_FALLBACK_ALGORITHM
    assert payload["candidate_universe"]["eligible_photo_count"] == 4
    assert len(payload["candidate_universe"]["decision_feature_snapshot_sha256"]) == 64
