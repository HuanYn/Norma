from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ai import app as app_module
from ai.index import AlbumIndexer
from ai.index.embedding import EmbeddingProvider
from ai.preferences import PreferenceService
from ai.preferences.model import load_preference_model
from ai.schemas import (
    PairwiseFeedbackRequest,
    SelectionReplacementRequest,
    SelectionRequest,
)
from ai.selection.parser import parse_selection_prompt
from ai.selection.replacement import ReplacementService
from ai.selection.service import SelectionService
from ai.storage import Database


class FakeSelectionProvider(EmbeddingProvider):
    name = "fake-selection-v1"
    dimension = 2

    def embed_image(self, path: Path) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        if "night" in text.casefold() or "夜景" in text:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        raise ValueError("unsupported test concept")


def _photo(path: Path, offset: int) -> None:
    image = Image.new("RGB", (480, 320), (30 + offset, 50, 90))
    draw = ImageDraw.Draw(image)
    for x in range(20 + offset, 450, 50):
        draw.line((x, 10, x, 310), fill=(220, 180, 80), width=7)
    image.save(path, "JPEG", quality=92)


def _album(tmp_path: Path) -> tuple[Database, str, dict[str, str]]:
    folder = tmp_path / "album"
    folder.mkdir()
    for index, name in enumerate(("a", "b", "c", "d", "e", "f")):
        _photo(folder / f"{name}.jpg", index * 3)
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexed = AlbumIndexer(database, data_dir).index(folder)
    ids = {photo.filename[0]: photo.id for photo in indexed.photos}

    settings = {
        "a": (90.0, "group-1", 0, np.asarray([1.0, 0.0], dtype=np.float32)),
        "b": (95.0, "group-1", 0, np.asarray([0.99, 0.01], dtype=np.float32)),
        "c": (80.0, None, 0, np.asarray([0.8, 0.2], dtype=np.float32)),
        "d": (70.0, None, 1, np.asarray([0.9, 0.1], dtype=np.float32)),
        "e": (40.0, None, 0, np.asarray([0.7, 0.3], dtype=np.float32)),
        "f": (90.0, None, 0, np.asarray([0.4, 0.6], dtype=np.float32)),
    }
    embedding_dir = data_dir / "test-embeddings"
    embedding_dir.mkdir()
    with database.connect() as connection:
        for name, (quality, group, reject, vector) in settings.items():
            path = embedding_dir / f"{name}.npy"
            normalized = vector / np.linalg.norm(vector)
            np.save(path, normalized.astype(np.float32), allow_pickle=False)
            connection.execute(
                """
                UPDATE photos SET quality_score = ?, similarity_group = ?,
                                  auto_reject = ?, embedding_path = ?,
                                  embedding_provider = 'fake-selection-v1',
                                  embedding_source_size = file_size,
                                  embedding_source_mtime_ns = source_mtime_ns
                WHERE id = ?
                """,
                (quality, group, reject, str(path.resolve()), ids[name]),
            )
    return database, indexed.album_id, ids


def test_parser_extracts_chinese_and_english_hard_constraints() -> None:
    chinese = parse_selection_prompt("选 8 张夜景，质量至少 65，相似组最多 2 张")
    assert chinese.target_count == 8
    assert chinese.min_quality == 65
    assert chinese.max_per_similarity_group == 2
    assert chinese.exclude_rejects

    english = parse_selection_prompt(
        "Pick 5 photos of night, quality at least 70, maximum 3 per similarity group, include blurry"
    )
    assert english.target_count == 5
    assert english.min_quality == 70
    assert english.max_per_similarity_group == 3
    assert not english.exclude_rejects


def test_selection_honors_all_hard_constraints_and_persists(tmp_path: Path) -> None:
    database, album_id, ids = _album(tmp_path)
    service = SelectionService(database, FakeSelectionProvider())
    response = service.select(
        SelectionRequest(
            album_id=album_id,
            prompt="选 3 张 night 夜景，质量至少 50，相似组最多 1 张",
        )
    )

    assert response.feasible
    assert len(response.selected) == 3
    selected_ids = {photo.photo_id for photo in response.selected}
    assert ids["d"] not in selected_ids  # auto reject
    assert ids["e"] not in selected_ids  # below quality floor
    assert len({ids["a"], ids["b"]} & selected_ids) == 1  # group cap
    assert all(photo.quality_score >= 50 for photo in response.selected)
    assert response.solver in {"deterministic-partition-greedy", "ortools-cp-sat"}

    with database.connect() as connection:
        row = connection.execute(
            "SELECT raw_prompt, parse_json, result_json FROM selections WHERE id = ?",
            (response.selection_id,),
        ).fetchone()
    assert row is not None
    assert "night" in row["raw_prompt"]
    assert '"target_count":3' in row["parse_json"]
    assert '"feasible":true' in row["result_json"]


def test_infeasible_constraints_return_no_partial_selection(tmp_path: Path) -> None:
    database, album_id, _ = _album(tmp_path)
    response = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(
            album_id=album_id,
            prompt="选 5 张 night，质量至少 50，相似组最多 1 张",
        )
    )
    assert not response.feasible
    assert response.selected == []
    assert response.solver_status == "infeasible"
    assert any("no partial selection" in warning for warning in response.warnings)


def test_quality_only_selection_is_explicit(tmp_path: Path) -> None:
    database, album_id, _ = _album(tmp_path)
    response = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(album_id=album_id, prompt="选 2 张")
    )
    assert response.feasible
    assert all(photo.semantic_score == 0 for photo in response.selected)
    assert any("quality only" in warning for warning in response.warnings)


def test_unknown_requested_concept_is_not_silently_ignored(tmp_path: Path) -> None:
    database, album_id, _ = _album(tmp_path)
    try:
        SelectionService(database, FakeSelectionProvider()).select(
            SelectionRequest(album_id=album_id, prompt="选 2 张猫")
        )
    except ValueError as error:
        assert "unsupported test concept" in str(error)
    else:
        raise AssertionError("unknown semantic request should fail explicitly")


def test_selection_requires_complete_quality_analysis(tmp_path: Path) -> None:
    folder = tmp_path / "deferred-quality"
    folder.mkdir()
    _photo(folder / "photo.jpg", 0)
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexed = AlbumIndexer(database, data_dir).index(folder, analyze_quality=False)

    try:
        SelectionService(database, FakeSelectionProvider()).select(
            SelectionRequest(album_id=indexed.album_id, prompt="选 1 张 night")
        )
    except ValueError as error:
        assert "run quality analysis first" in str(error)
    else:
        raise AssertionError("selection should reject an incomplete quality index")


def test_pairwise_feedback_updates_and_persists_local_model(tmp_path: Path) -> None:
    database, album_id, ids = _album(tmp_path)
    selection = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(album_id=album_id, prompt="选 3 张 night")
    )
    service = PreferenceService(database, FakeSelectionProvider())
    first = service.record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album_id,
            preferred_photo_id=ids["b"],
            rejected_photo_id=ids["e"],
            selection_id=selection.selection_id,
        )
    )
    second = service.record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album_id,
            preferred_photo_id=ids["b"],
            rejected_photo_id=ids["e"],
            selection_id=selection.selection_id,
        )
    )

    assert first.comparisons == 1
    assert second.comparisons == 2
    assert second.weights["quality"] > 0
    assert second.weights["semantic"] > 0
    model = load_preference_model(database)
    assert model.comparisons == 2
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 2

    personalized = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(album_id=album_id, prompt="选 3 张 night")
    )
    assert all(photo.preference_score != 0.5 for photo in personalized.selected)
    assert all(
        any("preference fit" in reason for reason in photo.reasons)
        for photo in personalized.selected
    )


def test_replacement_locks_existing_photos_and_preserves_constraints(
    tmp_path: Path,
) -> None:
    database, album_id, ids = _album(tmp_path)
    original = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(
            album_id=album_id,
            prompt="选 2 张 night，质量至少 50，相似组最多 1 张",
        )
    )
    assert ids["c"] in {photo.photo_id for photo in original.selected}
    result = ReplacementService(database, FakeSelectionProvider()).replace(
        original.selection_id,
        SelectionReplacementRequest(remove_photo_id=ids["c"]),
    )

    assert result.feasible
    assert result.replacement is not None
    assert result.replacement.photo_id == ids["f"]
    assert result.updated_selection is not None
    original_ids = {photo.photo_id for photo in original.selected}
    updated_ids = {photo.photo_id for photo in result.updated_selection.selected}
    assert updated_ids == (original_ids - {ids["c"]}) | {ids["f"]}
    assert len(updated_ids) == original.constraints.target_count
    assert result.updated_selection.constraints == original.constraints
    assert result.replacement_selection_id is not None
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0] == 2


def test_replacement_returns_infeasible_without_partial_result(tmp_path: Path) -> None:
    database, album_id, ids = _album(tmp_path)
    original = SelectionService(database, FakeSelectionProvider()).select(
        SelectionRequest(
            album_id=album_id,
            prompt="选 3 张 night，质量至少 50，相似组最多 1 张",
        )
    )
    result = ReplacementService(database, FakeSelectionProvider()).replace(
        original.selection_id,
        SelectionReplacementRequest(remove_photo_id=ids["c"]),
    )
    assert not result.feasible
    assert result.replacement is None
    assert result.updated_selection is None
    assert result.replacement_selection_id is None


def test_selection_and_preference_state_are_readable(
    tmp_path: Path, monkeypatch
) -> None:
    database, album_id, ids = _album(tmp_path)
    service = SelectionService(database, FakeSelectionProvider())
    selection = service.select(
        SelectionRequest(album_id=album_id, prompt="选 2 张 night")
    )
    assert service.get(selection.selection_id) == selection
    PreferenceService(database, FakeSelectionProvider()).record_pairwise(
        PairwiseFeedbackRequest(
            album_id=album_id,
            preferred_photo_id=ids["b"],
            rejected_photo_id=ids["e"],
            selection_id=selection.selection_id,
        )
    )

    monkeypatch.setattr(app_module, "database", database)
    with TestClient(app_module.app) as client:
        audit = client.get(f"/selections/{selection.selection_id}")
        preference = client.get("/preferences/local")
    assert audit.status_code == 200
    assert audit.json()["selection_id"] == selection.selection_id
    assert preference.status_code == 200
    assert preference.json()["comparisons"] == 1
