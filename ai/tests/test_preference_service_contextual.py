from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai import app as app_module
from ai.index.embedding import EmbeddingProvider
from ai.preferences.contextual import (
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    PROJECTION_ID,
)
from ai.preferences.model import load_preference_model
from ai.preferences.repository import PreferenceRepository
from ai.preferences.service import CONTEXTUAL_ALGORITHM, PreferenceService
from ai.schemas import PairwiseFeedbackRequest
from ai.storage import Database


class FakeOpenClipProvider(EmbeddingProvider):
    name = "openclip-test-512d-v1"
    dimension = 512

    def __init__(self, *, reject_query: bool = False) -> None:
        self.reject_query = reject_query

    def embed_image(self, path: Path) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        if self.reject_query:
            raise ValueError("test query rejected")
        return _unit(91).astype(np.float32)


def _unit(seed: int) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=512)
    return vector / np.linalg.norm(vector)


def _context(tmp_path: Path) -> tuple[Database, str, str, str, str]:
    database = Database(tmp_path / "data" / "norma.db")
    database.initialize()
    album_id = "album-contextual"
    selection_id = "selection-contextual"
    photo_ids = ("preferred-photo", "rejected-photo")
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO albums(id, name, source_path) VALUES (?, ?, ?)",
            (album_id, "Contextual", str(tmp_path / "album")),
        )
        connection.execute(
            """
            INSERT INTO selections(id, album_id, raw_prompt, parse_json, result_json)
            VALUES (?, ?, ?, '{}', ?)
            """,
            (
                selection_id,
                album_id,
                "选 2 张城市夜景，质量至少 60",
                json.dumps(
                    {
                        "user_id": "local",
                        "query_text": "选 2 张城市夜景，质量至少 60",
                        "provider_fingerprint": FakeOpenClipProvider.name,
                        "preference_model_id": None,
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        for index, photo_id in enumerate(photo_ids):
            source = tmp_path / f"source-{index}.jpg"
            source.write_bytes(f"public-test-photo-{index}".encode())
            stat = source.stat()
            embedding = tmp_path / f"embedding-{index}.npy"
            np.save(embedding, _unit(index + 1).astype(np.float32), allow_pickle=False)
            connection.execute(
                """
                INSERT INTO photos(
                    id, album_id, absolute_path, width, height,
                    quality_score, blur_score, auto_reject, metadata_json,
                    file_size, source_mtime_ns, embedding_path,
                    embedding_provider, embedding_source_size,
                    embedding_source_mtime_ns, embedding_source_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    photo_id,
                    album_id,
                    str(source),
                    640,
                    480,
                    90.0 - index * 20.0,
                    500.0 - index * 100.0,
                    index,
                    stat.st_size,
                    stat.st_mtime_ns,
                    str(embedding),
                    FakeOpenClipProvider.name,
                    stat.st_size,
                    stat.st_mtime_ns,
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                ),
            )
    return database, album_id, selection_id, photo_ids[0], photo_ids[1]


def _request(
    album_id: str,
    selection_id: str,
    preferred_id: str,
    rejected_id: str,
    *,
    choice: str = "preferred",
) -> PairwiseFeedbackRequest:
    return PairwiseFeedbackRequest(
        album_id=album_id,
        selection_id=selection_id,
        preferred_photo_id=preferred_id,
        rejected_photo_id=rejected_id,
        choice=choice,
    )


def test_openclip_feedback_persists_event_and_versions_laplace_model(
    tmp_path: Path,
) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)
    service = PreferenceService(database, FakeOpenClipProvider())
    request = _request(album_id, selection_id, preferred_id, rejected_id)

    first = service.record_pairwise(request)
    second = service.record_pairwise(request)
    repository = PreferenceRepository(database)
    events = repository.list_trainable_events(
        "local", FakeOpenClipProvider.name, FEATURE_SCHEMA
    )
    active = repository.load_active_model(
        "local", FakeOpenClipProvider.name, FEATURE_SCHEMA
    )

    assert first.contextual_event_id == first.feedback_id
    assert first.algorithm == CONTEXTUAL_ALGORITHM
    assert first.contextual_comparisons == 1
    assert first.contextual_trained is True
    assert second.contextual_comparisons == 2
    assert second.contextual_model_id != first.contextual_model_id
    assert [event.query_text for event in events] == [
        "选 2 张城市夜景，质量至少 60",
        "选 2 张城市夜景，质量至少 60",
    ]
    assert all(len(event.preferred_features) == FEATURE_DIMENSION for event in events)
    assert all(
        event.base_margin
        == pytest.approx(event.preferred_features[64] - event.rejected_features[64])
        for event in events
    )
    assert active is not None
    assert active.id == second.contextual_model_id
    assert active.projection_id == PROJECTION_ID
    assert len(active.mean) == FEATURE_DIMENSION
    assert np.asarray(active.covariance).shape == (
        FEATURE_DIMENSION,
        FEATURE_DIMENSION,
    )
    assert float(np.linalg.eigvalsh(active.covariance)[0]) >= -1e-10
    assert load_preference_model(database).comparisons == 2

    state = service.get_state("local")
    assert state.contextual_model_id == active.id
    assert state.contextual_comparisons == 2
    assert state.algorithm == CONTEXTUAL_ALGORITHM
    with database.connect() as connection:
        history = connection.execute(
            "SELECT active FROM preference_models ORDER BY created_at, id"
        ).fetchall()
        assert [row["active"] for row in history].count(1) == 1
        assert len(history) == 2
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 2


def test_nonbinary_choices_are_audited_but_never_train_either_model(
    tmp_path: Path,
) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)
    service = PreferenceService(database, FakeOpenClipProvider())

    for choice in ("tie", "skip", "both_bad"):
        result = service.record_pairwise(
            _request(
                album_id,
                selection_id,
                preferred_id,
                rejected_id,
                choice=choice,
            )
        )
        assert result.choice == choice
        assert result.contextual_trained is False
        assert result.contextual_comparisons == 0
        assert result.comparisons == 0

    repository = PreferenceRepository(database)
    events = repository.list_compatible_events(
        "local", FakeOpenClipProvider.name, FEATURE_SCHEMA
    )
    assert {event.choice for event in events} == {"tie", "skip", "both_bad"}
    assert len(events) == 3
    assert (
        repository.list_trainable_events(
            "local", FakeOpenClipProvider.name, FEATURE_SCHEMA
        )
        == []
    )
    assert (
        repository.load_active_model("local", FakeOpenClipProvider.name, FEATURE_SCHEMA)
        is None
    )
    assert load_preference_model(database).comparisons == 0
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 3


def test_stale_photo_cache_rejects_feedback_without_any_write(tmp_path: Path) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE photos SET embedding_source_size = embedding_source_size + 1
            WHERE id = ?
            """,
            (rejected_id,),
        )

    with pytest.raises(KeyError, match="current semantic cache"):
        PreferenceService(database, FakeOpenClipProvider()).record_pairwise(
            _request(album_id, selection_id, preferred_id, rejected_id)
        )

    with database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM preference_events").fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM user_preferences").fetchone()[0]
            == 0
        )


def test_openclip_query_failure_is_not_silently_downgraded(tmp_path: Path) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)

    with pytest.raises(ValueError, match="test query rejected"):
        PreferenceService(
            database, FakeOpenClipProvider(reject_query=True)
        ).record_pairwise(_request(album_id, selection_id, preferred_id, rejected_id))

    with database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM preference_events").fetchone()[0]
            == 0
        )


def test_contextual_feedback_rejects_selection_provider_drift(tmp_path: Path) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)
    with database.connect() as connection:
        connection.execute(
            "UPDATE selections SET result_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "user_id": "local",
                        "query_text": "选 2 张城市夜景",
                        "provider_fingerprint": "openclip-different-provider-v1",
                    },
                    ensure_ascii=False,
                ),
                selection_id,
            ),
        )

    with pytest.raises(ValueError, match="provider drift"):
        PreferenceService(database, FakeOpenClipProvider()).record_pairwise(
            _request(album_id, selection_id, preferred_id, rejected_id)
        )

    with database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM preference_events").fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


def test_event_is_durable_if_training_fails_after_event_insert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)

    def fail_training(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr("ai.preferences.service.train", fail_training)
    with pytest.raises(RuntimeError, match="optimizer failure"):
        PreferenceService(database, FakeOpenClipProvider()).record_pairwise(
            _request(album_id, selection_id, preferred_id, rejected_id)
        )

    with database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM preference_events").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM preference_models").fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM user_preferences").fetchone()[0]
            == 0
        )


def test_pairwise_and_state_http_api_expose_optional_contextual_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)
    provider = FakeOpenClipProvider()
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "embedding_provider", lambda: provider)

    with TestClient(app_module.app) as client:
        feedback = client.post(
            "/feedback/pairwise",
            json={
                "album_id": album_id,
                "selection_id": selection_id,
                "preferred_photo_id": preferred_id,
                "rejected_photo_id": rejected_id,
            },
        )
        state = client.get("/preferences/local")

    assert feedback.status_code == 200
    assert feedback.json()["algorithm"] == CONTEXTUAL_ALGORITHM
    assert feedback.json()["feature_schema"] == FEATURE_SCHEMA
    assert feedback.json()["contextual_comparisons"] == 1
    assert feedback.json()["legacy_audit_persisted"] is True
    assert state.status_code == 200
    assert state.json()["contextual_model_id"] == feedback.json()["contextual_model_id"]
    assert state.json()["contextual_comparisons"] == 1


def test_selection_feedback_rejects_cross_user_model_pollution(tmp_path: Path) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)
    request = _request(album_id, selection_id, preferred_id, rejected_id).model_copy(
        update={"user_id": "other-user"}
    )

    with pytest.raises(ValueError, match="belong to the local user"):
        PreferenceService(database, FakeOpenClipProvider()).record_pairwise(request)

    with database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM preference_events").fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


def test_contextual_success_survives_optional_legacy_audit_failure(
    tmp_path: Path,
) -> None:
    database, album_id, selection_id, preferred_id, rejected_id = _context(tmp_path)
    with database.connect() as connection:
        connection.executescript(
            """
            CREATE TRIGGER reject_legacy_feedback_audit
            BEFORE INSERT ON feedback
            BEGIN
                SELECT RAISE(ABORT, 'synthetic legacy audit failure');
            END;
            """
        )

    result = PreferenceService(database, FakeOpenClipProvider()).record_pairwise(
        _request(album_id, selection_id, preferred_id, rejected_id)
    )

    assert result.legacy_audit_persisted is False
    assert result.contextual_trained is True
    assert result.contextual_comparisons == 1
    assert result.comparisons == 1
    with database.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM preference_events").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM preference_models").fetchone()[0]
            == 1
        )
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0
