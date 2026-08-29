from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from ai.preferences.repository import (
    PreferenceEvent,
    PreferenceModelRecord,
    PreferenceRepository,
)
from ai.storage import Database


def _database(tmp_path: Path) -> Database:
    return Database(tmp_path / "data" / "norma.db")


def _event(
    event_id: str,
    *,
    user_id: str = "local",
    provider: str = "openclip:test",
    schema: str = "contextual-v1",
    choice: str = "preferred",
    created_at: str | None = "2026-08-28T10:00:00+00:00",
) -> PreferenceEvent:
    return PreferenceEvent(
        id=event_id,
        user_id=user_id,
        album_id="album-1",
        selection_id="selection-1",
        query_text="选两张夜景",
        preferred_photo_id=f"preferred-{event_id}",
        rejected_photo_id=f"rejected-{event_id}",
        choice=choice,
        provider_fingerprint=provider,
        feature_schema=schema,
        preferred_features=(0.1, 0.2),
        rejected_features=(-0.2, 0.4),
        base_margin=0.125,
        context={"z": 2, "a": {"source": "active-pair"}},
        model_id_at_display="display-model",
        created_at=created_at,
    )


def _model(
    model_id: str,
    *,
    user_id: str = "local",
    provider: str = "openclip:test",
    schema: str = "contextual-v1",
    mean: tuple[float, ...] = (0.25, -0.5),
) -> PreferenceModelRecord:
    dimension = len(mean)
    covariance = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )
    return PreferenceModelRecord(
        id=model_id,
        user_id=user_id,
        algorithm="bayesian-linear-pairwise-v1",
        provider_fingerprint=provider,
        feature_schema=schema,
        projection_id="projection-v1",
        mean=mean,
        covariance=covariance,
        training_pair_count=3,
        training_event_digest=f"digest-{model_id}",
        hyperparameters={"prior_precision": 2.0, "seed": 0},
        diagnostics={"converged": True, "loss": 0.42},
        created_at="2026-08-28T11:00:00+00:00",
    )


def test_fresh_schema_has_v10_preference_tables_and_no_history_fks(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    database.initialize()

    with database.connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        active_index = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_preference_models_one_active'
            """
        ).fetchone()
        event_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(preference_events)"
        ).fetchall()

    assert database.current_version() == 14
    assert {"preference_events", "preference_models"} <= tables
    assert active_index is not None
    assert "WHERE active = 1" in active_index["sql"]
    assert event_foreign_keys == []


def test_event_round_trip_is_canonical_and_database_immutable(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    repository = PreferenceRepository(database)
    event = _event("event-1")

    stored = repository.insert_event(event)

    assert stored == event
    assert repository.get_event(event.id) == event
    with database.connect() as connection:
        payload = connection.execute(
            """
            SELECT preferred_features_json, context_json
            FROM preference_events WHERE id = ?
            """,
            (event.id,),
        ).fetchone()
    assert payload["preferred_features_json"] == "[0.1,0.2]"
    assert payload["context_json"] == ('{"a":{"source":"active-pair"},"z":2}')

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with database.connect() as connection:
            connection.execute(
                "UPDATE preference_events SET query_text = 'changed' WHERE id = ?",
                (event.id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with database.connect() as connection:
            connection.execute(
                "DELETE FROM preference_events WHERE id = ?", (event.id,)
            )
    with pytest.raises(sqlite3.IntegrityError):
        repository.insert_event(event)

    assert repository.get_event(event.id) == event


def test_suggestion_feedback_key_is_one_shot_while_legacy_events_remain_allowed(
    tmp_path: Path,
) -> None:
    repository = PreferenceRepository(_database(tmp_path))
    first = replace(
        _event("suggestion-first", choice="tie"), suggestion_id="suggestion-1"
    )
    duplicate = replace(_event("suggestion-duplicate"), suggestion_id="suggestion-1")

    stored = repository.insert_event(first)
    assert stored.suggestion_id == "suggestion-1"
    with pytest.raises(sqlite3.IntegrityError, match="suggestion_id"):
        repository.insert_event(duplicate)

    repository.insert_event(_event("legacy-null-1"))
    repository.insert_event(_event("legacy-null-2"))
    assert repository.get_event("legacy-null-1").suggestion_id is None


def test_compatible_and_trainable_events_are_isolated_and_deterministic(
    tmp_path: Path,
) -> None:
    repository = PreferenceRepository(_database(tmp_path))
    same_time = "2026-08-28T10:00:00+00:00"
    for event in (
        _event("b", created_at=same_time),
        _event("a", created_at=same_time),
        _event("c", choice="tie", created_at="2026-08-28T10:01:00+00:00"),
        _event("other-provider", provider="openclip:other"),
        _event("other-schema", schema="contextual-v2"),
        _event("other-user", user_id="another-user"),
    ):
        repository.insert_event(event)

    compatible = repository.list_compatible_events(
        "local", "openclip:test", "contextual-v1"
    )
    trainable = repository.list_trainable_events(
        "local", "openclip:test", "contextual-v1"
    )

    assert [event.id for event in compatible] == ["a", "b", "c"]
    assert [event.id for event in trainable] == ["a", "b"]
    assert (
        repository.list_binary_events("local", "openclip:test", "contextual-v1")
        == trainable
    )


def test_activate_model_is_atomic_versioned_and_provider_scoped(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    repository = PreferenceRepository(database)
    repository.insert_event(_event("training-event"))

    first = repository.activate_model(_model("model-1"))
    second = repository.activate_model(_model("model-2"))
    other_provider = repository.activate_model(
        _model("model-other", provider="openclip:other")
    )

    assert first.active
    assert second.active
    assert other_provider.active
    assert (
        repository.load_active_model("local", "openclip:test", "contextual-v1")
        == second
    )
    assert (
        repository.load_active_model("local", "openclip:other", "contextual-v1")
        == other_provider
    )
    with database.connect() as connection:
        versions = connection.execute(
            """
            SELECT id, active FROM preference_models
            WHERE provider_fingerprint = 'openclip:test' ORDER BY id
            """
        ).fetchall()
    assert [(row["id"], row["active"]) for row in versions] == [
        ("model-1", 0),
        ("model-2", 1),
    ]

    # Deactivation and insertion share one transaction. The duplicate primary
    # key fails after the UPDATE, and rollback must leave model-2 active.
    with pytest.raises(sqlite3.IntegrityError):
        repository.activate_model(_model("model-1"))
    assert (
        repository.load_active_model("local", "openclip:test", "contextual-v1")
        == second
    )


def test_model_payload_is_immutable_except_repository_deactivation(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    repository = PreferenceRepository(database)
    repository.activate_model(_model("model-1"))

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with database.connect() as connection:
            connection.execute(
                "UPDATE preference_models SET mean_json = '[9,9]' WHERE id = 'model-1'"
            )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with database.connect() as connection:
            connection.execute("DELETE FROM preference_models WHERE id = 'model-1'")

    assert repository.load_active_model(
        "local", "openclip:test", "contextual-v1"
    ) == _model("model-1")


def test_repository_rejects_non_finite_or_inconsistent_arrays(
    tmp_path: Path,
) -> None:
    repository = PreferenceRepository(_database(tmp_path))

    with pytest.raises(ValueError, match="finite"):
        repository.insert_event(
            replace(_event("nan-event"), preferred_features=(float("nan"), 0.0))
        )
    with pytest.raises(ValueError, match="dimensions"):
        repository.insert_event(
            replace(_event("bad-dimension"), rejected_features=(0.0,))
        )
    with pytest.raises(ValueError, match="shape"):
        repository.activate_model(
            replace(_model("bad-covariance"), covariance=((1.0, 0.0),))
        )

    repository.insert_event(_event("dimension-source"))
    with pytest.raises(ValueError, match="does not match"):
        repository.activate_model(_model("wrong-model-dimension", mean=(0.0,)))
    assert (
        repository.load_active_model("local", "openclip:test", "contextual-v1") is None
    )


def test_v9_upgrade_preserves_legacy_user_preferences(tmp_path: Path) -> None:
    database = _database(tmp_path)
    database.path.parent.mkdir(parents=True)
    with sqlite3.connect(database.path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO schema_migrations(version) VALUES (9);
            CREATE TABLE user_preferences (
                user_id TEXT PRIMARY KEY,
                parameters_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO user_preferences(user_id, parameters_json)
            VALUES ('legacy-user', '{"version":1,"comparisons":2}');
            """
        )

    database.initialize()

    with database.connect() as connection:
        legacy = connection.execute(
            "SELECT parameters_json FROM user_preferences WHERE user_id = 'legacy-user'"
        ).fetchone()
        migrated = connection.execute(
            "SELECT 1 FROM preference_events LIMIT 1"
        ).fetchone()
    assert database.current_version() == 14
    assert legacy["parameters_json"] == '{"version":1,"comparisons":2}'
    assert migrated is None


def test_v10_upgrade_adds_one_shot_suggestion_key_without_mutating_events(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    database.path.parent.mkdir(parents=True)
    with sqlite3.connect(database.path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO schema_migrations(version) VALUES (10);
            CREATE TABLE preference_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                album_id TEXT NOT NULL,
                selection_id TEXT,
                query_text TEXT NOT NULL,
                preferred_photo_id TEXT NOT NULL,
                rejected_photo_id TEXT NOT NULL,
                choice TEXT NOT NULL,
                provider_fingerprint TEXT NOT NULL,
                feature_schema TEXT NOT NULL,
                preferred_features_json TEXT NOT NULL,
                rejected_features_json TEXT NOT NULL,
                base_margin REAL NOT NULL,
                context_json TEXT NOT NULL,
                model_id_at_display TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO preference_events(
                id, user_id, album_id, selection_id, query_text,
                preferred_photo_id, rejected_photo_id, choice,
                provider_fingerprint, feature_schema,
                preferred_features_json, rejected_features_json,
                base_margin, context_json, model_id_at_display
            ) VALUES (
                'legacy-event', 'local', 'album', 'selection', 'query',
                'left', 'right', 'tie', 'provider', 'schema',
                '[0.0]', '[1.0]', 0.0, '{}', NULL
            );
            """
        )

    database.initialize()

    with database.connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(preference_events)")
        }
        index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_preference_events_one_per_suggestion'"
        ).fetchone()
        legacy = connection.execute(
            "SELECT suggestion_id FROM preference_events WHERE id = 'legacy-event'"
        ).fetchone()

    assert database.current_version() == 14
    assert "suggestion_id" in columns
    assert index is not None
    assert "WHERE suggestion_id IS NOT NULL" in index["sql"]
    assert legacy["suggestion_id"] is None
