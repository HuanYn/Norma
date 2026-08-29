from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Mapping, Sequence

from ai.storage import Database


PREFERENCE_CHOICES = frozenset({"preferred", "tie", "skip", "both_bad"})
TRAINABLE_CHOICES = frozenset({"preferred"})


@dataclass(frozen=True, slots=True)
class PreferenceEvent:
    id: str
    user_id: str
    album_id: str
    selection_id: str | None
    query_text: str
    preferred_photo_id: str
    rejected_photo_id: str
    choice: str
    provider_fingerprint: str
    feature_schema: str
    preferred_features: tuple[float, ...]
    rejected_features: tuple[float, ...]
    base_margin: float
    context: Mapping[str, object]
    model_id_at_display: str | None
    suggestion_id: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class PreferenceModelRecord:
    id: str
    user_id: str
    algorithm: str
    provider_fingerprint: str
    feature_schema: str
    projection_id: str | None
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    training_pair_count: int
    training_event_digest: str
    hyperparameters: Mapping[str, object]
    diagnostics: Mapping[str, object]
    active: bool = True
    created_at: str | None = None


class PreferenceRepository:
    """Immutable preference event log and versioned posterior storage."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.initialize()

    def insert_event(self, event: PreferenceEvent) -> PreferenceEvent:
        normalized = _normalize_event(event)
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO preference_events(
                    id, user_id, album_id, selection_id, suggestion_id, query_text,
                    preferred_photo_id, rejected_photo_id, choice,
                    provider_fingerprint, feature_schema,
                    preferred_features_json, rejected_features_json,
                    base_margin, context_json, model_id_at_display, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE(?, CURRENT_TIMESTAMP)
                )
                """,
                (
                    normalized.id,
                    normalized.user_id,
                    normalized.album_id,
                    normalized.selection_id,
                    normalized.suggestion_id,
                    normalized.query_text,
                    normalized.preferred_photo_id,
                    normalized.rejected_photo_id,
                    normalized.choice,
                    normalized.provider_fingerprint,
                    normalized.feature_schema,
                    _canonical_json(
                        normalized.preferred_features, "preferred_features"
                    ),
                    _canonical_json(normalized.rejected_features, "rejected_features"),
                    normalized.base_margin,
                    _canonical_json(normalized.context, "context"),
                    normalized.model_id_at_display,
                    normalized.created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM preference_events WHERE id = ?", (normalized.id,)
            ).fetchone()
        if row is None:  # pragma: no cover - SQLite insert/read invariant
            raise RuntimeError(f"preference event was not persisted: {normalized.id}")
        return _event_from_row(row)

    def get_event(self, event_id: str) -> PreferenceEvent:
        _require_text(event_id, "event_id")
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM preference_events WHERE id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"preference event not found: {event_id}")
        return _event_from_row(row)

    def list_compatible_events(
        self,
        user_id: str,
        provider_fingerprint: str,
        feature_schema: str,
        *,
        choices: Sequence[str] | None = None,
    ) -> list[PreferenceEvent]:
        _require_text(user_id, "user_id")
        _require_text(provider_fingerprint, "provider_fingerprint")
        _require_text(feature_schema, "feature_schema")
        parameters: list[object] = [
            user_id,
            provider_fingerprint,
            feature_schema,
        ]
        where = "user_id = ? AND provider_fingerprint = ? AND feature_schema = ?"
        if choices is not None:
            normalized_choices = tuple(dict.fromkeys(choices))
            invalid = set(normalized_choices) - PREFERENCE_CHOICES
            if invalid:
                raise ValueError(
                    "unsupported preference choices: " + ", ".join(sorted(invalid))
                )
            if not normalized_choices:
                return []
            placeholders = ",".join("?" for _ in normalized_choices)
            where += f" AND choice IN ({placeholders})"
            parameters.extend(normalized_choices)
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM preference_events
                WHERE {where}
                ORDER BY created_at ASC, id ASC
                """,
                parameters,
            ).fetchall()
        events = [_event_from_row(row) for row in rows]
        dimensions = {len(event.preferred_features) for event in events}
        if len(dimensions) > 1:
            raise ValueError(
                "compatible preference events have inconsistent feature dimensions"
            )
        return events

    def list_trainable_events(
        self, user_id: str, provider_fingerprint: str, feature_schema: str
    ) -> list[PreferenceEvent]:
        return self.list_compatible_events(
            user_id,
            provider_fingerprint,
            feature_schema,
            choices=tuple(TRAINABLE_CHOICES),
        )

    def list_binary_events(
        self, user_id: str, provider_fingerprint: str, feature_schema: str
    ) -> list[PreferenceEvent]:
        return self.list_trainable_events(user_id, provider_fingerprint, feature_schema)

    def activate_model(self, model: PreferenceModelRecord) -> PreferenceModelRecord:
        normalized = _normalize_model(model)
        if not normalized.active:
            raise ValueError("a newly activated preference model must be active")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_model_dimension(connection, normalized)
            connection.execute(
                """
                UPDATE preference_models SET active = 0
                WHERE user_id = ? AND provider_fingerprint = ?
                  AND feature_schema = ? AND active = 1
                """,
                (
                    normalized.user_id,
                    normalized.provider_fingerprint,
                    normalized.feature_schema,
                ),
            )
            connection.execute(
                """
                INSERT INTO preference_models(
                    id, user_id, algorithm, provider_fingerprint,
                    feature_schema, projection_id, mean_json, covariance_json,
                    training_pair_count, training_event_digest,
                    hyperparameters_json, diagnostics_json, active, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1,
                    COALESCE(?, CURRENT_TIMESTAMP)
                )
                """,
                (
                    normalized.id,
                    normalized.user_id,
                    normalized.algorithm,
                    normalized.provider_fingerprint,
                    normalized.feature_schema,
                    normalized.projection_id,
                    _canonical_json(normalized.mean, "mean"),
                    _canonical_json(normalized.covariance, "covariance"),
                    normalized.training_pair_count,
                    normalized.training_event_digest,
                    _canonical_json(normalized.hyperparameters, "hyperparameters"),
                    _canonical_json(normalized.diagnostics, "diagnostics"),
                    normalized.created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM preference_models WHERE id = ?", (normalized.id,)
            ).fetchone()
        if row is None:  # pragma: no cover - SQLite insert/read invariant
            raise RuntimeError(f"preference model was not persisted: {normalized.id}")
        return _model_from_row(row)

    def load_active_model(
        self, user_id: str, provider_fingerprint: str, feature_schema: str
    ) -> PreferenceModelRecord | None:
        _require_text(user_id, "user_id")
        _require_text(provider_fingerprint, "provider_fingerprint")
        _require_text(feature_schema, "feature_schema")
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM preference_models
                WHERE user_id = ? AND provider_fingerprint = ?
                  AND feature_schema = ? AND active = 1
                """,
                (user_id, provider_fingerprint, feature_schema),
            ).fetchone()
        return _model_from_row(row) if row is not None else None

    @staticmethod
    def _validate_model_dimension(
        connection: sqlite3.Connection, model: PreferenceModelRecord
    ) -> None:
        rows = connection.execute(
            """
            SELECT preferred_features_json, rejected_features_json
            FROM preference_events
            WHERE user_id = ? AND provider_fingerprint = ? AND feature_schema = ?
            """,
            (model.user_id, model.provider_fingerprint, model.feature_schema),
        ).fetchall()
        expected = len(model.mean)
        for row in rows:
            preferred = _vector_from_json(
                row["preferred_features_json"], "preferred_features"
            )
            rejected = _vector_from_json(
                row["rejected_features_json"], "rejected_features"
            )
            if len(preferred) != expected or len(rejected) != expected:
                raise ValueError(
                    "preference model dimension does not match compatible events"
                )


def _normalize_event(event: PreferenceEvent) -> PreferenceEvent:
    _require_text(event.id, "event.id")
    _require_text(event.user_id, "event.user_id")
    _require_text(event.album_id, "event.album_id")
    _optional_text(event.selection_id, "event.selection_id")
    if not isinstance(event.query_text, str):
        raise ValueError("event.query_text must be a string")
    _require_text(event.preferred_photo_id, "event.preferred_photo_id")
    _require_text(event.rejected_photo_id, "event.rejected_photo_id")
    if event.preferred_photo_id == event.rejected_photo_id:
        raise ValueError("preferred and rejected photos must be different")
    if event.choice not in PREFERENCE_CHOICES:
        raise ValueError(f"unsupported preference choice: {event.choice}")
    _require_text(event.provider_fingerprint, "event.provider_fingerprint")
    _require_text(event.feature_schema, "event.feature_schema")
    preferred = _finite_vector(event.preferred_features, "preferred_features")
    rejected = _finite_vector(event.rejected_features, "rejected_features")
    if len(preferred) != len(rejected):
        raise ValueError("preferred and rejected feature dimensions must match")
    base_margin = _finite_number(event.base_margin, "base_margin")
    context = _json_mapping(event.context, "context")
    _optional_text(event.model_id_at_display, "event.model_id_at_display")
    _optional_text(event.suggestion_id, "event.suggestion_id")
    _optional_text(event.created_at, "event.created_at")
    return PreferenceEvent(
        id=event.id,
        user_id=event.user_id,
        album_id=event.album_id,
        selection_id=event.selection_id,
        query_text=event.query_text,
        preferred_photo_id=event.preferred_photo_id,
        rejected_photo_id=event.rejected_photo_id,
        choice=event.choice,
        provider_fingerprint=event.provider_fingerprint,
        feature_schema=event.feature_schema,
        preferred_features=preferred,
        rejected_features=rejected,
        base_margin=base_margin,
        context=context,
        model_id_at_display=event.model_id_at_display,
        suggestion_id=event.suggestion_id,
        created_at=event.created_at,
    )


def _normalize_model(model: PreferenceModelRecord) -> PreferenceModelRecord:
    _require_text(model.id, "model.id")
    _require_text(model.user_id, "model.user_id")
    _require_text(model.algorithm, "model.algorithm")
    _require_text(model.provider_fingerprint, "model.provider_fingerprint")
    _require_text(model.feature_schema, "model.feature_schema")
    _optional_text(model.projection_id, "model.projection_id")
    mean = _finite_vector(model.mean, "mean")
    covariance = _finite_matrix(model.covariance, len(mean), "covariance")
    if (
        isinstance(model.training_pair_count, bool)
        or not isinstance(model.training_pair_count, int)
        or model.training_pair_count < 0
    ):
        raise ValueError("training_pair_count must be a non-negative integer")
    _require_text(model.training_event_digest, "model.training_event_digest")
    hyperparameters = _json_mapping(model.hyperparameters, "hyperparameters")
    diagnostics = _json_mapping(model.diagnostics, "diagnostics")
    if not isinstance(model.active, bool):
        raise ValueError("model.active must be a boolean")
    _optional_text(model.created_at, "model.created_at")
    return PreferenceModelRecord(
        id=model.id,
        user_id=model.user_id,
        algorithm=model.algorithm,
        provider_fingerprint=model.provider_fingerprint,
        feature_schema=model.feature_schema,
        projection_id=model.projection_id,
        mean=mean,
        covariance=covariance,
        training_pair_count=model.training_pair_count,
        training_event_digest=model.training_event_digest,
        hyperparameters=hyperparameters,
        diagnostics=diagnostics,
        active=model.active,
        created_at=model.created_at,
    )


def _event_from_row(row: sqlite3.Row) -> PreferenceEvent:
    return _normalize_event(
        PreferenceEvent(
            id=row["id"],
            user_id=row["user_id"],
            album_id=row["album_id"],
            selection_id=row["selection_id"],
            query_text=row["query_text"],
            preferred_photo_id=row["preferred_photo_id"],
            rejected_photo_id=row["rejected_photo_id"],
            choice=row["choice"],
            provider_fingerprint=row["provider_fingerprint"],
            feature_schema=row["feature_schema"],
            preferred_features=_vector_from_json(
                row["preferred_features_json"], "preferred_features"
            ),
            rejected_features=_vector_from_json(
                row["rejected_features_json"], "rejected_features"
            ),
            base_margin=row["base_margin"],
            context=_mapping_from_json(row["context_json"], "context"),
            model_id_at_display=row["model_id_at_display"],
            suggestion_id=row["suggestion_id"],
            created_at=row["created_at"],
        )
    )


def _model_from_row(row: sqlite3.Row) -> PreferenceModelRecord:
    mean = _vector_from_json(row["mean_json"], "mean")
    return _normalize_model(
        PreferenceModelRecord(
            id=row["id"],
            user_id=row["user_id"],
            algorithm=row["algorithm"],
            provider_fingerprint=row["provider_fingerprint"],
            feature_schema=row["feature_schema"],
            projection_id=row["projection_id"],
            mean=mean,
            covariance=_matrix_from_json(
                row["covariance_json"], len(mean), "covariance"
            ),
            training_pair_count=row["training_pair_count"],
            training_event_digest=row["training_event_digest"],
            hyperparameters=_mapping_from_json(
                row["hyperparameters_json"], "hyperparameters"
            ),
            diagnostics=_mapping_from_json(row["diagnostics_json"], "diagnostics"),
            active=bool(row["active"]),
            created_at=row["created_at"],
        )
    )


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a numeric sequence")
    try:
        vector = tuple(_finite_number(value, label) for value in values)
    except TypeError as error:
        raise ValueError(f"{label} must be a numeric sequence") from error
    if not vector:
        raise ValueError(f"{label} must not be empty")
    return vector


def _finite_matrix(
    values: Sequence[Sequence[float]], dimension: int, label: str
) -> tuple[tuple[float, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a numeric matrix")
    try:
        rows = tuple(_finite_vector(row, label) for row in values)
    except TypeError as error:
        raise ValueError(f"{label} must be a numeric matrix") from error
    if len(rows) != dimension or any(len(row) != dimension for row in rows):
        raise ValueError(f"{label} must have shape ({dimension}, {dimension})")
    return rows


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must contain only numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must contain only finite numbers")
    return number


def _canonical_json(value: object, label: str) -> str:
    normalized = _normalize_json(value, label)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _mapping_from_json(payload: str, label: str) -> dict[str, object]:
    value = _json_from_text(payload, label)
    return _json_mapping(value, label)


def _vector_from_json(payload: str, label: str) -> tuple[float, ...]:
    value = _json_from_text(payload, label)
    if not isinstance(value, list):
        raise ValueError(f"stored {label} must be a JSON array")
    return _finite_vector(value, label)


def _matrix_from_json(
    payload: str, dimension: int, label: str
) -> tuple[tuple[float, ...], ...]:
    value = _json_from_text(payload, label)
    if not isinstance(value, list):
        raise ValueError(f"stored {label} must be a JSON matrix")
    return _finite_matrix(value, dimension, label)


def _json_from_text(payload: str, label: str) -> object:
    try:
        return json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"stored {label} is not valid JSON") from error


def _json_mapping(value: object, label: str) -> dict[str, object]:
    normalized = _normalize_json(value, label)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def _normalize_json(value: object, label: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} JSON object keys must be strings")
            normalized[key] = _normalize_json(item, label)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, label) for item in value]
    raise ValueError(f"{label} contains a non-JSON value: {type(value).__name__}")


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _optional_text(value: object, label: str) -> None:
    if value is not None:
        _require_text(value, label)
