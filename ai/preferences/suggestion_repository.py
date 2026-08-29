from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Mapping, Sequence

from ai.preferences.contextual import FEATURE_DIMENSION
from ai.storage import Database


@dataclass(frozen=True, slots=True)
class PreferenceSuggestionRecord:
    id: str
    selection_id: str
    album_id: str
    user_id: str
    query_text: str
    left_photo_id: str
    right_photo_id: str
    provider_fingerprint: str
    feature_schema: str
    projection_id: str
    model_id_at_display: str | None
    acquisition_version: str
    constraint_solver: str
    mode: str
    candidate_digest: str
    candidate_source_digest: str
    candidate_ids: tuple[str, ...]
    left_features: tuple[float, ...]
    right_features: tuple[float, ...]
    request: Mapping[str, object]
    diagnostics: Mapping[str, object]
    result: Mapping[str, object]
    created_at: str | None = None


class PreferenceSuggestionRepository:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.initialize()

    def insert(self, record: PreferenceSuggestionRecord) -> PreferenceSuggestionRecord:
        normalized = _normalize_record(record)
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO preference_suggestions(
                    id, selection_id, album_id, user_id, query_text,
                    left_photo_id, right_photo_id, provider_fingerprint,
                    feature_schema, projection_id, model_id_at_display,
                    acquisition_version, constraint_solver, mode,
                    candidate_digest, candidate_source_digest,
                    candidate_ids_json, left_features_json, right_features_json,
                    request_json, diagnostics_json, result_json, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP)
                )
                """,
                (
                    normalized.id,
                    normalized.selection_id,
                    normalized.album_id,
                    normalized.user_id,
                    normalized.query_text,
                    normalized.left_photo_id,
                    normalized.right_photo_id,
                    normalized.provider_fingerprint,
                    normalized.feature_schema,
                    normalized.projection_id,
                    normalized.model_id_at_display,
                    normalized.acquisition_version,
                    normalized.constraint_solver,
                    normalized.mode,
                    normalized.candidate_digest,
                    normalized.candidate_source_digest,
                    _canonical_json(normalized.candidate_ids, "candidate_ids"),
                    _canonical_json(normalized.left_features, "left_features"),
                    _canonical_json(normalized.right_features, "right_features"),
                    _canonical_json(normalized.request, "request"),
                    _canonical_json(normalized.diagnostics, "diagnostics"),
                    _canonical_json(normalized.result, "result"),
                    normalized.created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM preference_suggestions WHERE id = ?", (normalized.id,)
            ).fetchone()
        if row is None:  # pragma: no cover - SQLite insert/read invariant
            raise RuntimeError(
                f"preference suggestion was not persisted: {normalized.id}"
            )
        return _from_row(row)

    def get(self, suggestion_id: str) -> PreferenceSuggestionRecord:
        _require_text(suggestion_id, "suggestion_id")
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM preference_suggestions WHERE id = ?", (suggestion_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"preference suggestion not found: {suggestion_id}")
        return _from_row(row)

    def list_excluded_pairs(
        self,
        *,
        selection_id: str,
        user_id: str,
        provider_fingerprint: str,
        feature_schema: str,
        candidate_digest: str,
    ) -> list[tuple[str, str]]:
        for value, label in (
            (selection_id, "selection_id"),
            (user_id, "user_id"),
            (provider_fingerprint, "provider_fingerprint"),
            (feature_schema, "feature_schema"),
            (candidate_digest, "candidate_digest"),
        ):
            _require_text(value, label)
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT left_photo_id, right_photo_id
                FROM preference_suggestions
                WHERE selection_id = ? AND user_id = ?
                  AND provider_fingerprint = ? AND feature_schema = ?
                  AND candidate_digest = ?
                ORDER BY created_at, id
                """,
                (
                    selection_id,
                    user_id,
                    provider_fingerprint,
                    feature_schema,
                    candidate_digest,
                ),
            ).fetchall()
        return [(str(row["left_photo_id"]), str(row["right_photo_id"])) for row in rows]


def _normalize_record(record: PreferenceSuggestionRecord) -> PreferenceSuggestionRecord:
    for value, label in (
        (record.id, "record.id"),
        (record.selection_id, "record.selection_id"),
        (record.album_id, "record.album_id"),
        (record.user_id, "record.user_id"),
        (record.query_text, "record.query_text"),
        (record.left_photo_id, "record.left_photo_id"),
        (record.right_photo_id, "record.right_photo_id"),
        (record.provider_fingerprint, "record.provider_fingerprint"),
        (record.feature_schema, "record.feature_schema"),
        (record.projection_id, "record.projection_id"),
        (record.acquisition_version, "record.acquisition_version"),
        (record.constraint_solver, "record.constraint_solver"),
        (record.candidate_digest, "record.candidate_digest"),
        (record.candidate_source_digest, "record.candidate_source_digest"),
    ):
        _require_text(value, label)
    if record.left_photo_id == record.right_photo_id:
        raise ValueError("suggestion left and right photos must differ")
    if record.mode not in {"shortlist", "exhaustive"}:
        raise ValueError("suggestion mode must be shortlist or exhaustive")
    if record.model_id_at_display is not None:
        _require_text(record.model_id_at_display, "record.model_id_at_display")
    if record.created_at is not None:
        _require_text(record.created_at, "record.created_at")
    candidate_ids = _text_tuple(record.candidate_ids, "candidate_ids")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must not contain duplicates")
    if (
        record.left_photo_id not in candidate_ids
        or record.right_photo_id not in candidate_ids
    ):
        raise ValueError("suggestion pair must belong to candidate_ids")
    left_features = _finite_vector(record.left_features, "left_features")
    right_features = _finite_vector(record.right_features, "right_features")
    if (
        len(left_features) != FEATURE_DIMENSION
        or len(right_features) != FEATURE_DIMENSION
    ):
        raise ValueError(f"suggestion features must have dimension {FEATURE_DIMENSION}")
    return PreferenceSuggestionRecord(
        id=record.id,
        selection_id=record.selection_id,
        album_id=record.album_id,
        user_id=record.user_id,
        query_text=record.query_text,
        left_photo_id=record.left_photo_id,
        right_photo_id=record.right_photo_id,
        provider_fingerprint=record.provider_fingerprint,
        feature_schema=record.feature_schema,
        projection_id=record.projection_id,
        model_id_at_display=record.model_id_at_display,
        acquisition_version=record.acquisition_version,
        constraint_solver=record.constraint_solver,
        mode=record.mode,
        candidate_digest=record.candidate_digest,
        candidate_source_digest=record.candidate_source_digest,
        candidate_ids=candidate_ids,
        left_features=left_features,
        right_features=right_features,
        request=_json_mapping(record.request, "request"),
        diagnostics=_json_mapping(record.diagnostics, "diagnostics"),
        result=_json_mapping(record.result, "result"),
        created_at=record.created_at,
    )


def _from_row(row: sqlite3.Row) -> PreferenceSuggestionRecord:
    return _normalize_record(
        PreferenceSuggestionRecord(
            id=row["id"],
            selection_id=row["selection_id"],
            album_id=row["album_id"],
            user_id=row["user_id"],
            query_text=row["query_text"],
            left_photo_id=row["left_photo_id"],
            right_photo_id=row["right_photo_id"],
            provider_fingerprint=row["provider_fingerprint"],
            feature_schema=row["feature_schema"],
            projection_id=row["projection_id"],
            model_id_at_display=row["model_id_at_display"],
            acquisition_version=row["acquisition_version"],
            constraint_solver=row["constraint_solver"],
            mode=row["mode"],
            candidate_digest=row["candidate_digest"],
            candidate_source_digest=row["candidate_source_digest"],
            candidate_ids=_text_tuple_from_json(
                row["candidate_ids_json"], "candidate_ids"
            ),
            left_features=_vector_from_json(row["left_features_json"], "left_features"),
            right_features=_vector_from_json(
                row["right_features_json"], "right_features"
            ),
            request=_mapping_from_json(row["request_json"], "request"),
            diagnostics=_mapping_from_json(row["diagnostics_json"], "diagnostics"),
            result=_mapping_from_json(row["result_json"], "result"),
            created_at=row["created_at"],
        )
    )


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be non-empty text without edge whitespace")


def _text_tuple(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a string sequence")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{label} must be a string sequence") from error
    if not result:
        raise ValueError(f"{label} must not be empty")
    for value in result:
        _require_text(value, label)
    return result


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a numeric sequence")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric sequence") from error
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _canonical_json(value: object, label: str) -> str:
    normalized = _normalize_json(value, label)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _normalize_json(value: object, label: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item, label) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_json(item, label) for item in value]
    raise ValueError(f"{label} contains a non-JSON value")


def _json_mapping(value: object, label: str) -> dict[str, object]:
    normalized = _normalize_json(value, label)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def _json_from_text(payload: str, label: str) -> object:
    try:
        return json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"stored {label} is invalid JSON") from error


def _mapping_from_json(payload: str, label: str) -> dict[str, object]:
    return _json_mapping(_json_from_text(payload, label), label)


def _vector_from_json(payload: str, label: str) -> tuple[float, ...]:
    value = _json_from_text(payload, label)
    if not isinstance(value, list):
        raise ValueError(f"stored {label} must be a JSON array")
    return _finite_vector(value, label)


def _text_tuple_from_json(payload: str, label: str) -> tuple[str, ...]:
    value = _json_from_text(payload, label)
    if not isinstance(value, list):
        raise ValueError(f"stored {label} must be a JSON array")
    return _text_tuple(value, label)


__all__ = [
    "PreferenceSuggestionRecord",
    "PreferenceSuggestionRepository",
]
