"""Read-only, opt-in case-memory reranking over frozen OpenCLIP embeddings.

This is not DPO, model training, a learned user adapter, or VLM generation.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from ai.index.embedding import (
    EmbeddingProvider,
    embedding_cache_is_current,
    normalize_embedding,
    source_file_sha256,
)
from ai.storage import Database


MEMORY_EVIDENCE_SCHEMA = "openclip-case-memory-evidence-v1"
MEMORY_ALGORITHM = "frozen-openclip-case-memory-v1"
_INTERACTIVE_SOURCES = {"selection-pairwise-feedback", "pdrr-suggestion-feedback"}


class MemoryEvidenceUnavailableError(ValueError):
    """Optional snapshot capability is unavailable; feedback may still be logged."""


@dataclass(frozen=True, slots=True)
class MemoryCitation:
    event_id: str
    source: str
    query_similarity: float
    contribution: float


@dataclass(slots=True)
class MemoryRerankResult:
    enabled: bool
    deltas: dict[str, float]
    applied: bool = False
    allow_proxy: bool = False
    algorithm: str = MEMORY_ALGORITHM
    citations: dict[str, list[MemoryCitation]] = field(default_factory=dict)
    event_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reviewed_event_count: int = 0
    skipped_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _openclip_provider(provider: EmbeddingProvider) -> bool:
    return provider.name.casefold().startswith("openclip") and provider.dimension == 512


def _load_snapshot_photo(
    row: Mapping[str, object], provider: EmbeddingProvider
) -> tuple[np.ndarray, dict[str, object]]:
    if not embedding_cache_is_current(row, provider.name, strict_source_hash=True):
        raise ValueError("memory evidence requires a current content-verified embedding")
    path = Path(str(row["embedding_path"]))
    # A 512d .npy is only a few KB. Reject archives and unreasonable allocations.
    if path.suffix.casefold() != ".npy" or path.stat().st_size > 65_536:
        raise MemoryEvidenceUnavailableError("unsupported_embedding_file_budget")
    before = source_file_sha256(path)
    raw = np.load(path, allow_pickle=False)
    vector = normalize_embedding(raw, provider.dimension, label="memory embedding")
    after = source_file_sha256(path)
    if before != after or not embedding_cache_is_current(row, provider.name, strict_source_hash=True):
        raise ValueError("memory source or embedding changed while reading")
    return vector, {
        "photo_id": str(row["id"]),
        "source_sha256": str(row["embedding_source_sha256"]),
        "embedding_sha256": after,
    }


def build_memory_evidence(
    provider: EmbeddingProvider,
    query_vector: np.ndarray,
    preferred_row: Mapping[str, object],
    rejected_row: Mapping[str, object],
) -> dict[str, object]:
    """Snapshot new feedback's provenance without updating or training anything.

    Call while recording a *new* event. Never retrofit this evidence onto old
    events: present-day pixels cannot establish what was shown in the past.
    """
    if not _openclip_provider(provider):
        raise MemoryEvidenceUnavailableError("unsupported_embedding_provider")
    query = normalize_embedding(query_vector, provider.dimension, label="memory query")
    _, preferred = _load_snapshot_photo(preferred_row, provider)
    _, rejected = _load_snapshot_photo(rejected_row, provider)
    return {
        "schema": MEMORY_EVIDENCE_SCHEMA,
        "provider": provider.name,
        "dimension": provider.dimension,
        "query_vector": [float(value) for value in query],
        "preferred": preferred,
        "rejected": rejected,
    }


class MemoryReranker:
    def __init__(
        self,
        database: Database,
        provider: EmbeddingProvider,
        *,
        enabled: bool = False,
        allow_proxy: bool = False,
        max_events: int = 64,
        top_k: int = 8,
        max_delta: float = 0.15,
        min_query_similarity: float = 0.2,
    ) -> None:
        if not isinstance(enabled, bool) or not isinstance(allow_proxy, bool):
            raise ValueError("memory switches must be booleans")
        if isinstance(max_events, bool) or not isinstance(max_events, int) or not 1 <= max_events <= 256:
            raise ValueError("memory max_events must be an integer between 1 and 256")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 16:
            raise ValueError("memory top_k must be an integer between 1 and 16")
        if not math.isfinite(max_delta) or not 0 <= max_delta <= 0.15:
            raise ValueError("memory max_delta must be finite and between 0 and 0.15")
        if not math.isfinite(min_query_similarity) or not 0 <= min_query_similarity < 1:
            raise ValueError("memory min_query_similarity must be finite and between 0 and 1")
        self.database = database
        self.provider = provider
        self.enabled = enabled
        self.allow_proxy = allow_proxy
        self.max_events = max_events
        self.top_k = top_k
        self.max_delta = float(max_delta)
        self.min_query_similarity = float(min_query_similarity)

    def rerank(
        self,
        user_id: str,
        query_vector: np.ndarray,
        candidate_vectors: Mapping[str, np.ndarray],
    ) -> MemoryRerankResult:
        """Return only bounded soft-score deltas; caller must still solve constraints.

        The caller owns validating its current candidates. Historic case sources
        are independently revalidated here. No model call or DB write occurs.
        """
        result = MemoryRerankResult(
            enabled=self.enabled,
            allow_proxy=self.allow_proxy,
            deltas={photo_id: 0.0 for photo_id in candidate_vectors},
        )
        if not self.enabled:
            result.warnings.append("memory_disabled")
            return result
        if not _openclip_provider(self.provider):
            result.warnings.append("unsupported_embedding_provider")
            return result
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("memory user_id must be nonempty")
        query = normalize_embedding(query_vector, self.provider.dimension, label="memory query")
        candidates = {
            photo_id: normalize_embedding(vector, self.provider.dimension, label="memory candidate")
            for photo_id, vector in candidate_vectors.items()
        }
        if not candidates:
            return result
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM preference_events
                   WHERE user_id=? AND provider_fingerprint=? AND choice='preferred'
                   ORDER BY created_at DESC, id DESC LIMIT ?""",
                (user_id, self.provider.name, self.max_events + 1),
            ).fetchall()
        if len(rows) > self.max_events:
            result.warnings.append("memory_window_truncated")
        rows = rows[:self.max_events]
        result.reviewed_event_count = len(rows)
        ranked = []
        seen = set()
        for event in rows:
            try:
                context = json.loads(event["context_json"])
                if not isinstance(context, dict):
                    raise ValueError("invalid context")
                source = self._source(context)
                if source is None:
                    _skip(result, "excluded_provenance")
                    continue
                evidence = context.get("memory_evidence")
                if not isinstance(evidence, dict) or evidence.get("schema") != MEMORY_EVIDENCE_SCHEMA:
                    _skip(result, "missing_immutable_evidence")
                    continue
                if evidence.get("provider") != self.provider.name or evidence.get("dimension") != self.provider.dimension:
                    _skip(result, "incompatible_evidence")
                    continue
                history_query = normalize_embedding(
                    np.asarray(evidence["query_vector"]), self.provider.dimension,
                    label="historical memory query",
                )
                similarity = float(np.clip(np.dot(query, history_query), -1, 1))
                if similarity <= self.min_query_similarity:
                    _skip(result, "irrelevant_query")
                    continue
                key = (
                    event["preferred_photo_id"], event["rejected_photo_id"],
                    " ".join(str(event["query_text"]).split()).casefold(), source,
                )
                if key in seen:
                    _skip(result, "duplicate_case")
                    continue
                seen.add(key)
                ranked.append((similarity, str(event["id"]), source, event, evidence))
            except (ValueError, TypeError, KeyError, OverflowError):
                _skip(result, "invalid_evidence")
        # Only the most relevant valid cases count. Stable ID ordering resolves ties.
        ranked.sort(key=lambda item: (-item[0], item[1]))
        used = []
        photo_cache = {}
        for similarity, event_id, source, event, evidence in ranked:
            if len(used) == self.top_k:
                break
            try:
                preferred = self._case_photo(event, evidence, "preferred", photo_cache)
                rejected = self._case_photo(event, evidence, "rejected", photo_cache)
            except (ValueError, TypeError, KeyError, OSError, EOFError):
                _skip(result, "stale_or_invalid_source")
                continue
            weight = ((similarity - self.min_query_similarity) / (1 - self.min_query_similarity)) ** 2
            used.append((similarity, event_id, source, weight, preferred - rejected))
        if not used:
            result.warnings.append("no_relevant_valid_memory")
        normalizer = max(1.0, sum(item[3] for item in used))
        for photo_id, vector in candidates.items():
            citations = []
            delta = 0.0
            for similarity, event_id, source, weight, difference in used:
                contrast = float(np.clip(np.dot(vector, difference) / 2, -1, 1))
                contribution = self.max_delta * weight * contrast / normalizer
                delta += contribution
                if abs(contribution) > 1e-12:
                    citations.append(MemoryCitation(event_id, source, similarity, contribution))
            result.deltas[photo_id] = float(np.clip(delta, -self.max_delta, self.max_delta))
            if citations:
                result.citations[photo_id] = citations
        result.event_ids = [item[1] for item in used]
        result.applied = any(abs(delta) > 1e-12 for delta in result.deltas.values())
        for reason, count in sorted(result.skipped_counts.items()):
            result.warnings.append(f"{reason}:{count}")
        return result

    def _source(self, context: Mapping[str, object]) -> str | None:
        source = context.get("source")
        if source in _INTERACTIVE_SOURCES:
            # Explicit proxy flags always override a UI-like source label.
            if context.get("human_observed") is False or context.get("annotator") == "assistant_proxy":
                return None
            return "interactive_feedback"
        if (source == "assistant_proxy" and self.allow_proxy
                and context.get("authorized_by_user") is True
                and context.get("human_observed") is False):
            return "assistant_proxy"
        return None

    def _case_photo(self, event, evidence: Mapping[str, object], role: str, cache: dict) -> np.ndarray:
        photo_id = str(event[f"{role}_photo_id"])
        snapshot = evidence[role]
        if not isinstance(snapshot, dict) or snapshot.get("photo_id") != photo_id:
            raise ValueError("case identity mismatch")
        key = (photo_id, str(event["album_id"]))
        if key not in cache:
            with self.database.connect() as connection:
                row = connection.execute(
                    "SELECT * FROM photos WHERE id=? AND album_id=?", key
                ).fetchone()
            if row is None:
                raise ValueError("historic photo no longer exists")
            cache[key] = _load_snapshot_photo(row, self.provider)
        vector, current = cache[key]
        if (current["source_sha256"] != snapshot.get("source_sha256")
                or current["embedding_sha256"] != snapshot.get("embedding_sha256")):
            raise ValueError("historic source or vector changed")
        return vector


def _skip(result: MemoryRerankResult, reason: str) -> None:
    result.skipped_counts[reason] = result.skipped_counts.get(reason, 0) + 1
