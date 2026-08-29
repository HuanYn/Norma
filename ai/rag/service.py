from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ai.index.embedding import EmbeddingProvider
from ai.preferences.runtime import (
    IncompatiblePreferenceModelError,
    PreferenceRuntime,
    cosine_fallback_runtime,
    load_preference_runtime,
)
from ai.rag.engine import generate_grounded
from ai.rag.image_safety import (
    EvidenceImagePixelLimitError,
    InvalidEvidenceImageError,
    MAX_EVIDENCE_TOTAL_PIXELS,
    inspect_evidence_image,
)
from ai.rag.models import (
    EVIDENCE_SCHEMA,
    GroundedRAGRequest,
    RetrievalEvidence,
    build_evidence_bundle,
    canonical_sha256,
    evidence_descriptor,
    snapshot_image_file,
)
from ai.rag.providers import GroundedGenerationProvider
from ai.rag.security import contains_local_path, redact_local_paths
from ai.retrieval import RetrievalService
from ai.schemas import (
    AlbumRAGRequest,
    AlbumRAGResponse,
    AlbumSearchRequest,
    AlbumSearchResponse,
    RAGCitation,
    RAGClaim,
    RAGProvenance,
)
from ai.storage import Database


CANDIDATE_SNAPSHOT_SCHEMA = "norma-rag-candidate-decision-source-v2"
MAX_EVIDENCE_BYTES = 128 * 1024 * 1024
MAX_EMBEDDING_CACHE_BYTES = 1024 * 1024
_RAG_RUN_LOCK = threading.Lock()


class RAGDriftError(RuntimeError):
    """Retrieval evidence or its embedding snapshot changed during a run."""


class RAGEvidenceTooLargeError(ValueError):
    """The selected evidence exceeds the bounded local-VLM request budget."""


class RAGBusyError(RuntimeError):
    """Another process-local RAG run already owns the bounded CPU admission."""


@dataclass(frozen=True, slots=True)
class _DecisionSnapshot:
    user_id: str
    provider_fingerprint: str
    algorithm: str
    model_id: str | None
    comparisons: int
    feature_schema: str | None
    projection_id: str | None
    training_event_digest: str | None

    def digest_payload(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "comparisons": self.comparisons,
            "feature_schema": self.feature_schema,
            "model_id": self.model_id,
            "projection_id": self.projection_id,
            "provider_fingerprint": self.provider_fingerprint,
            "training_event_digest": self.training_event_digest,
            "user_id": self.user_id,
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    photo_id: str
    absolute_path: Path
    filename: str
    file_size: int
    source_mtime_ns: int
    embedding_provider: str
    embedding_source_size: int
    embedding_source_mtime_ns: int
    embedding_source_sha256: str
    embedding_sha256: str
    quality_score: float | None
    auto_reject: bool

    def digest_payload(self) -> dict[str, object]:
        return {
            "auto_reject": self.auto_reject,
            "embedding_provider": self.embedding_provider,
            "embedding_sha256": self.embedding_sha256,
            "embedding_source_mtime_ns": self.embedding_source_mtime_ns,
            "embedding_source_sha256": self.embedding_source_sha256,
            "embedding_source_size": self.embedding_source_size,
            "photo_id": self.photo_id,
            "quality_score": self.quality_score,
            "source_mtime_ns": self.source_mtime_ns,
            "source_size": self.file_size,
        }


@dataclass(frozen=True, slots=True)
class _CandidateSnapshot:
    candidates: tuple[_Candidate, ...]
    decision: _DecisionSnapshot
    source_digest: str

    @property
    def photo_ids(self) -> tuple[str, ...]:
        return tuple(candidate.photo_id for candidate in self.candidates)

    def by_id(self) -> dict[str, _Candidate]:
        return {candidate.photo_id: candidate for candidate in self.candidates}

    def embedding_digests(self) -> dict[str, str]:
        return {
            candidate.photo_id: candidate.embedding_sha256
            for candidate in self.candidates
        }


def _decision_from_runtime(runtime: PreferenceRuntime) -> _DecisionSnapshot:
    return _DecisionSnapshot(
        user_id=runtime.user_id,
        provider_fingerprint=runtime.provider_fingerprint,
        algorithm=runtime.algorithm,
        model_id=runtime.model_id,
        comparisons=runtime.comparisons,
        feature_schema=runtime.feature_schema,
        projection_id=runtime.projection_id,
        training_event_digest=runtime.training_event_digest,
    )


GenerationProviderFactory = Callable[[], GroundedGenerationProvider]


class GroundedRAGService:
    """Run learned retrieval, local VLM generation, and citation enforcement.

    The returned answer has referential citation/provenance validation only.
    Semantic entailment between cited pixels and generated claims is explicitly
    not verified.
    """

    def __init__(
        self,
        database: Database,
        retrieval: RetrievalService,
        generation_provider_factory: GenerationProviderFactory,
    ) -> None:
        self.database = database
        self.retrieval = retrieval
        self.embedding_provider: EmbeddingProvider = retrieval.provider
        self.generation_provider_factory = generation_provider_factory

    def run(self, album_id: str, request: AlbumRAGRequest) -> AlbumRAGResponse:
        if contains_local_path(request.user_id):
            raise ValueError("RAG user_id must not contain a local path")
        if not _RAG_RUN_LOCK.acquire(blocking=False):
            raise RAGBusyError("local RAG worker is busy; retry this request later")
        try:
            return self._run_admitted(album_id, request)
        finally:
            _RAG_RUN_LOCK.release()

    def _run_admitted(
        self, album_id: str, request: AlbumRAGRequest
    ) -> AlbumRAGResponse:
        before_search = self._candidate_snapshot(album_id, request.user_id)
        try:
            retrieval = self.retrieval.search(
                AlbumSearchRequest(
                    album_id=album_id,
                    query=request.query,
                    limit=request.top_k,
                    user_id=request.user_id,
                ),
                expected_embedding_sha256=before_search.embedding_digests(),
            )
        except OSError as error:
            raise RAGDriftError(
                "embedding cache changed while retrieval was running"
            ) from error
        except ValueError as error:
            if "cached embedding" in str(error).casefold():
                raise RAGDriftError(
                    "embedding cache changed while retrieval was running"
                ) from error
            raise
        after_search = self._candidate_snapshot(album_id, request.user_id)
        self._require_same_snapshot(before_search, after_search, "retrieval")
        self._require_search_decision(retrieval, after_search.decision)

        locked_snapshot = self._candidate_snapshot(album_id, request.user_id)
        self._require_same_snapshot(after_search, locked_snapshot, "RAG admission")
        return self._run_locked(
            album_id,
            request,
            retrieval,
            locked_snapshot,
        )

    def _run_locked(
        self,
        album_id: str,
        request: AlbumRAGRequest,
        retrieval: AlbumSearchResponse,
        locked_snapshot: _CandidateSnapshot,
    ) -> AlbumRAGResponse:
        by_id = locked_snapshot.by_id()
        if not retrieval.matches:
            raise ValueError("retrieval returned no evidence for grounded generation")
        evidence_size = 0
        for match in retrieval.matches:
            candidate = by_id.get(match.photo_id)
            if candidate is None:
                raise RAGDriftError(
                    f"retrieval returned photo outside candidate snapshot: {match.photo_id}"
                )
            evidence_size += candidate.file_size
        if evidence_size > MAX_EVIDENCE_BYTES:
            raise RAGEvidenceTooLargeError(
                "selected evidence exceeds the 128 MiB local-VLM request limit"
            )

        evidence: list[RetrievalEvidence] = []
        total_pixels = 0
        for rank, match in enumerate(retrieval.matches, start=1):
            candidate = by_id.get(match.photo_id)
            if candidate is None:
                raise RAGDriftError(
                    f"retrieval returned photo outside candidate snapshot: {match.photo_id}"
                )
            self._assert_source_current(candidate)
            try:
                image = snapshot_image_file(
                    candidate.absolute_path,
                    display_name=candidate.filename,
                )
            except ValueError as error:
                raise RAGDriftError(
                    f"evidence image changed while snapshotting photo_id {match.photo_id}"
                ) from error
            self._assert_source_current(candidate)
            if image.sha256 != candidate.embedding_source_sha256:
                raise RAGDriftError(
                    "evidence pixels do not match the source content used for the "
                    f"embedding of photo_id {match.photo_id}"
                )
            try:
                total_pixels += inspect_evidence_image(image.content)
            except EvidenceImagePixelLimitError as error:
                raise RAGEvidenceTooLargeError(str(error)) from error
            except InvalidEvidenceImageError as error:
                raise RAGDriftError(
                    f"evidence image is not decodable for photo_id {match.photo_id}"
                ) from error
            if total_pixels > MAX_EVIDENCE_TOTAL_PIXELS:
                raise RAGEvidenceTooLargeError(
                    "selected evidence exceeds the 96 megapixel aggregate decode limit"
                )
            evidence.append(
                RetrievalEvidence(
                    photo_id=match.photo_id,
                    image=image,
                    thumbnail=match.thumbnail_url,
                    retrieval_score=match.score,
                    rank=rank,
                    provider_fingerprint=retrieval.provider,
                )
            )

        self._verify_evidence_content(evidence, by_id)
        after_evidence = self._candidate_snapshot(album_id, request.user_id)
        self._require_same_snapshot(
            locked_snapshot, after_evidence, "evidence snapshot"
        )
        bundle = build_evidence_bundle(
            query=request.query,
            retrieval_provider_fingerprint=retrieval.provider,
            candidate_photo_ids=after_evidence.photo_ids,
            evidence=evidence,
            candidate_source_digest=after_evidence.source_digest,
        )
        grounded_request = GroundedRAGRequest(
            query=request.query,
            evidence=bundle,
            retrieval_provider_fingerprint=retrieval.provider,
            candidate_digest=bundle.provenance.candidate_digest,
        )
        provider = self.generation_provider_factory()
        answer = generate_grounded(provider, grounded_request)
        self._verify_evidence_content(evidence, by_id)
        after_generation = self._candidate_snapshot(album_id, request.user_id)
        self._require_same_snapshot(
            after_evidence,
            after_generation,
            "local VLM generation",
        )

        run_id = str(uuid.uuid4())
        response = AlbumRAGResponse(
            run_id=run_id,
            album_id=album_id,
            user_id=request.user_id,
            query=request.query,
            answer=answer.answer,
            claims=[
                RAGClaim(claim_id=claim.claim_id, text=claim.text)
                for claim in answer.claims
            ],
            citations=[
                RAGCitation(
                    claim_id=citation.claim_id,
                    photo_id=citation.photo_id,
                )
                for citation in answer.citations
            ],
            provenance=RAGProvenance(
                retrieval_provider_fingerprint=(
                    answer.provenance.retrieval_provider_fingerprint
                ),
                generation_provider_fingerprint=(
                    answer.provenance.generation_provider_fingerprint
                ),
                query_digest=answer.provenance.query_digest,
                candidate_digest=answer.provenance.candidate_digest,
                evidence_digest=answer.provenance.evidence_digest,
            ),
            retrieval=retrieval,
        )
        self._insert_run(response, request, bundle)
        return response

    def _candidate_snapshot(self, album_id: str, user_id: str) -> _CandidateSnapshot:
        decision = self._decision_snapshot(user_id)
        with self.database.connect() as connection:
            album = connection.execute(
                "SELECT 1 FROM albums WHERE id = ?", (album_id,)
            ).fetchone()
            rows = connection.execute(
                """
                SELECT id, absolute_path, file_size, source_mtime_ns,
                       quality_score, auto_reject,
                       embedding_path, embedding_provider,
                       embedding_source_size, embedding_source_mtime_ns,
                       embedding_source_sha256
                FROM photos WHERE album_id = ? ORDER BY id
                """,
                (album_id,),
            ).fetchall()
        if album is None or not rows:
            raise KeyError(f"album not found or empty: {album_id}")

        candidates = tuple(self._candidate_from_row(row) for row in rows)
        source_digest = canonical_sha256(
            {
                "album_id": album_id,
                "candidates": [candidate.digest_payload() for candidate in candidates],
                "decision": decision.digest_payload(),
                "schema": CANDIDATE_SNAPSHOT_SCHEMA,
            }
        )
        return _CandidateSnapshot(
            candidates=candidates,
            decision=decision,
            source_digest=source_digest,
        )

    def _candidate_from_row(self, row: Mapping[str, object]) -> _Candidate:
        photo_id = str(row["id"])
        source = Path(str(row["absolute_path"]))
        try:
            before = source.stat()
        except OSError as error:
            raise RAGDriftError(
                f"indexed source is unavailable for photo_id {photo_id}"
            ) from error
        source_size = row["file_size"]
        source_mtime = row["source_mtime_ns"]
        if source_size is None or source_mtime is None:
            raise KeyError(f"photo cache is incomplete for photo_id {photo_id}")
        if (before.st_size, before.st_mtime_ns) != (
            int(source_size),
            int(source_mtime),
        ):
            raise RAGDriftError(
                f"source changed since indexing for photo_id {photo_id}"
            )

        provider = row["embedding_provider"]
        embedding_path = row["embedding_path"]
        embedding_source_size = row["embedding_source_size"]
        embedding_source_mtime = row["embedding_source_mtime_ns"]
        embedding_source_sha256 = row["embedding_source_sha256"]
        if (
            provider != self.embedding_provider.name
            or not embedding_path
            or embedding_source_size != source_size
            or embedding_source_mtime != source_mtime
            or not isinstance(embedding_source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", embedding_source_sha256) is None
        ):
            raise KeyError(
                "album has no complete semantic cache for provider "
                f"{self.embedding_provider.name}; call the embed endpoint first"
            )
        embedding_sha256 = _stable_embedding_sha256(
            Path(str(embedding_path)),
            photo_id=photo_id,
            expected_dimension=self.embedding_provider.dimension,
        )
        try:
            after = source.stat()
        except OSError as error:
            raise RAGDriftError(
                f"source disappeared while validating photo_id {photo_id}"
            ) from error
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RAGDriftError(f"source changed while validating photo_id {photo_id}")
        return _Candidate(
            photo_id=photo_id,
            absolute_path=source,
            filename=source.name,
            file_size=int(source_size),
            source_mtime_ns=int(source_mtime),
            embedding_provider=str(provider),
            embedding_source_size=int(embedding_source_size),
            embedding_source_mtime_ns=int(embedding_source_mtime),
            embedding_source_sha256=embedding_source_sha256,
            embedding_sha256=embedding_sha256,
            quality_score=(
                float(row["quality_score"])
                if row["quality_score"] is not None
                else None
            ),
            auto_reject=bool(row["auto_reject"]),
        )

    def _decision_snapshot(self, user_id: str) -> _DecisionSnapshot:
        try:
            runtime = load_preference_runtime(
                self.database,
                self.embedding_provider,
                user_id=user_id,
            )
        except IncompatiblePreferenceModelError:
            runtime = cosine_fallback_runtime(
                self.embedding_provider,
                user_id=user_id,
            )
        return _decision_from_runtime(runtime)

    @staticmethod
    def _require_search_decision(
        retrieval: AlbumSearchResponse, expected: _DecisionSnapshot
    ) -> None:
        actual = {
            "algorithm": retrieval.algorithm,
            "comparisons": retrieval.preference_comparisons,
            "feature_schema": retrieval.feature_schema,
            "model_id": retrieval.preference_model_id,
            "projection_id": retrieval.projection_id,
            "provider_fingerprint": retrieval.provider,
            "user_id": retrieval.user_id,
        }
        expected_response = expected.digest_payload()
        expected_response.pop("training_event_digest")
        if actual != expected_response:
            raise RAGDriftError(
                "retrieval decision provenance drifted from the frozen runtime"
            )

    @staticmethod
    def _assert_source_current(candidate: _Candidate) -> None:
        try:
            current = candidate.absolute_path.stat()
        except OSError as error:
            raise RAGDriftError(
                f"evidence source is unavailable for photo_id {candidate.photo_id}"
            ) from error
        if (current.st_size, current.st_mtime_ns) != (
            candidate.file_size,
            candidate.source_mtime_ns,
        ):
            raise RAGDriftError(
                f"evidence source drift for photo_id {candidate.photo_id}"
            )

    @staticmethod
    def _require_same_snapshot(
        expected: _CandidateSnapshot,
        actual: _CandidateSnapshot,
        stage: str,
    ) -> None:
        if expected.source_digest != actual.source_digest:
            raise RAGDriftError(f"candidate universe drift during {stage}")

    @staticmethod
    def _verify_evidence_content(
        evidence: list[RetrievalEvidence],
        candidates: dict[str, _Candidate],
    ) -> None:
        """Re-read at most six evidence files to catch metadata-preserving swaps."""

        for item in evidence:
            candidate = candidates[item.photo_id]
            GroundedRAGService._assert_source_current(candidate)
            try:
                current = snapshot_image_file(
                    candidate.absolute_path,
                    display_name=candidate.filename,
                )
            except ValueError as error:
                raise RAGDriftError(
                    "evidence source changed while verifying frozen bytes for "
                    f"photo_id {item.photo_id}"
                ) from error
            if current.sha256 != item.image.sha256:
                raise RAGDriftError(
                    f"evidence source content drift for photo_id {item.photo_id}"
                )

    def _insert_run(
        self,
        response: AlbumRAGResponse,
        request: AlbumRAGRequest,
        bundle,
    ) -> None:
        persisted_query = redact_local_paths(request.query) or "[REDACTED]"
        request_payload = {
            "album_id": response.album_id,
            "query": persisted_query,
            "top_k": request.top_k,
            "user_id": request.user_id,
        }
        evidence_payload = {
            "candidate_photo_ids": bundle.candidate_photo_ids,
            "candidate_source_digest": bundle.candidate_source_digest,
            "items": [evidence_descriptor(item) for item in bundle.items],
            "schema": EVIDENCE_SCHEMA,
        }
        result_payload = response.model_dump(mode="json")
        result_payload["query"] = persisted_query
        result_payload["retrieval"]["query_text"] = persisted_query
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO rag_runs(
                    id, album_id, user_id, query_text,
                    retrieval_provider_fingerprint,
                    generation_provider_fingerprint,
                    candidate_digest, query_digest, evidence_digest,
                    evidence_json, result_json, request_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.run_id,
                    response.album_id,
                    response.user_id,
                    persisted_query,
                    response.provenance.retrieval_provider_fingerprint,
                    response.provenance.generation_provider_fingerprint,
                    response.provenance.candidate_digest,
                    response.provenance.query_digest,
                    response.provenance.evidence_digest,
                    _canonical_json(evidence_payload),
                    _canonical_json(result_payload),
                    _canonical_json(request_payload),
                ),
            )


def _stable_embedding_sha256(
    path: Path,
    *,
    photo_id: str,
    expected_dimension: int,
) -> str:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size < 1 or before.st_size > MAX_EMBEDDING_CACHE_BYTES:
                raise RAGDriftError(
                    f"embedding cache size is invalid for photo_id {photo_id}"
                )
            content = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise KeyError(
            f"embedding cache is unavailable for photo_id {photo_id}"
        ) from error
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RAGDriftError(
            f"embedding cache changed while validating photo_id {photo_id}"
        )
    try:
        vector = np.load(BytesIO(content), allow_pickle=False)
        array = np.asarray(vector)
    except (OSError, ValueError) as error:
        raise RAGDriftError(
            f"embedding cache is invalid for photo_id {photo_id}"
        ) from error
    if (
        array.shape != (expected_dimension,)
        or not np.issubdtype(array.dtype, np.number)
        or not np.all(np.isfinite(array))
        or float(np.linalg.norm(array.astype(np.float64))) <= 1e-12
    ):
        raise RAGDriftError(f"embedding cache is invalid for photo_id {photo_id}")
    return hashlib.sha256(content).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "CANDIDATE_SNAPSHOT_SCHEMA",
    "GroundedRAGService",
    "MAX_EVIDENCE_BYTES",
    "RAGBusyError",
    "RAGDriftError",
    "RAGEvidenceTooLargeError",
]
