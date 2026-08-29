from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ai.rag.security import contains_local_path


EVIDENCE_SCHEMA = "norma-grounded-evidence-v1"
OUTPUT_SCHEMA = "norma-citation-validated-answer-v1"
VALIDATION_SCOPE = "referential-citation-and-provenance-only-v1"


class GroundedRAGError(RuntimeError):
    """Base error for the grounded generation boundary."""


class EvidenceIntegrityError(GroundedRAGError):
    """Raised when a retrieval snapshot is incomplete or has drifted."""


class NoEvidenceError(GroundedRAGError):
    """Raised before generation when no evidence is available."""


class ProviderFailureError(GroundedRAGError):
    """Raised when the local generation provider fails."""


class VLMInputBudgetError(GroundedRAGError):
    """Raised when deterministic multimodal preprocessing exceeds its hard cap."""


class CitationValidationError(GroundedRAGError):
    """Raised by referential/citation checks, not semantic entailment checks."""


@dataclass(frozen=True, slots=True)
class EvidenceImageSnapshot:
    """Immutable original bytes retained for evidence identity and audit.

    A generation runtime may deterministically resize a decoded copy to enforce its
    visual-token budget; this snapshot and its SHA-256 never describe that derivative.
    """

    content: bytes
    byte_size: int
    sha256: str
    media_type: str
    display_name: str

    def __post_init__(self) -> None:
        _validate_image_snapshot(self)


@dataclass(frozen=True, slots=True)
class EvidenceRegion:
    label: str
    box: tuple[float, float, float, float]
    score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("region label must not be empty")
        if self.label != self.label.strip():
            raise ValueError("region label must not contain edge whitespace")
        if not isinstance(self.box, tuple) or len(self.box) != 4:
            raise ValueError("region box must be an immutable four-value tuple")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in self.box
        ):
            raise ValueError("region box must contain four finite values")
        left, top, right, bottom = self.box
        if left > right or top > bottom:
            raise ValueError("region box must use left <= right and top <= bottom")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise ValueError("region score must be a finite number")


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    photo_id: str
    image: EvidenceImageSnapshot
    thumbnail: str | None
    retrieval_score: float
    rank: int
    provider_fingerprint: str
    caption: str | None = None
    ocr_text: str | None = None
    regions: tuple[EvidenceRegion, ...] = ()

    def __post_init__(self) -> None:
        for value, label in (
            (self.photo_id, "photo_id"),
            (self.provider_fingerprint, "provider_fingerprint"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evidence {label} must not be empty")
            if value != value.strip():
                raise ValueError(f"evidence {label} must not contain edge whitespace")
            if contains_local_path(value):
                raise ValueError(f"evidence {label} must not contain a local path")
        if self.thumbnail is not None and (
            not isinstance(self.thumbnail, str)
            or not self.thumbnail.strip()
            or self.thumbnail != self.thumbnail.strip()
        ):
            raise ValueError(
                "evidence thumbnail must be a trimmed non-empty string when provided"
            )
        for value, label in (
            (self.caption, "caption"),
            (self.ocr_text, "ocr_text"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"evidence {label} must be a non-empty string when provided"
                )
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank < 1
        ):
            raise ValueError("evidence rank must be a positive integer")
        if (
            isinstance(self.retrieval_score, bool)
            or not isinstance(self.retrieval_score, (int, float))
            or not math.isfinite(self.retrieval_score)
        ):
            raise ValueError("evidence retrieval_score must be a finite number")
        if not isinstance(self.regions, tuple) or any(
            not isinstance(region, EvidenceRegion) for region in self.regions
        ):
            raise ValueError("evidence regions must be an immutable region tuple")
        if not isinstance(self.image, EvidenceImageSnapshot):
            raise ValueError("evidence image must be an immutable byte snapshot")
        _validate_image_snapshot(self.image)


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    schema: str
    retrieval_provider_fingerprint: str
    query_digest: str
    candidate_digest: str
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    query: str
    candidate_photo_ids: tuple[str, ...]
    items: tuple[RetrievalEvidence, ...]
    provenance: EvidenceProvenance
    candidate_source_digest: str | None = None

    @property
    def allowed_photo_ids(self) -> tuple[str, ...]:
        return tuple(item.photo_id for item in self.items)


@dataclass(frozen=True, slots=True)
class GroundedRAGRequest:
    """A request plus the live values against which its snapshot is checked."""

    query: str
    evidence: EvidenceBundle
    retrieval_provider_fingerprint: str
    candidate_digest: str


@dataclass(frozen=True, slots=True)
class GenerationProvenance:
    retrieval_provider_fingerprint: str
    generation_provider_fingerprint: str
    query_digest: str
    candidate_digest: str
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class GeneratedClaim:
    claim_id: str
    text: str


@dataclass(frozen=True, slots=True)
class GeneratedCitation:
    claim_id: str
    photo_id: str


@dataclass(frozen=True, slots=True)
class ProviderGenerationOutput:
    answer: str
    claims: tuple[GeneratedClaim, ...]
    citations: tuple[GeneratedCitation, ...]
    provenance: GenerationProvenance


@dataclass(frozen=True, slots=True)
class CitationValidatedAnswer:
    """Referentially validated output; semantic entailment remains unverified."""

    schema: str
    validation_scope: str
    answer: str
    claims: tuple[GeneratedClaim, ...]
    citations: tuple[GeneratedCitation, ...]
    provenance: GenerationProvenance


def build_evidence_bundle(
    *,
    query: str,
    retrieval_provider_fingerprint: str,
    candidate_photo_ids: Sequence[str],
    evidence: Sequence[RetrievalEvidence],
    candidate_source_digest: str | None = None,
) -> EvidenceBundle:
    """Create a deterministic, self-verifiable snapshot of retrieval evidence."""

    normalized_query = _required_text(query, "query")
    provider = _required_text(
        retrieval_provider_fingerprint, "retrieval_provider_fingerprint"
    )
    if contains_local_path(provider):
        raise ValueError("retrieval_provider_fingerprint must not contain a path")
    candidates = _canonical_ids(candidate_photo_ids, "candidate_photo_ids")
    source_digest = _optional_sha256(candidate_source_digest, "candidate_source_digest")
    candidate_set = set(candidates)
    ordered = tuple(sorted(evidence, key=lambda item: (item.rank, item.photo_id)))
    _validate_evidence_items(ordered, provider, candidate_set)

    provenance = EvidenceProvenance(
        schema=EVIDENCE_SCHEMA,
        retrieval_provider_fingerprint=provider,
        query_digest=query_digest(normalized_query),
        candidate_digest=candidate_digest(
            candidates, candidate_source_digest=source_digest
        ),
        evidence_digest=evidence_digest(ordered),
    )
    return EvidenceBundle(
        query=normalized_query,
        candidate_photo_ids=candidates,
        items=ordered,
        provenance=provenance,
        candidate_source_digest=source_digest,
    )


def snapshot_image_bytes(
    content: bytes,
    *,
    display_name: str,
    media_type: str,
) -> EvidenceImageSnapshot:
    """Freeze caller-owned bytes into the content-addressed VLM payload."""

    if not isinstance(content, bytes):
        raise ValueError("image snapshot content must be immutable bytes")
    frozen = bytes(content)
    return EvidenceImageSnapshot(
        content=frozen,
        byte_size=len(frozen),
        sha256=hashlib.sha256(frozen).hexdigest(),
        media_type=media_type,
        display_name=display_name,
    )


def snapshot_image_file(
    path: Path,
    *,
    display_name: str | None = None,
    media_type: str | None = None,
) -> EvidenceImageSnapshot:
    """Read a local file once; the path is discarded after creating the snapshot."""

    source = Path(path)
    try:
        with source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("image source must be a regular file")
            content = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ValueError(f"unable to snapshot local image: {error}") from error
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("image source changed while its bytes were being snapshotted")
    guessed_type, _ = mimetypes.guess_type(source.name)
    return snapshot_image_bytes(
        content,
        display_name=display_name or source.name,
        media_type=media_type or guessed_type or "image/octet-stream",
    )


def validate_evidence_bundle(bundle: EvidenceBundle) -> None:
    """Recompute every digest so dataclass replacement cannot bypass integrity."""

    try:
        rebuilt = build_evidence_bundle(
            query=bundle.query,
            retrieval_provider_fingerprint=(
                bundle.provenance.retrieval_provider_fingerprint
            ),
            candidate_photo_ids=bundle.candidate_photo_ids,
            evidence=bundle.items,
            candidate_source_digest=bundle.candidate_source_digest,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise EvidenceIntegrityError(str(error)) from error
    if bundle.provenance.schema != EVIDENCE_SCHEMA:
        raise EvidenceIntegrityError("evidence schema drift")
    for label in ("query_digest", "candidate_digest", "evidence_digest"):
        if getattr(bundle.provenance, label) != getattr(rebuilt.provenance, label):
            raise EvidenceIntegrityError(f"evidence {label} drift")
    if bundle.items != rebuilt.items:
        raise EvidenceIntegrityError("evidence ordering drift")


def query_digest(query: str) -> str:
    return canonical_sha256({"schema": EVIDENCE_SCHEMA, "query": query})


def candidate_digest(
    photo_ids: Sequence[str], *, candidate_source_digest: str | None = None
) -> str:
    ids = _canonical_ids(photo_ids, "candidate_photo_ids")
    source_digest = _optional_sha256(candidate_source_digest, "candidate_source_digest")
    payload: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "candidate_photo_ids": ids,
    }
    if source_digest is not None:
        payload["candidate_source_digest"] = source_digest
    return canonical_sha256(payload)


def evidence_digest(evidence: Sequence[RetrievalEvidence]) -> str:
    ordered = sorted(evidence, key=lambda item: (item.rank, item.photo_id))
    payload = [_evidence_digest_payload(item) for item in ordered]
    return canonical_sha256({"schema": EVIDENCE_SCHEMA, "evidence": payload})


def canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def canonical_answer(claims: Sequence[GeneratedClaim]) -> str:
    return "\n".join(f"{claim.text} [{claim.claim_id}]" for claim in claims)


def parse_provider_output(payload: Mapping[str, object]) -> ProviderGenerationOutput:
    """Strictly parse JSON-like model output without type coercion."""

    _exact_keys(payload, {"answer", "claims", "citations", "provenance"}, "output")
    answer = _mapping_text(payload, "answer", "output")
    raw_claims = _mapping_list(payload, "claims", "output")
    raw_citations = _mapping_list(payload, "citations", "output")
    raw_provenance = payload["provenance"]
    if not isinstance(raw_provenance, Mapping):
        raise CitationValidationError("output.provenance must be an object")

    claims: list[GeneratedClaim] = []
    for index, raw in enumerate(raw_claims):
        label = f"output.claims[{index}]"
        if not isinstance(raw, Mapping):
            raise CitationValidationError(f"{label} must be an object")
        _exact_keys(raw, {"claim_id", "text"}, label)
        claims.append(
            GeneratedClaim(
                claim_id=_mapping_text(raw, "claim_id", label),
                text=_mapping_text(raw, "text", label),
            )
        )

    citations: list[GeneratedCitation] = []
    for index, raw in enumerate(raw_citations):
        label = f"output.citations[{index}]"
        if not isinstance(raw, Mapping):
            raise CitationValidationError(f"{label} must be an object")
        _exact_keys(raw, {"claim_id", "photo_id"}, label)
        citations.append(
            GeneratedCitation(
                claim_id=_mapping_text(raw, "claim_id", label),
                photo_id=_mapping_text(raw, "photo_id", label),
            )
        )

    provenance_keys = {
        "retrieval_provider_fingerprint",
        "generation_provider_fingerprint",
        "query_digest",
        "candidate_digest",
        "evidence_digest",
    }
    _exact_keys(raw_provenance, provenance_keys, "output.provenance")
    provenance = GenerationProvenance(
        **{
            key: _mapping_text(raw_provenance, key, "output.provenance")
            for key in provenance_keys
        }
    )
    return ProviderGenerationOutput(
        answer=answer,
        claims=tuple(claims),
        citations=tuple(citations),
        provenance=provenance,
    )


def _validate_evidence_items(
    items: Sequence[RetrievalEvidence],
    provider: str,
    candidate_ids: set[str],
) -> None:
    photo_ids: set[str] = set()
    ranks: set[int] = set()
    for item in items:
        if not isinstance(item, RetrievalEvidence):
            raise ValueError("evidence items must be RetrievalEvidence values")
        _validate_image_snapshot(item.image)
        if item.photo_id in photo_ids:
            raise ValueError(f"duplicate evidence photo_id: {item.photo_id}")
        if item.rank in ranks:
            raise ValueError(f"duplicate evidence rank: {item.rank}")
        if item.photo_id not in candidate_ids:
            raise ValueError(
                f"evidence photo_id is outside candidate universe: {item.photo_id}"
            )
        if item.provider_fingerprint != provider:
            raise ValueError(
                "evidence provider fingerprint does not match retrieval snapshot"
            )
        photo_ids.add(item.photo_id)
        ranks.add(item.rank)


def _canonical_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    ids: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{label} must contain only strings")
        normalized = _required_text(value, label)
        if contains_local_path(normalized):
            raise ValueError(f"{label} must not contain local paths")
        ids.append(normalized)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(ids))


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    if value != value.strip():
        raise ValueError(f"{label} must not contain edge whitespace")
    return value


def _optional_sha256(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def evidence_descriptor(item: RetrievalEvidence) -> dict[str, object]:
    """Model-visible metadata; loader paths and raw image bytes are excluded."""

    return {
        "caption": item.caption,
        "image": {
            "byte_size": item.image.byte_size,
            "display_name": item.image.display_name,
            "media_type": item.image.media_type,
            "sha256": item.image.sha256,
        },
        "ocr_text": item.ocr_text,
        "photo_id": item.photo_id,
        "provider_fingerprint": item.provider_fingerprint,
        "rank": item.rank,
        "regions": [
            {"box": region.box, "label": region.label, "score": region.score}
            for region in item.regions
        ],
        "retrieval_score": item.retrieval_score,
    }


def _evidence_digest_payload(item: RetrievalEvidence) -> dict[str, object]:
    payload = evidence_descriptor(item)
    payload["thumbnail"] = item.thumbnail
    return payload


def _validate_image_snapshot(snapshot: EvidenceImageSnapshot) -> None:
    if not isinstance(snapshot.content, bytes):
        raise ValueError("image snapshot content must be immutable bytes")
    if (
        not isinstance(snapshot.byte_size, int)
        or isinstance(snapshot.byte_size, bool)
        or snapshot.byte_size < 1
        or snapshot.byte_size != len(snapshot.content)
    ):
        raise ValueError("image snapshot byte_size does not match content")
    actual_sha256 = hashlib.sha256(snapshot.content).hexdigest()
    if not isinstance(snapshot.sha256, str) or snapshot.sha256 != actual_sha256:
        raise ValueError("image snapshot sha256 does not match content")
    if (
        not isinstance(snapshot.media_type, str)
        or not snapshot.media_type.startswith("image/")
        or snapshot.media_type != snapshot.media_type.strip()
    ):
        raise ValueError("image snapshot media_type must be a trimmed image/* type")
    _validate_display_name(snapshot.display_name)


def _validate_display_name(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", ":", "\x00"))
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("image display_name must be a safe basename")


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CitationValidationError(
            f"{label} keys must be exactly {sorted(expected)}, got {sorted(actual)}"
        )


def _mapping_text(value: Mapping[str, object], key: str, label: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip():
        raise CitationValidationError(f"{label}.{key} must be a non-empty string")
    if item != item.strip():
        raise CitationValidationError(f"{label}.{key} must not contain edge whitespace")
    return item


def _mapping_list(value: Mapping[str, object], key: str, label: str) -> list[object]:
    item = value[key]
    if not isinstance(item, list):
        raise CitationValidationError(f"{label}.{key} must be an array")
    return item


__all__ = [
    "EVIDENCE_SCHEMA",
    "OUTPUT_SCHEMA",
    "VALIDATION_SCOPE",
    "CitationValidatedAnswer",
    "EvidenceBundle",
    "EvidenceImageSnapshot",
    "EvidenceIntegrityError",
    "EvidenceProvenance",
    "EvidenceRegion",
    "GeneratedCitation",
    "GeneratedClaim",
    "GenerationProvenance",
    "GroundedRAGError",
    "GroundedRAGRequest",
    "CitationValidationError",
    "NoEvidenceError",
    "ProviderFailureError",
    "ProviderGenerationOutput",
    "VLMInputBudgetError",
    "RetrievalEvidence",
    "build_evidence_bundle",
    "candidate_digest",
    "canonical_answer",
    "canonical_sha256",
    "evidence_digest",
    "evidence_descriptor",
    "parse_provider_output",
    "query_digest",
    "snapshot_image_bytes",
    "snapshot_image_file",
    "validate_evidence_bundle",
]
