from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, runtime_checkable

from ai.rag.models import (
    CitationValidationError,
    EvidenceImageSnapshot,
    EvidenceProvenance,
    GenerationProvenance,
    ProviderGenerationOutput,
    canonical_answer,
    parse_provider_output,
)


_JSON_FENCE = re.compile(r"\A```json[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\Z")


@dataclass(frozen=True, slots=True)
class ProviderImagePayload:
    photo_id: str
    image: EvidenceImageSnapshot


@dataclass(frozen=True, slots=True)
class ProviderGenerationRequest:
    system_prompt: str
    user_prompt: str
    images: tuple[ProviderImagePayload, ...]
    allowed_photo_ids: tuple[str, ...]
    provenance: EvidenceProvenance

    @property
    def image_contents(self) -> tuple[bytes, ...]:
        return tuple(item.image.content for item in self.images)


@runtime_checkable
class GroundedGenerationProvider(Protocol):
    name: str

    def generate(
        self, request: ProviderGenerationRequest
    ) -> ProviderGenerationOutput: ...


class Qwen3VLRuntime(Protocol):
    """Injected, already-loaded local runtime; this interface never downloads."""

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: tuple[ProviderImagePayload, ...],
        max_new_tokens: int,
        temperature: float,
    ) -> Mapping[str, object] | str: ...


class Qwen3VLLocalProvider:
    """Adapter for a Qwen3-VL-style runtime explicitly loaded from local files.

    Loading weights is intentionally outside this module.  API integration may
    inject a Transformers-backed runtime after verifying a local model path; this
    adapter itself has no Hub client, `from_pretrained`, or download side effect.
    """

    def __init__(
        self,
        *,
        provider_fingerprint: str,
        runtime: Qwen3VLRuntime,
        max_new_tokens: int = 768,
    ) -> None:
        if not provider_fingerprint.strip():
            raise ValueError("provider_fingerprint must not be empty")
        if provider_fingerprint != provider_fingerprint.strip():
            raise ValueError("provider_fingerprint must not contain edge whitespace")
        if isinstance(max_new_tokens, bool) or max_new_tokens < 64:
            raise ValueError("max_new_tokens must be an integer of at least 64")
        self.name = provider_fingerprint
        self.runtime = runtime
        self.max_new_tokens = max_new_tokens

    def generate(self, request: ProviderGenerationRequest) -> ProviderGenerationOutput:
        raw = self.runtime.generate_json(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            images=request.images,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        if isinstance(raw, str):
            raw = _parse_runtime_json(raw)
        if not isinstance(raw, Mapping):
            raise CitationValidationError("local VLM output must be a JSON object")
        expected_keys = {"claims", "citations"}
        if set(raw) != expected_keys:
            raise CitationValidationError(
                "local VLM output keys must be exactly "
                f"{sorted(expected_keys)}, got {sorted(raw)}"
            )

        # Parse the model-owned fields through the same strict nested schema, but
        # never ask the model to echo server-owned answer/provenance values.
        authoritative_provenance = GenerationProvenance(
            retrieval_provider_fingerprint=(
                request.provenance.retrieval_provider_fingerprint
            ),
            generation_provider_fingerprint=self.name,
            query_digest=request.provenance.query_digest,
            candidate_digest=request.provenance.candidate_digest,
            evidence_digest=request.provenance.evidence_digest,
        )
        parsed = parse_provider_output(
            {
                "answer": "server-constructed-placeholder",
                "claims": raw["claims"],
                "citations": raw["citations"],
                "provenance": {
                    "retrieval_provider_fingerprint": (
                        authoritative_provenance.retrieval_provider_fingerprint
                    ),
                    "generation_provider_fingerprint": (
                        authoritative_provenance.generation_provider_fingerprint
                    ),
                    "query_digest": authoritative_provenance.query_digest,
                    "candidate_digest": authoritative_provenance.candidate_digest,
                    "evidence_digest": authoritative_provenance.evidence_digest,
                },
            }
        )
        return ProviderGenerationOutput(
            answer=canonical_answer(parsed.claims),
            claims=parsed.claims,
            citations=parsed.citations,
            provenance=authoritative_provenance,
        )


class ScriptedGroundedProvider:
    """Deterministic fake provider for unit tests and offline integration tests."""

    def __init__(
        self,
        output: ProviderGenerationOutput
        | Callable[[ProviderGenerationRequest], ProviderGenerationOutput],
        *,
        name: str = "fake-grounded-v1",
    ) -> None:
        if not name.strip() or name != name.strip():
            raise ValueError("fake provider name must be a trimmed non-empty string")
        self.name = name
        self._output = output
        self.calls: list[ProviderGenerationRequest] = []

    def generate(self, request: ProviderGenerationRequest) -> ProviderGenerationOutput:
        if request.allowed_photo_ids != tuple(item.photo_id for item in request.images):
            raise AssertionError("provider request evidence allow-list drift")
        self.calls.append(request)
        if callable(self._output):
            return self._output(request)
        return self._output


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise CitationValidationError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _parse_runtime_json(raw: str) -> object:
    """Parse bare JSON or one exact JSON fence without extracting substrings."""

    candidate = raw.strip()
    if candidate.startswith("```"):
        fenced = _JSON_FENCE.fullmatch(candidate)
        if fenced is None:
            raise CitationValidationError(
                "local VLM output must be bare JSON or one complete ```json fence"
            )
        candidate = fenced.group("body")
    try:
        return json.loads(
            candidate,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise CitationValidationError(
            f"local VLM returned invalid JSON: {error}"
        ) from error


def _reject_nonfinite_constant(value: str) -> object:
    raise CitationValidationError(
        f"local VLM JSON contains a non-standard numeric constant: {value}"
    )


__all__ = [
    "GroundedGenerationProvider",
    "ProviderImagePayload",
    "ProviderGenerationRequest",
    "Qwen3VLLocalProvider",
    "Qwen3VLRuntime",
    "ScriptedGroundedProvider",
]
