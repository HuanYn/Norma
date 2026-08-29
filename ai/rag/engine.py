from __future__ import annotations

import re
from collections import Counter

from ai.rag.models import (
    OUTPUT_SCHEMA,
    VALIDATION_SCOPE,
    CitationValidatedAnswer,
    CitationValidationError,
    GeneratedCitation,
    GeneratedClaim,
    GenerationProvenance,
    GroundedRAGRequest,
    NoEvidenceError,
    ProviderFailureError,
    ProviderGenerationOutput,
    VLMInputBudgetError,
    canonical_answer,
    validate_evidence_bundle,
)
from ai.rag.prompting import build_prompt
from ai.rag.providers import (
    GroundedGenerationProvider,
    ProviderGenerationRequest,
    ProviderImagePayload,
)
from ai.rag.security import contains_local_path


_CLAIM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def generate_grounded(
    provider: GroundedGenerationProvider,
    request: GroundedRAGRequest,
) -> CitationValidatedAnswer:
    """Generate with referential safeguards and fail closed on citation drift.

    This core validates evidence bytes, references, and provenance.  It does not
    establish that a claim is semantically entailed by the cited pixels.
    """

    validate_citation_request(request)
    if not request.evidence.items:
        raise NoEvidenceError("grounded generation requires at least one evidence item")

    provider_name = getattr(provider, "name", None)
    if (
        not isinstance(provider_name, str)
        or not provider_name.strip()
        or provider_name != provider_name.strip()
    ):
        raise ValueError("generation provider needs a trimmed non-empty name")
    if contains_local_path(provider_name):
        raise ValueError("generation provider fingerprint must not contain a path")
    prompt = build_prompt(request.evidence, provider_name)
    provider_request = ProviderGenerationRequest(
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
        images=tuple(
            ProviderImagePayload(photo_id=item.photo_id, image=item.image)
            for item in request.evidence.items
        ),
        allowed_photo_ids=request.evidence.allowed_photo_ids,
        provenance=request.evidence.provenance,
    )
    try:
        output = provider.generate(provider_request)
    except (CitationValidationError, VLMInputBudgetError):
        raise
    except Exception as error:
        raise ProviderFailureError(
            f"grounded generation provider {provider_name} failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(output, ProviderGenerationOutput):
        raise CitationValidationError(
            "generation provider must return ProviderGenerationOutput"
        )
    validate_citation_output(output, request, provider_name)
    return CitationValidatedAnswer(
        schema=OUTPUT_SCHEMA,
        validation_scope=VALIDATION_SCOPE,
        answer=output.answer,
        claims=output.claims,
        citations=output.citations,
        provenance=output.provenance,
    )


def validate_citation_request(request: GroundedRAGRequest) -> None:
    validate_evidence_bundle(request.evidence)
    if request.query != request.evidence.query:
        raise CitationValidationError("request query drift from evidence snapshot")
    provenance = request.evidence.provenance
    if (
        request.retrieval_provider_fingerprint
        != provenance.retrieval_provider_fingerprint
    ):
        raise CitationValidationError(
            "request retrieval provider drift from evidence snapshot"
        )
    if request.candidate_digest != provenance.candidate_digest:
        raise CitationValidationError(
            "request candidate digest drift from evidence snapshot"
        )


def validate_citation_output(
    output: ProviderGenerationOutput,
    request: GroundedRAGRequest,
    generation_provider: str,
) -> None:
    if not isinstance(output.answer, str):
        raise CitationValidationError("generated answer must be a string")
    _reject_path_disclosure(output.answer, "generated answer")
    if not isinstance(output.claims, tuple) or any(
        not isinstance(claim, GeneratedClaim) for claim in output.claims
    ):
        raise CitationValidationError(
            "generated claims must be an immutable GeneratedClaim tuple"
        )
    if not isinstance(output.citations, tuple):
        raise CitationValidationError("generated citations must be an immutable tuple")
    if not isinstance(output.provenance, GenerationProvenance):
        raise CitationValidationError(
            "generated provenance must be GenerationProvenance"
        )
    if not output.claims:
        raise CitationValidationError(
            "provider returned no cited claims; refusing an unreferenced answer"
        )
    claim_ids: list[str] = []
    normalized_texts: list[str] = []
    for claim in output.claims:
        _validate_claim(claim)
        claim_ids.append(claim.claim_id)
        normalized_texts.append(" ".join(claim.text.casefold().split()))
    duplicate_ids = _duplicates(claim_ids)
    if duplicate_ids:
        raise CitationValidationError(
            f"duplicate claim IDs: {', '.join(duplicate_ids)}"
        )
    duplicate_texts = _duplicates(normalized_texts)
    if duplicate_texts:
        raise CitationValidationError("duplicate claims are not allowed")

    allowed = set(request.evidence.allowed_photo_ids)
    known_claims = set(claim_ids)
    citation_pairs: set[tuple[str, str]] = set()
    cited_claims: set[str] = set()
    for citation in output.citations:
        if not isinstance(citation, GeneratedCitation):
            raise CitationValidationError(
                "generated citations must contain only GeneratedCitation values"
            )
        if not isinstance(citation.claim_id, str) or not isinstance(
            citation.photo_id, str
        ):
            raise CitationValidationError(
                "generated citation identifiers must be strings"
            )
        _reject_path_disclosure(citation.claim_id, "citation claim_id")
        _reject_path_disclosure(citation.photo_id, "citation photo_id")
        if citation.claim_id not in known_claims:
            raise CitationValidationError(
                f"citation references unknown claim_id: {citation.claim_id}"
            )
        if citation.photo_id not in allowed:
            raise CitationValidationError(
                f"citation photo_id is outside evidence allow-list: {citation.photo_id}"
            )
        pair = (citation.claim_id, citation.photo_id)
        if pair in citation_pairs:
            raise CitationValidationError(
                "duplicate claim/photo citation pairs are not allowed"
            )
        citation_pairs.add(pair)
        cited_claims.add(citation.claim_id)
    uncited = [claim_id for claim_id in claim_ids if claim_id not in cited_claims]
    if uncited:
        raise CitationValidationError(
            f"claims without evidence citations: {', '.join(uncited)}"
        )

    expected_answer = canonical_answer(output.claims)
    if output.answer != expected_answer:
        raise CitationValidationError(
            "answer contains text outside the canonical cited-claim rendering"
        )

    expected_provenance = {
        "retrieval_provider_fingerprint": (
            request.evidence.provenance.retrieval_provider_fingerprint
        ),
        "generation_provider_fingerprint": generation_provider,
        "query_digest": request.evidence.provenance.query_digest,
        "candidate_digest": request.evidence.provenance.candidate_digest,
        "evidence_digest": request.evidence.provenance.evidence_digest,
    }
    for field, expected in expected_provenance.items():
        value = getattr(output.provenance, field)
        _reject_path_disclosure(value, f"generated {field}")
        if value != expected:
            raise CitationValidationError(f"generated {field} drift")


def _validate_claim(claim: GeneratedClaim) -> None:
    if not isinstance(claim.claim_id, str) or not _CLAIM_ID.fullmatch(claim.claim_id):
        raise CitationValidationError(
            "claim_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}"
        )
    if (
        not isinstance(claim.text, str)
        or not claim.text.strip()
        or claim.text != claim.text.strip()
    ):
        raise CitationValidationError("claim text must be a trimmed non-empty string")
    _reject_path_disclosure(claim.text, f"claim {claim.claim_id} text")


def _reject_path_disclosure(value: object, label: str) -> None:
    if isinstance(value, str) and contains_local_path(value):
        raise CitationValidationError(f"{label} contains a local path")


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


__all__ = [
    "generate_grounded",
    "validate_citation_output",
    "validate_citation_request",
]
