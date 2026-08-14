from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelectionIntent:
    target_count: int
    min_quality: float
    exclude_rejects: bool
    max_per_similarity_group: int


COUNT_PATTERNS = (
    re.compile(r"(?:选|挑|找|保留)\s*(\d+)\s*(?:张|幅)"),
    re.compile(r"\b(\d+)\s*(?:photos?|images?|shots?)\b", re.IGNORECASE),
)
QUALITY_PATTERNS = (
    re.compile(r"质量\s*(?:>=|≥|不低于|至少)\s*(\d+(?:\.\d+)?)"),
    re.compile(
        r"\bquality\s*(?:>=|≥|at\s+least)\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)
GROUP_PATTERNS = (
    re.compile(r"(?:每个)?相似(?:组|照片)?(?:最多|不超过)\s*(\d+)\s*张?"),
    re.compile(
        r"\bmax(?:imum)?\s+(\d+)\s+(?:per\s+)?similar(?:ity)?\s+group\b",
        re.IGNORECASE,
    ),
)


def parse_selection_prompt(prompt: str) -> SelectionIntent:
    normalized = " ".join(prompt.split())
    if not normalized:
        raise ValueError("selection prompt cannot be empty")

    target_count = int(_first_group(COUNT_PATTERNS, normalized) or 12)
    if not 1 <= target_count <= 50:
        raise ValueError("target photo count must be between 1 and 50")

    min_quality = float(_first_group(QUALITY_PATTERNS, normalized) or 0.0)
    if not 0.0 <= min_quality <= 100.0:
        raise ValueError("minimum quality must be between 0 and 100")

    max_per_group = int(_first_group(GROUP_PATTERNS, normalized) or 1)
    if not 1 <= max_per_group <= 10:
        raise ValueError("similarity-group limit must be between 1 and 10")

    include_rejects = bool(
        re.search(
            r"允许(?:模糊|废片|低质量)|包含(?:模糊|废片|低质量)|"
            r"\binclude\s+(?:rejects?|blurry|low[- ]quality)\b",
            normalized,
            re.IGNORECASE,
        )
    )
    return SelectionIntent(
        target_count=target_count,
        min_quality=min_quality,
        exclude_rejects=not include_rejects,
        max_per_similarity_group=max_per_group,
    )


def has_semantic_content(prompt: str) -> bool:
    residual = " ".join(prompt.split())
    for pattern in (*COUNT_PATTERNS, *QUALITY_PATTERNS, *GROUP_PATTERNS):
        residual = pattern.sub(" ", residual)
    residual = re.sub(
        r"允许(?:模糊|废片|低质量)|包含(?:模糊|废片|低质量)|"
        r"\binclude\s+(?:rejects?|blurry|low[- ]quality)\b",
        " ",
        residual,
        flags=re.IGNORECASE,
    )
    residual = re.sub(
        r"\b(?:pick|select|choose|find|keep|of|with|and|please)\b|"
        r"(?:请|帮我|选择|选|挑|找|保留|照片|图片)",
        " ",
        residual,
        flags=re.IGNORECASE,
    )
    residual = re.sub(r"[\s,，。.!！?？;；:：、]+", "", residual)
    return bool(residual)


def _first_group(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None
