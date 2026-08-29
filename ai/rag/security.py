from __future__ import annotations

import re


PATH_REDACTION = "[REDACTED_LOCAL_PATH]"
PATH_REDACTION_VERSION = "local-path-boundary-v5"


_HTTP_URL_PATTERN = re.compile(r"(?i)https?://[^\s\r\n\"'<>]+")
_LOCAL_PATH_PATTERNS = (
    re.compile(r"(?i)file:///?[^\r\n\"'<>]+"),
    # A drive path can legitimately touch prose (for example, ``locatedC:\\x``).
    # A word boundary/lookbehind here would miss both ASCII and CJK prefixes.
    re.compile(r"(?i)(?:[a-z]:[\\/])[^\r\n\"'<>|]*"),
    # Windows also treats C:folder\file as relative to drive C's working
    # directory.  Requiring a later separator avoids labels such as C:score.
    re.compile(r"(?i)(?:[a-z]:)[^\\/\r\n\"'<>|]+[\\/][^\r\n\"'<>|]*"),
    re.compile(r"(?<![\w:/\\])(?:\\\\|//)[^\r\n\"'<>]+"),
    # A single leading backslash is absolute to the current Windows drive.
    # Require at least a two-character root component to avoid common escaped
    # one-character literals such as ``\n`` being treated as paths.
    re.compile(r"(?<![\w\\])\\[^\s\\\"'<>|]{2,}(?:\\[^\s\\\"'<>|]+)*"),
    re.compile(r"(?<![\w])~[\\/][^\r\n\"'<>]*"),
    re.compile(r"(?:\.\.[\\/])+[^\r\n\"'<>|]*"),
    # Three-or-more slash components are path-like even when the first component
    # is ASCII/CJK prose (for example, ``located/vault/private``).  Exact
    # two-component Hub IDs such as ``owner/repository`` remain outside this
    # class, while URL and numeric-date exclusions are applied below.
    re.compile(r"(?<![:/])(?:[^\s/\\\"'<>]+/){2,}[^\s/\\\"'<>]+"),
    # Rooted paths after a clear start/whitespace/punctuation boundary may contain
    # spaces.  Consume through a quote/newline/shell delimiter so redaction cannot
    # leak the suffix after the first space (for example ``My Secret/photo.jpg``).
    re.compile(r"(?<![\w/])/(?:[^/\r\n\\\"'<>|]+/)+[^\r\n\\\"'<>|]+"),
    # Rooted POSIX paths may touch CJK prose, but a generic ``owner/repository``
    # identifier is not a local path.  Restrict the no-boundary form to common
    # filesystem roots and Windows drive-mount notation such as /E/Norma.  A
    # common root is sensitive on its own, so /etc and prefix/workspace are also
    # detected; the terminal guard prevents prefixes such as /database.
    re.compile(
        r"(?i)/(?:[a-z](?=/)|home|users?|usr|var|tmp|temp|etc|opt|mnt|media|srv|"
        r"root|private|volumes|app|workspace|data|photos?|code|projects?|models?)"
        r"(?:(?:/[^/\s\\\"'<>]+)+|(?=$|[^a-z0-9_.-]))"
    ),
    # Generic rooted paths are accepted at the start of a value or after ordinary
    # punctuation/whitespace.  Requiring a non-word predecessor keeps Hub IDs such
    # as Qwen/Qwen3-VL-2B-Instruct and owner/repository out of this class.
    re.compile(r"(?<![\w/])/[^\s/\\\"'<>]+(?=$|[\s,.;:!?()\[\]{}])"),
    re.compile(r"(?<![\w/])/(?:[^/\s\\\"'<>]+/)+[^/\s\\\"'<>]+"),
    re.compile(
        r"(?i)(?<![\w])(?:users?|home|appdata|temp|tmp|cache|\.cache)"
        r"(?:[\\/][^\s\"'<>]+)+"
    ),
)


def _overlaps(span: tuple[int, int], other: tuple[int, int]) -> bool:
    return span[0] < other[1] and other[0] < span[1]


def _is_numeric_slash_expression(value: str) -> bool:
    """Exclude ordinary fractions and slash-formatted dates from POSIX paths."""

    components = tuple(part for part in value.split("/") if part)
    return bool(components) and all(part.isdecimal() for part in components)


def _local_path_spans(text: str) -> tuple[tuple[int, int], ...]:
    url_spans = tuple(match.span() for match in _HTTP_URL_PATTERN.finditer(text))
    candidates: list[tuple[int, int]] = []
    for pattern in _LOCAL_PATH_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if any(_overlaps(span, url_span) for url_span in url_spans):
                continue
            if _is_numeric_slash_expression(match.group()):
                continue
            candidates.append(span)

    if not candidates:
        return ()

    merged: list[list[int]] = []
    for start, end in sorted(candidates):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def contains_local_path(text: str) -> bool:
    """Detect loader paths; this intentionally favors preventing disclosure."""

    if not isinstance(text, str):
        return False
    return bool(_local_path_spans(text))


def redact_local_paths(text: str | None) -> str | None:
    """Remove path-like local identifiers before untrusted metadata reaches a VLM."""

    if text is None:
        return None
    redacted = text
    for start, end in reversed(_local_path_spans(text)):
        redacted = redacted[:start] + PATH_REDACTION + redacted[end:]
    return redacted


__all__ = [
    "PATH_REDACTION",
    "PATH_REDACTION_VERSION",
    "contains_local_path",
    "redact_local_paths",
]
