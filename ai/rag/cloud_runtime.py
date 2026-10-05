"""Opt-in, bounded OpenAI-compatible vision calls over HTTPS.

Only retrieved evidence derivatives leave the machine. Cloud weight revisions
cannot be verified by this adapter; the fingerprint identifies the request and
preprocessing contract, not a pinned remote model snapshot.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import socket
import sys
from io import BytesIO
from time import monotonic
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image, ImageOps, __version__ as pillow_version

from ai.config import VLMThinkingMode
from ai.rag.image_safety import (
    MAX_EVIDENCE_IMAGE_PIXELS,
    MAX_EVIDENCE_TOTAL_PIXELS,
    inspect_evidence_image_dimensions,
)
from ai.rag.models import (
    CitationValidationError,
    ProviderGenerationOutput,
    VLMInputBudgetError,
)
from ai.rag.prompting import OUTPUT_CONTRACT_VERSION, prompt_contract_sha256
from ai.rag.providers import (
    ProviderGenerationRequest,
    ProviderImagePayload,
    Qwen3VLLocalProvider,
)
from ai.rag.security import contains_local_path, redact_local_paths


MAX_CLOUD_IMAGES = 6
MAX_IMAGE_SIDE = 1280
MAX_SOURCE_IMAGE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 128 * 1024 * 1024
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_PROMPT_BYTES = 64 * 1024
JPEG_QUALITY = 85
RUNTIME_VERSION = "stdlib-https-body-deadline-no-redirect-no-retry-v2"
PREPROCESS_VERSION = "exif-transpose-rgb-lanczos-jpeg85-no-metadata-v1"
PROMPT_SANITIZATION_VERSION = "local-path-redaction-control-label-aware-secret-mask-v1"
_PHOTO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,255}\Z")


class CloudVLMUnavailableError(RuntimeError):
    """Safe public failure: never includes credentials or a remote response body."""

    _MESSAGES = {
        "configuration": "cloud VLM configuration is invalid",
        "authentication": "cloud VLM authentication was rejected",
        "rate_limit": "cloud VLM rate limit or quota was reached",
        "timeout": "cloud VLM request timed out; no retry was attempted",
        "redirect": "cloud VLM redirect was refused",
        "network": "cloud VLM network request failed",
        "http": "cloud VLM returned an unsuccessful HTTP status",
        "response_size": "cloud VLM response exceeded the size limit",
        "response_format": "cloud VLM response did not match the expected protocol",
        "incomplete": "cloud VLM generation did not complete successfully",
    }

    def __init__(self, code: str) -> None:
        self.code = code if code in self._MESSAGES else "network"
        super().__init__(self._MESSAGES[self.code])


CloudTransport = Callable[..., bytes]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Refuse even same-origin redirects: never resend Authorization or pixels.
        raise CloudVLMUnavailableError("redirect")


def _https_transport(
    url: str,
    *,
    headers: Mapping[str, str],
    payload: bytes,
    timeout_seconds: float,
    response_limit_bytes: int,
) -> bytes:
    """Bound response-body reads by elapsed time as well as bytes.

    urllib's connect/TLS/response-header stages use socket operation timeouts;
    OS DNS lookup may have platform-specific timing. HTTPResponse may also parse
    chunk-size lines/trailers within read1 using socket-operation timeouts.
    This is not a strict total wall-clock guarantee for those stages. Body read
    iterations share the remaining deadline instead of resetting it per chunk.
    """

    deadline = monotonic() + timeout_seconds
    request = Request(url, data=payload, headers=dict(headers), method="POST")
    try:
        with build_opener(_NoRedirect()).open(
            request, timeout=timeout_seconds
        ) as response:
            if 300 <= response.status < 400:
                raise CloudVLMUnavailableError("redirect")
            if response.status != 200:
                raise CloudVLMUnavailableError(_http_error_code(response.status))
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except (ValueError, TypeError):
                    raise CloudVLMUnavailableError("response_format") from None
                if declared_size < 0 or declared_size > response_limit_bytes:
                    raise CloudVLMUnavailableError("response_size")
            return _read_response_body(response, deadline, response_limit_bytes)
    except CloudVLMUnavailableError:
        raise
    except HTTPError as error:
        code = _http_error_code(error.code)
        error.close()
        raise CloudVLMUnavailableError(code) from None
    except (TimeoutError, socket.timeout):
        raise CloudVLMUnavailableError("timeout") from None
    except URLError as error:
        code = "timeout" if isinstance(error.reason, TimeoutError) else "network"
        raise CloudVLMUnavailableError(code) from None
    except Exception:
        raise CloudVLMUnavailableError("network") from None


def _read_response_body(response, deadline: float, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise CloudVLMUnavailableError("timeout")
        # HTTPResponse.read1 performs at most one underlying buffered read.
        # Refresh the socket timeout so each body read consumes the same budget.
        if response.fp is not None:
            response.fp.raw._sock.settimeout(remaining)
        chunk = response.read1(min(64 * 1024, limit + 1 - size))
        if monotonic() >= deadline:
            raise CloudVLMUnavailableError("timeout")
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            raise CloudVLMUnavailableError("response_size")
        chunks.append(chunk)


def _http_error_code(status: int) -> str:
    if status in (401, 403):
        return "authentication"
    if status == 429:
        return "rate_limit"
    if 300 <= status < 400:
        return "redirect"
    return "http"


class OpenAICompatibleVisionRuntime:
    """Single non-streaming request; construction never makes a network call."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 60,
        json_response_format: bool = False,
        thinking_mode: VLMThinkingMode = "provider-default",
        transport: CloudTransport | None = None,
    ) -> None:
        self._endpoint = _endpoint(base_url)
        if (
            not isinstance(model, str)
            or _MODEL_ID.fullmatch(model) is None
            or not isinstance(api_key, str)
            or not api_key
            or len(api_key) > 4096
            or any(ord(character) < 33 or ord(character) > 126 for character in api_key)
            or api_key in self._endpoint
            or api_key in model
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 1 <= timeout_seconds <= 300
            or not isinstance(json_response_format, bool)
            or not isinstance(thinking_mode, str)
            or thinking_mode not in {"provider-default", "enabled", "disabled"}
        ):
            raise CloudVLMUnavailableError("configuration")
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = float(timeout_seconds)
        self._json_response_format = json_response_format
        self._thinking_mode = thinking_mode
        self._transport = transport or _https_transport
        contract = {
            "endpoint": self._endpoint,
            "model": model,
            "remote_weights": "unverified",
            "runtime": RUNTIME_VERSION,
            "python": sys.version.split()[0],
            "pillow": pillow_version,
            "prompt_sha256": prompt_contract_sha256(),
            "prompt_sanitization": PROMPT_SANITIZATION_VERSION,
            "output_contract": OUTPUT_CONTRACT_VERSION,
            "json_response_format": json_response_format,
            "thinking_mode": thinking_mode,
            "temperature": 0.0,
            "timeout_seconds": self._timeout_seconds,
            "preprocess": PREPROCESS_VERSION,
            "max_side": MAX_IMAGE_SIDE,
            "max_images": MAX_CLOUD_IMAGES,
            "jpeg_quality": JPEG_QUALITY,
            "max_source_bytes": MAX_SOURCE_IMAGE_BYTES,
            "max_source_total_bytes": MAX_SOURCE_TOTAL_BYTES,
            "max_source_pixels": MAX_EVIDENCE_IMAGE_PIXELS,
            "max_source_total_pixels": MAX_EVIDENCE_TOTAL_PIXELS,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_response_bytes": MAX_RESPONSE_BYTES,
            "max_prompt_bytes": MAX_PROMPT_BYTES,
        }
        contract_sha = hashlib.sha256(_json_bytes(contract)).hexdigest()
        self.provider_fingerprint = (
            f"cloud-vlm-openai-compatible-v1|contract_sha256={contract_sha}"
            "|remote_weights=unverified"
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: tuple[ProviderImagePayload, ...],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        if (
            temperature != 0.0
            or isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 64 <= max_new_tokens <= 1024
        ):
            raise CloudVLMUnavailableError("configuration")
        _validate_images(images)
        if any(self._api_key in item.photo_id for item in images):
            raise CitationValidationError(
                "cloud evidence ID contains forbidden content"
            )
        prompts = [system_prompt, user_prompt]
        if any(not isinstance(value, str) for value in prompts):
            raise VLMInputBudgetError("cloud prompts must be strings")
        if sum(len(value.encode("utf-8")) for value in prompts) > MAX_PROMPT_BYTES:
            raise VLMInputBudgetError("cloud prompts exceed the size limit")
        safe_system, safe_user = (
            _safe_prompt(value, self._api_key) for value in prompts
        )
        content = [{"type": "text", "text": safe_user}]
        for item in images:
            content.append(
                {"type": "text", "text": f"EVIDENCE_IMAGE photo_id={item.photo_id}"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _jpeg_data_url(item)},
                }
            )
        document = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": safe_system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if self._json_response_format:
            document["response_format"] = {"type": "json_object"}
        if self._thinking_mode != "provider-default":
            document["thinking"] = {"type": self._thinking_mode}
        payload = _json_bytes(document)
        if len(payload) > MAX_REQUEST_BYTES:
            raise VLMInputBudgetError("cloud request exceeds the upload size limit")
        try:
            raw = self._transport(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                payload=payload,
                timeout_seconds=self._timeout_seconds,
                response_limit_bytes=MAX_RESPONSE_BYTES,
            )
        except CloudVLMUnavailableError:
            raise
        except HTTPError as error:
            code = _http_error_code(error.code)
            error.close()
            raise CloudVLMUnavailableError(code) from None
        except (TimeoutError, socket.timeout):
            raise CloudVLMUnavailableError("timeout") from None
        except Exception:
            raise CloudVLMUnavailableError("network") from None
        if not isinstance(raw, bytes):
            raise CloudVLMUnavailableError("response_format")
        if len(raw) > MAX_RESPONSE_BYTES:
            raise CloudVLMUnavailableError("response_size")
        result = _completion_content(raw)
        if self._api_key in result or "data:image/" in result.casefold():
            raise CitationValidationError("cloud VLM output contains forbidden content")
        return result


class CloudVLMProvider(Qwen3VLLocalProvider):
    """Reuse the local adapter's schema while keeping cloud errors body-free."""

    runtime: OpenAICompatibleVisionRuntime

    def generate(self, request: ProviderGenerationRequest) -> ProviderGenerationOutput:
        if request.allowed_photo_ids != tuple(item.photo_id for item in request.images):
            raise CitationValidationError("cloud evidence allow-list drift")
        try:
            output = super().generate(request)
            decoded_strings = (
                output.answer,
                *(
                    value
                    for claim in output.claims
                    for value in (claim.claim_id, claim.text)
                ),
                *(
                    value
                    for citation in output.citations
                    for value in (citation.claim_id, citation.photo_id)
                ),
            )
            if any(
                self.runtime._api_key in value or "data:image/" in value.casefold()
                for value in decoded_strings
            ):
                raise CitationValidationError("cloud output contains forbidden content")
            allowed = set(request.allowed_photo_ids)
            claims = {claim.claim_id for claim in output.claims}
            if any(
                citation.photo_id not in allowed or citation.claim_id not in claims
                for citation in output.citations
            ):
                raise CitationValidationError("cloud output citation is not allowed")
            if contains_local_path(output.answer):
                raise CitationValidationError("cloud output contains a local path")
            return output
        except CitationValidationError:
            # Shared parser errors may contain an attacker-controlled JSON key.
            raise CitationValidationError(
                "cloud VLM output failed strict claims/citations validation"
            ) from None


def create_cloud_vlm_provider(
    base_url: str,
    model: str,
    api_key: str,
    *,
    max_new_tokens: int = 256,
    timeout_seconds: float = 60,
    json_response_format: bool = False,
    thinking_mode: VLMThinkingMode = "provider-default",
    transport: CloudTransport | None = None,
) -> CloudVLMProvider:
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 64 <= max_new_tokens <= 1024
    ):
        raise CloudVLMUnavailableError("configuration")
    runtime = OpenAICompatibleVisionRuntime(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        json_response_format=json_response_format,
        thinking_mode=thinking_mode,
        transport=transport,
    )
    return CloudVLMProvider(
        provider_fingerprint=(
            f"{runtime.provider_fingerprint}|max_new_tokens={max_new_tokens}"
        ),
        runtime=runtime,
        max_new_tokens=max_new_tokens,
    )


def _endpoint(base_url: str) -> str:
    if (
        not isinstance(base_url, str)
        or not base_url
        or len(base_url) > 2048
        or any(ord(character) <= 32 for character in base_url)
        or "\\" in base_url
    ):
        raise CloudVLMUnavailableError("configuration")
    try:
        parts = urlsplit(base_url)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or parts.port == 0
        ):
            raise ValueError
        path = parts.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        return urlunsplit(("https", parts.netloc.lower(), path, "", ""))
    except (ValueError, UnicodeError):
        raise CloudVLMUnavailableError("configuration") from None


def _safe_prompt(value: str, api_key: str) -> str:
    # The trusted label ends in "T:", and its explanation has "claims/citations".
    # Passing that whole line to path redaction resembles a Windows drive-relative
    # path. Preserve only the fixed label; redact the remaining text normally.
    prefix = "MODEL_OUTPUT_CONTRACT: "
    lines = []
    for line in value.replace(api_key, "[REDACTED_SECRET]").splitlines(keepends=True):
        if line.startswith(prefix):
            lines.append(prefix + redact_local_paths(line[len(prefix) :]))
        else:
            lines.append(redact_local_paths(line))
    return "".join(lines)


def _validate_images(images: tuple[ProviderImagePayload, ...]) -> None:
    if not 1 <= len(images) <= MAX_CLOUD_IMAGES:
        raise VLMInputBudgetError("cloud generation requires between 1 and 6 images")
    ids = [item.photo_id for item in images]
    if len(set(ids)) != len(ids) or any(
        _PHOTO_ID.fullmatch(value) is None for value in ids
    ):
        raise CitationValidationError("cloud image IDs must be unique safe identifiers")
    sizes = [len(item.image.content) for item in images]
    if (
        any(size > MAX_SOURCE_IMAGE_BYTES for size in sizes)
        or sum(sizes) > MAX_SOURCE_TOTAL_BYTES
    ):
        raise VLMInputBudgetError("cloud source image bytes exceed the decode budget")
    total_pixels = 0
    for item in images:
        try:
            width, height = inspect_evidence_image_dimensions(item.image.content)
        except ValueError:
            raise VLMInputBudgetError(
                "cloud evidence image is unsafe to decode"
            ) from None
        total_pixels += width * height
    if total_pixels > MAX_EVIDENCE_TOTAL_PIXELS:
        raise VLMInputBudgetError("cloud source image pixels exceed the decode budget")


def _jpeg_data_url(item: ProviderImagePayload) -> str:
    try:
        with Image.open(BytesIO(item.image.content)) as source:
            with ImageOps.exif_transpose(source) as oriented:
                with oriented.convert("RGB") as rgb:
                    rgb.thumbnail(
                        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS
                    )
                    # A fresh image has no inherited EXIF, ICC, XMP, or PNG text.
                    with Image.new("RGB", rgb.size) as clean:
                        clean.paste(rgb)
                        output = BytesIO()
                        clean.save(
                            output, format="JPEG", quality=JPEG_QUALITY, exif=b""
                        )
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(
            "ascii"
        )
    except (OSError, ValueError, SyntaxError):
        raise VLMInputBudgetError(
            "cloud evidence image could not be prepared"
        ) from None


def _completion_content(raw: bytes) -> str:
    try:
        envelope = json.loads(
            raw, object_pairs_hook=_unique_json, parse_constant=_invalid_constant
        )
        choices = envelope["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        if choice["finish_reason"] != "stop":
            raise CloudVLMUnavailableError("incomplete")
        message = choice["message"]
        if message.get("refusal") or message.get("tool_calls"):
            raise CloudVLMUnavailableError("incomplete")
        content = message["content"]
        if (
            message["role"] != "assistant"
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError
        return content
    except CloudVLMUnavailableError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
        raise CloudVLMUnavailableError("response_format") from None


def _unique_json(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError("nonfinite JSON value")


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "CloudVLMProvider",
    "CloudVLMUnavailableError",
    "OpenAICompatibleVisionRuntime",
    "create_cloud_vlm_provider",
]
