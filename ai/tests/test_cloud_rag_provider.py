from __future__ import annotations

import base64
import json
import socket
import traceback
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from PIL import Image, PngImagePlugin

from ai.rag import (
    CitationValidationError,
    GroundedRAGRequest,
    RetrievalEvidence,
    build_evidence_bundle,
    generate_grounded,
    snapshot_image_bytes,
)
from ai.rag import cloud_runtime as cloud
from ai.rag.models import VLMInputBudgetError
from ai.rag.prompting import build_prompt
from ai.rag.providers import ProviderGenerationRequest, ProviderImagePayload


API_KEY = "test-secret-do-not-log"
BASE_URL = "https://vision.example.test/v1"


def _image(*, metadata: bool = False, png: bool = False) -> bytes:
    with Image.new("RGB", (2000, 1000) if metadata else (24, 16), "navy") as image:
        buffer = BytesIO()
        if png:
            info = PngImagePlugin.PngInfo()
            info.add_text("location", "PRIVATE_METADATA")
            image.save(buffer, format="PNG", pnginfo=info)
        elif metadata:
            exif = Image.Exif()
            exif[270] = "PRIVATE_METADATA"
            exif[274] = 6
            image.save(buffer, format="JPEG", exif=exif, icc_profile=b"PRIVATE_ICC")
        else:
            image.save(buffer, format="JPEG")
        return buffer.getvalue()


def _bundle(*, content: bytes | None = None, count: int = 2):
    content = _image() if content is None else content
    return build_evidence_bundle(
        query="Which photo shows the sea?",
        retrieval_provider_fingerprint="test-local-clip",
        candidate_photo_ids=tuple(f"photo-{index}" for index in range(count + 1)),
        evidence=tuple(
            RetrievalEvidence(
                photo_id=f"photo-{index}",
                image=snapshot_image_bytes(
                    content, display_name="private.jpg", media_type="image/jpeg"
                ),
                thumbnail=None,
                retrieval_score=1.0 - index / 20,
                rank=index + 1,
                provider_fingerprint="test-local-clip",
                caption=r"Located at C:\Users\private\photo.jpg",
            )
            for index in range(count)
        ),
    )


def _request(bundle=None) -> ProviderGenerationRequest:
    bundle = _bundle() if bundle is None else bundle
    prompt = build_prompt(bundle, "unused")
    return ProviderGenerationRequest(
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
        images=tuple(
            ProviderImagePayload(item.photo_id, item.image) for item in bundle.items
        ),
        allowed_photo_ids=bundle.allowed_photo_ids,
        provenance=bundle.provenance,
    )


def _content(photo_id: str = "photo-0") -> str:
    return json.dumps(
        {
            "claims": [{"claim_id": "c1", "text": "The evidence is blue."}],
            "citations": [{"claim_id": "c1", "photo_id": photo_id}],
        }
    )


def _completion(content: str | None = None, *, finish_reason: str = "stop") -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": _content() if content is None else content,
                    },
                }
            ]
        }
    ).encode()


class RecordingTransport:
    def __init__(self, response: bytes | None = None):
        self.calls = []
        self.response = _completion() if response is None else response

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def _provider(transport=None, **kwargs):
    return cloud.create_cloud_vlm_provider(
        BASE_URL,
        "example-vl",
        API_KEY,
        transport=transport or RecordingTransport(),
        **kwargs,
    )


@pytest.mark.parametrize("png", [False, True])
def test_only_allowlisted_resized_metadata_free_images_leave_machine(png):
    transport = RecordingTransport()
    bundle = _bundle(content=_image(metadata=True, png=png))
    provider = _provider(transport)
    request = GroundedRAGRequest(
        query=bundle.query,
        evidence=bundle,
        retrieval_provider_fingerprint=bundle.provenance.retrieval_provider_fingerprint,
        candidate_digest=bundle.provenance.candidate_digest,
    )
    result = generate_grounded(provider, request)
    assert result.provenance.generation_provider_fingerprint == provider.name
    assert result.provenance.evidence_digest == bundle.provenance.evidence_digest
    assert len(transport.calls) == 1
    url, call = transport.calls[0]
    assert url == BASE_URL + "/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer " + API_KEY
    assert call["timeout_seconds"] == 60
    payload = call["payload"]
    for forbidden in (
        API_KEY,
        "PRIVATE_METADATA",
        "PRIVATE_ICC",
        "C:\\Users",
        "photo-2",
    ):
        assert forbidden.encode() not in payload
    body = json.loads(payload)
    assert body["stream"] is False
    assert body["max_tokens"] == 256
    assert "response_format" not in body
    content = body["messages"][1]["content"]
    assert content[0]["text"] == _request(bundle).user_prompt
    images = [
        part["image_url"]["url"] for part in content if part["type"] == "image_url"
    ]
    mapping = [part["text"] for part in content if part["type"] == "text"][1:]
    assert mapping == [
        "EVIDENCE_IMAGE photo_id=photo-0",
        "EVIDENCE_IMAGE photo_id=photo-1",
    ]
    assert len(images) == 2
    for data_url in images:
        prefix, encoded = data_url.split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            assert image.size == ((1280, 640) if png else (640, 1280))
            assert image.mode == "RGB"
            assert not image.getexif()
            assert "icc_profile" not in image.info
            assert "location" not in image.info


def test_fingerprint_includes_contract_but_no_secret_or_endpoint():
    first = _provider()
    changed_key = cloud.create_cloud_vlm_provider(
        BASE_URL, "example-vl", "different-secret"
    )
    changed_model = cloud.create_cloud_vlm_provider(BASE_URL, "example-vl-2", API_KEY)
    changed_endpoint = cloud.create_cloud_vlm_provider(
        "https://second.example.test/v1", "example-vl", API_KEY
    )
    changed_format = _provider(json_response_format=True)
    assert first.name == changed_key.name
    assert (
        len(
            {first.name, changed_model.name, changed_endpoint.name, changed_format.name}
        )
        == 4
    )
    assert "remote_weights=unverified" in first.name
    for value in (first.name, repr(first), repr(first.runtime)):
        assert API_KEY not in value
        assert BASE_URL not in value


def test_json_object_mode_is_explicitly_opt_in():
    transport = RecordingTransport()
    _provider(transport, json_response_format=True).generate(_request())
    assert json.loads(transport.calls[0][1]["payload"])["response_format"] == {
        "type": "json_object"
    }


def test_prompt_secret_is_redacted_and_secret_photo_id_is_not_uploaded():
    transport = RecordingTransport()
    provider = _provider(transport)
    request = _request()
    provider.generate(replace(request, user_prompt=request.user_prompt + API_KEY))
    assert API_KEY.encode() not in transport.calls[0][1]["payload"]
    assert b"[REDACTED_SECRET]" in transport.calls[0][1]["payload"]
    with pytest.raises(CitationValidationError):
        provider.generate(
            replace(
                request,
                images=(replace(request.images[0], photo_id=API_KEY),),
                allowed_photo_ids=(API_KEY,),
            )
        )
    assert len(transport.calls) == 1


def test_preprocess_and_package_version_affect_fingerprint(monkeypatch):
    first = _provider().name
    monkeypatch.setattr(cloud, "MAX_IMAGE_SIDE", 640)
    resized = _provider().name
    monkeypatch.setattr(cloud, "pillow_version", "test-future-version")
    upgraded = _provider().name
    assert len({first, resized, upgraded}) == 3


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/v1",
        "https://user:secret@example.test/v1",
        "https://example.test/v1?key=secret",
        "https://example.test/v1#secret",
        "https://example.test\\@evil.test",
        "https://example.test\n/v1",
        "file:///private",
        "https://example.test:0/v1",
    ],
)
def test_invalid_endpoints_fail_before_network(url):
    with pytest.raises(cloud.CloudVLMUnavailableError, match="configuration"):
        cloud.create_cloud_vlm_provider(url, "example-vl", API_KEY)


@pytest.mark.parametrize(
    "option",
    [
        {"max_new_tokens": True},
        {"max_new_tokens": 10},
        {"max_new_tokens": 128.5},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": True},
    ],
)
def test_invalid_generation_settings_fail_without_network(option):
    with pytest.raises(cloud.CloudVLMUnavailableError, match="configuration"):
        _provider(**option)


@pytest.mark.parametrize(
    "content",
    [
        '{"claims":[],"claims":[],"citations":[]}',
        '{"claims":[],"citations":[],"PRIVATE_REMOTE_BODY":"leak"}',
        '{"claims":[{"claim_id":"c1","text":"blue","extra":"leak"}],"citations":[]}',
        _content("outside-top-k"),
        _content().replace("c1", "unknown", 1),
        _content().replace("The evidence is blue.", "C:\\\\private\\\\photo.jpg"),
        _content().replace("The evidence is blue.", API_KEY),
        _content().replace("The evidence is blue.", "data:image/jpeg;base64,PRIVATE"),
        "```json\n{INVALID}\n```",
    ],
)
def test_bad_claims_and_citations_are_rejected_without_body_disclosure(content):
    provider = _provider(RecordingTransport(_completion(content)))
    with pytest.raises(CitationValidationError) as caught:
        provider.generate(_request())
    trace = "".join(traceback.format_exception(caught.value))
    for forbidden in ("PRIVATE_REMOTE_BODY", API_KEY, "data:image/", "outside-top-k"):
        assert forbidden not in trace


@pytest.mark.parametrize(
    "field", ["claim_text", "claim_id", "citation_claim_id", "citation_photo_id"]
)
def test_unicode_escaped_secrets_are_rejected_after_json_decoding(field):
    document = json.loads(_content())
    if field == "claim_text":
        document["claims"][0]["text"] = API_KEY
    elif field == "claim_id":
        document["claims"][0]["claim_id"] = API_KEY
    elif field == "citation_claim_id":
        document["citations"][0]["claim_id"] = API_KEY
    else:
        document["citations"][0]["photo_id"] = API_KEY
    escaped_key = "".join(f"\\u{ord(character):04x}" for character in API_KEY)
    content = json.dumps(document).replace(API_KEY, escaped_key)
    assert API_KEY not in content
    with pytest.raises(CitationValidationError) as caught:
        _provider(RecordingTransport(_completion(content))).generate(_request())
    assert API_KEY not in "".join(traceback.format_exception(caught.value))


def test_unicode_escaped_image_data_is_rejected_after_json_decoding():
    prefix = "DATA:IMAGE/JPEG;base64,PRIVATE"
    encoded = "".join(f"\\u{ord(character):04x}" for character in prefix)
    content = _content().replace("The evidence is blue.", encoded)
    with pytest.raises(CitationValidationError):
        _provider(RecordingTransport(_completion(content))).generate(_request())


def test_allowlist_mismatch_and_image_count_fail_before_upload():
    transport = RecordingTransport()
    provider = _provider(transport)
    with pytest.raises(CitationValidationError, match="allow-list"):
        provider.generate(replace(_request(), allowed_photo_ids=("photo-1", "photo-0")))
    with pytest.raises(VLMInputBudgetError, match="between 1 and 6"):
        provider.generate(_request(_bundle(count=7)))
    assert not transport.calls


def test_byte_pixel_and_upload_budgets_prevent_network(monkeypatch):
    transport = RecordingTransport()
    provider = _provider(transport)
    request = _request()
    monkeypatch.setattr(cloud, "MAX_SOURCE_IMAGE_BYTES", 1)
    with pytest.raises(VLMInputBudgetError, match="bytes"):
        provider.generate(request)
    monkeypatch.setattr(cloud, "MAX_SOURCE_IMAGE_BYTES", 32 * 1024 * 1024)
    monkeypatch.setattr(
        cloud, "inspect_evidence_image_dimensions", lambda content: (8000, 8000)
    )
    with pytest.raises(VLMInputBudgetError, match="pixels"):
        provider.generate(request)
    monkeypatch.setattr(
        cloud, "inspect_evidence_image_dimensions", lambda content: (24, 16)
    )
    monkeypatch.setattr(cloud, "MAX_REQUEST_BYTES", 1)
    with pytest.raises(VLMInputBudgetError, match="upload size"):
        provider.generate(request)
    assert not transport.calls


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "authentication"),
        (403, "authentication"),
        (429, "rate_limit"),
        (500, "http"),
        (302, "redirect"),
    ],
)
def test_http_errors_are_safe_and_never_retried(status, code):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise HTTPError(BASE_URL, status, API_KEY, {}, BytesIO(b"PRIVATE_REMOTE_BODY"))

    with pytest.raises(cloud.CloudVLMUnavailableError) as caught:
        _provider(fail).generate(_request())
    assert caught.value.code == code
    assert len(calls) == 1
    trace = "".join(traceback.format_exception(caught.value))
    assert API_KEY not in trace
    assert "PRIVATE_REMOTE_BODY" not in trace


@pytest.mark.parametrize("error", [TimeoutError(API_KEY), socket.timeout(API_KEY)])
def test_timeouts_are_safe_and_never_retried(error):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    with pytest.raises(cloud.CloudVLMUnavailableError) as caught:
        _provider(fail).generate(_request())
    assert caught.value.code == "timeout"
    assert len(calls) == 1
    assert API_KEY not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "response",
    [
        b"PRIVATE_REMOTE_BODY",
        b'{"choices":[],"choices":[]}',
        b'{"choices":NaN}',
        b"\xff",
        _completion(finish_reason="length"),
    ],
)
def test_invalid_envelopes_do_not_expose_body(response):
    with pytest.raises(cloud.CloudVLMUnavailableError) as caught:
        _provider(RecordingTransport(response)).generate(_request())
    assert caught.value.code in {"response_format", "incomplete"}
    assert "PRIVATE_REMOTE_BODY" not in str(caught.value)


def test_oversized_response_rejected():
    with pytest.raises(cloud.CloudVLMUnavailableError) as caught:
        _provider(RecordingTransport(b"x" * (cloud.MAX_RESPONSE_BYTES + 1))).generate(
            _request()
        )
    assert caught.value.code == "response_size"


def test_stdlib_transport_disables_redirects_and_bounds_reads(monkeypatch):
    class Response:
        status = 200
        headers = {}
        fp = SimpleNamespace(
            raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=lambda value: None))
        )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read1(self, limit):
            assert limit == 101
            return b"x" * limit

    class Opener:
        def open(self, request, timeout):
            assert request.get_method() == "POST"
            assert timeout == 20
            return Response()

    def build(*handlers):
        assert len(handlers) == 1
        with pytest.raises(cloud.CloudVLMUnavailableError) as caught:
            handlers[0].redirect_request(None, None, 307, "", {}, "https://evil.test")
        assert caught.value.code == "redirect"
        return Opener()

    monkeypatch.setattr(cloud, "build_opener", build)
    with pytest.raises(cloud.CloudVLMUnavailableError) as caught:
        cloud._https_transport(
            BASE_URL,
            headers={"Authorization": API_KEY},
            payload=b"{}",
            timeout_seconds=20,
            response_limit_bytes=100,
        )
    assert caught.value.code == "response_size"


def test_response_body_drip_cannot_extend_deadline(monkeypatch):
    now = [0.0]
    socket_timeouts = []
    reads = []
    closed = []

    class Response:
        status = 200
        headers = {}
        fp = SimpleNamespace(
            raw=SimpleNamespace(
                _sock=SimpleNamespace(settimeout=socket_timeouts.append)
            )
        )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def read1(self, limit):
            reads.append(limit)
            now[0] += 4.0
            return b"x"

    class Opener:
        def open(self, request, timeout):
            return Response()

    monkeypatch.setattr(cloud, "monotonic", lambda: now[0])
    monkeypatch.setattr(cloud, "build_opener", lambda *args: Opener())
    with pytest.raises(cloud.CloudVLMUnavailableError) as caught:
        cloud._https_transport(
            BASE_URL,
            headers={},
            payload=b"{}",
            timeout_seconds=10,
            response_limit_bytes=100,
        )
    assert caught.value.code == "timeout"
    assert socket_timeouts == [10.0, 6.0, 2.0]
    assert len(reads) == 3
    assert closed == [True]


def test_body_deadline_includes_time_spent_opening(monkeypatch):
    now = [0.0]

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read1(self, limit):
            raise AssertionError("expired request must not read the body")

    class Opener:
        def open(self, request, timeout):
            now[0] = 11.0
            return Response()

    monkeypatch.setattr(cloud, "monotonic", lambda: now[0])
    monkeypatch.setattr(cloud, "build_opener", lambda *args: Opener())
    with pytest.raises(cloud.CloudVLMUnavailableError) as caught:
        cloud._https_transport(
            BASE_URL,
            headers={},
            payload=b"{}",
            timeout_seconds=10,
            response_limit_bytes=100,
        )
    assert caught.value.code == "timeout"


def test_stdlib_transport_wraps_urlerror_without_disclosure(monkeypatch):
    class Opener:
        def open(self, *args, **kwargs):
            raise URLError(TimeoutError(API_KEY))

    monkeypatch.setattr(cloud, "build_opener", lambda *args: Opener())
    with pytest.raises(cloud.CloudVLMUnavailableError) as caught:
        cloud._https_transport(
            BASE_URL,
            headers={},
            payload=b"{}",
            timeout_seconds=20,
            response_limit_bytes=100,
        )
    assert caught.value.code == "timeout"
    assert API_KEY not in "".join(traceback.format_exception(caught.value))
