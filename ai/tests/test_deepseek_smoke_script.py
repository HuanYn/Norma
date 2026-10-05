from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from PIL import Image

from ai.rag import cloud_runtime as cloud
from scripts import check_deepseek as smoke


TEST_KEY = "fake-key-for-isolated-smoke-tests"


@pytest.fixture(autouse=True)
def isolated_smoke(monkeypatch, tmp_path):
    # Replace the module's environment object instead of inspecting or copying
    # the real process environment, which may contain an authorized live key.
    monkeypatch.setattr(smoke, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr(smoke, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(smoke.sys, "argv", ["check_deepseek.py"])

    def no_network(*args, **kwargs):
        pytest.fail("an unmocked network request must never be made")

    monkeypatch.setattr(smoke, "build_opener", no_network)
    monkeypatch.setattr(cloud, "build_opener", no_network)


class ModelsResponse(BytesIO):
    status = 200
    fp = None

    def __init__(self, model_ids):
        super().__init__(
            json.dumps({"data": [{"id": value} for value in model_ids]}).encode()
        )
        self.headers = {}


def models_opener(monkeypatch, model_ids):
    calls = []

    def open_request(request, *, timeout):
        calls.append((request, timeout))
        return ModelsResponse(model_ids)

    def build(*handlers):
        assert len(handlers) == 1
        assert isinstance(handlers[0], cloud._NoRedirect)
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(smoke, "build_opener", build)
    return calls


def read_report(tmp_path, capsys):
    captured = capsys.readouterr()
    report_path = tmp_path / ".norma" / "deepseek-smoke" / "result.json"
    stored = report_path.read_text(encoding="utf-8")
    assert TEST_KEY not in captured.out + captured.err + stored
    assert json.loads(captured.out) == json.loads(stored)
    assert captured.err == ""
    return json.loads(stored)


@pytest.mark.parametrize(
    "credential", [None, "", " leading-space", "line\nbreak", "x" * 4097]
)
def test_missing_or_invalid_credential_is_safe_and_does_not_call_network(
    monkeypatch, tmp_path, capsys, credential
):
    if credential is not None:
        smoke.os.environ["NORMA_VLM_API_KEY"] = credential
    monkeypatch.setattr(
        smoke,
        "create_cloud_vlm_provider",
        lambda *args, **kwargs: pytest.fail("generation must not be initialized"),
    )

    assert smoke.main() == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "configuration_required"}
    assert captured.err == ""
    assert not (tmp_path / ".norma").exists()
    if credential:
        assert credential not in captured.out


def test_model_not_listed_stops_before_generation(monkeypatch, tmp_path, capsys):
    smoke.os.environ["NORMA_VLM_API_KEY"] = TEST_KEY
    calls = models_opener(
        monkeypatch, ["other-model", TEST_KEY, "bad model", "unsafe\nmodel"]
    )
    monkeypatch.setattr(
        smoke,
        "create_cloud_vlm_provider",
        lambda *args, **kwargs: pytest.fail("must not switch or generate a model"),
    )

    assert smoke.main() == 1

    report = read_report(tmp_path, capsys)
    assert report["status"] == "vision_model_not_listed"
    assert report["models"] == ["other-model"]
    assert report["vision_attempts"] == 0
    assert report["private_images_uploaded"] == 0
    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == "https://api.deepseek.com/models"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer " + TEST_KEY
    assert timeout == 20
    assert not (
        tmp_path / ".norma" / "deepseek-smoke" / "synthetic-shapes.png"
    ).exists()


def test_http_error_never_discloses_body_url_message_or_key(
    monkeypatch, tmp_path, capsys
):
    smoke.os.environ["NORMA_VLM_API_KEY"] = TEST_KEY
    response_body = BytesIO(("PRIVATE_REMOTE_BODY " + TEST_KEY).encode())
    remote_error = HTTPError(
        "https://api.deepseek.com/models?private=" + TEST_KEY,
        401,
        "PRIVATE_REMOTE_MESSAGE " + TEST_KEY,
        {},
        response_body,
    )
    calls = []

    def fail(request, *, timeout):
        calls.append(request)
        raise remote_error

    monkeypatch.setattr(
        smoke, "build_opener", lambda *handlers: SimpleNamespace(open=fail)
    )

    assert smoke.main() == 1

    report = read_report(tmp_path, capsys)
    assert report["status"] == "http_error"
    assert report["stage"] == "models"
    assert report["http_status"] == 401
    assert report["vision_attempts"] == 0
    assert len(calls) == 1
    assert response_body.closed
    assert "PRIVATE_REMOTE" not in json.dumps(report)


def test_wrapped_vision_failure_keeps_only_safe_code(monkeypatch, tmp_path, capsys):
    smoke.os.environ["NORMA_VLM_API_KEY"] = TEST_KEY
    model_calls = models_opener(monkeypatch, [smoke.MODEL])
    vision_calls = []

    def transport(url, **kwargs):
        vision_calls.append(url)
        raise cloud.CloudVLMUnavailableError("authentication")

    monkeypatch.setattr(cloud, "_https_transport", transport)

    assert smoke.main() == 1

    report = read_report(tmp_path, capsys)
    assert report["status"] == "provider_error"
    assert report["stage"] == "vision"
    assert report["error_code"] == "authentication"
    assert report["vision_attempts"] == 1
    assert len(model_calls) == len(vision_calls) == 1


def test_success_sends_only_synthetic_pixels_and_writes_a_safe_report(
    monkeypatch, tmp_path, capsys
):
    smoke.os.environ["NORMA_VLM_API_KEY"] = TEST_KEY
    model_calls = models_opener(monkeypatch, [smoke.MODEL])
    vision_calls = []

    def transport(url, **kwargs):
        vision_calls.append((url, kwargs))
        content = {
            "claims": [{"claim_id": "c1", "text": "左侧是红色方形，右侧是蓝色圆形。"}],
            "citations": [{"claim_id": "c1", "photo_id": "synthetic-01"}],
        }
        return json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content, ensure_ascii=False),
                        },
                    }
                ]
            }
        ).encode()

    monkeypatch.setattr(cloud, "_https_transport", transport)
    with monkeypatch.context() as context:
        context.setattr(
            Path,
            "read_bytes",
            lambda *args, **kwargs: pytest.fail("smoke must not read photo files"),
        )
        assert smoke.main() == 0

    report = read_report(tmp_path, capsys)
    assert report["status"] == "vision_citation_smoke_passed"
    assert report["private_images_uploaded"] == 0
    assert report["vision_attempts"] == 1
    assert report["full_retrieval_test"] is False
    assert report["result"]["claims"][0]["text"] == "左侧是红色方形，右侧是蓝色圆形。"
    assert report["result"]["citations"] == [
        {"claim_id": "c1", "photo_id": "synthetic-01"}
    ]
    assert len(model_calls) == len(vision_calls) == 1
    url, request = vision_calls[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer " + TEST_KEY
    assert request["timeout_seconds"] == 60
    assert request["response_limit_bytes"] == cloud.MAX_RESPONSE_BYTES
    assert TEST_KEY.encode() not in request["payload"]
    document = json.loads(request["payload"])
    assert document["model"] == smoke.MODEL
    assert document["thinking"] == {"type": "disabled"}
    assert document["response_format"] == {"type": "json_object"}
    assert document["stream"] is False
    assert document["max_tokens"] == 384
    images = [
        item["image_url"]["url"]
        for item in document["messages"][1]["content"]
        if item["type"] == "image_url"
    ]
    assert len(images) == 1
    prefix, encoded = images[0].split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    with Image.open(BytesIO(base64.b64decode(encoded))) as picture:
        assert picture.size == (320, 180)
        red = picture.getpixel((70, 90))
        blue = picture.getpixel((245, 90))
        assert red[0] > 240 and red[1] < 15 and red[2] < 15
        assert blue[0] < 15 and blue[1] < 15 and blue[2] > 240
        assert not picture.getexif()
    artifacts = tmp_path / ".norma" / "deepseek-smoke"
    assert {path.name for path in artifacts.iterdir()} == {
        "result.json",
        "synthetic-shapes.png",
    }
