from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai import app as app_module
from ai import cli
from ai.config import Settings, load_settings
from ai.rag import cloud_runtime as cloud
from ai.tests.test_rag_http_integration import _album


SECRET = "integration-api-key-must-not-appear"
ENDPOINT = "https://vision.example.test/v1"
MODEL = "example-cloud-vision"


def _settings(data_dir: Path, **changes) -> Settings:
    base = Settings(
        host="127.0.0.1",
        port=8765,
        data_dir=data_dir,
        log_level="INFO",
        vlm_provider="openai-compatible",
        vlm_base_url=ENDPOINT,
        vlm_model=MODEL,
        vlm_api_key=SECRET,
        vlm_max_new_tokens=384,
        vlm_timeout_seconds=45,
        preference_mode="record-only",
    )
    return replace(base, **changes)


def _environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {
        "NORMA_DATA_DIR": str(tmp_path / "data"),
        "NORMA_VLM_PROVIDER": "openai-compatible",
        "NORMA_VLM_BASE_URL": ENDPOINT,
        "NORMA_VLM_MODEL": MODEL,
        "NORMA_VLM_API_KEY": SECRET,
        "NORMA_VLM_MAX_NEW_TOKENS": "384",
        "NORMA_VLM_TIMEOUT_SECONDS": "45",
        "NORMA_PREFERENCE_MODE": "record-only",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _configure(monkeypatch, database, settings, embedding_provider):
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "settings", settings)
    monkeypatch.setattr(app_module, "embedding_provider", lambda: embedding_provider)


def test_cloud_environment_configuration_and_repr(tmp_path, monkeypatch):
    _environment(monkeypatch, tmp_path)
    settings = load_settings()
    assert settings.vlm_provider == "openai-compatible"
    assert settings.vlm_base_url == ENDPOINT
    assert settings.vlm_model == MODEL
    assert settings.vlm_api_key == SECRET
    assert settings.vlm_timeout_seconds == 45
    assert settings.vlm_max_new_tokens == 384
    assert settings.vlm_configured is True
    assert settings.preference_mode == "record-only"
    assert SECRET not in repr(settings)


def test_defaults_require_explicit_cloud_configuration(tmp_path, monkeypatch):
    _environment(monkeypatch, tmp_path)
    for key in (
        "NORMA_VLM_PROVIDER",
        "NORMA_VLM_BASE_URL",
        "NORMA_VLM_MODEL",
        "NORMA_VLM_API_KEY",
        "NORMA_VLM_MAX_NEW_TOKENS",
        "NORMA_VLM_TIMEOUT_SECONDS",
        "NORMA_PREFERENCE_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings()
    assert settings.vlm_provider == "openai-compatible"
    assert settings.vlm_configured is False
    assert settings.vlm_api_key == ""
    assert settings.preference_mode == "record-only"


@pytest.mark.parametrize("missing", ["vlm_base_url", "vlm_model", "vlm_api_key"])
def test_cloud_configuration_presence_requires_all_three_fields(tmp_path, missing):
    assert _settings(tmp_path, **{missing: "  "}).vlm_configured is False


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("NORMA_VLM_PROVIDER", "unknown"),
        ("NORMA_VLM_TIMEOUT_SECONDS", "0"),
        ("NORMA_VLM_TIMEOUT_SECONDS", "181"),
        ("NORMA_VLM_MAX_NEW_TOKENS", "1025"),
    ],
)
def test_cloud_environment_rejects_invalid_options(
    tmp_path, monkeypatch, variable, value
):
    _environment(monkeypatch, tmp_path)
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError):
        load_settings()


def test_cli_data_directory_override_preserves_cloud_and_preference_settings(
    tmp_path, monkeypatch
):
    _environment(monkeypatch, tmp_path)
    original = load_settings()
    new_directory = tmp_path / "cli-override"
    actual = cli._settings(new_directory)
    assert actual.data_dir == new_directory.resolve()
    for field in fields(Settings):
        if field.name != "data_dir":
            assert getattr(actual, field.name) == getattr(original, field.name)


@pytest.mark.parametrize("mode", ["openai-compatible", "local"])
def test_factory_chooses_only_configured_generation_mode(tmp_path, monkeypatch, mode):
    settings = _settings(tmp_path, vlm_provider=mode)
    monkeypatch.setattr(app_module, "settings", settings)
    calls = []
    sentinel = object()

    def local(path, **kwargs):
        calls.append(("local", (path,), kwargs))
        return sentinel

    def remote(*args, **kwargs):
        calls.append(("openai-compatible", args, kwargs))
        return sentinel

    monkeypatch.setattr(app_module, "create_local_qwen3vl_provider", local)
    monkeypatch.setattr(cloud, "create_cloud_vlm_provider", remote)
    assert app_module.rag_generation_provider() is sentinel
    assert len(calls) == 1
    chosen, args, kwargs = calls[0]
    assert chosen == mode
    if mode == "openai-compatible":
        assert args == (ENDPOINT, MODEL, SECRET)
        assert kwargs == {"max_new_tokens": 384, "timeout_seconds": 45}
    else:
        assert args == (settings.local_vlm_model_dir,)
        assert kwargs == {"max_new_tokens": 384}


@pytest.mark.parametrize("configured", [False, True])
def test_health_and_capabilities_report_mode_without_keys_or_network(
    tmp_path, monkeypatch, configured
):
    database, data_dir, _, _, embedding_provider = _album(tmp_path)
    settings = _settings(data_dir, vlm_api_key=SECRET if configured else "")
    _configure(monkeypatch, database, settings, embedding_provider)

    def forbidden(*args, **kwargs):
        raise AssertionError("status endpoints must not initialize a cloud client")

    monkeypatch.setattr(cloud, "create_cloud_vlm_provider", forbidden)
    with TestClient(app_module.app) as client:
        for route in ("/health", "/capabilities"):
            response = client.get(route)
            assert response.status_code == 200
            payload = response.json()
            assert payload["vlm_provider"] == "openai-compatible"
            assert payload["vlm_configured"] is configured
            assert payload["preference_training_enabled"] is False
            assert payload["preference_mode"] == "record-only"
            for private in (SECRET, ENDPOINT, "vlm_api_key"):
                assert private not in response.text


def test_http_cloud_rag_persists_only_validated_result_without_training(
    tmp_path, monkeypatch
):
    database, data_dir, album_id, _, embedding_provider = _album(tmp_path)
    _configure(monkeypatch, database, _settings(data_dir), embedding_provider)
    calls = []

    def transport(url, **kwargs):
        calls.append((url, kwargs))
        payload = json.loads(kwargs["payload"])
        photo_ids = [
            part["text"].removeprefix("EVIDENCE_IMAGE photo_id=")
            for part in payload["messages"][1]["content"]
            if part["type"] == "text"
            and part["text"].startswith("EVIDENCE_IMAGE photo_id=")
        ]
        assert len(photo_ids) == 2
        assert payload["model"] == MODEL
        assert kwargs["headers"]["Authorization"] == "Bearer " + SECRET
        content = json.dumps(
            {
                "claims": [{"claim_id": "c1", "text": "这张照片具有蓝色区域。"}],
                "citations": [{"claim_id": "c1", "photo_id": photo_ids[0]}],
            }
        )
        return json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ]
            }
        ).encode()

    monkeypatch.setattr(cloud, "_https_transport", transport)
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag", json={"query": "哪些照片有蓝色？", "top_k": 2}
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(calls) == 1
    assert len(payload["retrieval"]["matches"]) == 2
    assert payload["retrieval"]["preference_model_id"] is None
    assert payload["retrieval"]["preference_comparisons"] == 0
    assert payload["validation_level"] == "citation-referential-only"
    assert payload["semantic_entailment_verified"] is False
    assert payload["provenance"]["generation_provider_fingerprint"].startswith(
        "cloud-vlm-openai-compatible-v1|"
    )
    with database.connect() as connection:
        rows = connection.execute("SELECT * FROM rag_runs").fetchall()
        assert len(rows) == 1
        assert rows[0]["id"] == payload["run_id"]
        audit = " ".join(str(value) for value in tuple(rows[0]))
        assert len(json.loads(rows[0]["evidence_json"])["items"]) == 2
        assert json.loads(rows[0]["result_json"])["run_id"] == payload["run_id"]
    for private in (SECRET, ENDPOINT, "data:image/", str(tmp_path)):
        assert private not in audit
        assert private not in response.text


@pytest.mark.parametrize(
    "changes",
    [
        {"vlm_api_key": ""},
        {"vlm_base_url": "http://vision.example.test/v1"},
        {"vlm_model": ""},
    ],
)
def test_missing_or_invalid_cloud_configuration_returns_503_without_calling_network(
    tmp_path, monkeypatch, changes
):
    database, data_dir, album_id, _, embedding_provider = _album(tmp_path)
    _configure(
        monkeypatch, database, _settings(data_dir, **changes), embedding_provider
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid configuration must not reach the network")

    monkeypatch.setattr(cloud, "_https_transport", forbidden)
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag", json={"query": "蓝色的照片", "top_k": 2}
        )
    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "cloud VLM configuration is invalid"
    assert SECRET not in response.text
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_cloud_timeout_is_safe_http_503_without_run_or_retry(tmp_path, monkeypatch):
    database, data_dir, album_id, _, embedding_provider = _album(tmp_path)
    _configure(monkeypatch, database, _settings(data_dir), embedding_provider)
    calls = []

    def timeout(*args, **kwargs):
        calls.append(1)
        raise TimeoutError(SECRET)

    monkeypatch.setattr(cloud, "_https_transport", timeout)
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag", json={"query": "蓝色的照片", "top_k": 2}
        )
    assert response.status_code == 503, response.text
    assert len(calls) == 1
    assert SECRET not in response.text
    assert "timed out" in response.json()["detail"]
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0
