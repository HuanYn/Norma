"""Offline HTTP contract tests for the three-day demo integration."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from ai import demo_api
from ai.config import Settings
from ai.rag.transformers_runtime import LocalVLMUnavailableError
from ai.people.indexer import PeopleIndexer
from ai.people.labels import PersonLabelService
from ai.schemas import SelectionReplacementRequest
from ai.selection.replacement import ReplacementService
from ai.selection.service import SelectionService
from ai.selection.structured import StructuredSelectionParser
from ai.tests.test_selection import FakeSelectionProvider, _album
from ai.tests.test_person_selection import QuotaFaceProvider
from ai.tests.test_structured_selection import FakeTextRuntime, document
from ai.video.models import decode_image


class TrackingEmbedding(FakeSelectionProvider):
    def __init__(self):
        self.queries = []

    def embed_text(self, text):
        self.queries.append(text)
        return super().embed_text(text)


class FakeWorker:
    def __init__(self):
        self.calls = []
        self.job_id = "a" * 32

    def submit(self, content, options, *, upload_confirmed):
        self.calls.append(("submit", content, options, upload_confirmed))
        # Exercise the actual client's image precondition without any network.
        decode_image(base64.b64encode(content).decode("ascii"))
        return {"id": self.job_id, "state": "queued", "progress": 0}

    def health(self):
        self.calls.append(("health",))
        return {"status": "ok"}

    def status(self, job_id):
        self.calls.append(("status", job_id))
        return {"id": job_id, "state": "queued"}

    def cancel(self, job_id):
        self.calls.append(("cancel", job_id))
        return {"id": job_id, "state": "cancel_requested"}

    def download(self, job_id, destination):
        self.calls.append(("download", job_id, destination))
        destination.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    database, album_id, ids = _album(tmp_path)
    settings = Settings(
        host="127.0.0.1", port=8765, data_dir=tmp_path / "data", log_level="INFO",
        vlm_provider="openai-compatible", vlm_base_url="https://example.invalid",
        vlm_model="test-model", vlm_api_key="unit-test-secret",
        video_base_url="http://127.0.0.1:9876",
    )
    embedding = TrackingEmbedding()
    service = SelectionService(database, embedding)
    worker = FakeWorker()
    values = SimpleNamespace(database=database, album_id=album_id, ids=ids,
        settings=settings, embedding=embedding, service=service, worker=worker,
        worker_factories=[])

    def factory(current_settings):
        values.worker_factories.append(current_settings)
        return worker

    monkeypatch.setattr(demo_api, "_client", factory)
    app = FastAPI()
    app.include_router(demo_api.create_demo_router(
        lambda: database, lambda: values.settings, lambda: service,
    ))
    values.client = TestClient(app, raise_server_exceptions=False)
    yield values
    values.client.close()


def _parser(monkeypatch, value):
    runtime = FakeTextRuntime(value)
    parser = StructuredSelectionParser(runtime)
    factories = []
    def factory(settings):
        factories.append(settings)
        return parser
    monkeypatch.setattr(demo_api, "_parser", factory)
    return runtime, factories


def _video_request(values, **changes):
    payload = {"album_id": values.album_id, "photo_id": values.ids["a"],
               "upload_confirmed": True, "options": {"prompt": "slow camera push"}}
    payload.update(changes)
    return payload


@pytest.mark.parametrize("endpoint", ["/selection/parse", "/selections/structured"])
def test_cloud_consent_absent_never_constructs_or_calls_parser(setup, monkeypatch, endpoint):
    runtime, factories = _parser(monkeypatch, document())
    payload = {"prompt": "夜景"}
    if "structured" in endpoint:
        payload["album_id"] = setup.album_id
    response = setup.client.post(endpoint, json=payload)
    assert response.status_code == 400
    assert not factories and not runtime.calls and not setup.embedding.queries


@pytest.mark.parametrize("value", ["true", 1, None])
def test_cloud_consent_is_not_string_or_number_coerced(setup, monkeypatch, value):
    runtime, factories = _parser(monkeypatch, document())
    response = setup.client.post("/selection/parse", json={"prompt": "夜景", "allow_cloud": value})
    assert response.status_code == 422
    assert not factories and not runtime.calls


def test_explicit_consent_returns_parse_without_running_selection(setup, monkeypatch):
    runtime, _ = _parser(monkeypatch, document(semantic_query="夜景", style_preferences=[]))
    response = setup.client.post("/selection/parse", json={"prompt": "夜景", "allow_cloud": True})
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["provenance"]["provider_fingerprint"] == runtime.provider_fingerprint
    assert len(runtime.calls) == 1 and not setup.embedding.queries
    with setup.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0] == 0


def test_structured_selection_uses_semantic_text_constraints_and_persists_audit(setup, monkeypatch):
    value = document(semantic_query="night", style_preferences=["暖色"])
    value["hard_constraints"].update(target_count=3, min_quality=50.0)
    value["constraint_evidence"] = [
        {"field": "target_count", "source_text": "挑选三张"},
        {"field": "min_quality", "source_text": "质量至少50"},
    ]
    runtime, _ = _parser(monkeypatch, value)
    original = "挑选三张，质量至少50，night，暖色"
    response = setup.client.post("/selections/structured", json={
        "album_id": setup.album_id, "prompt": original, "allow_cloud": True,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(runtime.calls) == 1
    assert result["feasible"] and len(result["selected"]) == 3
    assert setup.embedding.queries == ["night 暖色"]
    assert result["prompt"] == original and result["query_text"] == "night 暖色"
    selected_ids = {item["photo_id"] for item in result["selected"]}
    assert setup.ids["d"] not in selected_ids  # auto reject
    assert setup.ids["e"] not in selected_ids  # quality floor
    assert len({setup.ids["a"], setup.ids["b"]} & selected_ids) == 1
    assert result["constraints"]["target_count"] == 3
    assert result["intent_provenance"]["provider_fingerprint"] == runtime.provider_fingerprint
    stored = setup.service.get(result["selection_id"])
    assert stored.intent_provenance == result["intent_provenance"]
    assert stored.query_text == "night 暖色"
    with setup.database.connect() as connection:
        row = connection.execute("SELECT raw_prompt,parse_json,result_json FROM selections WHERE id=?",
                                 (result["selection_id"],)).fetchone()
    assert row["raw_prompt"] == original
    assert json.loads(row["parse_json"])["target_count"] == 3
    assert json.loads(row["result_json"])["intent_provenance"] == result["intent_provenance"]


def test_structured_quality_only_does_not_embed_new_count_wording(setup, monkeypatch):
    value = document(semantic_query="", style_preferences=[])
    value["hard_constraints"]["target_count"] = 2
    value["constraint_evidence"] = [{"field": "target_count", "source_text": "挑选两张"}]
    _parser(monkeypatch, value)
    response = setup.client.post("/selections/structured", json={
        "album_id": setup.album_id, "prompt": "挑选两张", "allow_cloud": True,
    })
    assert response.status_code == 200
    assert response.json()["feasible"]
    assert response.json()["query_text"] is None and not setup.embedding.queries


@pytest.mark.parametrize("semantic", [False, True])
def test_structured_replacement_retains_parse_provenance_and_query_mode(setup, monkeypatch, semantic):
    value = document(semantic_query="night" if semantic else "", style_preferences=[])
    value["hard_constraints"]["target_count"] = 2
    value["constraint_evidence"] = [{"field": "target_count", "source_text": "挑选两张"}]
    _parser(monkeypatch, value)
    response = setup.client.post("/selections/structured", json={
        "album_id": setup.album_id, "prompt": "挑选两张" + ("，night" if semantic else ""),
        "allow_cloud": True,
    })
    assert response.status_code == 200
    original = response.json()
    replaced = ReplacementService(setup.database, setup.embedding).replace(
        original["selection_id"],
        SelectionReplacementRequest(remove_photo_id=original["selected"][0]["photo_id"]),
    )
    assert replaced.feasible
    assert replaced.updated_selection.intent_provenance == original["intent_provenance"]
    assert replaced.updated_selection.query_text == original["query_text"]
    assert setup.embedding.queries == (["night", "night"] if semantic else [])
    assert setup.service.get(replaced.updated_selection.selection_id).intent_provenance == original["intent_provenance"]


def test_structured_person_quota_uses_existing_named_face_evidence(setup, monkeypatch):
    indexed = PeopleIndexer(setup.database, setup.settings.data_dir, QuotaFaceProvider()).index(setup.album_id)
    cluster = next(item for item in indexed.clusters if any(face.photo_id == setup.ids["c"] for face in item.faces))
    PersonLabelService(setup.database).set(setup.album_id, cluster.cluster_id, "Me")
    value = document(semantic_query="", style_preferences=[])
    value["hard_constraints"].update(target_count=2, person_minimums={"me": 2})
    value["constraint_evidence"] = [
        {"field": "target_count", "source_text": "挑选两张"},
        {"field": "person_minimums", "source_text": "最少两张有我"},
    ]
    _parser(monkeypatch, value)
    response = setup.client.post("/selections/structured", json={
        "album_id": setup.album_id, "prompt": "挑选两张，最少两张有我", "allow_cloud": True,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["feasible"] and len(result["selected"]) == 2
    assert all(item["photo_id"] in {setup.ids[name] for name in ["c", "e", "f"]} for item in result["selected"])
    assert result["constraints"]["person_minimums"] == {"me": 2}
    assert result["people_snapshot_sha256"]


def test_unsupported_requirement_is_visible_and_blocks_selection(setup, monkeypatch):
    value = document(unsupported_hard_requirements=[{"text": "不要自拍", "reason": "No verified selfie evidence"}])
    runtime, _ = _parser(monkeypatch, value)
    response = setup.client.post("/selections/structured", json={
        "album_id": setup.album_id, "prompt": "不要自拍", "allow_cloud": True,
    })
    assert response.status_code == 422
    assert response.json()["detail"]["document"]["unsupported_hard_requirements"][0]["text"] == "不要自拍"
    assert len(runtime.calls) == 1 and not setup.embedding.queries
    with setup.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0] == 0


def test_local_model_missing_is_service_unavailable_not_server_error(setup, monkeypatch):
    setup.settings = replace(setup.settings, vlm_provider="local")
    def unavailable(_):
        raise LocalVLMUnavailableError("Model missing at E:/private/model")
    monkeypatch.setattr(demo_api, "_parser", unavailable)
    response = setup.client.post("/selection/parse", json={"prompt": "夜景"})
    assert response.status_code == 503
    assert "E:/private" not in response.text


@pytest.mark.parametrize("value", [False, "true", 1, None])
def test_video_requires_literal_confirmation_before_worker_use(setup, value):
    response = setup.client.post("/demo/video/jobs", json=_video_request(setup, upload_confirmed=value))
    assert response.status_code == 422
    assert not setup.worker.calls and not setup.worker_factories


def test_video_requires_confirmation_field(setup):
    value = _video_request(setup)
    del value["upload_confirmed"]
    assert setup.client.post("/demo/video/jobs", json=value).status_code == 422
    assert not setup.worker.calls


@pytest.mark.parametrize("key,value", [("image_path", "E:/private/image.jpg"), ("worker_url", "https://other.invalid"), ("image_base64", "not-allowed")])
def test_video_endpoint_never_accepts_arbitrary_path_url_or_image_bytes(setup, key, value):
    response = setup.client.post("/demo/video/jobs", json=_video_request(setup, **{key: value}))
    assert response.status_code == 422
    assert not setup.worker.calls and not setup.worker_factories


def test_video_photo_must_belong_to_requested_album(setup):
    response = setup.client.post("/demo/video/jobs", json=_video_request(setup, album_id="different-album"))
    assert response.status_code == 404
    assert not setup.worker.calls


def test_video_refuses_changed_indexed_source_before_upload(setup):
    with setup.database.connect() as connection:
        row = connection.execute("SELECT absolute_path FROM photos WHERE id=?", (setup.ids["a"],)).fetchone()
    source = Path(row["absolute_path"])
    source.write_bytes(source.read_bytes() + b"changed")
    response = setup.client.post("/demo/video/jobs", json=_video_request(setup))
    assert response.status_code == 409
    assert not setup.worker.calls


def test_video_refuses_source_changed_during_preparation(setup, monkeypatch):
    with setup.database.connect() as connection:
        row = connection.execute("SELECT absolute_path FROM photos WHERE id=?", (setup.ids["a"],)).fetchone()
    path = Path(row["absolute_path"])
    original = demo_api.ImageOps.exif_transpose
    def change_after_decode(source):
        oriented = original(source)
        path.write_bytes(path.read_bytes() + b"source-mutated-during-preparation")
        return oriented
    monkeypatch.setattr(demo_api.ImageOps, "exif_transpose", change_after_decode)
    response = setup.client.post("/demo/video/jobs", json=_video_request(setup))
    assert response.status_code == 409
    assert not setup.worker.calls


def _replace_indexed_photo(setup, image, *, exif=None):
    with setup.database.connect() as connection:
        row = connection.execute("SELECT absolute_path FROM photos WHERE id=?", (setup.ids["a"],)).fetchone()
        source = Path(row["absolute_path"])
        image.save(source, format="JPEG", **({"exif": exif} if exif else {}))
        stat = source.stat()
        connection.execute("UPDATE photos SET file_size=?,source_mtime_ns=? WHERE id=?",
                           (stat.st_size, stat.st_mtime_ns, setup.ids["a"]))


def test_video_upload_is_bounded_oriented_metadata_free_derivative(setup):
    exif = Image.Exif()
    exif[0x0112] = 6  # Rotate 90 degrees before export.
    exif[0x010E] = "private location and camera note"
    _replace_indexed_photo(setup, Image.new("RGB", (1600, 1000), "red"), exif=exif)
    response = setup.client.post("/demo/video/jobs", json=_video_request(setup))
    assert response.status_code == 202, response.text
    kind, content, options, confirmed = setup.worker.calls[0]
    assert kind == "submit" and confirmed is True
    assert isinstance(content, bytes) and b"private location" not in content
    with Image.open(io.BytesIO(content)) as uploaded:
        assert uploaded.format == "JPEG"
        assert uploaded.size == (800, 1280)
        assert not uploaded.getexif()
        assert "icc_profile" not in uploaded.info and "exif" not in uploaded.info
    with setup.database.connect() as connection:
        row = connection.execute("SELECT album_id,photo_id FROM demo_video_links_v1 WHERE job_id=?",
                                 (setup.worker.job_id,)).fetchone()
    assert row["album_id"] == setup.album_id and row["photo_id"] == setup.ids["a"]


def test_video_tiny_indexed_image_is_rejected_as_client_error(setup):
    _replace_indexed_photo(setup, Image.new("RGB", (16, 16), "red"))
    response = setup.client.post("/demo/video/jobs", json=_video_request(setup))
    assert 400 <= response.status_code < 500, response.text


@pytest.mark.parametrize("method,suffix", [("get", ""), ("post", "/cancel"), ("get", "/artifact")])
def test_foreign_remote_job_cannot_be_queried_cancelled_or_downloaded(setup, method, suffix):
    response = getattr(setup.client, method)("/demo/video/jobs/" + "b" * 32 + suffix)
    assert response.status_code == 404
    assert not setup.worker.calls and not setup.worker_factories


def test_owned_video_job_can_be_polled_cancelled_and_cached_locally(setup):
    assert setup.client.post("/demo/video/jobs", json=_video_request(setup)).status_code == 202
    base = "/demo/video/jobs/" + setup.worker.job_id
    assert setup.client.get(base).status_code == 200
    assert setup.client.post(base + "/cancel").status_code == 200
    first = setup.client.get(base + "/artifact")
    second = setup.client.get(base + "/artifact")
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert first.headers["content-type"] == "video/mp4"
    downloads = [call for call in setup.worker.calls if call[0] == "download"]
    assert len(downloads) == 1
    assert downloads[0][2] == setup.settings.data_dir / "video-artifacts" / (setup.worker.job_id + ".mp4")


def test_unconfigured_video_status_does_not_touch_worker(setup):
    setup.settings = replace(setup.settings, video_base_url="")
    response = setup.client.get("/demo/video/status")
    assert response.json() == {"configured": False, "reachable": False}
    assert not setup.worker.calls and not setup.worker_factories
