from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai import app as app_module
from ai.config import Settings, load_settings
from ai.index import AlbumIndexer
from ai.index.embedding import EmbeddingProvider
from ai.preferences.service import PreferenceService
from ai.rag.models import (
    GeneratedCitation,
    GeneratedClaim,
    GenerationProvenance,
    ProviderGenerationOutput,
    VLMInputBudgetError,
    canonical_answer,
    snapshot_image_bytes,
)
from ai.rag.providers import ProviderImagePayload, ScriptedGroundedProvider
from ai.rag import prompting as rag_prompting
from ai.rag.security import PATH_REDACTION
from ai.rag import transformers_runtime as runtime_module
from ai.rag import service as rag_service_module
from ai.rag.service import GroundedRAGService, RAGBusyError
from ai.rag.transformers_runtime import (
    LocalVLMUnavailableError,
    TransformersQwen3VLRuntime,
    create_local_qwen3vl_provider,
)
from ai.retrieval import RetrievalService
from ai.schemas import (
    AlbumRAGRequest,
    AlbumSearchRequest,
    PairwiseFeedbackRequest,
    SelectionRequest,
)
from ai.selection import SelectionService
from ai.storage import Database


GENERATION_PROVIDER = "fake-qwen3-vl-local-v1"


def test_rag_http_error_redaction_consumes_posix_path_suffix_after_spaces() -> None:
    detail = app_module._safe_rag_error(
        RuntimeError("provider failed at /home/alice/My Secret/photo.jpg")
    )

    assert PATH_REDACTION in detail
    assert "Secret/photo.jpg" not in detail


class RAGOpenClipProvider(EmbeddingProvider):
    name = "openclip-rag-test-512d-v1"
    dimension = 512

    def embed_image(self, path: Path) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[0] = 1.0
        return vector


def _vector(cosine: float, axis: int) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = cosine
    vector[axis] = np.sqrt(1.0 - cosine**2)
    return vector


def _album(
    tmp_path: Path,
) -> tuple[Database, Path, str, dict[str, str], RAGOpenClipProvider]:
    folder = tmp_path / "album"
    folder.mkdir()
    for index, name in enumerate(("a", "b", "c", "d")):
        Image.new(
            "RGB",
            (96, 64),
            (30 + index * 35, 70 + index * 10, 110),
        ).save(folder / f"{name}.jpg", "JPEG")
    data_dir = tmp_path / "data"
    database = Database(data_dir / "norma.db")
    indexed = AlbumIndexer(database, data_dir).index(folder)
    ids = {photo.filename[0]: photo.id for photo in indexed.photos}
    provider = RAGOpenClipProvider()
    vectors = {
        "a": _vector(0.84, 1),
        "b": _vector(0.82, 2),
        "c": _vector(0.60, 3),
        "d": _vector(0.40, 4),
    }
    embedding_dir = data_dir / "rag-embeddings"
    embedding_dir.mkdir()
    with database.connect() as connection:
        for name, vector in vectors.items():
            embedding_path = embedding_dir / f"{name}.npy"
            np.save(embedding_path, vector, allow_pickle=False)
            connection.execute(
                """
                UPDATE photos SET embedding_path = ?, embedding_provider = ?,
                    embedding_source_size = file_size,
                    embedding_source_mtime_ns = source_mtime_ns,
                    embedding_source_sha256 = ?,
                    quality_score = ?, auto_reject = 0
                WHERE id = ?
                """,
                (
                    str(embedding_path.resolve()),
                    provider.name,
                    hashlib.sha256((folder / f"{name}.jpg").read_bytes()).hexdigest(),
                    20.0 + ord(name),
                    ids[name],
                ),
            )
    return database, data_dir, indexed.album_id, ids, provider


def _valid_provider() -> ScriptedGroundedProvider:
    def output(request) -> ProviderGenerationOutput:
        photo_id = request.allowed_photo_ids[0]
        claims = (GeneratedClaim("c1", f"{photo_id} 是检索到的图像证据。"),)
        citations = (GeneratedCitation("c1", photo_id),)
        provenance = GenerationProvenance(
            retrieval_provider_fingerprint=(
                request.provenance.retrieval_provider_fingerprint
            ),
            generation_provider_fingerprint=GENERATION_PROVIDER,
            query_digest=request.provenance.query_digest,
            candidate_digest=request.provenance.candidate_digest,
            evidence_digest=request.provenance.evidence_digest,
        )
        return ProviderGenerationOutput(
            answer=canonical_answer(claims),
            claims=claims,
            citations=citations,
            provenance=provenance,
        )

    return ScriptedGroundedProvider(output, name=GENERATION_PROVIDER)


def _settings(data_dir: Path, *, vlm_model_path: Path | None = None) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8765,
        data_dir=data_dir,
        log_level="INFO",
        embedding_provider="openclip-multilingual",
        vlm_model_path=vlm_model_path,
    )


def _configure_app(
    monkeypatch: pytest.MonkeyPatch,
    database: Database,
    data_dir: Path,
    provider: EmbeddingProvider,
    generation_provider_factory: Callable[[], object],
) -> None:
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "settings", _settings(data_dir))
    monkeypatch.setattr(app_module, "embedding_provider", lambda: provider)
    monkeypatch.setattr(
        app_module,
        "rag_generation_provider",
        generation_provider_factory,
    )


def _fake_pinned_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes]]:
    model_dir = (tmp_path / "qwen").resolve()
    model_dir.mkdir()
    contents = {
        name: (
            b"fake-local-weights"
            if name == "model.safetensors"
            else f"fake:{name}".encode()
        )
        for name in runtime_module._PINNED_RUNTIME_ASSET_NAMES
    }
    for name, content in contents.items():
        (model_dir / name).write_bytes(content)
    manifest_path = tmp_path / "pinned-test-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "norma-pinned-model-manifest-v1",
                "model_id": runtime_module.QWEN3_VL_MODEL_ID,
                "revision": runtime_module.QWEN3_VL_MODEL_REVISION,
                "files": [
                    {
                        "name": name,
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "source": "test",
                    }
                    for name, content in sorted(contents.items())
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_module, "_PINNED_MANIFEST_PATH", manifest_path)
    return model_dir, contents


def test_rag_endpoint_uses_learned_search_and_persists_immutable_safe_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, ids, provider = _album(tmp_path)
    selection = SelectionService(database, provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 1 city photo")
    )
    for _ in range(5):
        learned = PreferenceService(database, provider).record_pairwise(
            PairwiseFeedbackRequest(
                album_id=album_id,
                selection_id=selection.selection_id,
                preferred_photo_id=ids["b"],
                rejected_photo_id=ids["a"],
            )
        )
    fake = _valid_provider()
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)

    with TestClient(app_module.app) as client:
        first = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "适合城市回顾的照片", "top_k": 2, "user_id": "local"},
        )
        second = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "适合城市回顾的照片", "top_k": 2, "user_id": "local"},
        )

    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["validation_level"] == "citation-referential-only"
    assert payload["semantic_entailment_verified"] is False
    assert payload["retrieval"]["preference_model_id"] == learned.contextual_model_id
    assert payload["retrieval"]["preference_comparisons"] == 5
    assert payload["retrieval"]["algorithm"] == (
        "openclip-contextual-posterior-utility-v1"
    )
    assert len(payload["retrieval"]["matches"]) == 2
    assert (
        payload["provenance"]["candidate_digest"]
        == second.json()["provenance"]["candidate_digest"]
    )
    assert (
        payload["provenance"]["query_digest"]
        == second.json()["provenance"]["query_digest"]
    )
    assert (
        payload["provenance"]["evidence_digest"]
        == second.json()["provenance"]["evidence_digest"]
    )

    with database.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM rag_runs ORDER BY created_at, id"
        ).fetchall()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE rag_runs SET query_text = 'tampered' WHERE id = ?",
                (payload["run_id"],),
            )
        with pytest.raises(sqlite3.OperationalError, match="rowid"):
            connection.execute(
                """INSERT OR REPLACE INTO rag_runs(
                       rowid, id, album_id, user_id, query_text,
                       retrieval_provider_fingerprint,
                       generation_provider_fingerprint,
                       candidate_digest, query_digest, evidence_digest,
                       evidence_json, result_json, request_json, created_at
                   )
                   SELECT 1, 'replacement-id', album_id, user_id, 'tampered',
                          retrieval_provider_fingerprint,
                          generation_provider_fingerprint,
                          candidate_digest, query_digest, evidence_digest,
                          evidence_json, result_json, request_json, created_at
                   FROM rag_runs LIMIT 1"""
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "DELETE FROM rag_runs WHERE id = ?", (payload["run_id"],)
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                """INSERT OR REPLACE INTO rag_runs(
                       id, album_id, user_id, query_text,
                       retrieval_provider_fingerprint,
                       generation_provider_fingerprint,
                       candidate_digest, query_digest, evidence_digest,
                       evidence_json, result_json, request_json
                   ) VALUES (?, 'album', 'user', 'tampered', 'r', 'g',
                             'c', 'q', 'e', '{}', '{}', '{}')""",
                (payload["run_id"],),
            )

    assert len(rows) == 2
    stored = rows[0]
    evidence = json.loads(stored["evidence_json"])
    assert len(evidence["candidate_photo_ids"]) == 4
    assert len(evidence["items"]) == 2
    assert all("content" not in item["image"] for item in evidence["items"])
    audit_text = "\n".join(
        str(stored[field]) for field in ("evidence_json", "result_json", "request_json")
    )
    with database.connect() as connection:
        source_paths = [
            row["absolute_path"]
            for row in connection.execute(
                "SELECT absolute_path FROM photos WHERE album_id = ?", (album_id,)
            )
        ]
    assert all(source not in audit_text for source in source_paths)
    assert str(data_dir.resolve()) not in audit_text


def test_missing_local_model_returns_503_without_import_or_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, _, provider = _album(tmp_path)
    missing = (tmp_path / "missing-qwen").resolve()
    _configure_app(
        monkeypatch,
        database,
        data_dir,
        provider,
        lambda: app_module.create_local_qwen3vl_provider(missing),
    )

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片"},
        )

    assert response.status_code == 503
    assert "local Qwen3-VL" in response.json()["detail"]
    assert not missing.exists()
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_rag_http_persistence_redacts_path_like_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, _, provider = _album(tmp_path)
    fake = _valid_provider()
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)
    query = "请处理 located/vault/private 和 /workspace"

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": query, "user_id": "portfolio-user_01"},
        )

    assert response.status_code == 200, response.text
    with database.connect() as connection:
        row = connection.execute(
            "SELECT query_text, request_json, result_json FROM rag_runs"
        ).fetchone()
    assert row is not None
    persisted = "\n".join(str(row[field]) for field in row.keys())
    assert query not in persisted
    assert "/workspace" not in persisted
    assert "located/vault/private" not in persisted
    assert PATH_REDACTION in persisted


def test_rag_http_rejects_path_like_user_id_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, _, provider = _album(tmp_path)
    fake = _valid_provider()
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)
    local_identity = r"C:\Users\alice"

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "user_id": local_identity},
        )

    assert response.status_code == 422
    assert local_identity not in response.text
    assert fake.calls == []
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_interactive_search_does_not_reread_album_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, _, provider = _album(tmp_path)
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.suffix.casefold() in {".jpg", ".jpeg"}:
            raise AssertionError("interactive search must not hash original photos")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    response = RetrievalService(database, data_dir, provider).search(
        AlbumSearchRequest(
            album_id=album_id,
            query="城市照片",
        )
    )

    assert response.matches


def test_source_change_after_search_returns_409_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, ids, provider = _album(tmp_path)
    retrieval = RetrievalService(database, data_dir, provider)
    original_search = retrieval.search
    fake = _valid_provider()

    def changing_search(request, **kwargs):
        result = original_search(request, **kwargs)
        with database.connect() as connection:
            row = connection.execute(
                "SELECT absolute_path, source_mtime_ns FROM photos WHERE id = ?",
                (ids["a"],),
            ).fetchone()
        target = Path(row["absolute_path"])
        Image.new("RGB", (97, 65), (250, 20, 20)).save(target, "JPEG")
        changed_ns = int(row["source_mtime_ns"]) + 1_000_000_000
        os.utime(target, ns=(changed_ns, changed_ns))
        return result

    retrieval.search = changing_search  # type: ignore[method-assign]
    service = GroundedRAGService(database, retrieval, lambda: fake)
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "settings", _settings(data_dir))
    monkeypatch.setattr(app_module, "rag_service", lambda: service)

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片"},
        )

    assert response.status_code == 409
    assert fake.calls == []
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_same_stat_top_image_replacement_is_rejected_by_embedding_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, ids, provider = _album(tmp_path)
    retrieval = RetrievalService(database, data_dir, provider)
    original_search = retrieval.search
    with database.connect() as connection:
        row = connection.execute(
            "SELECT absolute_path FROM photos WHERE id = ?", (ids["a"],)
        ).fetchone()
    target = Path(row["absolute_path"])
    original = target.read_bytes()
    original_stat = target.stat()
    replacement = bytearray(original)
    replacement[100] ^= 1
    with Image.open(BytesIO(replacement)) as image:
        image.load()
    assert bytes(replacement) != original
    assert len(replacement) == len(original)

    def changing_search(request, **kwargs):
        result = original_search(request, **kwargs)
        target.write_bytes(replacement)
        os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return result

    retrieval.search = changing_search  # type: ignore[method-assign]
    fake = _valid_provider()
    service = GroundedRAGService(database, retrieval, lambda: fake)
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "settings", _settings(data_dir))
    monkeypatch.setattr(app_module, "rag_service", lambda: service)

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "top_k": 1},
        )

    assert response.status_code == 409
    assert fake.calls == []
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_corrupt_non_top_embedding_returns_409_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, ids, provider = _album(tmp_path)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT embedding_path FROM photos WHERE id = ?", (ids["d"],)
        ).fetchone()
    Path(row["embedding_path"]).write_bytes(b"not-a-valid-npy")
    fake = _valid_provider()
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "top_k": 1},
        )

    assert response.status_code == 409
    assert fake.calls == []


def test_temporary_valid_embedding_aba_is_rejected_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, ids, provider = _album(tmp_path)
    retrieval = RetrievalService(database, data_dir, provider)
    original_search = retrieval.search
    with database.connect() as connection:
        row = connection.execute(
            "SELECT embedding_path FROM photos WHERE id = ?", (ids["d"],)
        ).fetchone()
    embedding_path = Path(row["embedding_path"])
    original_bytes = embedding_path.read_bytes()
    original_stat = embedding_path.stat()
    alternate = np.zeros(provider.dimension, dtype=np.float32)
    alternate[7] = 1.0
    alternate_path = tmp_path / "alternate.npy"
    np.save(alternate_path, alternate, allow_pickle=False)
    alternate_bytes = alternate_path.read_bytes()
    assert len(alternate_bytes) == len(original_bytes)

    def aba_search(request, **kwargs):
        embedding_path.write_bytes(alternate_bytes)
        os.utime(
            embedding_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        try:
            return original_search(request, **kwargs)
        finally:
            embedding_path.write_bytes(original_bytes)
            os.utime(
                embedding_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

    retrieval.search = aba_search  # type: ignore[method-assign]
    fake = _valid_provider()
    service = GroundedRAGService(database, retrieval, lambda: fake)
    monkeypatch.setattr(app_module, "database", database)
    monkeypatch.setattr(app_module, "settings", _settings(data_dir))
    monkeypatch.setattr(app_module, "rag_service", lambda: service)

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "top_k": 1},
        )

    assert response.status_code == 409
    assert fake.calls == []
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_source_change_during_generation_returns_409_without_audit_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, ids, provider = _album(tmp_path)
    valid = _valid_provider()

    def mutate_then_generate(request) -> ProviderGenerationOutput:
        with database.connect() as connection:
            row = connection.execute(
                "SELECT absolute_path, source_mtime_ns FROM photos WHERE id = ?",
                (ids["d"],),
            ).fetchone()
        target = Path(row["absolute_path"])
        Image.new("RGB", (98, 66), (20, 250, 20)).save(target, "JPEG")
        changed_ns = int(row["source_mtime_ns"]) + 1_000_000_000
        os.utime(target, ns=(changed_ns, changed_ns))
        return valid.generate(request)

    fake = ScriptedGroundedProvider(mutate_then_generate, name=GENERATION_PROVIDER)
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "top_k": 1},
        )

    assert response.status_code == 409
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "quality_score = quality_score + 1",
        "auto_reject = CASE auto_reject WHEN 0 THEN 1 ELSE 0 END",
    ),
)
def test_ranking_input_change_during_generation_is_409_without_audit(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, ids, provider = _album(tmp_path)
    valid = _valid_provider()

    def mutate_then_generate(request) -> ProviderGenerationOutput:
        with database.connect() as connection:
            connection.execute(
                f"UPDATE photos SET {mutation} WHERE id = ?", (ids["d"],)
            )
        return valid.generate(request)

    fake = ScriptedGroundedProvider(mutate_then_generate, name=GENERATION_PROVIDER)
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "top_k": 1},
        )

    assert response.status_code == 409
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_active_preference_model_change_during_generation_is_409(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, ids, provider = _album(tmp_path)
    selection = SelectionService(database, provider).select(
        SelectionRequest(album_id=album_id, prompt="Select 1 city photo")
    )
    for _ in range(5):
        PreferenceService(database, provider).record_pairwise(
            PairwiseFeedbackRequest(
                album_id=album_id,
                selection_id=selection.selection_id,
                preferred_photo_id=ids["b"],
                rejected_photo_id=ids["a"],
            )
        )
    valid = _valid_provider()

    def deactivate_then_generate(request) -> ProviderGenerationOutput:
        with database.connect() as connection:
            connection.execute(
                "UPDATE preference_models SET active = 0 WHERE active = 1"
            )
        return valid.generate(request)

    fake = ScriptedGroundedProvider(deactivate_then_generate, name=GENERATION_PROVIDER)
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "top_k": 1},
        )

    assert response.status_code == 409
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_evidence_byte_budget_rejects_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, _, provider = _album(tmp_path)
    fake = _valid_provider()
    monkeypatch.setattr(rag_service_module, "MAX_EVIDENCE_BYTES", 1)
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "top_k": 1},
        )

    assert response.status_code == 413
    assert fake.calls == []


def test_vlm_visual_token_overflow_maps_413_without_audit_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, _, provider = _album(tmp_path)

    def reject_visual_input(_request) -> ProviderGenerationOutput:
        raise VLMInputBudgetError(
            "local Qwen3-VL visual input exceeds the 4096-token hard limit"
        )

    fake = ScriptedGroundedProvider(reject_visual_input, name=GENERATION_PROVIDER)
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "top_k": 1},
        )

    assert response.status_code == 413
    assert "4096-token" in response.json()["detail"]
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_decompression_bomb_pixel_budget_rejects_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, ids, provider = _album(tmp_path)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT absolute_path FROM photos WHERE id = ?", (ids["a"],)
        ).fetchone()
    target = Path(row["absolute_path"])
    bomb_path = tmp_path / "compressed-bomb.png"
    Image.new("1", (9000, 8000), 1).save(bomb_path, "PNG")
    target.write_bytes(bomb_path.read_bytes())
    source_stat = target.stat()
    source_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE photos SET file_size = ?, source_mtime_ns = ?,
                embedding_source_size = ?, embedding_source_mtime_ns = ?,
                embedding_source_sha256 = ?
            WHERE id = ?
            """,
            (
                source_stat.st_size,
                source_stat.st_mtime_ns,
                source_stat.st_size,
                source_stat.st_mtime_ns,
                source_sha256,
                ids["a"],
            ),
        )
    fake = _valid_provider()
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)

    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片", "top_k": 1},
        )

    assert response.status_code == 413
    assert fake.calls == []


def test_process_admission_rejects_second_run_before_candidate_io(
    tmp_path: Path,
) -> None:
    database, data_dir, album_id, _, provider = _album(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    valid = _valid_provider()

    def blocking_generate(request) -> ProviderGenerationOutput:
        entered.set()
        assert release.wait(timeout=5)
        return valid.generate(request)

    first_provider = ScriptedGroundedProvider(
        blocking_generate,
        name=GENERATION_PROVIDER,
    )
    first = GroundedRAGService(
        database,
        RetrievalService(database, data_dir, provider),
        lambda: first_provider,
    )
    second = GroundedRAGService(
        database,
        RetrievalService(database, data_dir, provider),
        _valid_provider,
    )
    second_candidate_io = False

    def forbidden_candidate_io(*_args, **_kwargs):
        nonlocal second_candidate_io
        second_candidate_io = True
        raise AssertionError("busy request must not inspect candidates")

    second._candidate_snapshot = forbidden_candidate_io  # type: ignore[method-assign]
    request = AlbumRAGRequest(query="城市照片", top_k=1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(first.run, album_id, request)
        assert entered.wait(timeout=5)
        with pytest.raises(RAGBusyError, match="busy"):
            second.run(album_id, request)
        assert second_candidate_io is False
        release.set()
        assert future.result(timeout=5).answer


@pytest.mark.parametrize("invalid_kind", ("forged", "uncited", "path-leak"))
def test_invalid_provider_output_returns_422_and_is_not_persisted(
    invalid_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, data_dir, album_id, _, provider = _album(tmp_path)

    def invalid_output(request) -> ProviderGenerationOutput:
        claim_text = (
            r"本地位置是 C:\Users\alice\secret.jpg"
            if invalid_kind == "path-leak"
            else "这是一个模型结论。"
        )
        claims = (GeneratedClaim("c1", claim_text),)
        if invalid_kind == "uncited":
            citations: tuple[GeneratedCitation, ...] = ()
        elif invalid_kind == "forged":
            citations = (GeneratedCitation("c1", "forged-photo-id"),)
        else:
            citations = (GeneratedCitation("c1", request.allowed_photo_ids[0]),)
        return ProviderGenerationOutput(
            answer=canonical_answer(claims),
            claims=claims,
            citations=citations,
            provenance=GenerationProvenance(
                retrieval_provider_fingerprint=(
                    request.provenance.retrieval_provider_fingerprint
                ),
                generation_provider_fingerprint=GENERATION_PROVIDER,
                query_digest=request.provenance.query_digest,
                candidate_digest=request.provenance.candidate_digest,
                evidence_digest=request.provenance.evidence_digest,
            ),
        )

    fake = ScriptedGroundedProvider(invalid_output, name=GENERATION_PROVIDER)
    _configure_app(monkeypatch, database, data_dir, provider, lambda: fake)
    with TestClient(app_module.app) as client:
        response = client.post(
            f"/albums/{album_id}/rag",
            json={"query": "城市照片"},
        )

    assert response.status_code == 422
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM rag_runs").fetchone()[0] == 0


def test_transformers_runtime_uses_only_local_cpu_deterministic_multi_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, contents = _fake_pinned_model(tmp_path, monkeypatch)
    weights = contents["model.safetensors"]
    calls: dict[str, object] = {}

    class Processor:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls["processor_load"] = (path, kwargs)
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return {
                "input_ids": [[1, 2]],
                "image_grid_thw": [[1, 2, 2], [1, 2, 2]],
            }

        def batch_decode(self, values, **kwargs):
            calls["decoded_values"] = values
            return ['{"claims":[],"citations":[]}']

    class Model:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            calls["model_load"] = (path, kwargs)
            return cls()

        def to(self, device: str):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return [[1, 2, 3]]

    fake_transformers = SimpleNamespace(
        AutoProcessor=Processor,
        Qwen3VLForConditionalGeneration=Model,
    )
    fake_torch = SimpleNamespace(inference_mode=nullcontext)
    runtime = TransformersQwen3VLRuntime(
        model_dir,
        dependency_loader=lambda: (fake_transformers, fake_torch),
    )
    image_bytes = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(image_bytes, "JPEG")
    payloads = tuple(
        ProviderImagePayload(
            photo_id=f"photo-{index}",
            image=snapshot_image_bytes(
                image_bytes.read_bytes(),
                display_name=f"photo-{index}.jpg",
                media_type="image/jpeg",
            ),
        )
        for index in (1, 2)
    )

    fingerprint_before = runtime.provider_fingerprint
    output = runtime.generate_json(
        system_prompt="system",
        user_prompt="user",
        images=payloads,
        max_new_tokens=256,
        temperature=0.0,
    )

    assert output == '{"claims":[],"citations":[]}'
    assert runtime.provider_fingerprint == fingerprint_before
    assert runtime.loaded is True
    assert calls["device"] == "cpu"
    assert calls["eval"] is True
    for key in ("processor_load", "model_load"):
        loaded_path, kwargs = calls[key]
        assert loaded_path == str(model_dir)
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
    generate = calls["generate"]
    assert generate["do_sample"] is False
    assert generate["max_new_tokens"] == 256
    messages = calls["messages"]
    assert messages[0]["role"] == "system"
    image_parts = [part for part in messages[1]["content"] if part["type"] == "image"]
    assert len(image_parts) == 2
    assert all(part["image"].mode == "RGB" for part in image_parts)
    expected_prefix = hashlib.sha256(weights).hexdigest()[:20]
    assert f"weights_sha256={expected_prefix}" in runtime.provider_fingerprint
    assert "manifest_sha256=" in runtime.provider_fingerprint
    for component in (
        "transformers=",
        "torch=",
        "torchvision=",
        "tokenizers=",
        "safetensors=",
        "numpy=",
        "numeric_threading=",
        "jinja2=",
        "pillow=",
        "python=",
    ):
        assert component in runtime.provider_fingerprint
    provider = create_local_qwen3vl_provider(model_dir, max_new_tokens=384)
    assert provider.name.endswith("|max_new_tokens=384")


@pytest.mark.parametrize(
    "changed_distribution",
    [
        "transformers",
        "torch",
        "torchvision",
        "tokenizers",
        "safetensors",
        "numpy",
        "Jinja2",
        "Pillow",
    ],
)
def test_qwen_runtime_dependency_abi_changes_provider_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_distribution: str,
) -> None:
    model_dir, _ = _fake_pinned_model(tmp_path, monkeypatch)

    monkeypatch.setattr(runtime_module, "package_version", lambda _name: "1.0.0")
    baseline = TransformersQwen3VLRuntime(model_dir).provider_fingerprint

    monkeypatch.setattr(
        runtime_module,
        "package_version",
        lambda name: "2.0.0" if name == changed_distribution else "1.0.0",
    )
    changed = TransformersQwen3VLRuntime(model_dir).provider_fingerprint

    assert changed != baseline


def test_qwen_prompt_contract_changes_provider_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, _ = _fake_pinned_model(tmp_path, monkeypatch)
    baseline = TransformersQwen3VLRuntime(model_dir).provider_fingerprint

    monkeypatch.setattr(
        rag_prompting,
        "SYSTEM_PROMPT",
        rag_prompting.SYSTEM_PROMPT + "\nContract mutation for identity test.",
    )
    changed = TransformersQwen3VLRuntime(model_dir).provider_fingerprint

    assert changed != baseline
    assert "prompt_contract_sha256=" in changed


def test_six_high_resolution_images_are_deterministically_scaled_under_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, _ = _fake_pinned_model(tmp_path, monkeypatch)
    source = tmp_path / "high-resolution.jpg"
    Image.new("RGB", (2048, 1536), (20, 30, 40)).save(source, "JPEG")
    snapshot = source.read_bytes()
    payloads = tuple(
        ProviderImagePayload(
            photo_id=f"photo-{index}",
            image=snapshot_image_bytes(
                snapshot,
                display_name=f"photo-{index}.jpg",
                media_type="image/jpeg",
            ),
        )
        for index in range(6)
    )
    observed_sizes: list[tuple[int, int]] = []

    class Processor:
        def apply_chat_template(self, messages, **_kwargs):
            images = [
                part["image"]
                for part in messages[1]["content"]
                if part["type"] == "image"
            ]
            observed_sizes.extend(image.size for image in images)
            return {
                "input_ids": [[1]],
                "image_grid_thw": [
                    [1, image.height // 16, image.width // 16] for image in images
                ],
            }

        def batch_decode(self, _values, **_kwargs):
            return ['{"claims":[],"citations":[]}']

    class Model:
        def generate(self, **_kwargs):
            return [[1, 2]]

    runtime = TransformersQwen3VLRuntime(model_dir)
    runtime._processor = Processor()
    runtime._model = Model()
    runtime._torch = SimpleNamespace(inference_mode=nullcontext)

    output = runtime.generate_json(
        system_prompt="system",
        user_prompt="user",
        images=payloads,
        max_new_tokens=64,
        temperature=0.0,
    )

    assert output == '{"claims":[],"citations":[]}'
    assert len(observed_sizes) == 6
    assert all(width % 32 == height % 32 == 0 for width, height in observed_sizes)
    assert all(width < 2048 and height < 1536 for width, height in observed_sizes)
    visual_tokens = sum(width * height // (32 * 32) for width, height in observed_sizes)
    assert visual_tokens <= runtime_module.PREFLIGHT_VISUAL_TOKEN_BUDGET


def test_rounding_preflight_and_processor_grid_hard_cap_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = runtime_module._target_image_dimensions((4033, 3025), 640)
    assert target[0] % 32 == target[1] % 32 == 0
    assert target[0] * target[1] // (32 * 32) <= 640

    model_dir, _ = _fake_pinned_model(tmp_path, monkeypatch)
    source = tmp_path / "image.jpg"
    Image.new("RGB", (512, 512), (20, 30, 40)).save(source, "JPEG")
    payload = ProviderImagePayload(
        photo_id="photo",
        image=snapshot_image_bytes(
            source.read_bytes(),
            display_name="photo.jpg",
            media_type="image/jpeg",
        ),
    )
    model_called = False

    class Processor:
        def apply_chat_template(self, _messages, **_kwargs):
            return {
                "input_ids": [[1]],
                # 1 * 2 * 8194 / merge_size^2 = 4097 visual tokens.
                "image_grid_thw": [[1, 2, 8194]],
            }

    class Model:
        def generate(self, **_kwargs):
            nonlocal model_called
            model_called = True
            return [[1, 2]]

    runtime = TransformersQwen3VLRuntime(model_dir)
    runtime._processor = Processor()
    runtime._model = Model()
    runtime._torch = SimpleNamespace(inference_mode=nullcontext)

    with pytest.raises(VLMInputBudgetError, match="4096-token"):
        runtime.generate_json(
            system_prompt="system",
            user_prompt="user",
            images=(payload,),
            max_new_tokens=64,
            temperature=0.0,
        )

    assert model_called is False


@pytest.mark.parametrize(
    "model_inputs",
    (
        {"input_ids": [[1]]},
        {"input_ids": [[1]], "image_grid_thw": [[1, 3, 4]]},
        {"input_ids": [[1]], "image_grid_thw": [[1, 2, 2], [1, 2, 2]]},
    ),
)
def test_processor_grid_missing_malformed_or_wrong_count_is_rejected(
    model_inputs: dict[str, object],
) -> None:
    with pytest.raises(VLMInputBudgetError):
        runtime_module._validate_processor_visual_tokens(model_inputs, image_count=1)


@pytest.mark.parametrize(
    "asset_name",
    ("model.safetensors", "config.json", "tokenizer.json"),
)
def test_runtime_rehash_rejects_same_stat_asset_tamper_before_lazy_load(
    asset_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, _ = _fake_pinned_model(tmp_path, monkeypatch)
    loader_called = False

    def forbidden_loader():
        nonlocal loader_called
        loader_called = True
        raise AssertionError("dependencies must not load after manifest drift")

    runtime = TransformersQwen3VLRuntime(
        model_dir,
        dependency_loader=forbidden_loader,
    )
    target = model_dir / asset_name
    original = target.read_bytes()
    original_stat = target.stat()
    tampered = bytearray(original)
    tampered[0] ^= 1
    target.write_bytes(tampered)
    os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(LocalVLMUnavailableError, match="manifest drift"):
        runtime._ensure_loaded()

    assert loader_called is False
    assert runtime.loaded is False
    assert runtime.full_manifest_verification_count == 1
    assert runtime.full_manifest_verification_ms >= 0.0


@pytest.mark.parametrize(
    "unexpected_name",
    ("special_tokens_map.json", "chat_template.jinja"),
)
def test_runtime_closed_world_rejects_unpinned_transformers_assets(
    unexpected_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, _ = _fake_pinned_model(tmp_path, monkeypatch)
    (model_dir / unexpected_name).write_text("{}", encoding="utf-8")

    with pytest.raises(LocalVLMUnavailableError, match="does not exactly match"):
        TransformersQwen3VLRuntime(model_dir)


def test_runtime_post_load_manifest_check_does_not_publish_tampered_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, _ = _fake_pinned_model(tmp_path, monkeypatch)
    target = model_dir / "config.json"
    original_stat = target.stat()

    class Processor:
        @classmethod
        def from_pretrained(cls, _path: str, **_kwargs):
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, _path: str, **_kwargs):
            content = bytearray(target.read_bytes())
            content[0] ^= 1
            target.write_bytes(content)
            os.utime(
                target,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            return cls()

        def to(self, _device: str):
            return self

        def eval(self) -> None:
            return None

    fake_transformers = SimpleNamespace(
        AutoProcessor=Processor,
        Qwen3VLForConditionalGeneration=Model,
    )
    runtime = TransformersQwen3VLRuntime(
        model_dir,
        dependency_loader=lambda: (fake_transformers, SimpleNamespace()),
    )

    with pytest.raises(LocalVLMUnavailableError, match="manifest drift"):
        runtime._ensure_loaded()

    assert runtime.loaded is False
    assert runtime.full_manifest_verification_count == 2


def test_default_manifest_pins_official_weight_and_is_packaged() -> None:
    manifest_path = Path(runtime_module.__file__).with_name("qwen3_vl_2b_manifest.json")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {item["name"]: item for item in document["files"]}

    assert document["revision"] == "89644892e4d85e24eaac8bacfd4f463576704203"
    assert files["model.safetensors"] == {
        "name": "model.safetensors",
        "size": 4255140312,
        "sha256": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
        "source": "huggingface",
    }
    pyproject = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["tool"]["setuptools"]["package-data"]["ai.rag"] == [
        "qwen3_vl_2b_manifest.json"
    ]
    assert (
        "transformers>=4.57,<6"
        in pyproject["project"]["optional-dependencies"]["multimodal"]
    )


def test_v11_to_v14_migration_creates_immutable_rag_runs(tmp_path: Path) -> None:
    database = Database(tmp_path / "norma.db")
    with sqlite3.connect(database.path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations(
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO schema_migrations(version) VALUES (11);
            """
        )

    database.initialize()

    with database.connect() as connection:
        table = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'rag_runs'"
        ).fetchone()
        triggers = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND tbl_name = 'rag_runs'"
            )
        }
    assert database.current_version() == 14
    assert table is not None
    assert "WITHOUT ROWID" in table["sql"].upper()
    assert triggers == {
        "rag_runs_no_update",
        "rag_runs_no_replace",
        "rag_runs_no_delete",
    }


def test_v13_to_v14_preserves_audit_rows_and_removes_rowid(tmp_path: Path) -> None:
    database = Database(tmp_path / "norma.db")
    database.initialize()
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO rag_runs(
                id, album_id, user_id, query_text,
                retrieval_provider_fingerprint,
                generation_provider_fingerprint,
                candidate_digest, query_digest, evidence_digest,
                evidence_json, result_json, request_json
            ) VALUES ('run-old', 'album', 'local', 'query', 'retrieval',
                      'generation', ?, ?, ?, '{}', '{}', '{}')
            """,
            ("a" * 64, "b" * 64, "c" * 64),
        )
        connection.execute("DROP TRIGGER rag_runs_no_update")
        connection.execute("DROP TRIGGER rag_runs_no_replace")
        connection.execute("DROP TRIGGER rag_runs_no_delete")
        connection.execute("DROP INDEX idx_rag_runs_context")
        connection.execute("CREATE TABLE rag_runs_v13 AS SELECT * FROM rag_runs")
        connection.execute("DROP TABLE rag_runs")
        connection.execute("ALTER TABLE rag_runs_v13 RENAME TO rag_runs")
        connection.execute("DELETE FROM schema_migrations WHERE version = 14")

    database.initialize()

    with database.connect() as connection:
        table_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'rag_runs'"
        ).fetchone()["sql"]
        row = connection.execute(
            "SELECT id, query_text, candidate_digest FROM rag_runs"
        ).fetchone()
        with pytest.raises(sqlite3.OperationalError, match="rowid"):
            connection.execute("SELECT rowid FROM rag_runs")

    assert database.current_version() == 14
    assert "WITHOUT ROWID" in table_sql.upper()
    assert tuple(row) == ("run-old", "query", "a" * 64)


def test_v12_to_v13_preserves_photos_but_invalidates_legacy_vector_binding(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "norma.db")
    database.initialize()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO albums(id, name, source_path) VALUES ('a', 'A', 'source')"
        )
        connection.execute(
            """
            INSERT INTO photos(
                id, album_id, absolute_path, file_size, source_mtime_ns,
                embedding_path, embedding_provider,
                embedding_source_size, embedding_source_mtime_ns,
                embedding_source_sha256
            ) VALUES ('p', 'a', 'photo.jpg', 10, 20, 'vector.npy',
                      'openclip-test', 10, 20, ?)
            """,
            ("a" * 64,),
        )
        connection.execute("ALTER TABLE photos DROP COLUMN embedding_source_sha256")
        connection.execute("DELETE FROM schema_migrations WHERE version >= 13")

    database.initialize()

    with database.connect() as connection:
        row = connection.execute(
            """SELECT id, embedding_path, embedding_provider,
                      embedding_source_size, embedding_source_mtime_ns,
                      embedding_source_sha256
               FROM photos WHERE id = 'p'"""
        ).fetchone()
    assert database.current_version() == 14
    assert tuple(row) == ("p", "vector.npy", "openclip-test", 10, 20, None)


def test_relative_transformers_model_path_is_rejected_before_loading(
    tmp_path: Path,
) -> None:
    with pytest.raises(LocalVLMUnavailableError, match="explicit local directory"):
        TransformersQwen3VLRuntime(Path("relative-model"))


def test_vlm_settings_use_local_default_and_enforce_token_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORMA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("NORMA_VLM_MODEL_PATH", raising=False)
    monkeypatch.delenv("NORMA_VLM_MAX_NEW_TOKENS", raising=False)

    settings = load_settings()

    assert settings.vlm_max_new_tokens == 256
    assert (
        settings.local_vlm_model_dir
        == (
            tmp_path
            / "data"
            / "models"
            / "qwen3-vl"
            / "Qwen3-VL-2B-Instruct-modelscope"
        ).resolve()
    )
    monkeypatch.setenv("NORMA_VLM_MAX_NEW_TOKENS", "63")
    with pytest.raises(ValueError, match="between 64 and 1024"):
        load_settings()
