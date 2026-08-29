from __future__ import annotations

import copy
import hashlib
import os
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai import app as app_module
from ai.index import openclip_provider as openclip_provider_module
from ai.index.embedding import (
    OPENCLIP_RAW_PROVIDER_NAME,
    OPENCLIP_RAW_V2_PROVIDER_NAME,
    embedding_cache_is_current,
    openclip_provider_name,
)
from ai.index.openclip_identity import (
    OpenClipIdentityError,
    canonical_openclip_provider_name,
    load_pinned_openclip_manifest,
    openclip_runtime_abi_versions,
    verify_pinned_openclip_cache,
)
from ai.index.openclip_provider import OpenClipMultilingualProvider
from ai.index.openclip_provider import (
    _configure_huggingface_cache,
    _validated_local_text_config,
)
from ai.provider_runtime import EmbeddingWarmupManager


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _small_manifest() -> tuple[dict[str, object], dict[str, bytes]]:
    original, _ = load_pinned_openclip_manifest()
    manifest = copy.deepcopy(original)
    contents = {
        "open_clip_pytorch_model.bin": b"pinned-openclip-weight",
        "config.json": b"pinned-config",
        "sentencepiece.bpe.model": b"pinned-sentencepiece",
        "tokenizer.json": b"pinned-tokenizer",
        "tokenizer_config.json": b"pinned-tokenizer-config",
    }
    weight = manifest["model"]["weight"]  # type: ignore[index]
    weight["size"] = len(contents[weight["name"]])
    weight["sha256"] = _sha256(contents[weight["name"]])
    for item in manifest["tokenizer"]["files"]:  # type: ignore[index]
        content = contents[item["name"]]
        item["size"] = len(content)
        item["sha256"] = _sha256(content)
    return manifest, contents


def _write_snapshot(
    cache: Path, manifest: dict[str, object], contents: dict[str, bytes]
) -> None:
    for section, files in (
        (manifest["model"], [manifest["model"]["weight"]]),  # type: ignore[index]
        (manifest["tokenizer"], manifest["tokenizer"]["files"]),  # type: ignore[index]
    ):
        repository = section["repository"]
        revision = section["revision"]
        snapshot = (
            cache / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
        )
        snapshot.mkdir(parents=True, exist_ok=True)
        for item in files:
            (snapshot / item["name"]).write_bytes(contents[item["name"]])


def test_provider_fingerprint_binds_manifest_runtime_and_query_contract() -> None:
    manifest_a, _ = _small_manifest()
    manifest_b = copy.deepcopy(manifest_a)
    manifest_b["model"]["weight"]["sha256"] = "f" * 64  # type: ignore[index]
    runtime_a = openclip_runtime_abi_versions(manifest_a)
    runtime_b = dict(runtime_a)
    runtime_b["ftfy"] = runtime_a["ftfy"] + "+different"
    query = {"normalization": "test-contract-v1"}

    name_a = canonical_openclip_provider_name(
        "raw-multilingual",
        query_contract=query,
        document=manifest_a,
        runtime_versions=runtime_a,
    )
    name_b = canonical_openclip_provider_name(
        "raw-multilingual",
        query_contract=query,
        document=manifest_b,
        runtime_versions=runtime_a,
    )
    name_runtime_b = canonical_openclip_provider_name(
        "raw-multilingual",
        query_contract=query,
        document=manifest_a,
        runtime_versions=runtime_b,
    )
    name_query_b = canonical_openclip_provider_name(
        "raw-multilingual",
        query_contract={"normalization": "test-contract-v2"},
        document=manifest_a,
        runtime_versions=runtime_a,
    )
    name_threading_b = canonical_openclip_provider_name(
        "raw-multilingual",
        query_contract=query,
        document=manifest_a,
        runtime_versions=runtime_a,
        numeric_threading="test-numeric-threading-v2",
    )

    assert len({name_a, name_b, name_runtime_b, name_query_b, name_threading_b}) == 5
    assert "-v3-m" in name_a
    assert name_a.startswith("openclip-")
    assert manifest_a["model"]["name"] == manifest_b["model"]["name"]  # type: ignore[index]
    assert (
        manifest_a["model"]["pretrained_tag"]
        == manifest_b["model"][  # type: ignore[index]
            "pretrained_tag"
        ]
    )
    row_from_a = {"embedding_path": "vector.npy", "embedding_provider": name_a}
    assert not embedding_cache_is_current(row_from_a, name_b)


def test_old_raw_v2_cache_is_invalidated_by_canonical_v3_identity() -> None:
    row = {
        "embedding_path": "old-vector.npy",
        "embedding_provider": OPENCLIP_RAW_V2_PROVIDER_NAME,
    }
    assert OPENCLIP_RAW_PROVIDER_NAME != OPENCLIP_RAW_V2_PROVIDER_NAME
    assert not embedding_cache_is_current(row, OPENCLIP_RAW_PROVIDER_NAME)


def test_pinned_snapshot_rejects_same_size_content_substitution(tmp_path: Path) -> None:
    manifest, contents = _small_manifest()
    cache = tmp_path / "openclip"
    _write_snapshot(cache, manifest, contents)
    snapshot = verify_pinned_openclip_cache(cache, document=manifest)
    assert snapshot.weight_path.is_file()
    assert snapshot.tokenizer_dir.is_dir()

    weight = snapshot.weight_path
    original = weight.read_bytes()
    weight.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(OpenClipIdentityError, match="manifest verification"):
        verify_pinned_openclip_cache(cache, document=manifest)


@pytest.mark.parametrize("extra_name", ["special_tokens_map.json", "added_tokens.json"])
def test_pinned_snapshot_rejects_unpinned_tokenizer_assets(
    tmp_path: Path, extra_name: str
) -> None:
    manifest, contents = _small_manifest()
    cache = tmp_path / "openclip"
    _write_snapshot(cache, manifest, contents)
    tokenizer = manifest["tokenizer"]  # type: ignore[assignment]
    snapshot = (
        cache
        / f"models--{tokenizer['repository'].replace('/', '--')}"
        / "snapshots"
        / tokenizer["revision"]
    )
    (snapshot / extra_name).write_text("{}", encoding="utf-8")

    with pytest.raises(OpenClipIdentityError, match="unpinned asset"):
        verify_pinned_openclip_cache(cache, document=manifest)


def test_hf_text_encoder_config_is_local_only_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    manifest, contents = _small_manifest()
    cache = tmp_path / "openclip"
    _write_snapshot(cache, manifest, contents)
    snapshot = verify_pinned_openclip_cache(cache, document=manifest)
    calls: list[tuple[str, dict[str, object]]] = []
    contract = manifest["tokenizer"]["model_config_contract"]  # type: ignore[index]

    def _local_config(source: str, **kwargs: object) -> SimpleNamespace:
        calls.append((source, kwargs))
        return SimpleNamespace(**contract)

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(_local_config),
    )
    _configure_huggingface_cache(cache)
    text_config = _validated_local_text_config(
        transformers, snapshot.tokenizer_dir, manifest
    )

    assert calls == [
        (
            str(snapshot.tokenizer_dir.resolve()),
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]
    assert Path(text_config["hf_model_name"]) == snapshot.tokenizer_dir.resolve()
    assert Path(text_config["hf_tokenizer_name"]) == snapshot.tokenizer_dir.resolve()
    assert text_config["hf_model_pretrained"] is False
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_instance_identity_separates_cpu_and_cuda_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu_name = openclip_provider_name("raw-multilingual", "cpu")
    cuda_name = openclip_provider_name("raw-multilingual", "cuda")
    assert cpu_name != cuda_name
    assert "-cpu-v3-" in cpu_name
    assert "-cuda-v3-" in cuda_name

    monkeypatch.setattr(
        openclip_provider_module,
        "resolve_openclip_backend",
        lambda _requested: "cuda",
    )
    provider = OpenClipMultilingualProvider(
        cache_dir=tmp_path,
        device="auto",
        batch_size=1,
    )
    assert provider.name == cuda_name


def test_openclip_manifest_is_declared_as_wheel_package_data() -> None:
    project_root = Path(__file__).resolve().parents[2]
    with (project_root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    assert pyproject["tool"]["setuptools"]["package-data"]["ai.index"] == [
        "openclip_xlm_roberta_manifest.json"
    ]


def test_openclip_manifest_failure_is_visible_in_health_and_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = OpenClipMultilingualProvider(
        cache_dir=tmp_path,
        device="cpu",
        batch_size=1,
    )
    manager = EmbeddingWarmupManager(lambda: provider)
    submitted = manager.submit()
    assert submitted.provider == OPENCLIP_RAW_PROVIDER_NAME
    assert submitted.warmup_state == "loading"
    for _ in range(200):
        status = manager.status()
        if status.warmup_state == "failed":
            break
        time.sleep(0.01)

    assert status.warmup_state == "failed"
    assert status.loaded is False
    assert "pinned snapshot verification failed" in str(status.error)

    class _Database:
        @staticmethod
        def current_version() -> int:
            return 13

    monkeypatch.setattr(app_module, "database", _Database())
    monkeypatch.setattr(app_module, "embedding_provider", lambda: provider)
    monkeypatch.setattr(app_module, "embedding_warmup", manager)
    health = app_module.health()
    readiness = app_module.get_embedding_provider_status()
    assert health.embedding_provider == OPENCLIP_RAW_PROVIDER_NAME
    assert readiness.provider == health.embedding_provider
    assert readiness.warmup_state == "failed"
    assert readiness.loaded is False
