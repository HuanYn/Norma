from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ai.config import load_settings
from ai.index.embedding import (
    OPENCLIP_LEGACY_BRIDGE_PROVIDER_NAME,
    OPENCLIP_RAW_PROVIDER_NAME,
    EmbeddingProvider,
    create_embedding_provider,
    embedding_provider_capabilities,
)
from ai.index.openclip_provider import (
    _bridge_chinese_query,
    _configure_huggingface_cache,
    _validated_rows,
)


class TinyProvider(EmbeddingProvider):
    name = "tiny"
    dimension = 2

    def embed_image(self, path: Path) -> np.ndarray:
        return np.asarray([float(len(path.name)), 1.0], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_default_batch_api_preserves_provider_compatibility() -> None:
    provider = TinyProvider()
    vectors = provider.embed_images([Path("a.jpg"), Path("long-name.jpg")])
    assert len(vectors) == 2
    assert vectors[0].shape == (2,)
    assert vectors[1][0] > vectors[0][0]


def test_openclip_factory_is_lazy_and_versioned(tmp_path: Path) -> None:
    provider = create_embedding_provider(
        "openclip-multilingual",
        cache_dir=tmp_path,
        device="cpu",
        batch_size=4,
    )
    assert provider.name == OPENCLIP_RAW_PROVIDER_NAME
    assert provider.dimension == 512
    assert provider.batch_size == 4
    assert provider.query_mode == "raw-multilingual"
    assert provider._model is None


def test_openclip_factory_aliases_select_distinct_query_versions(
    tmp_path: Path,
) -> None:
    for alias in ("openclip", "openclip-multilingual", OPENCLIP_RAW_PROVIDER_NAME):
        provider = create_embedding_provider(alias, cache_dir=tmp_path, device="cpu")
        assert provider.name == OPENCLIP_RAW_PROVIDER_NAME
        assert provider._prepare_text_query("城市 夜景") == "城市 夜景"

    for alias in (
        "openclip-legacy-bridge",
        "openclip-multilingual-legacy",
        "openclip-xlm-roberta-base-vit-b-32-laion5b-v1",
    ):
        provider = create_embedding_provider(alias, cache_dir=tmp_path, device="cpu")
        assert provider.name == OPENCLIP_LEGACY_BRIDGE_PROVIDER_NAME
        assert provider._prepare_text_query("城市夜景摄影").startswith("a photo of ")


def test_default_provider_is_model_backed_multilingual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NORMA_EMBEDDING_PROVIDER", raising=False)
    assert load_settings().embedding_provider == "openclip-multilingual"


def test_openclip_output_validation_normalizes_and_rejects_bad_shapes() -> None:
    rows = _validated_rows(np.ones((2, 512), dtype=np.float32), 512)
    assert len(rows) == 2
    assert np.linalg.norm(rows[0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="expected"):
        _validated_rows(np.ones((2, 16), dtype=np.float32), 512)
    invalid = np.ones((1, 512), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _validated_rows(invalid, 512)
    with pytest.raises(ValueError, match="zero embedding"):
        _validated_rows(np.zeros((1, 512), dtype=np.float32), 512)


def test_provider_capabilities_distinguish_default_baseline_and_legacy() -> None:
    capabilities = embedding_provider_capabilities("openclip-multilingual")
    by_id = {item["id"]: item for item in capabilities}

    baseline = by_id["lightweight"]
    assert baseline["model_backed"] is False
    assert baseline["default"] is False
    assert baseline["baseline"] is True
    assert baseline["legacy"] is False

    default = by_id["openclip-multilingual"]
    assert default["name"] == OPENCLIP_RAW_PROVIDER_NAME
    assert default["model_backed"] is True
    assert default["default"] is True
    assert default["baseline"] is False
    assert default["legacy"] is False
    assert default["query_mode"] == "raw-multilingual"
    assert default["active"] is True
    assert default["install_extra"] == "multimodal"

    legacy = by_id["openclip-legacy-bridge"]
    assert legacy["name"] == OPENCLIP_LEGACY_BRIDGE_PROVIDER_NAME
    assert legacy["model_backed"] is True
    assert legacy["default"] is False
    assert legacy["baseline"] is False
    assert legacy["legacy"] is True
    assert legacy["query_mode"] == "legacy-chinese-keyword-bridge"
    assert legacy["active"] is False


def test_huggingface_cache_is_scoped_to_norma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    cache = tmp_path / "models" / "openclip"
    _configure_huggingface_cache(cache)
    assert cache.is_dir()
    assert Path(os.environ["HF_HOME"]) == cache.resolve()
    assert Path(os.environ["HF_HUB_CACHE"]) == cache.resolve()


def test_chinese_query_bridge_is_explicit_and_bounded() -> None:
    bridged = _bridge_chinese_query("城市夜景摄影")
    assert bridged.startswith("a photo of ")
    assert "night" in bridged
    assert "architecture" in bridged
    assert _bridge_chinese_query("未收录的新概念") == "未收录的新概念"
    assert _bridge_chinese_query("night architecture") == "night architecture"
