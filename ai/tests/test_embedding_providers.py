from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ai.index.embedding import (
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
    assert provider.name == "openclip-xlm-roberta-base-vit-b-32-laion5b-v1"
    assert provider.dimension == 512
    assert provider.batch_size == 4
    assert provider._model is None


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


def test_provider_capabilities_keep_optional_model_explicit() -> None:
    capabilities = embedding_provider_capabilities("openclip-multilingual")
    assert capabilities[0]["model_backed"] is False
    assert capabilities[1]["model_backed"] is True
    assert capabilities[1]["active"] is True
    assert capabilities[1]["install_extra"] == "multimodal"


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
