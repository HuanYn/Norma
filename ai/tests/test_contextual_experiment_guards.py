from __future__ import annotations

from pathlib import Path

import pytest

from figures.benchmark_contextual_preference_simulation import (
    OPENCLIP_CACHE_MARKERS,
    _validate_run_paths,
)


def test_contextual_benchmark_rejects_reuse_output_self_reference(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.json"
    result.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must be different files"):
        _validate_run_paths(
            cache_dir=tmp_path / "models",
            output=result,
            reuse_embeddings_from=result,
        )


def test_contextual_benchmark_requires_parent_model_cache_root(
    tmp_path: Path,
) -> None:
    provider_cache = tmp_path / "models" / "openclip"
    provider_cache.mkdir(parents=True)
    for marker in OPENCLIP_CACHE_MARKERS:
        (provider_cache / marker).mkdir()

    _validate_run_paths(
        cache_dir=tmp_path / "models",
        output=tmp_path / "result.json",
        reuse_embeddings_from=None,
    )

    with pytest.raises(FileNotFoundError, match="parent Norma model-cache root"):
        _validate_run_paths(
            cache_dir=provider_cache,
            output=tmp_path / "wrong-result.json",
            reuse_embeddings_from=None,
        )
