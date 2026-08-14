from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps

from ai.index.embedding import (
    CONCEPT_TERMS,
    SEMANTIC_DIMENSIONS,
    EmbeddingProvider,
    EmbeddingProviderUnavailableError,
    _contains_term,
    normalize_embedding,
)


class OpenClipMultilingualProvider(EmbeddingProvider):
    """Lazy multilingual OpenCLIP inference with provider-versioned identity."""

    name = "openclip-xlm-roberta-base-vit-b-32-laion5b-v1"
    dimension = 512
    model_name = "xlm-roberta-base-ViT-B-32"
    pretrained = "laion5b_s13b_b90k"

    def __init__(self, *, cache_dir: Path, device: str, batch_size: int) -> None:
        self.cache_dir = cache_dir.resolve()
        self.requested_device = device
        self.batch_size = batch_size
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._device or "cpu"

    def embed_image(self, path: Path) -> np.ndarray:
        return self.embed_images([path])[0]

    def embed_images(self, paths: Sequence[Path]) -> list[np.ndarray]:
        if not paths:
            return []
        self._ensure_loaded()
        vectors: list[np.ndarray] = []
        with self._lock, self._torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                batch_paths = paths[start : start + self.batch_size]
                tensors = [
                    self._preprocess(_read_verified(path)) for path in batch_paths
                ]
                batch = self._torch.stack(tensors).to(self._device)
                encoded = self._model.encode_image(batch)
                array = encoded.float().cpu().numpy()
                vectors.extend(_validated_rows(array, self.dimension))
        return vectors

    def embed_text(self, text: str) -> np.ndarray:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("text query must not be empty")
        normalized = _bridge_chinese_query(normalized)
        self._ensure_loaded()
        with self._lock, self._torch.inference_mode():
            tokens = self._tokenizer([normalized]).to(self._device)
            encoded = self._model.encode_text(tokens)
            array = encoded.float().cpu().numpy()
        return _validated_rows(array, self.dimension)[0]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            model_cache = self.cache_dir / "openclip"
            _configure_huggingface_cache(model_cache)
            try:
                import open_clip
                import torch
            except ImportError as error:
                raise EmbeddingProviderUnavailableError(
                    "OpenCLIP dependencies are missing; install `.[multimodal]`"
                ) from error
            device = self._resolve_device(torch)
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device=device,
                    cache_dir=str(model_cache),
                )
                tokenizer = open_clip.get_tokenizer(
                    self.model_name,
                    cache_dir=str(model_cache),
                )
            except Exception as error:
                raise EmbeddingProviderUnavailableError(
                    "OpenCLIP model could not be loaded; verify network/model cache and "
                    "that Torch companion packages match the installed Torch build: "
                    f"{error}"
                ) from error
            model.eval()
            self._torch = torch
            self._device = device
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer

    def _resolve_device(self, torch: Any) -> str:
        requested = self.requested_device
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested not in {"cpu", "cuda"}:
            raise ValueError("embedding device must be auto, cpu, or cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            raise EmbeddingProviderUnavailableError(
                "CUDA was requested but is not available"
            )
        return requested


def _configure_huggingface_cache(model_cache: Path) -> None:
    """Keep OpenCLIP's implicit Transformers lookups in Norma's model cache."""

    resolved = model_cache.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(resolved)
    os.environ["HF_HUB_CACHE"] = str(resolved)


def _read_verified(path: Path) -> Image.Image:
    before = path.stat()
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed during embedding: {path}")
    return image


def _validated_rows(array: np.ndarray, dimension: int) -> list[np.ndarray]:
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(
            f"OpenCLIP returned shape {array.shape}, expected (*, {dimension})"
        )
    return [
        normalize_embedding(row, dimension, label="OpenCLIP embedding") for row in array
    ]


def _bridge_chinese_query(text: str) -> str:
    """Map recognized Chinese concepts to transparent English CLIP prompts."""

    if not any("\u4e00" <= character <= "\u9fff" for character in text):
        return text
    normalized = text.casefold()
    concepts = [
        concept.replace("_", " ")
        for concept in SEMANTIC_DIMENSIONS
        if any(
            _contains_term(normalized, term.casefold())
            for term in CONCEPT_TERMS[concept]
            if any("\u4e00" <= character <= "\u9fff" for character in term)
        )
    ]
    if not concepts:
        return text
    return "a photo of " + ", ".join(dict.fromkeys(concepts))
