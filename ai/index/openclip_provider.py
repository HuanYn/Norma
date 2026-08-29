from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps

from ai.index.embedding import (
    CONCEPT_TERMS,
    OPENCLIP_LEGACY_BRIDGE_PROVIDER_NAME,
    OPENCLIP_RAW_PROVIDER_NAME,
    SEMANTIC_DIMENSIONS,
    EmbeddingProvider,
    EmbeddingProviderUnavailableError,
    _contains_term,
    normalize_embedding,
    openclip_provider_name,
)
from ai.index.openclip_identity import (
    OPENCLIP_MODEL_NAME,
    OPENCLIP_PRETRAINED_TAG,
    OpenClipIdentityError,
    load_pinned_openclip_manifest,
    openclip_runtime_abi_versions,
    resolve_openclip_backend,
    verify_pinned_openclip_cache,
)
from ai.numeric_runtime import (
    NumericRuntimeConflictError,
    ensure_torch_numpy_runtime_compatible,
)


class OpenClipMultilingualProvider(EmbeddingProvider):
    """Lazy multilingual OpenCLIP using the model's native text encoder."""

    name = OPENCLIP_RAW_PROVIDER_NAME
    dimension = 512
    model_name = OPENCLIP_MODEL_NAME
    pretrained = OPENCLIP_PRETRAINED_TAG
    query_mode = "raw-multilingual"

    def __init__(self, *, cache_dir: Path, device: str, batch_size: int) -> None:
        self.cache_dir = cache_dir.resolve()
        self.requested_device = device
        try:
            self._resolved_backend = resolve_openclip_backend(device)
        except NumericRuntimeConflictError as error:
            raise EmbeddingProviderUnavailableError(str(error)) from error
        self.name = openclip_provider_name(self.query_mode, self._resolved_backend)
        self.batch_size = batch_size
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._manifest, self.manifest_sha256 = load_pinned_openclip_manifest()
        self._runtime_abi = openclip_runtime_abi_versions(self._manifest)

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._device or "cpu"

    @property
    def model_backed(self) -> bool:
        return True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def runtime_device(self) -> str | None:
        return self._device

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
        prepared = self._prepare_text_query(text)
        self._ensure_loaded()
        with self._lock, self._torch.inference_mode():
            tokens = self._tokenizer([prepared]).to(self._device)
            encoded = self._model.encode_text(tokens)
            array = encoded.float().cpu().numpy()
        return _validated_rows(array, self.dimension)[0]

    def _prepare_text_query(self, text: str) -> str:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("text query must not be empty")
        return normalized

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                ensure_torch_numpy_runtime_compatible()
            except NumericRuntimeConflictError as error:
                raise EmbeddingProviderUnavailableError(str(error)) from error
            model_cache = self.cache_dir / "openclip"
            _configure_huggingface_cache(model_cache)
            try:
                snapshot = verify_pinned_openclip_cache(model_cache)
            except OpenClipIdentityError as error:
                raise EmbeddingProviderUnavailableError(
                    "OpenCLIP pinned snapshot verification failed; no mutable Hub "
                    f"fallback was attempted: {error}"
                ) from error
            try:
                import open_clip
                import torch
                import transformers
                from open_clip.tokenizer import HFTokenizer
            except ImportError as error:
                raise EmbeddingProviderUnavailableError(
                    "OpenCLIP dependencies are missing; install `.[multimodal]`"
                ) from error
            if openclip_runtime_abi_versions(self._manifest) != self._runtime_abi:
                raise EmbeddingProviderUnavailableError(
                    "OpenCLIP runtime ABI changed after provider fingerprinting"
                )
            device = self._resolve_device(torch)
            try:
                _validate_openclip_metadata(open_clip, self._manifest)
                preprocess_contract = self._manifest["preprocess"]
                tokenizer_contract = self._manifest["tokenizer"]
                local_text_config = _validated_local_text_config(
                    transformers,
                    snapshot.tokenizer_dir,
                    self._manifest,
                )
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=str(snapshot.weight_path),
                    pretrained_text=False,
                    precision=self._manifest["inference"]["precision"],
                    device=device,
                    cache_dir=str(model_cache),
                    image_mean=tuple(preprocess_contract["mean"]),
                    image_std=tuple(preprocess_contract["std"]),
                    image_interpolation=preprocess_contract["interpolation"],
                    image_resize_mode=preprocess_contract["resize_mode"],
                    weights_only=self._manifest["inference"]["weights_only"],
                    text_cfg=local_text_config,
                )
                tokenizer = HFTokenizer(
                    str(snapshot.tokenizer_dir),
                    context_length=tokenizer_contract["context_length"],
                    clean=tokenizer_contract["clean"],
                    strip_sep_token=tokenizer_contract["strip_sep_token"],
                    tokenizer_mode=tokenizer_contract["tokenizer_mode"],
                    local_files_only=tokenizer_contract["local_files_only"],
                )
                verify_pinned_openclip_cache(model_cache)
            except Exception as error:
                raise EmbeddingProviderUnavailableError(
                    "OpenCLIP model could not be loaded from the pinned local snapshot; "
                    "no mutable Hub fallback was attempted: "
                    f"{type(error).__name__}: {error}"
                ) from error
            model.eval()
            self._torch = torch
            self._device = device
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer

    def _resolve_device(self, torch: Any) -> str:
        requested = self.requested_device
        if requested not in {"cpu", "cuda"}:
            current = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested == "cuda" and not torch.cuda.is_available():
            raise EmbeddingProviderUnavailableError(
                "CUDA was requested but is not available"
            )
        else:
            current = requested
        if current != self._resolved_backend:
            raise EmbeddingProviderUnavailableError(
                "OpenCLIP backend changed after provider fingerprinting"
            )
        return current


class OpenClipLegacyBridgeProvider(OpenClipMultilingualProvider):
    """Ablation-only provider reproducing the bounded Chinese keyword bridge."""

    name = OPENCLIP_LEGACY_BRIDGE_PROVIDER_NAME
    query_mode = "legacy-chinese-keyword-bridge"

    def _prepare_text_query(self, text: str) -> str:
        return _bridge_chinese_query(super()._prepare_text_query(text))


def _configure_huggingface_cache(model_cache: Path) -> None:
    """Scope Hugging Face state and prohibit mutable network fallbacks."""

    resolved = model_cache.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(resolved)
    os.environ["HF_HUB_CACHE"] = str(resolved)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _validated_local_text_config(
    transformers: Any,
    tokenizer_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Bind both HFTextEncoder config and tokenizer to one verified local snapshot."""

    resolved = tokenizer_dir.resolve(strict=True)
    contract = manifest["tokenizer"]
    config = transformers.AutoConfig.from_pretrained(
        str(resolved),
        local_files_only=True,
        trust_remote_code=False,
    )
    actual = {
        key: getattr(config, key, None) for key in contract["model_config_contract"]
    }
    if actual != contract["model_config_contract"]:
        raise OpenClipIdentityError(
            "local XLM-R model config differs from the pinned manifest"
        )
    text_config = dict(manifest["model"]["config"]["text_cfg"])
    text_config.update(
        {
            "context_length": contract["context_length"],
            "hf_model_name": str(resolved),
            "hf_model_pretrained": False,
            "hf_tokenizer_name": str(resolved),
        }
    )
    return text_config


def _validate_openclip_metadata(open_clip: Any, manifest: dict[str, Any]) -> None:
    """Ensure the installed OpenCLIP registry still describes the pinned contract."""

    actual_model = open_clip.get_model_config(manifest["model"]["name"])
    if actual_model != manifest["model"]["config"]:
        raise OpenClipIdentityError(
            "installed OpenCLIP model metadata differs from the pinned manifest"
        )
    actual_pretrained = open_clip.get_pretrained_cfg(
        manifest["model"]["name"], manifest["model"]["pretrained_tag"]
    )
    preprocess = manifest["preprocess"]
    expected = {
        "repository": manifest["model"]["repository"],
        "resize_mode": preprocess["resize_mode"],
        "interpolation": preprocess["interpolation"],
        "mean": preprocess["mean"],
        "std": preprocess["std"],
    }
    actual = {
        "repository": str(actual_pretrained.get("hf_hub", "")).rstrip("/"),
        "resize_mode": actual_pretrained.get("resize_mode"),
        "interpolation": actual_pretrained.get("interpolation"),
        "mean": list(actual_pretrained.get("mean", ())),
        "std": list(actual_pretrained.get("std", ())),
    }
    if actual != expected:
        raise OpenClipIdentityError(
            "installed OpenCLIP pretrained metadata differs from the pinned manifest"
        )


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
