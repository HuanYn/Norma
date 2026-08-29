from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Mapping

from ai.numeric_runtime import (
    ensure_torch_numpy_runtime_compatible,
    numeric_threading_contract,
)


OPENCLIP_MODEL_NAME = "xlm-roberta-base-ViT-B-32"
OPENCLIP_PRETRAINED_TAG = "laion5b_s13b_b90k"
OPENCLIP_MODEL_REPOSITORY = "laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k"
OPENCLIP_MODEL_REVISION = "506d40eb551f4801a1c27fc20a31c7b8f590deda"
OPENCLIP_TOKENIZER_REPOSITORY = "xlm-roberta-base"
OPENCLIP_TOKENIZER_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
_MANIFEST_PATH = Path(__file__).with_name("openclip_xlm_roberta_manifest.json")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_IGNORED_SNAPSHOT_DOCUMENTS = frozenset({".gitattributes", "README.md"})
_QUERY_MODE_SLUGS = {
    "raw-multilingual": "raw",
    "legacy-chinese-keyword-bridge": "zh-bridge",
}


class OpenClipIdentityError(RuntimeError):
    """The pinned OpenCLIP identity or its local snapshot is not trustworthy."""


@dataclass(frozen=True, slots=True)
class PinnedOpenClipSnapshot:
    weight_path: Path
    tokenizer_dir: Path


def load_pinned_openclip_manifest() -> tuple[dict[str, Any], str]:
    """Load and strictly validate Norma's version-controlled OpenCLIP manifest."""

    try:
        document = json.loads(_MANIFEST_PATH.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise OpenClipIdentityError(
            "the version-controlled OpenCLIP manifest is unavailable"
        ) from error
    return parse_pinned_openclip_manifest(document)


def parse_pinned_openclip_manifest(
    document: object,
) -> tuple[dict[str, Any], str]:
    """Validate a manifest document and return its canonical content digest."""

    if not isinstance(document, dict) or set(document) != {
        "schema",
        "model",
        "preprocess",
        "tokenizer",
        "inference",
        "runtime_identity",
    }:
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")
    if document["schema"] != "norma-openclip-pinned-manifest-v1":
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")
    _validate_model(document["model"])
    _validate_preprocess(document["preprocess"])
    _validate_tokenizer(document["tokenizer"])
    _validate_inference(document["inference"])
    _validate_runtime_identity(document["runtime_identity"])
    canonical = _canonical_json(document)
    return document, hashlib.sha256(canonical).hexdigest()


def canonical_openclip_provider_name(
    query_mode: str,
    *,
    query_contract: Mapping[str, object],
    backend: str = "cpu",
    document: object | None = None,
    runtime_versions: Mapping[str, str] | None = None,
    numeric_threading: str | None = None,
) -> str:
    """Build identity from the pin, query contract, and direct Python runtime ABI."""

    if query_mode not in _QUERY_MODE_SLUGS:
        raise OpenClipIdentityError("unsupported OpenCLIP query contract")
    if not query_contract or not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in query_contract.items()
    ):
        raise OpenClipIdentityError("invalid OpenCLIP query contract")
    if backend not in {"cpu", "cuda", "unavailable"}:
        raise OpenClipIdentityError("invalid OpenCLIP inference backend")
    manifest, manifest_sha256 = (
        load_pinned_openclip_manifest()
        if document is None
        else parse_pinned_openclip_manifest(document)
    )
    expected_runtime = openclip_runtime_abi_versions(manifest)
    runtime = (
        dict(runtime_versions) if runtime_versions is not None else expected_runtime
    )
    if set(runtime) != set(expected_runtime) or not all(
        isinstance(value, str) and value for value in runtime.values()
    ):
        raise OpenClipIdentityError("invalid OpenCLIP runtime identity")
    threading_contract = numeric_threading or numeric_threading_contract()
    if not isinstance(threading_contract, str) or not threading_contract:
        raise OpenClipIdentityError("invalid OpenCLIP numeric threading identity")
    identity = {
        "manifest_sha256": manifest_sha256,
        "backend": backend,
        "numeric_threading": threading_contract,
        "query_mode": query_mode,
        "query_contract": dict(sorted(query_contract.items())),
        "runtime": dict(sorted(runtime.items())),
    }
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    slug = _QUERY_MODE_SLUGS[query_mode]
    return (
        "openclip-xlm-roberta-base-vit-b-32-laion5b-"
        f"{slug}-{backend}-v3-m{manifest_sha256[:16]}-i{identity_sha256[:24]}"
    )


def openclip_runtime_abi_versions(
    document: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Resolve curated package versions directly affecting model/token transforms."""

    manifest = document
    if manifest is None:
        manifest, _ = load_pinned_openclip_manifest()
    runtime_identity = manifest["runtime_identity"]
    versions = {
        distribution: _distribution_version(distribution)
        for distribution in runtime_identity["distributions"]
    }
    versions["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return versions


def resolve_openclip_backend(requested_device: str) -> str:
    """Resolve auto/cuda without loading OpenCLIP weights or model architecture."""

    ensure_torch_numpy_runtime_compatible()
    requested = requested_device.strip().casefold()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("embedding device must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "unavailable"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "unavailable"
    return "cuda" if torch.cuda.is_available() else "cpu"


def verify_pinned_openclip_cache(
    model_cache: Path,
    *,
    document: object | None = None,
) -> PinnedOpenClipSnapshot:
    """Hash every pinned model/tokenizer asset in a fixed local Hub snapshot."""

    manifest, _ = (
        load_pinned_openclip_manifest()
        if document is None
        else parse_pinned_openclip_manifest(document)
    )
    cache_root = model_cache.resolve()
    model = manifest["model"]
    tokenizer = manifest["tokenizer"]
    model_runtime_names = {model["weight"]["name"]}
    tokenizer_runtime_names = {item["name"] for item in tokenizer["files"]}
    model_root, model_snapshot = _snapshot_directory(
        cache_root,
        model["repository"],
        model["revision"],
        model_runtime_names,
    )
    tokenizer_root, tokenizer_snapshot = _snapshot_directory(
        cache_root,
        tokenizer["repository"],
        tokenizer["revision"],
        tokenizer_runtime_names,
    )
    weight_path = _verified_snapshot_file(model_root, model_snapshot, model["weight"])
    for expected in tokenizer["files"]:
        _verified_snapshot_file(tokenizer_root, tokenizer_snapshot, expected)
    return PinnedOpenClipSnapshot(
        weight_path=weight_path,
        tokenizer_dir=tokenizer_snapshot,
    )


def _snapshot_directory(
    cache_root: Path,
    repository: str,
    revision: str,
    runtime_names: set[str],
) -> tuple[Path, Path]:
    repository_root = cache_root / f"models--{repository.replace('/', '--')}"
    snapshot = repository_root / "snapshots" / revision
    try:
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
        resolved_root = repository_root.resolve(strict=True)
        resolved_snapshot = snapshot.resolve(strict=True)
    except OSError as error:
        raise OpenClipIdentityError(
            "the pinned OpenCLIP snapshot is missing from the local model cache"
        ) from error
    if not resolved_snapshot.is_relative_to(resolved_root):
        raise OpenClipIdentityError("the pinned OpenCLIP snapshot escaped its cache")
    try:
        entries = tuple(resolved_snapshot.iterdir())
    except OSError as error:
        raise OpenClipIdentityError(
            "the pinned OpenCLIP snapshot cannot be enumerated"
        ) from error
    actual_runtime_names: set[str] = set()
    for entry in entries:
        if entry.name in _IGNORED_SNAPSHOT_DOCUMENTS:
            try:
                resolved_entry = entry.resolve(strict=True)
            except OSError as error:
                raise OpenClipIdentityError(
                    "the pinned OpenCLIP documentation entry is invalid"
                ) from error
            if not resolved_entry.is_file() or not resolved_entry.is_relative_to(
                resolved_root
            ):
                raise OpenClipIdentityError(
                    "the pinned OpenCLIP documentation entry escaped its cache"
                )
            continue
        if entry.name not in runtime_names or not entry.is_file():
            raise OpenClipIdentityError(
                f"the pinned OpenCLIP snapshot contains unpinned asset {entry.name}"
            )
        actual_runtime_names.add(entry.name)
    if actual_runtime_names != runtime_names:
        raise OpenClipIdentityError("the pinned OpenCLIP snapshot is incomplete")
    return resolved_root, resolved_snapshot


def _verified_snapshot_file(
    repository_root: Path,
    snapshot: Path,
    expected: Mapping[str, object],
) -> Path:
    path = snapshot / str(expected["name"])
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise OpenClipIdentityError(
            "the pinned OpenCLIP snapshot is incomplete"
        ) from error
    if not resolved.is_file() or not resolved.is_relative_to(repository_root):
        raise OpenClipIdentityError("the pinned OpenCLIP asset escaped its cache")
    actual_size, actual_sha256 = _stable_file_hash(path)
    if actual_size != expected["size"] or actual_sha256 != expected["sha256"]:
        raise OpenClipIdentityError(
            "the local OpenCLIP snapshot failed pinned manifest verification"
        )
    return path


def _stable_file_hash(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before_path = path.stat()
            before_fd = os.fstat(handle.fileno())
            if before_fd.st_size <= 0:
                raise OpenClipIdentityError("the pinned OpenCLIP asset is empty")
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            after_fd = os.fstat(handle.fileno())
            after_path = path.stat()
    except OpenClipIdentityError:
        raise
    except OSError as error:
        raise OpenClipIdentityError(
            "the pinned OpenCLIP asset could not be fingerprinted"
        ) from error
    snapshots = {
        _stat_identity(before_path),
        _stat_identity(before_fd),
        _stat_identity(after_fd),
        _stat_identity(after_path),
    }
    if len(snapshots) != 1:
        raise OpenClipIdentityError(
            "the pinned OpenCLIP asset changed while being fingerprinted"
        )
    return after_fd.st_size, digest.hexdigest()


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _validate_model(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "name",
        "pretrained_tag",
        "repository",
        "revision",
        "weight",
        "config",
    }:
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")
    if (
        value["name"] != OPENCLIP_MODEL_NAME
        or value["pretrained_tag"] != OPENCLIP_PRETRAINED_TAG
        or value["repository"] != OPENCLIP_MODEL_REPOSITORY
        or value["revision"] != OPENCLIP_MODEL_REVISION
        or _REVISION_RE.fullmatch(value["revision"]) is None
    ):
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")
    _validate_file(value["weight"], expected_name="open_clip_pytorch_model.bin")
    expected_config = {
        "embed_dim": 512,
        "vision_cfg": {
            "image_size": 224,
            "layers": 12,
            "patch_size": 32,
            "width": 768,
        },
        "text_cfg": {
            "hf_model_name": "xlm-roberta-base",
            "hf_pooler_type": "mean_pooler",
            "hf_tokenizer_name": "xlm-roberta-base",
        },
    }
    if value["config"] != expected_config:
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")


def _validate_preprocess(value: object) -> None:
    expected = {
        "image_size": 224,
        "resize_mode": "shortest",
        "interpolation": "bicubic",
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    }
    if value != expected:
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")


def _validate_tokenizer(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "repository",
        "revision",
        "context_length",
        "clean",
        "strip_sep_token",
        "tokenizer_mode",
        "local_files_only",
        "model_source",
        "model_config_contract",
        "files",
    }:
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")
    if (
        value["repository"] != OPENCLIP_TOKENIZER_REPOSITORY
        or value["revision"] != OPENCLIP_TOKENIZER_REVISION
        or _REVISION_RE.fullmatch(value["revision"]) is None
        or value["context_length"] != 77
        or value["clean"] != "whitespace"
        or value["strip_sep_token"] is not False
        or value["tokenizer_mode"] is not None
        or value["local_files_only"] is not True
        or value["model_source"] != "verified-local-snapshot"
        or value["model_config_contract"]
        != {
            "model_type": "xlm-roberta",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "max_position_embeddings": 514,
            "vocab_size": 250002,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 1,
            "layer_norm_eps": 1e-5,
        }
        or not isinstance(value["files"], list)
    ):
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")
    expected_names = {
        "config.json",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    seen: set[str] = set()
    for item in value["files"]:
        _validate_file(item)
        name = item["name"]
        if name in seen:
            raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")
        seen.add(name)
    if seen != expected_names:
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")


def _validate_file(value: object, *, expected_name: str | None = None) -> None:
    if not isinstance(value, dict) or set(value) != {"name", "size", "sha256"}:
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")
    name, size, sha256 = value["name"], value["size"], value["sha256"]
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or (expected_name is not None and name != expected_name)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(sha256, str)
        or _SHA256_RE.fullmatch(sha256) is None
    ):
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")


def _validate_inference(value: object) -> None:
    if value != {"precision": "fp32", "weights_only": True, "eval_mode": True}:
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")


def _validate_runtime_identity(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"python", "distributions"}
        or value["python"] != "full-version"
        or not isinstance(value["distributions"], list)
        or value["distributions"]
        != [
            "open-clip-torch",
            "torch",
            "torchvision",
            "transformers",
            "tokenizers",
            "sentencepiece",
            "ftfy",
            "Pillow",
            "numpy",
        ]
    ):
        raise OpenClipIdentityError("the pinned OpenCLIP manifest is invalid")


def _distribution_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "missing"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OpenClipIdentityError(
            "the pinned OpenCLIP manifest is invalid"
        ) from error


__all__ = [
    "OPENCLIP_MODEL_NAME",
    "OPENCLIP_MODEL_REPOSITORY",
    "OPENCLIP_MODEL_REVISION",
    "OPENCLIP_PRETRAINED_TAG",
    "OPENCLIP_TOKENIZER_REPOSITORY",
    "OPENCLIP_TOKENIZER_REVISION",
    "OpenClipIdentityError",
    "PinnedOpenClipSnapshot",
    "canonical_openclip_provider_name",
    "load_pinned_openclip_manifest",
    "openclip_runtime_abi_versions",
    "parse_pinned_openclip_manifest",
    "resolve_openclip_backend",
    "verify_pinned_openclip_cache",
]
