from __future__ import annotations

import hashlib
import json
import math
import operator
import sys
import threading
import time
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Callable, Mapping

from PIL import Image

from ai.numeric_runtime import (
    NumericRuntimeConflictError,
    ensure_torch_numpy_runtime_compatible,
    numeric_threading_contract,
)
from ai.rag.image_safety import (
    decode_evidence_image_rgb,
    inspect_evidence_image_dimensions,
)
from ai.rag.models import VLMInputBudgetError
from ai.rag.prompting import prompt_contract_sha256
from ai.rag.providers import ProviderImagePayload, Qwen3VLLocalProvider


QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3_VL_MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
RUNTIME_VERSION = "transformers-cpu-local-only-v3"
PREPROCESS_VERSION = "pil-rgb-waterfill-p16-m2-pre3840-hard4096-v2"
QWEN_VISION_PATCH_SIZE = 16
QWEN_VISION_MERGE_SIZE = 2
QWEN_VISION_GRID_PIXELS = QWEN_VISION_PATCH_SIZE * QWEN_VISION_MERGE_SIZE
MIN_IMAGE_VISUAL_TOKENS = 64
PREFLIGHT_VISUAL_TOKEN_BUDGET = 3840
MAX_VISUAL_TOKENS = 4096
MAX_IMAGE_ASPECT_RATIO = 200.0
_WEIGHT_NAME = "model.safetensors"
_PINNED_MANIFEST_PATH = Path(__file__).with_name("qwen3_vl_2b_manifest.json")
_PINNED_RUNTIME_ASSET_NAMES = frozenset(
    {
        "chat_template.json",
        "config.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        _WEIGHT_NAME,
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
)
_IGNORED_MODEL_DOCUMENTATION_NAMES = frozenset({".gitattributes", "README.md"})
_RUNTIME_ASSET_SUFFIXES = frozenset(
    {".json", ".txt", ".model", ".jinja", ".safetensors"}
)


class LocalVLMUnavailableError(RuntimeError):
    """The explicitly configured local VLM cannot be used without downloading."""


DependencyLoader = Callable[[], tuple[object, object]]


class TransformersQwen3VLRuntime:
    """Lazy CPU-only Qwen3-VL runtime that can only load an on-disk directory.

    It performs retrieval-augmented multimodal generation, but it does not verify
    semantic entailment. Referential citation validation remains the service/core's
    responsibility.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        dependency_loader: DependencyLoader | None = None,
    ) -> None:
        _ensure_numeric_runtime()
        raw_path = Path(model_path)
        if not raw_path.is_absolute():
            raise LocalVLMUnavailableError(
                "NORMA_VLM_MODEL_PATH must resolve to an explicit local directory"
            )
        self.model_path = raw_path.resolve()
        (
            self._weight_path,
            self._weight_stat,
            weight_sha256,
            self._asset_stats,
            manifest_sha256,
            self._content_hashes,
        ) = _validate_and_fingerprint_model(self.model_path)
        safe_model_id = QWEN3_VL_MODEL_ID.replace("/", "%2F")
        runtime_abi = _runtime_abi_versions()
        prompt_sha256 = prompt_contract_sha256()
        self.provider_fingerprint = (
            "qwen3-vl-local-v1"
            f"|model={safe_model_id}"
            f"|revision={QWEN3_VL_MODEL_REVISION}"
            f"|weights_sha256={weight_sha256[:20]}"
            f"|manifest_sha256={manifest_sha256}"
            f"|runtime={RUNTIME_VERSION}"
            f"|preprocess={PREPROCESS_VERSION}"
            f"|prompt_contract_sha256={prompt_sha256}"
            f"|transformers={runtime_abi['transformers']}"
            f"|torch={runtime_abi['torch']}"
            f"|torchvision={runtime_abi['torchvision']}"
            f"|tokenizers={runtime_abi['tokenizers']}"
            f"|safetensors={runtime_abi['safetensors']}"
            f"|numpy={runtime_abi['numpy']}"
            f"|numeric_threading={numeric_threading_contract()}"
            f"|jinja2={runtime_abi['jinja2']}"
            f"|pillow={runtime_abi['pillow']}"
            f"|python={runtime_abi['python']}"
        )
        self._dependency_loader = dependency_loader or _load_dependencies
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._processor: object | None = None
        self._model: object | None = None
        self._torch: object | None = None
        self._full_manifest_verification_count = 0
        self._full_manifest_verification_ms = 0.0

    @property
    def loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    @property
    def full_manifest_verification_count(self) -> int:
        return self._full_manifest_verification_count

    @property
    def full_manifest_verification_ms(self) -> float:
        """Time spent on the two fail-closed hashes around the first load."""

        return self._full_manifest_verification_ms

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: tuple[ProviderImagePayload, ...],
        max_new_tokens: int,
        temperature: float,
    ) -> Mapping[str, object] | str:
        if temperature != 0.0:
            raise ValueError("Qwen3-VL runtime only supports deterministic generation")
        if not images:
            raise ValueError("Qwen3-VL runtime requires at least one evidence image")
        self._assert_model_snapshot_unchanged()
        self._ensure_loaded()
        with self._generation_lock:
            return self._generate_loaded(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images,
                max_new_tokens=max_new_tokens,
            )

    def _generate_loaded(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: tuple[ProviderImagePayload, ...],
        max_new_tokens: int,
    ) -> str:
        assert self._processor is not None
        assert self._model is not None

        pil_images = _prepare_images_for_vlm(images)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in pil_images),
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        try:
            model_inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            for image in pil_images:
                image.close()
        _validate_processor_visual_tokens(model_inputs, len(images))
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to("cpu")
        elif isinstance(model_inputs, Mapping):
            model_inputs = {
                key: value.to("cpu") if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }
        input_ids = model_inputs["input_ids"]
        inference_mode = getattr(self._torch, "inference_mode", nullcontext)
        with inference_mode():
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output_ids[len(source_ids) :]
            for source_ids, output_ids in zip(input_ids, generated, strict=True)
        ]
        decoded = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded or not isinstance(decoded[0], str):
            raise RuntimeError("local Qwen3-VL returned no decodable output")
        self._assert_model_snapshot_unchanged()
        return decoded[0].strip()

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            _ensure_numeric_runtime()
            self._verify_content_manifest("before local model load")
            try:
                transformers, torch = self._dependency_loader()
                processor_class = getattr(transformers, "AutoProcessor")
                model_class = getattr(
                    transformers,
                    "Qwen3VLForConditionalGeneration",
                    None,
                ) or getattr(transformers, "AutoModelForImageTextToText")
                processor = processor_class.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model = model_class.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model = model.to("cpu")
                model.eval()
                self._verify_content_manifest("after local model load")
            except LocalVLMUnavailableError:
                raise
            except Exception as error:
                raise LocalVLMUnavailableError(
                    "local Qwen3-VL dependencies or weights could not be loaded; "
                    "no Hub fallback was attempted: "
                    f"{type(error).__name__}: {error}"
                ) from error
            self._processor = processor
            self._model = model
            self._torch = torch

    def _verify_content_manifest(self, stage: str) -> None:
        started = time.perf_counter()
        try:
            current_paths = set(_model_asset_paths(self.model_path))
            expected_paths = set(self._content_hashes)
            if current_paths != expected_paths:
                raise LocalVLMUnavailableError(
                    f"local Qwen3-VL model manifest drift {stage}"
                )
            for path, expected_sha256 in self._content_hashes.items():
                _, current_sha256 = _stable_model_file_hash(
                    path,
                    error_label="model manifest",
                )
                if current_sha256 != expected_sha256:
                    raise LocalVLMUnavailableError(
                        f"local Qwen3-VL model manifest drift {stage}"
                    )
        finally:
            self._full_manifest_verification_count += 1
            self._full_manifest_verification_ms += round(
                (time.perf_counter() - started) * 1000,
                3,
            )

    def _assert_model_snapshot_unchanged(self) -> None:
        try:
            current = self._weight_path.stat()
        except OSError as error:
            raise LocalVLMUnavailableError(
                "the configured local Qwen3-VL weights are no longer available"
            ) from error
        current_snapshot = (current.st_size, current.st_mtime_ns)
        if current_snapshot != self._weight_stat:
            raise LocalVLMUnavailableError(
                "the configured local Qwen3-VL weights changed after fingerprinting"
            )
        for asset_path, expected in self._asset_stats.items():
            try:
                asset = asset_path.stat()
            except OSError as error:
                raise LocalVLMUnavailableError(
                    "the configured local Qwen3-VL preprocessing assets changed"
                ) from error
            if (asset.st_size, asset.st_mtime_ns) != expected:
                raise LocalVLMUnavailableError(
                    "the configured local Qwen3-VL preprocessing assets changed"
                )


_RUNTIME_LOCK = threading.Lock()
_RUNTIMES: dict[Path, TransformersQwen3VLRuntime] = {}


def create_local_qwen3vl_provider(
    model_path: Path,
    *,
    max_new_tokens: int = 256,
) -> Qwen3VLLocalProvider:
    """Return a process-local singleton runtime without eagerly loading weights."""

    _ensure_numeric_runtime()
    if isinstance(max_new_tokens, bool) or not 64 <= max_new_tokens <= 1024:
        raise ValueError("max_new_tokens must be between 64 and 1024")
    path = Path(model_path)
    if not path.is_absolute():
        raise LocalVLMUnavailableError(
            "NORMA_VLM_MODEL_PATH must resolve to an explicit local directory"
        )
    resolved = path.resolve()
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(resolved)
        if runtime is None:
            runtime = TransformersQwen3VLRuntime(resolved)
            _RUNTIMES[resolved] = runtime
    return Qwen3VLLocalProvider(
        provider_fingerprint=(
            f"{runtime.provider_fingerprint}|max_new_tokens={max_new_tokens}"
        ),
        runtime=runtime,
        max_new_tokens=max_new_tokens,
    )


def _validate_and_fingerprint_model(
    model_path: Path,
) -> tuple[
    Path,
    tuple[int, int],
    str,
    dict[Path, tuple[int, int]],
    str,
    dict[Path, str],
]:
    if not model_path.is_dir():
        raise LocalVLMUnavailableError(
            "local Qwen3-VL model directory is missing; set NORMA_VLM_MODEL_PATH "
            "to a complete local snapshot"
        )
    pinned, manifest_sha256 = _load_pinned_manifest()
    expected_names = set(pinned)
    actual_paths = _model_asset_paths(model_path)
    actual_by_name = {path.name: path for path in actual_paths}
    if set(actual_by_name) != expected_names:
        raise LocalVLMUnavailableError(
            "local Qwen3-VL snapshot does not exactly match the pinned manifest"
        )

    weight_path = actual_by_name[_WEIGHT_NAME]
    asset_stats: dict[Path, tuple[int, int]] = {}
    content_hashes: dict[Path, str] = {}
    weight_stat: tuple[int, int] | None = None
    for name in sorted(expected_names):
        asset_path = actual_by_name[name]
        expected_size, expected_sha256 = pinned[name]
        stat_snapshot, actual_sha256 = _stable_model_file_hash(
            asset_path,
            error_label="pinned model asset",
        )
        if stat_snapshot[0] != expected_size or actual_sha256 != expected_sha256:
            raise LocalVLMUnavailableError(
                "local Qwen3-VL snapshot failed pinned manifest verification"
            )
        content_hashes[asset_path] = expected_sha256
        if name == _WEIGHT_NAME:
            weight_stat = stat_snapshot
        else:
            asset_stats[asset_path] = stat_snapshot
    assert weight_stat is not None
    return (
        weight_path,
        weight_stat,
        pinned[_WEIGHT_NAME][1],
        asset_stats,
        manifest_sha256,
        content_hashes,
    )


def _load_pinned_manifest() -> tuple[dict[str, tuple[int, str]], str]:
    try:
        raw = _PINNED_MANIFEST_PATH.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise LocalVLMUnavailableError(
            "the version-controlled Qwen3-VL pinned manifest is unavailable"
        ) from error
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "model_id", "revision", "files"}
        or document["schema"] != "norma-pinned-model-manifest-v1"
        or document["model_id"] != QWEN3_VL_MODEL_ID
        or document["revision"] != QWEN3_VL_MODEL_REVISION
        or not isinstance(document["files"], list)
    ):
        raise LocalVLMUnavailableError("the pinned Qwen3-VL manifest is invalid")
    files: dict[str, tuple[int, str]] = {}
    for item in document["files"]:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "size",
            "sha256",
            "source",
        }:
            raise LocalVLMUnavailableError("the pinned Qwen3-VL manifest is invalid")
        name = item["name"]
        size = item["size"]
        sha256 = item["sha256"]
        source = item["source"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in files
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(source, str)
            or not source
        ):
            raise LocalVLMUnavailableError("the pinned Qwen3-VL manifest is invalid")
        files[name] = (size, sha256)
    if _WEIGHT_NAME not in files:
        raise LocalVLMUnavailableError("the pinned Qwen3-VL manifest is invalid")
    if set(files) != _PINNED_RUNTIME_ASSET_NAMES:
        raise LocalVLMUnavailableError("the pinned Qwen3-VL manifest is invalid")
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return files, hashlib.sha256(canonical).hexdigest()


def _model_asset_paths(model_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    try:
        entries = sorted(model_path.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise LocalVLMUnavailableError(
            "local Qwen3-VL snapshot directory cannot be enumerated"
        ) from error
    for path in entries:
        if path.name in _IGNORED_MODEL_DOCUMENTATION_NAMES:
            if not path.is_file():
                raise LocalVLMUnavailableError(
                    "local Qwen3-VL snapshot contains an unexpected directory"
                )
            continue
        if not path.is_file() or path.suffix.casefold() not in _RUNTIME_ASSET_SUFFIXES:
            raise LocalVLMUnavailableError(
                f"local Qwen3-VL snapshot contains unpinned asset {path.name}"
            )
        paths.append(path)
    return tuple(paths)


def _stable_model_file_hash(
    path: Path,
    *,
    error_label: str,
) -> tuple[tuple[int, int], str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = path.stat()
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            after = path.stat()
    except OSError as error:
        raise LocalVLMUnavailableError(
            f"local Qwen3-VL {error_label} could not be fingerprinted"
        ) from error
    before_snapshot = (before.st_size, before.st_mtime_ns)
    after_snapshot = (after.st_size, after.st_mtime_ns)
    if before_snapshot != after_snapshot:
        raise LocalVLMUnavailableError(
            f"local Qwen3-VL {error_label} changed while being fingerprinted"
        )
    return after_snapshot, digest.hexdigest()


def _runtime_abi_versions() -> dict[str, str]:
    versions = {
        "transformers": _distribution_version("transformers"),
        "torch": _distribution_version("torch"),
        "torchvision": _distribution_version("torchvision"),
        "tokenizers": _distribution_version("tokenizers"),
        "safetensors": _distribution_version("safetensors"),
        "numpy": _distribution_version("numpy"),
        "jinja2": _distribution_version("Jinja2"),
        "pillow": _distribution_version("Pillow"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    return {
        key: "".join(
            character
            if character.isalnum() or character in {".", "+", "_", "-"}
            else "_"
            for character in value
        )
        for key, value in versions.items()
    }


def _distribution_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "missing"


def _prepare_images_for_vlm(
    payloads: tuple[ProviderImagePayload, ...],
) -> tuple[Image.Image, ...]:
    """Create bounded model-input derivatives without changing evidence identity."""

    dimensions = tuple(
        inspect_evidence_image_dimensions(payload.image.content) for payload in payloads
    )
    budgets = _allocate_visual_token_budgets(dimensions)
    prepared: list[Image.Image] = []
    try:
        for payload, dimensions_item, budget in zip(
            payloads, dimensions, budgets, strict=True
        ):
            source = _decode_image(payload)
            target_size = _target_image_dimensions(dimensions_item, budget)
            if source.size == target_size:
                prepared.append(source)
                continue
            try:
                resized = source.resize(target_size, Image.Resampling.LANCZOS)
            finally:
                source.close()
            prepared.append(resized)
    except BaseException:
        for image in prepared:
            image.close()
        raise
    return tuple(prepared)


def _allocate_visual_token_budgets(
    dimensions: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    if not dimensions:
        raise VLMInputBudgetError("local Qwen3-VL requires image evidence")
    capacities = tuple(_image_token_capacity(item) for item in dimensions)
    budgets = [0] * len(capacities)
    active = list(range(len(capacities)))
    remaining = PREFLIGHT_VISUAL_TOKEN_BUDGET
    while active:
        share = remaining // len(active)
        satisfied = [index for index in active if capacities[index] <= share]
        if satisfied:
            for index in satisfied:
                budgets[index] = capacities[index]
                remaining -= capacities[index]
                active.remove(index)
            continue
        base, extra = divmod(remaining, len(active))
        for offset, index in enumerate(active):
            budgets[index] = base + int(offset < extra)
        remaining = 0
        break
    if any(budget < MIN_IMAGE_VISUAL_TOKENS for budget in budgets):
        raise VLMInputBudgetError(
            "local Qwen3-VL visual budget cannot safely represent every image"
        )
    if sum(budgets) > PREFLIGHT_VISUAL_TOKEN_BUDGET:
        raise VLMInputBudgetError("local Qwen3-VL preflight visual budget overflow")
    return tuple(budgets)


def _image_token_capacity(dimensions: tuple[int, int]) -> int:
    width, height = dimensions
    _validate_image_geometry(width, height)
    aligned_width = max(
        QWEN_VISION_GRID_PIXELS,
        round(width / QWEN_VISION_GRID_PIXELS) * QWEN_VISION_GRID_PIXELS,
    )
    aligned_height = max(
        QWEN_VISION_GRID_PIXELS,
        round(height / QWEN_VISION_GRID_PIXELS) * QWEN_VISION_GRID_PIXELS,
    )
    tokens = (aligned_width // QWEN_VISION_GRID_PIXELS) * (
        aligned_height // QWEN_VISION_GRID_PIXELS
    )
    return min(
        PREFLIGHT_VISUAL_TOKEN_BUDGET,
        max(MIN_IMAGE_VISUAL_TOKENS, tokens),
    )


def _target_image_dimensions(
    dimensions: tuple[int, int], token_budget: int
) -> tuple[int, int]:
    width, height = dimensions
    _validate_image_geometry(width, height)
    target_tokens = min(token_budget, _image_token_capacity(dimensions))
    if target_tokens < MIN_IMAGE_VISUAL_TOKENS:
        raise VLMInputBudgetError("local Qwen3-VL per-image visual budget is too small")
    aspect = width / height
    best: tuple[tuple[float, int, int, int], tuple[int, int]] | None = None
    for grid_height in range(1, target_tokens + 1):
        ideal_width = aspect * grid_height
        candidates = {
            max(1, math.floor(ideal_width)),
            max(1, math.ceil(ideal_width)),
            max(1, round(ideal_width)),
        }
        for grid_width in candidates:
            tokens = grid_width * grid_height
            if not MIN_IMAGE_VISUAL_TOKENS <= tokens <= target_tokens:
                continue
            distortion = abs(math.log((grid_width / grid_height) / aspect))
            key = (distortion, -tokens, grid_height, grid_width)
            candidate = (
                grid_width * QWEN_VISION_GRID_PIXELS,
                grid_height * QWEN_VISION_GRID_PIXELS,
            )
            if best is None or key < best[0]:
                best = (key, candidate)
    if best is None:
        raise VLMInputBudgetError(
            "local Qwen3-VL could not fit image geometry into its visual budget"
        )
    return best[1]


def _validate_image_geometry(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise VLMInputBudgetError("local Qwen3-VL image dimensions must be positive")
    if max(width, height) / min(width, height) > MAX_IMAGE_ASPECT_RATIO:
        raise VLMInputBudgetError(
            "local Qwen3-VL image aspect ratio exceeds the safe limit"
        )


def _validate_processor_visual_tokens(model_inputs: object, image_count: int) -> int:
    try:
        grid = model_inputs["image_grid_thw"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise VLMInputBudgetError(
            "local Qwen3-VL processor omitted image_grid_thw"
        ) from error
    if hasattr(grid, "detach"):
        grid = grid.detach()
    if hasattr(grid, "cpu"):
        grid = grid.cpu()
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if not isinstance(grid, (list, tuple)) or len(grid) != image_count:
        raise VLMInputBudgetError(
            "local Qwen3-VL processor returned an invalid image grid count"
        )
    total = 0
    for row in grid:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise VLMInputBudgetError(
                "local Qwen3-VL processor returned malformed image_grid_thw"
            )
        values: list[int] = []
        for value in row:
            if isinstance(value, bool):
                raise VLMInputBudgetError(
                    "local Qwen3-VL processor returned a non-integer image grid"
                )
            try:
                integer = operator.index(value)
            except TypeError as error:
                raise VLMInputBudgetError(
                    "local Qwen3-VL processor returned a non-integer image grid"
                ) from error
            if integer < 1:
                raise VLMInputBudgetError(
                    "local Qwen3-VL processor returned a non-positive image grid"
                )
            values.append(integer)
        temporal, grid_height, grid_width = values
        if (
            grid_height % QWEN_VISION_MERGE_SIZE != 0
            or grid_width % QWEN_VISION_MERGE_SIZE != 0
        ):
            raise VLMInputBudgetError(
                "local Qwen3-VL image grid is incompatible with merge_size=2"
            )
        tokens = temporal * grid_height * grid_width // (QWEN_VISION_MERGE_SIZE**2)
        total += tokens
        if total > MAX_VISUAL_TOKENS:
            raise VLMInputBudgetError(
                "local Qwen3-VL visual input exceeds the 4096-token hard limit"
            )
    return total


def _decode_image(payload: ProviderImagePayload) -> Image.Image:
    try:
        return decode_evidence_image_rgb(payload.image.content)
    except Exception as error:
        raise ValueError(
            f"evidence image for photo_id {payload.photo_id} is not decodable"
        ) from error


def _load_dependencies() -> tuple[object, object]:
    _ensure_numeric_runtime()
    try:
        import torch
        import transformers
    except ImportError as error:
        raise LocalVLMUnavailableError(
            "local Qwen3-VL requires the multimodal transformers/torch dependencies"
        ) from error
    return transformers, torch


def _ensure_numeric_runtime() -> None:
    try:
        ensure_torch_numpy_runtime_compatible()
    except NumericRuntimeConflictError as error:
        raise LocalVLMUnavailableError(str(error)) from error


__all__ = [
    "LocalVLMUnavailableError",
    "MAX_VISUAL_TOKENS",
    "PREFLIGHT_VISUAL_TOKEN_BUDGET",
    "PREPROCESS_VERSION",
    "QWEN3_VL_MODEL_ID",
    "QWEN3_VL_MODEL_REVISION",
    "RUNTIME_VERSION",
    "TransformersQwen3VLRuntime",
    "create_local_qwen3vl_provider",
]
