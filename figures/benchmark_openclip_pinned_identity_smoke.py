"""Run a local-only smoke test for the pinned multilingual OpenCLIP provider.

The report verifies the identity/cache boundary and basic image/text inference.
It is a single-machine engineering smoke, not a retrieval-quality benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import threading
import time
from collections.abc import Iterator
from pathlib import Path

# Import Norma before NumPy so the Windows MKL/OpenMP contract is established
# before NumPy selects a native threading backend.
from ai.index.openclip_provider import OpenClipMultilingualProvider
from ai.numeric_runtime import numeric_threading_contract
import numpy as np


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector_record(vector: np.ndarray) -> dict[str, object]:
    array = np.asarray(vector, dtype=np.float32)
    return {
        "dimension": int(array.size),
        "finite": bool(np.all(np.isfinite(array))),
        "l2_norm": float(np.linalg.norm(array)),
        "sha256_float32": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _sample_rss(stop: threading.Event, samples: list[int]) -> None:
    try:
        import psutil
    except ImportError:
        return
    process = psutil.Process()
    samples.append(process.memory_info().rss)
    while not stop.wait(0.05):
        samples.append(process.memory_info().rss)
    samples.append(process.memory_info().rss)


@contextlib.contextmanager
def _prevent_system_sleep() -> Iterator[None]:
    """Keep local CPU inference out of Windows idle sleep during wall timing."""

    if os.name != "nt":
        yield
        return

    import ctypes

    es_continuous = 0x80000000
    es_system_required = 0x00000001
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_thread_execution_state = kernel32.SetThreadExecutionState
    set_thread_execution_state.argtypes = [ctypes.c_uint]
    set_thread_execution_state.restype = ctypes.c_uint
    if not set_thread_execution_state(es_continuous | es_system_required):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        if not set_thread_execution_state(es_continuous):
            raise ctypes.WinError(ctypes.get_last_error())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    image_path = args.image.resolve(strict=True)
    query_zh = "哥特式教堂的石雕屋顶细节"
    query_en = "stone carvings and Gothic cathedral roof details"
    started = time.perf_counter()
    stop = threading.Event()
    rss_samples: list[int] = []
    sampler = threading.Thread(
        target=_sample_rss, args=(stop, rss_samples), daemon=True
    )
    sampler.start()
    status = "failed"
    error: str | None = None
    observations: dict[str, object] = {}
    provider: OpenClipMultilingualProvider | None = None
    try:
        with _prevent_system_sleep():
            provider_started = time.perf_counter()
            provider = OpenClipMultilingualProvider(
                cache_dir=args.cache_root.resolve(),
                device="cpu",
                batch_size=1,
            )
            provider_initialization_seconds = time.perf_counter() - provider_started

            text_started = time.perf_counter()
            zh_first = provider.embed_text(query_zh)
            cold_text_seconds = time.perf_counter() - text_started

            text_started = time.perf_counter()
            zh_repeat = provider.embed_text(query_zh)
            repeat_text_seconds = time.perf_counter() - text_started

            text_started = time.perf_counter()
            en = provider.embed_text(query_en)
            english_text_seconds = time.perf_counter() - text_started

            image_started = time.perf_counter()
            image_vector = provider.embed_image(image_path)
            image_seconds = time.perf_counter() - image_started
            observations = {
                "provider_initialization_seconds": provider_initialization_seconds,
                "cold_text_seconds": cold_text_seconds,
                "repeat_text_seconds": repeat_text_seconds,
                "english_text_seconds": english_text_seconds,
                "image_seconds": image_seconds,
                "chinese": _vector_record(zh_first),
                "chinese_repeat": _vector_record(zh_repeat),
                "english": _vector_record(en),
                "image": _vector_record(image_vector),
                "repeat_cosine": float(np.dot(zh_first, zh_repeat)),
                "zh_en_cosine": float(np.dot(zh_first, en)),
                "zh_image_cosine": float(np.dot(zh_first, image_vector)),
            }
            status = "passed"
    except Exception as exc:  # report exact local failure without hiding it
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        sampler.join(timeout=1)
        report = {
            "benchmark": "openclip-pinned-v3-local-smoke-20260829",
            "claim_boundary": (
                "identity/cache and single-image/text inference smoke only; "
                "not a retrieval-quality or cross-machine reproducibility result"
            ),
            "status": status,
            "error": error,
            "duration_seconds": time.perf_counter() - started,
            "peak_rss_bytes": max(rss_samples) if rss_samples else None,
            "provider_fingerprint": provider.name if provider else None,
            "manifest_sha256": provider.manifest_sha256 if provider else None,
            "resolved_device": provider.device
            if provider and provider.is_loaded
            else None,
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numeric_threading_contract": numeric_threading_contract(),
                "sleep_inhibition": (
                    "windows-es-system-required" if os.name == "nt" else "not-required"
                ),
                "torch_threads": args.threads,
                **{
                    name: _distribution_version(name)
                    for name in (
                        "open-clip-torch",
                        "torch",
                        "torchvision",
                        "transformers",
                        "tokenizers",
                        "sentencepiece",
                        "ftfy",
                        "Pillow",
                        "numpy",
                    )
                },
            },
            "image": {
                "source": "Wikimedia Commons CC BY-SA 3.0 smoke fixture",
                "byte_size": image_path.stat().st_size,
                "sha256": _sha256(image_path),
            },
            "queries": {"zh": query_zh, "en": query_en},
            "observations": observations,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
