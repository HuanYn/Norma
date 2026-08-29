"""Run a real, local-only Qwen3-VL grounded-generation smoke test.

This benchmark validates model loading, multimodal input, strict JSON parsing,
content-addressed evidence, citation allow-listing, and server-owned provenance.
It does not measure retrieval quality or semantic entailment.
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

from ai.numeric_runtime import numeric_threading_contract
from ai.rag.engine import generate_grounded
from ai.rag.models import (
    GroundedRAGRequest,
    RetrievalEvidence,
    build_evidence_bundle,
    snapshot_image_file,
)
from ai.rag.security import redact_local_paths
from ai.rag.transformers_runtime import create_local_qwen3vl_provider


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _sample_rss(stop: threading.Event, samples: list[int]) -> None:
    """Sample process RSS when psutil is available; never affect the smoke."""

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
    """Keep long CPU inference out of Windows idle sleep during wall timing."""

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
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    model_path = args.model_path.resolve(strict=True)
    image_path = args.image.resolve(strict=True)
    weight_path = model_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("model.safetensors is missing from --model-path")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    image = snapshot_image_file(
        image_path,
        display_name="public-smoke.jpg",
        media_type="image/jpeg",
    )
    evidence = RetrievalEvidence(
        photo_id="public-smoke-001",
        image=image,
        thumbnail=None,
        retrieval_score=1.0,
        rank=1,
        provider_fingerprint="public-smoke-retrieval-v1",
    )
    query = (
        "请用一个简短、可核对的 claim 描述图中最明显的建筑元素和可见材质；"
        "不要推断具体地点，也不要输出本地路径。"
    )
    bundle = build_evidence_bundle(
        query=query,
        retrieval_provider_fingerprint=evidence.provider_fingerprint,
        candidate_photo_ids=(evidence.photo_id,),
        evidence=(evidence,),
        candidate_source_digest=image.sha256,
    )
    request = GroundedRAGRequest(
        query=query,
        evidence=bundle,
        retrieval_provider_fingerprint=evidence.provider_fingerprint,
        candidate_digest=bundle.provenance.candidate_digest,
    )
    report: dict[str, object] = {
        "benchmark": "qwen3vl-grounded-local-smoke-v5",
        "claim_boundary": (
            "generation/citation/provenance smoke only; retrieval quality and "
            "semantic entailment are not evaluated"
        ),
        "environment": {
            "machine": platform.machine(),
            "numpy": _distribution_version("numpy"),
            "numeric_threading_contract": numeric_threading_contract(),
            "jinja2": _distribution_version("Jinja2"),
            "platform": platform.platform(),
            "pillow": _distribution_version("Pillow"),
            "python": platform.python_version(),
            "sleep_inhibition": (
                "windows-es-system-required" if os.name == "nt" else "not-required"
            ),
            "safetensors": _distribution_version("safetensors"),
            "tokenizers": _distribution_version("tokenizers"),
            "torch": torch.__version__,
            "torchvision": _distribution_version("torchvision"),
            "transformers": _distribution_version("transformers"),
            "torch_threads": args.threads,
        },
        "image": {
            "byte_size": image.byte_size,
            "media_type": image.media_type,
            "sha256": image.sha256,
            "source": "Wikimedia Commons CC BY-SA 3.0 smoke fixture",
        },
        "model": {
            "file_count": sum(1 for item in model_path.iterdir() if item.is_file()),
            "weight_byte_size": weight_path.stat().st_size,
            "weight_sha256": _sha256(weight_path),
        },
        "max_new_tokens": args.max_new_tokens,
        "query": query,
        "repeats": args.repeats,
    }
    overall_started = time.perf_counter()
    rss_stop = threading.Event()
    rss_samples: list[int] = []
    rss_thread = threading.Thread(
        target=_sample_rss,
        args=(rss_stop, rss_samples),
        daemon=True,
    )
    rss_thread.start()
    runs: list[dict[str, object]] = []
    try:
        provider_started = time.perf_counter()
        provider = create_local_qwen3vl_provider(
            model_path,
            max_new_tokens=args.max_new_tokens,
        )
        provider_initialization_seconds = time.perf_counter() - provider_started
        with _prevent_system_sleep():
            for repeat_index in range(args.repeats):
                run_started = time.perf_counter()
                answer = generate_grounded(provider, request)
                runs.append(
                    {
                        "answer": answer.answer,
                        "citations": [
                            {"claim_id": item.claim_id, "photo_id": item.photo_id}
                            for item in answer.citations
                        ],
                        "claims": [
                            {"claim_id": item.claim_id, "text": item.text}
                            for item in answer.claims
                        ],
                        "duration_seconds": time.perf_counter() - run_started,
                        "repeat_index": repeat_index,
                        "validation_scope": answer.validation_scope,
                    }
                )
    except Exception as error:
        rss_stop.set()
        rss_thread.join(timeout=1)
        report.update(
            {
                "duration_seconds": time.perf_counter() - overall_started,
                "error": redact_local_paths(f"{type(error).__name__}: {error}"),
                "failed_repeat_index": len(runs),
                "peak_rss_bytes": max(rss_samples) if rss_samples else None,
                "runs": runs,
                "status": "failed",
            }
        )
        _write_report(args.output, report)
        raise

    rss_stop.set()
    rss_thread.join(timeout=1)
    runtime = provider.runtime
    canonical_runs = [
        (item["answer"], item["claims"], item["citations"]) for item in runs
    ]

    report.update(
        {
            "deterministic_replay": len(set(map(repr, canonical_runs))) == 1,
            "duration_seconds": time.perf_counter() - overall_started,
            "full_manifest_verification_count": getattr(
                runtime, "full_manifest_verification_count", None
            ),
            "full_manifest_verification_ms": getattr(
                runtime, "full_manifest_verification_ms", None
            ),
            "generation_provider_fingerprint": (
                answer.provenance.generation_provider_fingerprint
            ),
            "peak_rss_bytes": max(rss_samples) if rss_samples else None,
            "provider_initialization_seconds": provider_initialization_seconds,
            "provenance": {
                "candidate_digest": answer.provenance.candidate_digest,
                "evidence_digest": answer.provenance.evidence_digest,
                "query_digest": answer.provenance.query_digest,
                "retrieval_provider_fingerprint": (
                    answer.provenance.retrieval_provider_fingerprint
                ),
            },
            "runs": runs,
            "status": "passed",
            "validation_scope": answer.validation_scope,
        }
    )
    _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
