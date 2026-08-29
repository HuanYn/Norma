"""Capture one bounded Qwen3-VL raw response on the public smoke fixture.

This is a diagnosis artifact, not a production parser.  It records the model
text only after local-path redaction so output-contract failures can be fixed
without weakening the fail-closed adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from ai.rag.models import (
    GroundedRAGRequest,
    RetrievalEvidence,
    build_evidence_bundle,
    snapshot_image_file,
)
from ai.rag.prompting import build_prompt
from ai.rag.providers import ProviderImagePayload
from ai.rag.security import redact_local_paths
from ai.rag.transformers_runtime import create_local_qwen3vl_provider


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    image = snapshot_image_file(
        args.image.resolve(strict=True),
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
        "请用一个简短、可核对的 claim 描述图中最明显的主体和可见特征；"
        "不要推断身份，也不要输出本地路径。"
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
    provider = create_local_qwen3vl_provider(
        args.model_path.resolve(strict=True),
        max_new_tokens=args.max_new_tokens,
    )
    prompt = build_prompt(request.evidence, provider.name)
    started = time.perf_counter()
    raw = provider.runtime.generate_json(
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
        images=(ProviderImagePayload(photo_id=evidence.photo_id, image=image),),
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    duration = time.perf_counter() - started
    if not isinstance(raw, str):
        raw_text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        raw_type = type(raw).__name__
    else:
        raw_text = raw
        raw_type = "str"
    redacted = redact_local_paths(raw_text) or ""
    report: dict[str, object] = {
        "benchmark": "qwen3vl-raw-output-diagnostic-v1",
        "claim_boundary": "public-image output-contract diagnosis only",
        "duration_seconds": duration,
        "max_new_tokens": args.max_new_tokens,
        "provider_fingerprint": provider.name,
        "raw_output": redacted,
        "raw_output_length": len(raw_text),
        "raw_output_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "raw_output_type": raw_type,
        "status": "captured",
    }
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
