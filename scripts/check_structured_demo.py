"""One local-only, controlled-public-prompt smoke run with raw output capture.

The UTF-8 prompt lives in this source file, not a PowerShell -c argument. The
trace contains this public example only, never settings, credentials or images.
Existing traces are retained; there are no automatic model retries.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.numeric_runtime import configure_numeric_runtime  # noqa: E402

configure_numeric_runtime()

from ai.config import load_settings  # noqa: E402
from ai.rag.transformers_runtime import create_local_qwen3vl_provider  # noqa: E402
from ai.selection.structured import (  # noqa: E402
    CONTRACT_VERSION,
    MAX_NEW_TOKENS,
    StructuredIntentError,
    StructuredSelectionParser,
)

PUBLIC_PROMPT = "选3张建筑，偏暖色电影感"
TRACE_PATH = ROOT / ".norma" / "local-intent-smoke.json"


class TracingLocalRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.provider_fingerprint = runtime.provider_fingerprint
        self.raw_output = None
        self.generation_seconds = None

    def generate_text_json(self, **kwargs):
        started = time.perf_counter()
        try:
            self.raw_output = self.runtime.generate_text_json(**kwargs)
            return self.raw_output
        finally:
            self.generation_seconds = round(time.perf_counter() - started, 3)


def main() -> int:
    started = time.perf_counter()
    record = {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "prompt": PUBLIC_PROMPT,
        "prompt_source": "UTF-8 Python source literal",
        "inference": "local-pinned-Qwen3-VL-2B-CPU-only",
        "max_new_tokens": MAX_NEW_TOKENS,
        "contract_version": CONTRACT_VERSION,
        "status": "unavailable",
    }
    traced = None
    try:
        settings = load_settings()
        provider = create_local_qwen3vl_provider(
            settings.local_vlm_model_dir, max_new_tokens=MAX_NEW_TOKENS,
        )
        traced = TracingLocalRuntime(provider.runtime)
        result = StructuredSelectionParser(traced).parse(PUBLIC_PROMPT)
        record.update(status=result.status, document=result.document.model_dump(),
                      provenance=result.provenance.model_dump())
    except Exception as error:
        record["error_type"] = type(error).__name__
        # Only the public structured-validation exception messages are retained;
        # loader/runtime errors may contain local filesystem paths.
        record["error"] = str(error) if isinstance(error, StructuredIntentError) else "Local runtime unavailable or failed"
    if traced is not None:
        record["raw_model_output"] = traced.raw_output
        record["generation_seconds"] = traced.generation_seconds
    record["total_seconds"] = round(time.perf_counter() - started, 3)
    trace = {"schema": "norma-local-intent-smoke-v1", "runs": []}
    if TRACE_PATH.exists():
        existing = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
        if existing.get("schema") != trace["schema"] or not isinstance(existing.get("runs"), list):
            raise RuntimeError("Refusing to overwrite an unknown existing trace")
        trace = existing
    trace["runs"].append(record)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in ["status", "error_type", "generation_seconds", "total_seconds"]}))
    return 0 if record["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
