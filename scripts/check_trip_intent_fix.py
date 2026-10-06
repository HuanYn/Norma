"""CPU-only reproduction of the reported trip intent, retaining raw outputs."""

import argparse
import json
from pathlib import Path
import sys
import time
import faulthandler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.rag.transformers_runtime import create_local_qwen3vl_provider
import ai.selection.compact as compact
from ai.workflow import normalize_query

CANDIDATE = compact.SYSTEM


def main():
    faulthandler.dump_traceback_later(45, repeat=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--baseline-report",
        type=Path,
        required=True,
        help="Preserved diagnostic JSON containing the original before-system prompt",
    )
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    print("Loading CPU runtime", flush=True)
    import torch

    torch.set_num_threads(8)
    runtime = create_local_qwen3vl_provider(
        Path(".norma/data/models/qwen3-vl/Qwen3-VL-2B-Instruct-modelscope").resolve()
    ).runtime
    print("Runtime verified", flush=True)
    rows = []

    class Logged:
        provider_fingerprint = runtime.provider_fingerprint
        raw = None

        def generate_text_json(self, **kw):
            self.raw = runtime.generate_text_json(**kw)
            return self.raw

    logged = Logged()
    parser = compact.CompactSelectionParser(logged)
    original = compact.SYSTEM
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    before_system = next(
        row["system"] for row in baseline["runs"] if row["label"] == "before"
    )
    prompt = normalize_query("挑出最适合这趟旅游展示的9张照片", 9, 102)[0]
    for label, system, text in [
        ("before", before_system, prompt),
        ("candidate", CANDIDATE, prompt),
        ("guard", CANDIDATE, "选4张，不要自拍"),
    ]:
        compact.SYSTEM = system
        started = time.perf_counter()
        logged.raw = None
        row = dict(label=label, prompt=text, system=system)
        try:
            result = parser.parse(text)
            row.update(status=result.status, document=result.document.model_dump())
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        row.update(raw=logged.raw, seconds=time.perf_counter() - started)
        rows.append(row)
        (args.output / "result.json").write_text(
            json.dumps(
                dict(scope="CPU model diagnostic, no images or GPU service", runs=rows),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    compact.SYSTEM = original
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
