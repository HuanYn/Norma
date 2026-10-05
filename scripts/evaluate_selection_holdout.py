"""Evaluate a frozen external pair manifest and separately saved predictions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.selection.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior results: choose a new output path")
    raw = args.labels.read_bytes()
    document = json.loads(raw)
    document["label_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    prediction = json.loads(args.predictions.read_bytes())
    if prediction["label_manifest_sha256"] != document["label_manifest_sha256"]:
        raise ValueError("Predictions were not bound to this frozen label manifest")
    if prediction["method_revision"] != document["method_revision"]:
        raise ValueError("Method revision differs from the frozen protocol")
    report = evaluate(document, prediction["scores"])
    report["predictions_sha256"] = hashlib.sha256(
        args.predictions.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
