"""Record real same-input legacy/learned scores, not a quality-improvement claim."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.aesthetics.provider import PyiqaMusiqProvider, file_sha256
from ai.index.quality import analyze_quality
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provider = PyiqaMusiqProvider(args.models)
    with Image.open(args.image) as image:
        legacy = analyze_quality(image)
    runs = []
    for label in ("first_load", "warm"):
        started = time.perf_counter()
        scores = provider.score(args.image)
        runs.append(
            {
                "run": label,
                "seconds": time.perf_counter() - started,
                "scores": scores.as_dict(),
            }
        )
    result = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": args.image.name,
        "input_sha256": file_sha256(args.image),
        "legacy_quality": legacy.quality_score,
        "learned_provider": provider.fingerprint,
        "device": "cpu",
        "python": platform.python_version(),
        "runs": runs,
        "claim_boundary": "One public-image engineering smoke; score scales differ. No human quality or preference improvement is measured.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
