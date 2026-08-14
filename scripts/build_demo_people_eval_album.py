from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.people import create_face_provider


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a controlled people-clustering fixture from public portraits"
    )
    parser.add_argument("--source", type=Path, default=Path(".norma/demo-portraits"))
    parser.add_argument(
        "--output", type=Path, default=Path(".norma/demo-portraits-eval")
    )
    parser.add_argument("--duplicates", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.duplicates <= 12:
        parser.error("--duplicates must be between 1 and 12")

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    originals = sorted(source.glob("*.jpg"))
    if not originals:
        parser.error("source album has no JPEGs")

    provider = create_face_provider("opencv-haar")
    single_face = [path for path in originals if len(provider.detect(path)) == 1]
    if len(single_face) < args.duplicates:
        parser.error(
            f"only {len(single_face)} single-face images available; "
            f"requested {args.duplicates} duplicates"
        )

    for path in originals:
        shutil.copy2(path, output / path.name)

    derived: list[dict[str, str]] = []
    for index, path in enumerate(single_face[: args.duplicates], start=1):
        filename = f"eval_person_duplicate_{index:02d}.jpg"
        shutil.copy2(path, output / filename)
        derived.append(
            {"file": filename, "source": path.name, "transform": "exact-copy"}
        )

    attribution = source / "ATTRIBUTION.json"
    if attribution.exists():
        shutil.copy2(attribution, output / "ATTRIBUTION.json")
    (output / "PEOPLE_EVAL_DERIVATIONS.json").write_text(
        json.dumps(
            {
                "notice": "Controlled exact copies for local face-clustering regression only.",
                "detector": provider.name,
                "source_album": str(source),
                "derived_images": derived,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Built {len(originals) + len(derived)}-image people evaluation album "
        f"at {output}"
    )


if __name__ == "__main__":
    main()
