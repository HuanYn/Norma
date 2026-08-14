from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a controlled quality/similarity evaluation album from public JPEGs"
    )
    parser.add_argument("--source", type=Path, default=Path(".norma/demo-album"))
    parser.add_argument("--output", type=Path, default=Path(".norma/demo-album-eval"))
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    originals = sorted(source.glob("*.jpg"))
    if len(originals) < 12:
        parser.error("source album must contain at least 12 JPEGs")

    for path in originals:
        shutil.copy2(path, output / path.name)

    derived: list[dict[str, str]] = []
    for index, path in enumerate(originals[:4], start=1):
        name = f"eval_duplicate_{index:02d}.jpg"
        shutil.copy2(path, output / name)
        derived.append({"file": name, "source": path.name, "transform": "exact-copy"})

    transforms = (
        (originals[4], "eval_blur_01.jpg", "gaussian-blur-28"),
        (originals[5], "eval_blur_02.jpg", "gaussian-blur-34"),
        (originals[6], "eval_dark_01.jpg", "brightness-0.04"),
        (originals[7], "eval_dark_02.jpg", "brightness-0.07"),
        (originals[8], "eval_bright_01.jpg", "brightness-3.8"),
    )
    for source_path, filename, transform in transforms:
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
            if transform.startswith("gaussian"):
                radius = float(transform.rsplit("-", 1)[1])
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            else:
                factor = float(transform.rsplit("-", 1)[1])
                image = ImageEnhance.Brightness(image).enhance(factor)
            image.save(output / filename, "JPEG", quality=90)
        derived.append(
            {"file": filename, "source": source_path.name, "transform": transform}
        )

    attribution_source = source / "ATTRIBUTION.json"
    if attribution_source.exists():
        shutil.copy2(attribution_source, output / "ATTRIBUTION.json")
    (output / "EVAL_DERIVATIONS.json").write_text(
        json.dumps(
            {
                "notice": "Controlled derivatives for local quality and duplicate-detection testing only.",
                "source_album": str(source),
                "derived_images": derived,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Built {len(list(output.glob('*.jpg')))}-image evaluation album at {output}")


if __name__ == "__main__":
    main()
