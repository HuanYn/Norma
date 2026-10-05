"""Copy existing DEVELOPMENT photos only into an ignored workflow demo album.

Never use the *-eval folders. Originals and their rights metadata stay intact.
"""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / ".norma/workflow-demo-104"
    target.mkdir(exist_ok=False)
    entries = []
    for name in ("demo-album", "demo-portraits"):
        source = ROOT / ".norma" / name
        attribution = json.loads((source / "ATTRIBUTION.json").read_text(encoding="utf-8"))
        by_name = {v["file"]: v for v in attribution["images"]}
        for image in sorted(source.glob("*.jpg")):
            if image.name not in by_name:
                raise ValueError("Missing source attribution")
            filename = f"{name}-{image.name}"
            shutil.copy2(image, target / filename)
            entries.append({"file": filename, "source": name, "attribution": by_name[image.name]})
    for name in ("mountain", "temple"):
        source = ROOT / ".norma/user-world-demo-20261005/inputs" / (name + ".jpg")
        filename = "user-" + source.name
        shutil.copy2(source, target / filename)
        entries.append({"file": filename, "source": "user-authorized-private-demo", "rights": "Private; not a public dataset or redistribution license"})
    (target / "ATTRIBUTION.json").write_text(json.dumps({"scope": "development-demo-not-heldout", "images": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(target), "photos": len(entries), "originals_modified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
