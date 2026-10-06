"""Prepare an explicit review manifest or serve an existing local blind-review bundle."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.selection.review import build_bundle, create_review_app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--port", type=int, default=8772)
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    if not 1024 <= a.port <= 65535:
        p.error("Use a local unprivileged port")
    if a.manifest:
        build_bundle(json.loads(a.manifest.read_text(encoding="utf-8")), a.bundle)
    if not a.prepare_only:
        import uvicorn

        uvicorn.run(create_review_app(a.bundle), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
