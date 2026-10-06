"""Provision pinned OpenCLIP and MUSIQ assets, or verify them offline."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.index.openclip_identity import (  # noqa: E402
    load_pinned_openclip_manifest,
    verify_pinned_openclip_cache,
)
from ai.aesthetics.provider import WEIGHTS, file_sha256  # noqa: E402


def install(cache: Path, *, offline=False):
    cache = cache.resolve()
    manifest, _ = load_pinned_openclip_manifest()
    if not offline:
        from huggingface_hub import snapshot_download

        for spec, patterns in (
            (manifest["model"], [manifest["model"]["weight"]["name"]]),
            (manifest["tokenizer"], [f["name"] for f in manifest["tokenizer"]["files"]]),
        ):
            snapshot_download(
                repo_id=spec["repository"], revision=spec["revision"],
                allow_patterns=patterns, cache_dir=str(cache / "openclip"),
            )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/download_aesthetics_models.py"),
             "--destination", str(cache / "musiq")], check=True,
        )
    verify_pinned_openclip_cache(cache / "openclip")
    for filename, expected in WEIGHTS.values():
        if file_sha256(cache / "musiq" / filename) != expected:
            raise ValueError(f"MUSIQ checksum mismatch: {filename}")
    print("OpenCLIP and both MUSIQ checkpoints verified.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".norma/data/models")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    install(args.cache_dir, offline=args.offline)


if __name__ == "__main__":
    main()
