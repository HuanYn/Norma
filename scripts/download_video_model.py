"""Explicit, resumable download of a pinned official Wan inference checkpoint.

No import-time network or inference. Run only in the isolated video environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    destination = args.destination.expanduser().resolve()
    print(f"Downloading {MODEL_ID}@{REVISION} to {destination}", flush=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=destination,
        max_workers=4,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.jinja",
            "README.md",
            "LICENSE*",
        ],
    )
    print("Snapshot download completed; inference is not yet verified.", flush=True)


if __name__ == "__main__":
    main()
