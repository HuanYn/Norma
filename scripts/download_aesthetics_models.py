"""Explicitly fetch and verify official MUSIQ weights for the optional scorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.aesthetics.provider import (
    WEIGHTS,
    WEIGHTS_REPOSITORY,
    WEIGHTS_REVISION,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    for filename, digest in WEIGHTS.values():
        result = Path(
            hf_hub_download(
                repo_id=WEIGHTS_REPOSITORY,
                filename=filename,
                revision=WEIGHTS_REVISION,
                local_dir=args.destination.resolve(),
            )
        )
        if file_sha256(result) != digest:
            raise RuntimeError(f"Checkpoint hash mismatch: {filename}")
        print(f"Verified {filename}", flush=True)


if __name__ == "__main__":
    main()
