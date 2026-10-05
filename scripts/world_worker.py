"""Start the private LingBot session worker on one explicitly bound idle GPU."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.video_worker import read_private_token  # noqa: E402
from ai.exploration.runtime import WorldRuntime  # noqa: E402
from ai.exploration.server import create_app  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.startswith("GPU-") or "," in gpu:
        parser.error("Explicit single GPU UUID binding required")
    occupied, utilization = map(
        int,
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).split(","),
    )
    if occupied >= 500 or utilization > 5:
        parser.error("Selected GPU is not idle; do not stop other tasks")
    if not 1024 <= args.port <= 65535:
        parser.error("Invalid port")
    import uvicorn

    app = create_app(
        args.data_dir,
        WorldRuntime(args.source, args.models),
        read_private_token(args.token_file),
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
