"""Start local world UI; requires the authenticated private worker via SSH tunnel."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.video_worker import read_private_token  # noqa: E402
from ai.exploration.web import create_web  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--token-file", type=Path, default=ROOT / ".norma/video-worker.token"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    import uvicorn

    uvicorn.run(
        create_web(read_private_token(args.token_file), ROOT, port=args.port),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
