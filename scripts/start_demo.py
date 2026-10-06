"""Start the isolated demo without replacing the existing library service."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def default_web_dist(root=ROOT):
    preview = root / ".norma/demo-web-dist"
    return preview if (preview / "index.html").is_file() else root / "ai/web_dist"


def demo_environment(data_dir, web_dist, provider=None, environ=None):
    """Keep explicit user settings; isolation paths are controlled by CLI."""
    env = dict(os.environ if environ is None else environ)
    defaults = {
        "NORMA_MODEL_CACHE_DIR": str(ROOT / ".norma/data/models"),
        "NORMA_VLM_PROVIDER": "local",
        "NORMA_VLM_MODEL_PATH": str(
            ROOT / ".norma/data/models/qwen3-vl/Qwen3-VL-2B-Instruct-modelscope"
        ),
        "NORMA_PREFERENCE_MODE": "record-only",
        "NORMA_VIDEO_BASE_URL": "http://127.0.0.1:8766",
        "NORMA_VIDEO_TOKEN_FILE": str(ROOT / ".norma/video-worker.token"),
    }
    for name, value in defaults.items():
        env.setdefault(name, value)
    env.update(
        NORMA_DATA_DIR=str(data_dir.resolve()), NORMA_WEB_DIST=str(web_dist.resolve())
    )
    if provider is not None:
        env["NORMA_VLM_PROVIDER"] = provider
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".norma/demo-data")
    parser.add_argument("--web-dist", type=Path, default=default_web_dist())
    parser.add_argument(
        "--vlm-provider",
        choices=["local", "openai-compatible"],
        help="Explicit override; otherwise preserve NORMA_VLM_PROVIDER",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not (args.web_dist / "index.html").is_file():
        parser.error("Build first: pnpm exec vite build --outDir .norma/demo-web-dist")
    env = demo_environment(args.data_dir, args.web_dist, args.vlm_provider)
    print(f"Isolated Norma demo: http://127.0.0.1:{args.port}", flush=True)
    print(
        "Local model parsing is optional; video requires the SSH tunnel and remote weights.",
        flush=True,
    )
    try:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "ai",
                "web",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            cwd=ROOT,
            env=env,
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
