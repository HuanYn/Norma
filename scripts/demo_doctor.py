"""Read-only demo preflight. Never loads models, calls cloud APIs or starts jobs."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.start_demo import ROOT, demo_environment
from scripts.video_worker import read_private_token


def inspect(env, port=8767):
    checks = []

    def add(name, ok, detail, required=False):
        checks.append(
            dict(
                name=name,
                status="ok" if ok else "missing",
                detail=detail,
                required=required,
            )
        )

    add(
        "frontend",
        (Path(env["NORMA_WEB_DIST"]) / "index.html").is_file(),
        "Compiled web entry",
        True,
    )
    for name in [
        "fastapi",
        "uvicorn",
        "torch",
        "transformers",
        "open_clip_torch",
        "pyiqa",
    ]:
        try:
            add(name, True, importlib.metadata.version(name), True)
        except importlib.metadata.PackageNotFoundError:
            add(name, False, "Install the pinned project dependencies", True)
    provider = env["NORMA_VLM_PROVIDER"]
    if provider == "local":
        directory = Path(env["NORMA_VLM_MODEL_PATH"])
        present = (directory / "config.json").is_file() and any(
            directory.glob("*.safetensors")
        )
        add(
            "semantic_parser",
            present,
            "Local weight presence only; not a load/inference check",
        )
    else:
        configured = all(
            env.get(k, "").strip()
            for k in ["NORMA_VLM_BASE_URL", "NORMA_VLM_MODEL", "NORMA_VLM_API_KEY"]
        )
        add(
            "semantic_parser",
            configured,
            "Cloud configuration presence only; no network authentication performed",
        )
    musiq = Path(
        env.get(
            "NORMA_AESTHETICS_MODEL_DIR",
            str(Path(env["NORMA_MODEL_CACHE_DIR"]) / "musiq"),
        )
    )
    add(
        "musiq_weights",
        musiq.is_dir() and any(musiq.rglob("*.pth")),
        "Weight presence only; not integrity verification",
        True,
    )
    token = None
    try:
        token = read_private_token(Path(env["NORMA_VIDEO_TOKEN_FILE"]))
        add("worker_credential", True, "Configured; secret never included")
    except (OSError, ValueError):
        add("worker_credential", False, "Private token file is missing or invalid")
    with httpx.Client(timeout=5, trust_env=False, follow_redirects=False) as client:
        for name, url, headers in [
            ("web_service", f"http://127.0.0.1:{port}/health", {}),
            (
                "world_slot_1",
                "http://127.0.0.1:8769/status",
                {"Authorization": "Bearer " + token} if token else {},
            ),
            (
                "world_slot_2",
                "http://127.0.0.1:8770/status",
                {"Authorization": "Bearer " + token} if token else {},
            ),
        ]:
            try:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                raw = response.json()
                safe = {
                    k: raw[k]
                    for k in ("active", "max_steps", "decode_mode", "vlm_provider")
                    if k in raw
                }
                add(name, True, safe)
            except (httpx.HTTPError, ValueError):
                add(
                    name,
                    False,
                    "Not available; check service/tunnel/credentials without exposing tokens",
                )
    return {
        "scope": "read-only presence/reachability, NOT model-quality or end-to-end acceptance",
        "provider": provider,
        "checks": checks,
        "required_pass": all(c["status"] == "ok" for c in checks if c["required"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or (args.output and args.output.exists()):
        parser.error("Valid port and a new output file required")
    env = demo_environment(ROOT / ".norma/demo-data", ROOT / ".norma/demo-web-dist")
    report = inspect(env, args.port)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if report["required_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
