"""Collect a known public demo job; never submit, retry or upload a new image."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / ".norma/video-demo-collection")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.job_id):
        parser.error("Invalid job identifier")
    if not 0 <= args.wait_seconds <= 1800:
        parser.error("wait-seconds must be between 0 and 1800")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"http://127.0.0.1:8767/demo/video/jobs/{args.job_id}"
    deadline = time.monotonic() + args.wait_seconds
    states = []
    previous = None
    with httpx.Client(timeout=60, trust_env=False, follow_redirects=False) as client:
        while True:
            response = client.get(endpoint)
            response.raise_for_status()
            job = response.json()
            state = {key: job.get(key) for key in ("status", "stage", "step", "total_steps", "step_percent")}
            if state != previous:
                states.append({"observed_at": datetime.now(timezone.utc).isoformat(), **state})
                print(json.dumps(state), flush=True)
                previous = state
            if job["status"] in {"completed", "failed", "cancelled", "interrupted"} or time.monotonic() >= deadline:
                break
            time.sleep(5)
        report = {"observed_at": datetime.now(timezone.utc).isoformat(),
                  "scope": "Known public demo image. Engineering smoke, not an independent video quality evaluation.",
                  "job": job, "observed_states": states}
        if job["status"] == "completed":
            response = client.get(endpoint + "/artifact")
            response.raise_for_status()
            raw = response.content
            if len(raw) < 32 or raw[4:8] != b"ftyp":
                raise RuntimeError("Completed artifact is not an MP4")
            output = args.output_dir / f"{args.job_id}.mp4"
            if output.exists() and output.read_bytes() != raw:
                raise RuntimeError("Refusing to replace a different existing artifact")
            if not output.exists():
                output.write_bytes(raw)
            report["artifact"] = {"file": output.name, "bytes": len(raw),
                                  "sha256": hashlib.sha256(raw).hexdigest()}
        destination = args.output_dir / f"{args.job_id}.json"
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {destination}", flush=True)
    return 0 if job["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
