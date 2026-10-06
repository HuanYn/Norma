"""Real two-action smoke. Uses one user-authorized photo on the user's server.

Second action is POSTed only after the first result completes and is downloaded.
No invented renderer, no pre-supplied trajectory, no automatic POST retry.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.video_worker import read_private_token  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--steps", type=int, choices=range(2, 15), default=2)
    parser.add_argument(
        "--token-file", type=Path, default=Path(".norma/video-worker.token")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "events": [], "steps": []}
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}",
        timeout=30,
        trust_env=False,
        headers={"Authorization": "Bearer " + read_private_token(args.token_file)},
    )

    def save():
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    def post(path, body=None):
        response = client.post(path, json=body)
        response.raise_for_status()
        return response.json()

    def ready(sid):
        end = time.monotonic() + 900
        previous = None
        while time.monotonic() < end:
            response = client.get(f"/sessions/{sid}")
            response.raise_for_status()
            record = response.json()
            phase = (record["status"], record["phase"])
            if phase != previous:
                print(phase, flush=True)
                report["events"].append({"at": time.time(), "status": phase})
                previous = phase
                save()
            if record["status"] == "ready":
                return record
            if record["status"] in {"failed", "closed", "interrupted"}:
                raise RuntimeError(record.get("error", record["status"]))
            time.sleep(2)
        raise TimeoutError("World operation timeout")

    sid = None
    try:
        save()
        record = post(
            "/sessions",
            {
                "image_base64": base64.b64encode(args.image.read_bytes()).decode(),
                "prompt": "A photorealistic landscape. Preserve solid terrain, stable lighting and coherent scene structure. Camera motion follows the supplied camera controls.",
                "upload_confirmed": True,
            },
        )
        sid = record["id"]
        report["session_id"] = sid
        save()
        ready(sid)
        for sequence in range(args.steps):
            action = ("forward", "right", "look_left", "backward")[sequence % 4]
            post(f"/sessions/{sid}/steps", {"sequence": sequence, "action": action})
            record = ready(sid)
            result = record["history"][-1]
            assert result["sequence"] == sequence and result["action"] == action
            response = client.get(f"/sessions/{sid}/artifacts/{result['artifact']}")
            response.raise_for_status()
            assert hashlib.sha256(response.content).hexdigest() == result["sha256"]
            (args.output / result["artifact"]).write_bytes(response.content)
            report["steps"].append(result)
            save()
        first, second = report["steps"][:2]
        assert first["kv_cache_identity"] == second["kv_cache_identity"]
        assert first["frames"] == 21 and second["frames"] == 45
        assert second["prefix_max_abs_error"] <= 2 / 255
        assert second["kv_global_end"] > first["kv_global_end"]
        assert report["steps"][-1]["frames"] == args.steps * 24 - 3
        report["status"] = "passed"
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        if sid:
            try:
                post(f"/sessions/{sid}/close")
            except Exception:
                report["close_request_failed"] = True
        save()
        client.close()


if __name__ == "__main__":
    main()
