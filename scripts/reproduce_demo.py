"""Run real CPU photo preparation, MUSIQ curation and text selection in isolation.

Uses the same HTTP routes as the website through an in-process ASGI client.
Weights are provisioned separately. No cloud API, face analysis or GPU worker.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def image_hashes(album):
    return {
        str(p.relative_to(album)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(album.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    }


def wait_job(client, job, prefix, timeout):
    deadline = time.monotonic() + timeout
    try:
        while job["status"] not in {"completed", "failed", "cancelled"}:
            if time.monotonic() >= deadline:
                raise TimeoutError("Preparation timed out; cancellation requested")
            time.sleep(0.5)
            response = client.get(f"{prefix}/{job['id']}")
            response.raise_for_status()
            job = response.json()
    except BaseException:
        client.post(f"{prefix}/{job['id']}/cancel")
        raise
    if job["status"] != "completed":
        raise RuntimeError(job.get("error") or job["status"])
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--album", type=Path, default=ROOT / ".norma/public-smoke")
    parser.add_argument("--models", type=Path, default=ROOT / ".norma/data/models")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default="选1张建筑照片")
    parser.add_argument("--min-aesthetic", type=float, default=1.0)
    parser.add_argument("--min-technical", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    album = args.album.resolve()
    before = image_hashes(album)
    if not before:
        parser.error("Album needs JPG/JPEG photos; run download_public_smoke_image.py first")
    output = (args.output or ROOT / ".norma" / ("reproduce-" + uuid.uuid4().hex[:12])).resolve()
    # A fresh private directory prevents accidentally opening a user's existing DB.
    output.mkdir(parents=True, exist_ok=False)
    for key in list(os.environ):
        if key.startswith("NORMA_"):
            os.environ.pop(key)
    os.environ.update(
        NORMA_DATA_DIR=str(output / "data"),
        NORMA_MODEL_CACHE_DIR=str(args.models.resolve()),
        NORMA_EMBEDDING_PROVIDER="openclip-multilingual",
        NORMA_EMBEDDING_DEVICE="cpu", NORMA_AESTHETICS_DEVICE="cpu",
        NORMA_PREFERENCE_MODE="record-only", NORMA_VLM_PROVIDER="local",
        NORMA_LOG_LEVEL="WARNING",
    )
    from ai.app import app
    from fastapi.testclient import TestClient

    report = {"status": "running", "input_count": len(before), "prompt": args.prompt,
              "device": "cpu", "thresholds": {"min_aesthetic": args.min_aesthetic,
              "min_technical": args.min_technical}, "scope": "real-model photo workflow"}
    started = time.perf_counter()
    try:
        with TestClient(app) as client:
            def post(path, data):
                response = client.post(path, json=data)
                response.raise_for_status()
                return response.json()

            print("Preparing quality and OpenCLIP embeddings...", flush=True)
            prepared = wait_job(client, post("/jobs/prepare", {
                "folder": str(album), "include_quality": True,
                "include_embeddings": True, "include_people": False,
            }), "/jobs", args.timeout)
            album_id = prepared["result"]["album"]["album_id"]
            print("Evaluating both MUSIQ models...", flush=True)
            assessed = wait_job(client, post(f"/albums/{album_id}/aesthetics/jobs", {
                "force": False,
            }), "/aesthetics/jobs", args.timeout)
            curated = post(f"/workflow/albums/{album_id}/curations", report["thresholds"])
            print("Solving the requested photo collection...", flush=True)
            selected = post(f"/workflow/curations/{curated['id']}/select", {
                "prompt": args.prompt, "use_language_model": False,
                "use_preference_memory": False,
            })
            if not selected["feasible"] or len(selected["selected"]) != selected["constraints"]["target_count"]:
                raise RuntimeError("Selection did not satisfy the requested count")
            if before != image_hashes(album):
                raise RuntimeError("Input photos changed during reproduction")
            report.update(status="passed", originals_unchanged=True, kept=len(curated["kept"]),
                          hidden=len(curated["hidden"]), preparation=prepared, assessment=assessed,
                          selection=selected)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{report['status']}: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
