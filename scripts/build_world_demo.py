"""Publish two validated LOCAL outputs into the existing localhost demo's static root.

No upload, remote API or generation. Photos/videos stay in ignored runtime data.
Duration checks validate delivery, not perceptual quality or interactivity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def validate_video(folder: Path) -> dict:
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    video = folder / "offline-camera-smoke.mp4"
    if result.get("status") != "completed" or not video.is_file():
        raise ValueError("Only successfully completed model outputs may be published")
    if hashlib.sha256(video.read_bytes()).hexdigest() != result.get("video_sha256"):
        raise ValueError("Video checksum mismatch")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration",
        "-of", "json", str(video),
    ], text=True))
    stream = probe["streams"][0]
    frames = int(stream["nb_read_frames"])
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    fps = numerator / denominator
    duration = float(stream["duration"])
    if frames != result["requested_frames"] or fps != 16 or duration < 10:
        raise ValueError("Delivery needs real requested frames,16fps and at least10s")
    if [3, frames, stream["height"], stream["width"]] != result["output_shape_cthw"]:
        raise ValueError("Saved video dimensions disagree with model output")
    return {"width": stream["width"], "height": stream["height"], "frames": frames,
            "fps": fps, "duration": duration, "generation_seconds": result["generation_seconds"],
            "sha256": result["video_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mountain", type=Path, required=True)
    parser.add_argument("--temple", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / ".norma/demo-web-dist/world-demo")
    args = parser.parse_args()
    folders = {"mountain": args.mountain, "temple": args.temple}
    # Validate BOTH before modifying the gallery; never publish a half-finished run.
    metadata = {name: validate_video(folder) for name, folder in folders.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, folder in folders.items():
        shutil.copyfile(folder / "offline-camera-smoke.mp4", args.output / f"{name}.mp4")
        shutil.copyfile(folder / "input-crop.png", args.output / f"{name}.png")
    template = (ROOT / "scripts/templates/world_demo.html").read_text(encoding="utf-8")
    (args.output / "index.html").write_text(template, encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
