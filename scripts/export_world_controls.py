"""Add an evidence-bound action HUD to existing videos, without model inference."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


KEYS = [
    ("forward", "W", 1, 0),
    ("left", "A", 0, 1),
    ("backward", "S", 1, 1),
    ("right", "D", 2, 1),
    ("look_up", "↑", 1, 0),
    ("look_left", "←", 0, 1),
    ("look_down", "↓", 1, 1),
    ("look_right", "→", 2, 1),
]


def probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--font", type=Path, default=Path("C:/Windows/Fonts/seguisym.ttf")
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("A passed source report is required")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.font, args.output / "hud-font.ttf")
    receipts = []
    for index, world in enumerate(report["worlds"], 1):
        source = (args.report.parent / f"world-{index}.mp4").resolve()
        history = world["history"]
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if source_sha != history[-1]["sha256"]:
            raise ValueError("Source video does not match its action receipt")
        before = probe(source)
        if int(before["nb_read_frames"]) != history[-1]["frames"]:
            raise ValueError("Receipt/video frame mismatch")
        # Overlay in the original frame: never scale, crop, or add sidebars.
        filters = []
        width, height = before["width"], before["height"]
        key_size = max(16, min(48, width // 12))
        gap, margin = max(3, key_size // 8), max(8, width // 32)
        pad_width = 3 * key_size + 2 * gap
        right_x = width - margin - pad_width
        top_y = height - margin - 2 * key_size - gap

        def box(x, y, w, h, color, condition=None):
            value = f"drawbox=x={x}:y={y}:w={w}:h={h}:color={color}:t=fill"
            filters.append(value + (f":enable='{condition}'" if condition else ""))

        def text(value, x, y, size=20):
            filters.append(
                f"drawtext=fontfile=hud-font.ttf:text='{value}':x={x}:y={y}:fontsize={size}:fontcolor=white"
            )

        for i, (action, key, col, row) in enumerate(KEYS):
            x = (margin if i < 4 else right_x) + col * (key_size + gap)
            y = top_y + row * (key_size + gap)
            box(x, y, key_size, key_size, "0x293748@0.8")
            start = 0
            for step in history:
                end = step["frames"]
                if end <= start or step["action"] not in {k[0] for k in KEYS} | {
                    "hold"
                }:
                    raise ValueError("Invalid action timeline")
                if step["action"] == action:
                    box(
                        x,
                        y,
                        key_size,
                        key_size,
                        "0x36ad90@0.95",
                        f"gte(n,{start})*lt(n,{end})",
                    )
                start = end
            text(
                key,
                f"{x}+{key_size / 2}-text_w/2",
                f"{y}+{key_size / 2}-text_h/2",
                round(key_size * 0.65),
            )
        target = args.output.resolve() / f"world-{index}-controls.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-i",
                str(source),
                "-vf",
                ",".join(filters),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ],
            cwd=args.output,
            check=True,
        )
        after = probe(target)
        if any(
            before[k] != after[k]
            for k in ("width", "height", "nb_read_frames", "r_frame_rate", "duration")
        ):
            raise ValueError("Export changed the source dimensions or timeline")
        receipts.append(
            dict(
                source_sha256=source_sha,
                output=str(target),
                output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                video=after,
                actions=[
                    {k: s[k] for k in ("action", "frames", "duration")} for s in history
                ],
            )
        )
    result = dict(
        scope="CPU-only HUD overlay, recorded actions, no new generation or interactive MP4",
        status="passed",
        videos=receipts,
    )
    (args.output / "receipt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
