"""Bounded OFFLINE LingBot-v2 smoke test; not an interactive-world service.

Requires verified model assets, an explicitly pinned upstream checkout and one
explicitly bound, idle GPU. Uses an assumed camera, never claims reconstructed
geometry, visual control quality, persistent sessions or real-time operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SOURCE_REVISION = "1895d300d8ac936401689b26389f51cbd36530eb"


def aligned_frame_count(value: str) -> int:
    frames = int(value)
    if not 9 <= frames <= 357 or (frames - 1) % 4 or ((frames - 1) // 4 + 1) % 3:
        raise argparse.ArgumentTypeError("frames must align to VAE stride4 and latent chunk3, within9..357")
    return frames


def portrait_geometry(width: int, original_width: int, original_height: int) -> tuple[int, int]:
    if width % 16 or not 256 <= width <= 512 or original_width <= 0 or original_height <= 0:
        raise ValueError("Invalid portrait preview geometry")
    height = max(256, min(960, round(width * original_height / original_width / 16) * 16))
    return width, height


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=aligned_frame_count, default=21)
    parser.add_argument("--portrait-width", type=int, choices=[256, 320, 384, 448, 480, 512])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--motion", choices=["forward", "right"], default="forward")
    parser.add_argument("--local-attn-size", type=int, choices=[6, 18], default=6)
    parser.add_argument("--sink-size", type=int, choices=[0, 6], default=0)
    args = parser.parse_args()
    if args.sink_size >= args.local_attn_size:
        parser.error("sink size must be smaller than attention window")
    prompt = (args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else
              "A photorealistic street scene. The camera moves slowly forward. Stable buildings and natural daylight.")
    if not prompt or len(prompt) > 3000:
        parser.error("prompt must contain1..3000 characters")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.startswith("GPU-") or "," in gpu:
        parser.error("Bind exactly one GPU UUID in CUDA_VISIBLE_DEVICES")
    revision = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != SOURCE_REVISION:
        parser.error("Unexpected upstream revision")
    dirty = subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if dirty:
        parser.error("Upstream tracked source must be unmodified for this baseline")
    manifest = args.models / "download-manifest.json"
    if not manifest.is_file():
        parser.error("Complete download/integrity manifest is required")
    free, used, occupied = map(int, subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=memory.free,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(","))
    if free < 20 * 1024 or used > 5 or occupied >= 500:
        parser.error("GPU must be idle with at least 20 GiB free; do not stop other jobs")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    report = {
        "status": "running", "scope": "offline prescribed camera trajectory ONLY",
        "source_revision": revision, "gpu_uuid": gpu,
        "input_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "requested_frames": args.frames, "chunk_size": 3, "max_area": 832 * 480,
        "local_attn_size": args.local_attn_size, "sink_size": args.sink_size, "seed": 42,
        "camera": "Assumed fx=fy=600, cx=416, cy=240 at832x480; 0.25 arbitrary forward units",
        "image_preprocess": "Center crop to832x480; preserve original input unchanged",
        "interactive": False, "visual_control_validated": False,
        "prompt": prompt, "motion": args.motion,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    started = time.perf_counter()
    result = 1
    try:
        import numpy as np
        import torch
        import flash_attn
        from PIL import Image, ImageOps
        from flash_attn import flash_attn_func

        torch.set_num_threads(8)
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
                      flash_attn=flash_attn.__version__, gpu_name=torch.cuda.get_device_name(0))
        # Small real CUDA operator validation before loading model weights.
        probe = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
        actual = flash_attn_func(probe, probe, probe)
        if not bool(torch.isfinite(actual).all()):
            raise RuntimeError("FlashAttention operator produced non-finite output")
        torch.cuda.synchronize()
        del probe, actual
        report["flash_attention_operator_passed"] = True
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], args.frames, axis=0)
        poses[:, 2 if args.motion == "forward" else 0, 3] = np.linspace(0, 0.25, args.frames)
        action_path = args.output / "camera"
        action_path.mkdir()
        np.save(action_path / "poses.npy", poses)
        intrinsics = [[600, 600, 416, 240]]
        with Image.open(args.image) as original:
            original = ImageOps.exif_transpose(original)
            report["original_dimensions_wh"] = list(original.size)
            if args.portrait_width:
                size = portrait_geometry(args.portrait_width, *original.size)
                image = ImageOps.pad(original.convert("RGB"), size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
                focal = size[0] / (2 * math.tan(math.radians(30)))
                intrinsics = [[focal * 832 / size[0], focal * 480 / size[1], 416, 240]]
                # +1 avoids the verified float-floor grid underflow (480 ->464).
                report["max_area"] = size[0] * size[1] + 1
                report["image_preprocess"] = "Full photo contained in aligned portrait canvas; minimal black padding; no crop/stretch"
                report["camera"] = "Assumed horizontal FOV60 degrees; intrinsics transformed to upstream832x480 reference; not recovered geometry"
                report["input_canvas_wh"] = list(size)
            else:
                image = ImageOps.fit(original.convert("RGB"), (832, 480), method=Image.Resampling.LANCZOS)
        np.save(action_path / "intrinsics.npy", np.array(intrinsics, dtype=np.float32))
        image.save(args.output / "input-crop.png")
        sys.path.insert(0, str(args.source.resolve()))
        import wan
        from wan.configs import WAN_CONFIGS
        from wan.utils.utils import save_video

        class RejectIncompleteWeights(logging.Handler):
            def emit(self, record):
                if any(part in record.getMessage() for part in ("Missing keys when loading DiT", "Unexpected keys when loading DiT")):
                    raise RuntimeError("Refusing baseline with missing/unexpected DiT weights")

        logging.getLogger().addHandler(RejectIncompleteWeights())
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        model = wan.WanI2VCausal(
            config=WAN_CONFIGS["i2v-1.3B"], checkpoint_dir=str(args.models),
            assets_dir=str(args.models / "assets"), t5_cpu=True,
            convert_model_dtype=True, local_attn_size=args.local_attn_size, sink_size=args.sink_size,
            infer_mode="causal_fast",
        )
        torch.cuda.synchronize()
        report["model_load_seconds"] = time.perf_counter() - load_start
        generation_start = time.perf_counter()
        video = model.generate(
            prompt, image, action_path=str(action_path), chunk_size=3, max_area=report["max_area"],
            frame_num=args.frames, shift=5.0, seed=42, offload_model=True,
        )
        torch.cuda.synchronize()
        report["generation_seconds"] = time.perf_counter() - generation_start
        if video.ndim != 4 or video.shape[1] != args.frames or not bool(torch.isfinite(video).all()):
            raise RuntimeError("Unexpected frame count/shape or non-finite output")
        if args.portrait_width and tuple(video.shape[-2:][::-1]) != tuple(report["input_canvas_wh"]):
            raise RuntimeError("Actual output dimensions differ from explicitly aligned portrait canvas")
        report["output_shape_cthw"] = list(video.shape)
        report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["peak_scope"] = "This process from before model load; not whole GPU or nvidia-smi peak"
        output = args.output / "offline-camera-smoke.mp4"
        save_video(tensor=video[None], save_file=str(output), fps=16,
                   nrow=1, normalize=True, value_range=(-1, 1))
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError("Video encoder did not produce an output")
        report.update(status="completed", playback_fps=16, video_bytes=output.stat().st_size,
                      video_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        result = 0
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        report["total_seconds"] = time.perf_counter() - started
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
