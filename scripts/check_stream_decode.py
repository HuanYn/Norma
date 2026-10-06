"""Single GPU, same-latent equivalence/timing probe. No image/model downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.exploration.stream_decode import StreamingVAEDecoder


def pixel_error(reference, current):
    import torch
    if reference.shape != current.shape or not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(current).all()):
        raise ValueError("Non-finite or shape-mismatched decode output")
    return float((reference.cpu() - current.cpu()).abs().max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--vae", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--steps", type=int, default=14)
    args = p.parse_args()
    if not 1 <= args.steps <= 14 or args.height not in [256, 512, 640]:
        p.error("Bounded probe only")
    if (
        subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        != "1895d300d8ac936401689b26389f51cbd36530eb"
    ):
        raise RuntimeError("Source revision mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    import torch
    from wan.modules.vae2_1 import Wan2_1_VAE

    torch.set_num_threads(8)
    # Production WanI2VCausal constructs the VAE with its float32 default.
    # Do not benchmark BF16 as if it were the shipping decoder's precision.
    vae = Wan2_1_VAE(vae_pth=str(args.vae), dtype=torch.float32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)
    latent = torch.randn(
        16,
        args.steps * 6,
        args.height // 8,
        384 // 8,
        device="cuda",
        generator=generator,
    )
    report = {
        "scope": "same synthetic latents; decoder equivalence and timing ONLY, not generated visual quality",
        "height": args.height,
        "width": 384,
        "steps": args.steps,
        "seed": 42,
        "dtype": "float32-production-default",
        "runs": [],
        "passed": False,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "command": sys.argv,
                    "upstream_status": subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"], text=True),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "stream_decoder_sha256": hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("ai/exploration/stream_decode.py").read_bytes()).hexdigest()},
    }
    # Warm both paths before collecting timings; keep a dedicated stream cache.
    with torch.inference_mode():
        vae.decode([latent[:, :1]])
    streaming = StreamingVAEDecoder(vae)
    prefix = []
    for step in range(1, args.steps + 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        chunk = streaming.append(latent[:, (step - 1) * 6 : step * 6])
        torch.cuda.synchronize()
        stream_seconds = time.perf_counter() - t
        prefix.append(chunk.cpu())
        current = torch.cat(prefix, dim=1)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            reference = vae.decode([latent[:, : step * 6]])[0]
        torch.cuda.synchronize()
        reference_seconds = time.perf_counter() - t
        error = pixel_error(reference, current)
        row = {
            "step": step,
            "frames": current.shape[1],
            "stream_decode_seconds": stream_seconds,
            "prefix_decode_seconds": reference_seconds,
            "max_abs_error": error,
            "pass": error <= 2 / 255,
        }
        report["runs"].append(row)
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)
        del reference, chunk, current
        if not row["pass"]:
            raise RuntimeError("Streaming differs from full-prefix decode")
    report["passed"] = True
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["vae_sha256"] = hashlib.sha256(args.vae.read_bytes()).hexdigest()
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
