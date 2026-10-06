"""Stateful, bounded LingBot-v2 adapter (single session, single GPU).

CC BY-NC-SA 4.0. Inference adaptation referencing Robbyant/LingBot-World-v2
1895d300d8ac936401689b26389f51cbd36530eb (wan/image2video.py).
Upstream attribution/license: https://github.com/Robbyant/lingbot-world-v2
No upstream files or weights are redistributed here. Non-commercial use only.

DiT KV/cross-attention, RNG, camera and generated latents persist BETWEEN actions.
The default prefix decoder is the reference; optional stream mode preserves the
causal VAE feature cache between steps. Neither restarts from the last frame.
The bounded longer trajectory remains non-real-time and needs visual evaluation.
"""

from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from ai.exploration.controls import CameraAction, _local_transform

SOURCE_REVISION = "1895d300d8ac936401689b26389f51cbd36530eb"
MAX_STEPS = 7
HARD_MAX_STEPS = 14
LATENTS_PER_STEP = 6
MAX_LATENTS = MAX_STEPS * LATENTS_PER_STEP
MAX_FRAMES = (MAX_LATENTS - 1) * 4 + 1


class WorldRuntime:
    def __init__(
        self, source: Path, models: Path, *, max_steps=MAX_STEPS, decode_mode="prefix"
    ):
        if type(max_steps) is not int or not 1 <= max_steps <= HARD_MAX_STEPS:
            raise ValueError("World steps must be in 1..14")
        if decode_mode not in {"prefix", "stream"}:
            raise ValueError("Unknown VAE decoding mode")
        self.max_steps, self.decode_mode = max_steps, decode_mode
        self.source, self.models = source.resolve(), models.resolve()
        self.pipe = None
        self.state = None

    def begin(self, image: Image.Image, prompt: str, output: Path, progress) -> dict:
        import torch

        if (
            subprocess.check_output(
                ["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True
            ).strip()
            != SOURCE_REVISION
        ):
            raise RuntimeError("Unexpected upstream source revision")
        if not (self.models / "download-manifest.json").is_file():
            raise RuntimeError("Verified model download manifest is missing")
        sys.path.insert(0, str(self.source)) if str(
            self.source
        ) not in sys.path else None
        from wan import WanI2VCausal
        from wan.configs import WAN_CONFIGS

        torch.set_num_threads(8)
        progress("loading_model")
        started = time.perf_counter()
        if self.pipe is None:
            self.pipe = WanI2VCausal(
                config=WAN_CONFIGS["i2v-1.3B"],
                checkpoint_dir=str(self.models),
                assets_dir=str(self.models / "assets"),
                t5_cpu=True,
                convert_model_dtype=True,
                local_attn_size=18,
                sink_size=6,
            )
        pipe = self.pipe
        width = 384
        height = max(256, min(640, round(width * image.height / image.width / 16) * 16))
        image = ImageOps.pad(
            image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS
        )
        image.save(output / "input.png")
        progress("encoding_input")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            context = [
                v.to(pipe.device)
                for v in pipe.text_encoder([prompt], torch.device("cpu"))
            ]
            pixels = (
                torch.from_numpy(np.asarray(image).copy())
                .permute(2, 0, 1)
                .float()
                .to(pipe.device)
                / 127.5
                - 1
            )
            max_latents = self.max_steps * LATENTS_PER_STEP
            max_frames = (max_latents - 1) * 4 + 1
            padded = torch.zeros(3, max_frames, height, width, device=pipe.device)
            padded[:, 0] = pixels
            encoded = pipe.vae.encode([padded])[0]
            mask = torch.zeros(
                4, max_latents, height // 8, width // 8, device=pipe.device
            )
            mask[:, 0] = 1
            condition = torch.cat((mask, encoded))
            del pixels, padded, encoded, mask
            tokens = height * width // 256
            config = pipe.model.config
            kv = pipe._initialize_self_kv_cache(
                num_layers=config.num_layers,
                shape=[
                    1,
                    tokens * 18,
                    config.num_heads,
                    config.dim // config.num_heads,
                ],
                dtype=pipe.pipe_dtype,
                device=pipe.device,
            )
            cross = pipe._initialize_crossattn_cache(
                num_layers=config.num_layers,
                shape=[1, 512, config.num_heads, config.dim // config.num_heads],
                dtype=pipe.pipe_dtype,
                device=pipe.device,
            )
        pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=5.0)
        self.state = dict(
            context=context,
            condition=condition,
            kv=kv,
            cross=cross,
            cross_ready=False,
            generator=torch.Generator(device=pipe.device).manual_seed(42),
            timesteps=pipe.scheduler.timesteps[[0, 250, 500, 750]],
            tokens=tokens,
            latents=[],
            pose=np.eye(4),
            width=width,
            height=height,
            steps=0,
            last_video=None,
            output=output,
        )
        if self.decode_mode == "stream":
            from ai.exploration.stream_decode import StreamingVAEDecoder

            self.state["decoder"] = StreamingVAEDecoder(pipe.vae)
        torch.cuda.synchronize()
        return {
            "width": width,
            "height": height,
            "max_steps": self.max_steps,
            "max_duration": max_frames / 16,
            "initialization_seconds": time.perf_counter() - started,
            "source_revision": SOURCE_REVISION,
            "runtime": "persistent-dit-kv-" + self.decode_mode + "-vae-v2",
            "decode_mode": self.decode_mode,
        }

    def step(self, action: str, cancelled, progress) -> dict:
        import torch
        from einops import rearrange
        from wan.utils.cam_utils import compute_relative_poses, get_plucker_embeddings
        from wan.utils.utils import save_video

        pipe, state = self.pipe, self.state
        if state is None or state["steps"] >= self.max_steps:
            raise RuntimeError("No active session or session frame limit reached")
        started = time.perf_counter()
        width, height = state["width"], state["height"]
        offset = sum(z.shape[1] for z in state["latents"])
        magnitude = (
            0.0 if action == "hold" else 5.0 if action.startswith("look_") else 0.25
        )
        requested = CameraAction(
            sequence=state["steps"], action=action, frames=8, magnitude=magnitude
        )
        fractions = (
            np.linspace(0, 1, LATENTS_PER_STEP)
            if offset == 0
            else np.linspace(0, 1, LATENTS_PER_STEP + 1)
        )
        poses = np.stack(
            [state["pose"] @ _local_transform(requested, float(f)) for f in fractions]
        )
        relative = compute_relative_poses(
            torch.from_numpy(poses).float(), framewise=True
        )
        if offset:
            relative = relative[1:]
        focal = width / (2 * math.tan(math.radians(30)))
        intrinsics = torch.tensor(
            [[focal, focal, width / 2, height / 2]], device=pipe.device
        ).repeat(6, 1)
        with torch.inference_mode(), torch.autocast("cuda", dtype=pipe.param_dtype):
            rays = get_plucker_embeddings(
                relative.to(pipe.device), intrinsics, height, width
            )
            rays = rearrange(rays, "f (h a) (w b) c -> 1 (c a b) f h w", a=8, b=8).to(
                pipe.param_dtype
            )
            for local_start in (0, 3):
                if cancelled():
                    raise InterruptedError("Session cancelled; discard mutated caches")
                progress("generating", local_start // 3, 2)
                start = offset + local_start
                x = torch.randn(
                    16,
                    3,
                    height // 8,
                    width // 8,
                    device=pipe.device,
                    generator=state["generator"],
                )
                kwargs = dict(
                    context=state["context"],
                    seq_len=3 * state["tokens"],
                    y=[state["condition"][:, start : start + 3]],
                    dit_cond_dict={
                        "c2ws_plucker_emb": (rays[:, :, local_start : local_start + 3],)
                    },
                    kv_cache=state["kv"],
                    crossattn_cache=state["cross"],
                    current_start=start * state["tokens"],
                    max_attention_size=18 * state["tokens"],
                    frame_seqlen=state["tokens"],
                )
                for index, timestep in enumerate(state["timesteps"]):
                    if cancelled():
                        raise InterruptedError(
                            "Session cancelled; discard mutated caches"
                        )
                    flow = pipe.model(
                        x=[x],
                        t=timestep[None].to(pipe.device),
                        cross_attn_first_call=not state["cross_ready"],
                        **kwargs,
                    )[0]
                    state["cross_ready"] = True
                    clean = pipe._convert_flow_pred_to_x0(
                        flow_pred=flow,
                        xt=x,
                        timestep=timestep,
                        scheduler=pipe.scheduler,
                    )
                    if index < 3:
                        noise = torch.randn(
                            clean.shape,
                            device=pipe.device,
                            dtype=clean.dtype,
                            generator=state["generator"],
                        )
                        x = pipe.scheduler.add_noise(
                            clean, noise, state["timesteps"][index + 1]
                        )
                pipe.model(
                    x=[clean],
                    t=torch.zeros_like(state["timesteps"][:1], device=pipe.device),
                    cross_attn_first_call=False,
                    **kwargs,
                )
                state["latents"].append(clean.detach())
            progress("decoding")
            torch.cuda.synchronize()
            denoise_seconds = time.perf_counter() - started
            decode_started = time.perf_counter()
            if self.decode_mode == "stream":
                new_frames = (
                    state["decoder"]
                    .append(torch.cat(state["latents"][-2:], dim=1))
                    .cpu()
                )
                video = (
                    new_frames
                    if state["last_video"] is None
                    else torch.cat((state["last_video"], new_frames), dim=1)
                )
            else:
                video = pipe.vae.decode([torch.cat(state["latents"], dim=1)])[0].cpu()
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - decode_started
        expected_frames = (offset + 6 - 1) * 4 + 1
        if tuple(video.shape) != (3, expected_frames, height, width) or not bool(
            torch.isfinite(video).all()
        ):
            raise RuntimeError("Invalid decoded frame sequence")
        prefix_error = None
        if state["last_video"] is not None:
            previous = state["last_video"]
            prefix_error = float((video[:, : previous.shape[1]] - previous).abs().max())
            # One normalized 8-bit code difference is allowed for kernel rounding.
            if prefix_error > 2 / 255:
                raise RuntimeError(
                    "Previously displayed prefix changed beyond quantization tolerance"
                )
        if cancelled():
            raise InterruptedError("Cancelled before artifact publication")
        progress("saving")
        sequence = state["steps"]
        target = state["output"] / f"step-{sequence:03d}.mp4"
        save_video(
            tensor=video[None],
            save_file=str(target),
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("Video export failed")
        state.update(last_video=video, pose=poses[-1], steps=sequence + 1)
        return {
            "sequence": sequence,
            "action": action,
            "frames": expected_frames,
            "duration": expected_frames / 16,
            "seconds": time.perf_counter() - started,
            "denoise_seconds": denoise_seconds,
            "decode_seconds": decode_seconds,
            "decode_mode": self.decode_mode,
            "prefix_check_scope": "copied unchanged; equivalence measured separately"
            if self.decode_mode == "stream"
            else "redecoded prefix compared to prior preview",
            "prefix_max_abs_error": prefix_error,
            "latent_offset": offset + 6,
            "camera_pose": poses[-1].tolist(),
            "artifact": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "kv_global_end": int(state["kv"][0]["global_end_index"].item()),
            "kv_cache_identity": id(state["kv"]),
        }

    def close(self):
        self.state = None
        self.pipe = None
        gc.collect()
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
