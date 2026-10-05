from __future__ import annotations

import gc
from pathlib import Path
from threading import Event
from typing import Callable

from PIL import Image, ImageOps

from .models import VideoOptions

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
Progress = Callable[[str, int | None, int | None], None]


class GenerationCancelled(Exception):
    pass


class VideoModelUnavailable(RuntimeError):
    pass


class WanGenerator:
    """Lazy, sequential offloaded Wan 2.2 5B I2V inference, never a mock video."""

    def __init__(self, model_path: str = MODEL_ID, *, allow_downloads: bool = False,
                 revision: str | None = None, offload: str = "model") -> None:
        if offload not in {"model", "sequential"}:
            raise ValueError("offload must be model or sequential")
        self.model_path = model_path
        self.allow_downloads = allow_downloads
        self.revision = revision
        self.offload = offload
        self._pipe = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            if not torch.cuda.is_available():
                raise VideoModelUnavailable("CUDA GPU unavailable; video generation has not run")
            if not torch.cuda.is_bf16_supported():
                raise VideoModelUnavailable("This worker requires a CUDA GPU supporting bfloat16")
            loading = {"local_files_only": not self.allow_downloads}
            if self.revision:
                loading["revision"] = self.revision
            vae = AutoencoderKLWan.from_pretrained(
                self.model_path, subfolder="vae", torch_dtype=torch.float32, **loading
            )
            # The checkpoint's model_index names WanPipeline (T2V). Explicitly select
            # the I2V class: the 5B branch uses expand_timesteps and no CLIP encoder.
            pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_path, vae=vae, image_encoder=None, image_processor=None,
                torch_dtype=torch.bfloat16, expand_timesteps=True, **loading
            )
            # Diffusers 0.35.1's tiled Wan VAE path skips patchify/unpatchify.
            # TI2V-5B uses patch_size=2 and a 12-channel encoder: tiling sends
            # raw RGB (3 channels) into it and crashes before denoising. Keep
            # the supported non-tiled path; use small previews + CPU offload.
            # Re-enable only after a separately verified dependency upgrade.
            pipe.vae.disable_tiling()
            if self.offload == "sequential":
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
            self._pipe = pipe
            return pipe
        except VideoModelUnavailable:
            raise
        except Exception as exc:
            # Do not expose cache paths, environment values or vendor exceptions
            # through the HTTP API. Operators can inspect installed dependencies.
            raise VideoModelUnavailable(
                "Cannot load Wan I2V. Check video dependencies, CUDA and cached weights; "
                "downloads are disabled unless explicitly enabled"
            ) from exc

    def generate(self, image: Image.Image, options: VideoOptions, output: Path,
                 progress: Progress, cancelled: Event) -> dict:
        import importlib.metadata

        def check_cancel() -> None:
            if cancelled.is_set():
                raise GenerationCancelled()

        check_cancel()
        progress("loading_model", None, None)
        pipe = self._load()
        import torch
        from diffusers.utils import export_to_video

        check_cancel()
        # Fit rather than stretch. The persisted job records the exact center-crop
        # output dimensions; clients should preview this crop before submission.
        frame = ImageOps.fit(image, (options.width, options.height), method=Image.Resampling.LANCZOS)
        progress("encoding", None, None)

        def step_end(pipeline, step_index, timestep, tensors):
            check_cancel()
            progress("denoising", step_index + 1, options.num_inference_steps)
            if step_index + 1 == options.num_inference_steps:
                progress("decoding", None, None)
            return tensors

        try:
            torch.cuda.reset_peak_memory_stats()
            frames = pipe(
                image=frame, prompt=options.prompt, negative_prompt=options.negative_prompt,
                width=options.width, height=options.height, num_frames=options.num_frames,
                num_inference_steps=options.num_inference_steps,
                guidance_scale=options.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(options.seed),
                callback_on_step_end=step_end,
            ).frames[0]
            check_cancel()
            progress("exporting", None, None)
            export_to_video(frames, str(output), fps=24)
            check_cancel()
            return {
                "model_id": MODEL_ID, "model_revision": self.revision,
                "pipeline": "WanImageToVideoPipeline", "fps": 24,
                "frame_count": len(frames), "duration_seconds": len(frames) / 24,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_version": torch.__version__,
                "diffusers_version": importlib.metadata.version("diffusers"),
                "offload": self.offload,
                "vae_tiling": False,
            }
        except torch.cuda.OutOfMemoryError as exc:
            raise VideoModelUnavailable(
                "CUDA out of memory. Reduce dimensions or frames, or configure sequential offload"
            ) from exc
        finally:
            # Callback cancellation bypasses Diffusers' normal end-of-call hooks.
            pipe.maybe_free_model_hooks()
            gc.collect()
            torch.cuda.empty_cache()
