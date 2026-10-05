from __future__ import annotations

import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from ai.video.generator import GenerationCancelled, VideoModelUnavailable, WanGenerator
from ai.video.models import VideoOptions


@pytest.fixture
def inference_modules(monkeypatch):
    calls = []
    torch = ModuleType("torch")
    torch.float32 = "fp32"
    torch.bfloat16 = "bf16"
    torch.__version__ = "fixture"
    class OutOfMemoryError(Exception):
        pass
    torch.cuda = SimpleNamespace(
        is_available=lambda: True, is_bf16_supported=lambda: True,
        reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 123,
        empty_cache=lambda: None, OutOfMemoryError=OutOfMemoryError,
    )
    class Seed:
        def __init__(self, device):
            calls.append(("device", device))
        def manual_seed(self, seed):
            calls.append(("seed", seed))
            return self
    torch.Generator = Seed
    class VAE:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("vae_load", args, kwargs))
            return VAE()
        def enable_tiling(self):
            pytest.fail("Wan5B patchified VAE must not use broken diffusers 0.35.1 tiling")
        def disable_tiling(self):
            calls.append(("tiling_disabled",))
    class Pipeline:
        vae = VAE()
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("pipe_load", args, kwargs))
            return Pipeline()
        def enable_model_cpu_offload(self):
            calls.append(("model_offload",))
        def enable_sequential_cpu_offload(self):
            calls.append(("sequential_offload",))
        def maybe_free_model_hooks(self):
            calls.append(("cleanup",))
        def __call__(self, **kwargs):
            calls.append(("generate", kwargs))
            for index in range(kwargs["num_inference_steps"]):
                kwargs["callback_on_step_end"](self, index, None, {})
            return SimpleNamespace(frames=[[kwargs["image"]] * kwargs["num_frames"]])
    diffusers = ModuleType("diffusers")
    diffusers.AutoencoderKLWan = VAE
    diffusers.WanImageToVideoPipeline = Pipeline
    utils = ModuleType("diffusers.utils")
    def export(frames, destination, fps):
        calls.append(("export", len(frames), fps))
    utils.export_to_video = export
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    monkeypatch.setattr("importlib.metadata.version", lambda package: "fixture")
    return calls, torch


def test_i2v_5b_loads_correct_class_and_offloads(tmp_path, inference_modules):
    calls, torch = inference_modules
    generator = WanGenerator("/operator/model", revision="pinned-hash")
    progress = []
    metadata = generator.generate(
        Image.new("RGB", (800, 600)), VideoOptions(prompt="Slow pan", num_inference_steps=4),
        tmp_path / "output.mp4", lambda *args: progress.append(args), threading.Event(),
    )
    vae = next(c for c in calls if c[0] == "vae_load")[2]
    pipe = next(c for c in calls if c[0] == "pipe_load")[2]
    assert vae["torch_dtype"] == "fp32"
    assert pipe["torch_dtype"] == "bf16"
    assert pipe["expand_timesteps"] is True
    assert pipe["image_encoder"] is None and pipe["image_processor"] is None
    assert pipe["local_files_only"] is True and pipe["revision"] == "pinned-hash"
    assert ("model_offload",) in calls and ("cleanup",) in calls
    assert ("tiling_disabled",) in calls
    assert ("export", 49, 24) in calls
    assert metadata["duration_seconds"] == 49 / 24
    assert metadata["vae_tiling"] is False
    request = next(c for c in calls if c[0] == "generate")[1]
    assert request["image"].size == (640, 384)
    assert request["guidance_scale"] == 5.0
    assert progress[0] == ("loading_model", None, None)
    assert ("denoising", 4, 4) in progress
    assert ("decoding", None, None) in progress
    assert progress[-1] == ("exporting", None, None)


def test_cancel_callback_does_not_export_and_cleans_hooks(tmp_path, inference_modules):
    calls, torch = inference_modules
    cancelled = threading.Event()
    def progress(stage, step, total):
        if stage == "denoising":
            cancelled.set()
    with pytest.raises(GenerationCancelled):
        WanGenerator().generate(Image.new("RGB", (500, 500)), VideoOptions(prompt="Pan"),
                                tmp_path / "output.mp4", progress, cancelled)
    assert ("cleanup",) in calls
    assert not any(c[0] == "export" for c in calls)


def test_no_cuda_reports_unavailable_before_model_load(inference_modules):
    calls, torch = inference_modules
    torch.cuda.is_available = lambda: False
    with pytest.raises(VideoModelUnavailable, match="CUDA GPU unavailable"):
        WanGenerator()._load()
    assert not calls


def test_alternative_offload_and_explicit_download(inference_modules):
    calls, torch = inference_modules
    generator = WanGenerator(allow_downloads=True, offload="sequential")
    generator._load()
    generator._load()
    assert sum(c[0] == "pipe_load" for c in calls) == 1
    assert ("sequential_offload",) in calls
    assert next(c for c in calls if c[0] == "pipe_load")[2]["local_files_only"] is False


def test_real_diffusers_0351_patchified_vae_requires_non_tiled_encode():
    """Optional real CPU regression: tiny random VAE, no checkpoint or GPU.

    The three-channel failure reproduces the actual dependency implementation,
    not our injected pipeline fixture. Run in the server's video environment.
    """
    diffusers = pytest.importorskip("diffusers")
    if diffusers.__version__ != "0.35.1":
        pytest.skip("Regression targets the pinned 0.35.1 dependency")
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vae = diffusers.AutoencoderKLWan(
            base_dim=8, decoder_base_dim=8, z_dim=4,
            dim_mult=[1, 2, 4, 4], num_res_blocks=1,
            in_channels=12, out_channels=12, is_residual=True,
            patch_size=2, scale_factor_temporal=4, scale_factor_spatial=16,
            latents_mean=[0.0] * 4, latents_std=[1.0] * 4,
        ).to(device="cpu", dtype=torch.float32)
        image = torch.zeros(1, 3, 1, 32, 32, device="cpu")
        vae.enable_tiling(
            tile_sample_min_height=16, tile_sample_min_width=16,
            tile_sample_stride_height=16, tile_sample_stride_width=16,
        )
        with torch.no_grad(), pytest.raises(RuntimeError, match="12 channels.*3 channels"):
            vae.encode(image)
        vae.disable_tiling()
        with torch.no_grad():
            encoded = vae.encode(image).latent_dist.mode()
        assert tuple(encoded.shape) == (1, 4, 1, 2, 2)
    finally:
        torch.set_num_threads(previous_threads)
