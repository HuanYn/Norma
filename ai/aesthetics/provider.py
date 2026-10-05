from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


PYIQA_VERSION = "0.1.16"
WEIGHTS_REPOSITORY = "chaofengc/IQA-PyTorch-Weights"
WEIGHTS_REVISION = "0df2df423c65f6a64209309695f3845727431027"
# Full hashes from the upstream Hugging Face LFS manifest, not filename prefixes.
WEIGHTS = {
    "musiq": (
        "musiq_koniq_ckpt-e95806b9.pth",
        "e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa",
    ),
    "musiq-ava": (
        "musiq_ava_ckpt-e8d3f067.pth",
        "e8d3f0671965dfe4301a1f7a4aedc8f49be8fed33b82dd5d116a9eb3ce2bd16b",
    ),
}


class AestheticsProviderUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AssessmentScores:
    technical_quality: float
    aesthetic_quality: float
    technical_model: str = "MUSIQ-KonIQ10k"
    aesthetic_model: str = "MUSIQ-AVA"
    technical_scale: str = "approximately 0..100; higher is better"
    aesthetic_scale: str = "1..10; higher is better"

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.technical_quality, self.aesthetic_quality
        )):
            raise ValueError("assessment model returned non-finite scores")
        if not 1 <= self.aesthetic_quality <= 10:
            raise ValueError("assessment aesthetic score is outside the AVA scale")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class AssessmentProvider(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def status(self) -> dict[str, object]: ...

    def score(self, path: Path) -> AssessmentScores: ...


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PyiqaMusiqProvider:
    """Pinned local MUSIQ checkpoints. No model download or rule-based fallback.

    The AVA checkpoint has a ten-class head, unlike KonIQ's regression head.
    Explicit ``num_class`` is essential when a local checkpoint path is passed.
    """

    def __init__(
        self, model_dir: Path, *, device: str = "cpu", max_side: int = 1024
    ) -> None:
        if device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("aesthetics device must be cpu, cuda, or cuda:0")
        if not 224 <= max_side <= 2048:
            raise ValueError("aesthetics max_side must be between 224 and 2048")
        self.model_dir = model_dir.resolve()
        self.device = device
        self.max_side = max_side
        self._models: dict[str, object] = {}
        self._torch = None
        self._lock = threading.RLock()
        identity = {
            "implementation": "norma-musiq-dual-v1",
            "pyiqa": PYIQA_VERSION,
            "weights": WEIGHTS,
            "preprocess": "exif-transpose-rgb-lanczos-long-edge-no-upscale-v1",
            "max_side": max_side,
            "device": device,
            "dtype": "float32",
        }
        self._fingerprint = "musiq-dual-v1:" + hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def status(self) -> dict[str, object]:
        """A cheap configuration check, not an inference or integrity probe."""
        try:
            version = importlib.metadata.version("pyiqa")
        except importlib.metadata.PackageNotFoundError:
            version = None
        missing = [name for name, _ in WEIGHTS.values()
                   if not (self.model_dir / name).is_file()]
        configured = version == PYIQA_VERSION and not missing
        return {
            "provider": self.fingerprint,
            "model_backed": True,
            "configured": configured,
            "loaded": len(self._models) == 2,
            "readiness": (
                "loaded" if len(self._models) == 2 else
                "weights_present_unverified" if configured else "unavailable"
            ),
            "required_pyiqa_version": PYIQA_VERSION,
            "installed_pyiqa_version": version,
            "missing_weights": missing,
            "device": self.device,
            "max_side": self.max_side,
            "automatic_download": False,
            "personalized": False,
            "license_note": "Research/demo use only after reviewing pyiqa and weight terms; not a commercial clearance.",
        }

    def warmup(self) -> None:
        with self._lock:
            if len(self._models) == 2:
                return
            if not self.status()["configured"]:
                raise AestheticsProviderUnavailableError(
                    "Learned assessment is not configured: install pyiqa==0.1.16 "
                    "and provision both pinned local MUSIQ checkpoints."
                )
            # Verify before passing pickle checkpoints to upstream torch.load.
            for metric, (filename, expected) in WEIGHTS.items():
                try:
                    actual = file_sha256(self.model_dir / filename)
                except OSError:
                    raise AestheticsProviderUnavailableError(
                        f"Local checkpoint is unreadable for {metric}."
                    ) from None
                if actual != expected:
                    raise AestheticsProviderUnavailableError(
                        f"Checkpoint integrity verification failed for {metric}."
                    )
            try:
                import torch
                import pyiqa

                if self.device.startswith("cuda") and not torch.cuda.is_available():
                    raise AestheticsProviderUnavailableError("Requested CUDA is unavailable.")
                loaded = {}
                for metric, (filename, _) in WEIGHTS.items():
                    loaded[metric] = pyiqa.create_metric(
                        metric,
                        device=torch.device(self.device),
                        pretrained=False,
                        pretrained_model_path=str(self.model_dir / filename),
                        num_class=10 if metric == "musiq-ava" else 1,
                    )
                self._models = loaded
                self._torch = torch
            except AestheticsProviderUnavailableError:
                raise
            except Exception:
                # Avoid returning dependency internals/local paths to API clients.
                raise AestheticsProviderUnavailableError(
                    "Learned assessment initialization failed; check the pinned "
                    "pyiqa/PyTorch environment and device compatibility."
                ) from None

    def score(self, path: Path) -> AssessmentScores:
        # One caller at a time: bounded GPU memory and safe shared model reuse.
        with self._lock:
            self.warmup()
            import numpy as np
            from PIL import Image, ImageOps

            with Image.open(path) as original:
                if original.width * original.height > 50_000_000:
                    raise ValueError("assessment image exceeds the 50 megapixel budget")
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
                array = np.asarray(image, dtype=np.float32).copy() / 255.0
            tensor = self._torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(self.device)
            try:
                with self._torch.inference_mode():
                    technical = float(self._models["musiq"](tensor).item())
                    aesthetic = float(self._models["musiq-ava"](tensor).item())
            except Exception:
                raise AestheticsProviderUnavailableError(
                    "Learned assessment inference failed; reduce the image budget "
                    "or check model/device compatibility."
                ) from None
            return AssessmentScores(technical, aesthetic)
