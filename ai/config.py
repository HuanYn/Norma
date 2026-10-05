from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


PreferenceMode = Literal["record-only", "adaptive"]
VLMProviderMode = Literal["openai-compatible", "local"]


def validate_preference_mode(value: str) -> PreferenceMode:
    if value not in {"record-only", "adaptive"}:
        raise ValueError("NORMA_PREFERENCE_MODE must be 'record-only' or 'adaptive'")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    log_level: str
    embedding_provider: str = "openclip-multilingual"
    face_provider: str = "opencv-yunet-sface"
    embedding_device: str = "auto"
    embedding_batch_size: int = 8
    model_cache_root: Path | None = None
    prewarm_embedding: bool = False
    cache_budget_bytes: int | None = None
    vlm_model_path: Path | None = None
    vlm_max_new_tokens: int = 256
    preference_mode: PreferenceMode = "record-only"
    vlm_provider: VLMProviderMode = "openai-compatible"
    vlm_base_url: str = ""
    vlm_model: str = ""
    vlm_api_key: str = field(default="", repr=False)
    vlm_timeout_seconds: int = 60

    def __post_init__(self) -> None:
        validate_preference_mode(self.preference_mode)
        if self.vlm_provider not in {"openai-compatible", "local"}:
            raise ValueError(
                "NORMA_VLM_PROVIDER must be 'openai-compatible' or 'local'"
            )
        if not 1 <= self.vlm_timeout_seconds <= 180:
            raise ValueError("NORMA_VLM_TIMEOUT_SECONDS must be between 1 and 180")

    @property
    def vlm_configured(self) -> bool:
        """Configuration presence only, never a remote authentication probe."""
        return self.vlm_provider == "local" or all(
            value.strip()
            for value in (self.vlm_base_url, self.vlm_model, self.vlm_api_key)
        )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "norma.db"

    @property
    def model_cache_dir(self) -> Path:
        return (self.model_cache_root or (self.data_dir / "models")).resolve()

    @property
    def local_vlm_model_dir(self) -> Path:
        """Explicit local Qwen3-VL directory; this is never treated as a Hub ID."""

        return (
            self.vlm_model_path
            or self.data_dir / "models" / "qwen3-vl" / "Qwen3-VL-2B-Instruct-modelscope"
        ).resolve()


def load_settings() -> Settings:
    model_cache = os.getenv("NORMA_MODEL_CACHE_DIR")
    vlm_model_path = os.getenv("NORMA_VLM_MODEL_PATH")
    vlm_max_new_tokens = int(os.getenv("NORMA_VLM_MAX_NEW_TOKENS", "256"))
    if not 64 <= vlm_max_new_tokens <= 1024:
        raise ValueError("NORMA_VLM_MAX_NEW_TOKENS must be between 64 and 1024")
    cache_budget_gb = os.getenv("NORMA_CACHE_BUDGET_GB")
    cache_budget_bytes = None
    if cache_budget_gb:
        value = float(cache_budget_gb)
        if value <= 0:
            raise ValueError("NORMA_CACHE_BUDGET_GB must be greater than zero")
        cache_budget_bytes = round(value * 1024**3)
    return Settings(
        host=os.getenv("NORMA_HOST", "127.0.0.1"),
        port=int(os.getenv("NORMA_PORT", "8765")),
        data_dir=Path(os.getenv("NORMA_DATA_DIR", ".norma/data")).resolve(),
        log_level=os.getenv("NORMA_LOG_LEVEL", "INFO").upper(),
        embedding_provider=os.getenv(
            "NORMA_EMBEDDING_PROVIDER", "openclip-multilingual"
        ),
        face_provider=os.getenv("NORMA_FACE_PROVIDER", "opencv-yunet-sface"),
        embedding_device=os.getenv("NORMA_EMBEDDING_DEVICE", "auto"),
        embedding_batch_size=int(os.getenv("NORMA_EMBEDDING_BATCH_SIZE", "8")),
        model_cache_root=Path(model_cache).resolve() if model_cache else None,
        prewarm_embedding=os.getenv("NORMA_PREWARM_EMBEDDING", "0").strip().casefold()
        in {"1", "true", "yes", "on"},
        cache_budget_bytes=cache_budget_bytes,
        vlm_model_path=(Path(vlm_model_path).resolve() if vlm_model_path else None),
        vlm_max_new_tokens=vlm_max_new_tokens,
        preference_mode=validate_preference_mode(
            os.getenv("NORMA_PREFERENCE_MODE", "record-only")
        ),
        vlm_provider=os.getenv("NORMA_VLM_PROVIDER", "openai-compatible"),
        vlm_base_url=os.getenv("NORMA_VLM_BASE_URL", ""),
        vlm_model=os.getenv("NORMA_VLM_MODEL", ""),
        vlm_api_key=os.getenv("NORMA_VLM_API_KEY", ""),
        vlm_timeout_seconds=int(os.getenv("NORMA_VLM_TIMEOUT_SECONDS", "60")),
    )
