from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


PreferenceMode = Literal["record-only", "adaptive"]
VLMProviderMode = Literal["openai-compatible", "local"]
VLMThinkingMode = Literal["provider-default", "enabled", "disabled"]


def validate_preference_mode(value: str) -> PreferenceMode:
    if value not in {"record-only", "adaptive"}:
        raise ValueError("NORMA_PREFERENCE_MODE must be 'record-only' or 'adaptive'")
    return value  # type: ignore[return-value]


def validate_vlm_thinking_mode(value: str) -> VLMThinkingMode:
    if not isinstance(value, str) or value not in {
        "provider-default",
        "enabled",
        "disabled",
    }:
        raise ValueError(
            "NORMA_VLM_THINKING_MODE must be 'provider-default', 'enabled', or 'disabled'"
        )
    return value  # type: ignore[return-value]


def _vlm_json_response_format_from_environment() -> bool:
    value = os.getenv("NORMA_VLM_JSON_RESPONSE_FORMAT", "0").strip().casefold()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError("NORMA_VLM_JSON_RESPONSE_FORMAT must be 0, 1, false, or true")
    return value in {"1", "true"}


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
    vlm_thinking_mode: VLMThinkingMode = "provider-default"
    vlm_json_response_format: bool = False
    aesthetics_model_path: Path | None = None
    aesthetics_device: str = "cpu"
    video_base_url: str = ""
    video_worker_token_file: Path | None = None
    web_dist_path: Path | None = None

    def __post_init__(self) -> None:
        validate_preference_mode(self.preference_mode)
        if self.vlm_provider not in {"openai-compatible", "local"}:
            raise ValueError(
                "NORMA_VLM_PROVIDER must be 'openai-compatible' or 'local'"
            )
        if not 1 <= self.vlm_timeout_seconds <= 180:
            raise ValueError("NORMA_VLM_TIMEOUT_SECONDS must be between 1 and 180")
        validate_vlm_thinking_mode(self.vlm_thinking_mode)
        if not isinstance(self.vlm_json_response_format, bool):
            raise ValueError("NORMA_VLM_JSON_RESPONSE_FORMAT must be a boolean")
        if self.aesthetics_device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("NORMA_AESTHETICS_DEVICE must be cpu, cuda, or cuda:0")

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

    @property
    def aesthetics_model_dir(self) -> Path:
        return (self.aesthetics_model_path or self.model_cache_dir / "musiq").resolve()


def load_settings() -> Settings:
    model_cache = os.getenv("NORMA_MODEL_CACHE_DIR")
    vlm_model_path = os.getenv("NORMA_VLM_MODEL_PATH")
    aesthetics_path = os.getenv("NORMA_AESTHETICS_MODEL_DIR")
    video_token_file = os.getenv("NORMA_VIDEO_TOKEN_FILE")
    web_dist_path = os.getenv("NORMA_WEB_DIST")
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
        vlm_thinking_mode=validate_vlm_thinking_mode(
            os.getenv("NORMA_VLM_THINKING_MODE", "provider-default")
        ),
        vlm_json_response_format=_vlm_json_response_format_from_environment(),
        aesthetics_model_path=Path(aesthetics_path).resolve()
        if aesthetics_path
        else None,
        aesthetics_device=os.getenv("NORMA_AESTHETICS_DEVICE", "cpu"),
        video_base_url=os.getenv("NORMA_VIDEO_BASE_URL", ""),
        video_worker_token_file=Path(video_token_file).resolve()
        if video_token_file
        else None,
        web_dist_path=Path(web_dist_path).resolve() if web_dist_path else None,
    )
