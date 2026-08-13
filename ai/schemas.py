from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    service: str = "norma-ai"
    status: str
    schema_version: int


class CapabilitiesResponse(BaseModel):
    image_types: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg"])
    original_policy: str = "read-only"
    embedding_provider: str = "deterministic-fallback"
    milestones: dict[str, str]

