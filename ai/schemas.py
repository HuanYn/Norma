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


class AlbumIndexRequest(BaseModel):
    folder: str = Field(min_length=1)
    name: str | None = Field(default=None, max_length=200)


class PhotoSummary(BaseModel):
    id: str
    filename: str
    absolute_path: str
    thumbnail_path: str
    thumbnail_url: str
    width: int
    height: int
    file_size: int
    capture_time: str | None
    quality_score: float
    blur_score: float
    similarity_group: str | None
    auto_reject: bool
    reject_reason: str | None
    quality_flags: list[str]


class AlbumIndexResponse(BaseModel):
    album_id: str
    name: str
    source_path: str
    total: int
    rejected: int
    similar_groups: int
    duration_ms: int
    provider: str
    photos: list[PhotoSummary]
    errors: list[str] = Field(default_factory=list)
