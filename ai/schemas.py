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


class AlbumEmbeddingResponse(BaseModel):
    album_id: str
    count: int
    provider: str
    dimension: int
    duration_ms: int


class AlbumSearchRequest(BaseModel):
    album_id: str = Field(min_length=1)
    query: str | None = None
    reference_photo_id: str | None = None
    limit: int = Field(default=20, ge=1, le=50)
    subset_photo_ids: list[str] | None = None


class SearchMatch(BaseModel):
    photo_id: str
    filename: str
    thumbnail_url: str
    score: float
    quality_score: float
    auto_reject: bool
    similarity_group: str | None


class AlbumSearchResponse(BaseModel):
    album_id: str
    mode: str
    provider: str
    matches: list[SearchMatch]


class FaceSummary(BaseModel):
    face_id: str
    photo_id: str
    box: list[int]
    thumbnail_url: str


class PersonClusterSummary(BaseModel):
    cluster_id: str
    label: str
    faces: list[FaceSummary]


class PeopleIndexResponse(BaseModel):
    album_id: str
    total_faces: int
    cluster_count: int
    provider: str
    duration_ms: int
    clusters: list[PersonClusterSummary]
