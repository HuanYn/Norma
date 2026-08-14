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


class EmbeddingProviderCapability(BaseModel):
    id: str
    name: str
    dimension: int
    available: bool
    model_backed: bool
    multilingual: str
    active: bool
    install_extra: str | None


class EmbeddingProviderListResponse(BaseModel):
    items: list[EmbeddingProviderCapability]


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
    computed_count: int
    reused_count: int
    rejected: int
    similar_groups: int
    duration_ms: int
    provider: str
    photos: list[PhotoSummary]
    errors: list[str] = Field(default_factory=list)


class AlbumSummary(BaseModel):
    id: str
    name: str
    source_path: str
    created_at: str
    indexed_at: str | None
    photo_count: int
    rejected_count: int
    embedded_count: int
    embedding_provider: str | None
    face_count: int
    selection_count: int


class AlbumListResponse(BaseModel):
    items: list[AlbumSummary]
    total: int
    limit: int
    offset: int


class AlbumPhotoListResponse(BaseModel):
    album: AlbumSummary
    items: list[PhotoSummary]
    total: int
    limit: int
    offset: int


class SelectionHistoryItem(BaseModel):
    id: str
    album_id: str
    prompt: str
    created_at: str
    feasible: bool
    selected_count: int
    solver: str | None
    solver_status: str | None


class SelectionHistoryResponse(BaseModel):
    items: list[SelectionHistoryItem]
    total: int
    limit: int
    offset: int


class AlbumEmbeddingResponse(BaseModel):
    album_id: str
    count: int
    computed_count: int
    reused_count: int
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


class PrepareJobRequest(BaseModel):
    folder: str = Field(min_length=1)
    name: str | None = Field(default=None, max_length=200)
    include_people: bool = True


class JobResponse(BaseModel):
    id: str
    job_type: str
    status: str
    stage: str
    progress: float = Field(ge=0, le=1)
    payload: dict[str, object]
    result: dict[str, object] | None
    error: str | None
    cancel_requested: bool
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int
    limit: int
    offset: int


class SelectionRequest(BaseModel):
    album_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=1000)
    subset_photo_ids: list[str] | None = None


class SelectionConstraints(BaseModel):
    target_count: int
    min_quality: float
    exclude_rejects: bool
    max_per_similarity_group: int


class SelectedPhoto(BaseModel):
    photo_id: str
    filename: str
    thumbnail_url: str
    total_score: float
    semantic_score: float
    preference_score: float = 0.5
    quality_score: float
    similarity_group: str | None
    reasons: list[str]


class SelectionResponse(BaseModel):
    selection_id: str
    album_id: str
    prompt: str
    constraints: SelectionConstraints
    feasible: bool
    candidate_count: int
    solver: str
    solver_status: str
    duration_ms: int
    selected: list[SelectedPhoto]
    warnings: list[str] = Field(default_factory=list)


class PairwiseFeedbackRequest(BaseModel):
    album_id: str = Field(min_length=1)
    preferred_photo_id: str = Field(min_length=1)
    rejected_photo_id: str = Field(min_length=1)
    selection_id: str | None = None
    user_id: str = Field(default="local", min_length=1, max_length=100)


class PreferenceModelResponse(BaseModel):
    feedback_id: str
    user_id: str
    comparisons: int
    probability_before: float
    feature_difference: dict[str, float]
    weights: dict[str, float]


class PreferenceStateResponse(BaseModel):
    user_id: str
    comparisons: int
    weights: dict[str, float]


class SelectionReplacementRequest(BaseModel):
    remove_photo_id: str = Field(min_length=1)


class SelectionReplacementResponse(BaseModel):
    previous_selection_id: str
    replacement_selection_id: str | None
    feasible: bool
    removed_photo_id: str
    replacement: SelectedPhoto | None
    updated_selection: SelectionResponse | None
    duration_ms: int
    explanation: list[str]
