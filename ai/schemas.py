from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    service: str = "norma-ai"
    status: str
    schema_version: int
    embedding_provider: str
    face_provider: str


class CapabilitiesResponse(BaseModel):
    image_types: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg"])
    original_policy: str = "read-only"
    embedding_provider: str
    milestones: dict[str, str]


class EmbeddingProviderCapability(BaseModel):
    id: str
    name: str
    dimension: int
    available: bool
    model_backed: bool
    default: bool
    baseline: bool
    legacy: bool
    multilingual: str
    query_mode: str | None = None
    active: bool
    install_extra: str | None


class EmbeddingProviderListResponse(BaseModel):
    items: list[EmbeddingProviderCapability]


class EmbeddingProviderStatusResponse(BaseModel):
    provider: str
    dimension: int
    model_backed: bool
    loaded: bool
    device: str | None
    warmup_state: str
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class EmbeddingProviderWarmupResponse(BaseModel):
    provider: str
    dimension: int
    model_backed: bool
    loaded_before: bool
    loaded_after: bool
    device: str | None
    duration_ms: int


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
    quality_score: float | None
    blur_score: float | None
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
    quality_count: int
    rejected_count: int
    similar_group_count: int
    embedded_count: int
    embedding_provider: str | None
    face_count: int
    people_processed_count: int
    people_provider: str | None
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
    user_id: str = Field(default="local", min_length=1, max_length=100)


class SearchMatch(BaseModel):
    photo_id: str
    filename: str
    thumbnail_url: str
    score: float
    quality_score: float | None
    auto_reject: bool
    similarity_group: str | None
    semantic_score: float | None = None
    preference_residual: float = 0.0


class AlbumSearchResponse(BaseModel):
    album_id: str
    mode: str
    provider: str
    matches: list[SearchMatch]
    user_id: str = "local"
    query_text: str | None = None
    algorithm: str = "legacy-cosine-v1"
    preference_model_id: str | None = None
    preference_comparisons: int = 0
    feature_schema: str | None = None
    projection_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AlbumRAGRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=6)
    user_id: str = Field(default="local", min_length=1, max_length=100)


class RAGClaim(BaseModel):
    claim_id: str
    text: str


class RAGCitation(BaseModel):
    claim_id: str
    photo_id: str


class RAGProvenance(BaseModel):
    retrieval_provider_fingerprint: str
    generation_provider_fingerprint: str
    query_digest: str
    candidate_digest: str
    evidence_digest: str


class AlbumRAGResponse(BaseModel):
    """Retrieval + local VLM output with referential citation enforcement only."""

    run_id: str
    album_id: str
    user_id: str
    query: str
    answer: str
    claims: list[RAGClaim]
    citations: list[RAGCitation]
    provenance: RAGProvenance
    retrieval: AlbumSearchResponse
    validation_level: Literal["citation-referential-only"] = "citation-referential-only"
    semantic_entailment_verified: Literal[False] = False


class EvaluationQueryCreateRequest(BaseModel):
    album_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1, max_length=500)
    notes: str | None = Field(default=None, max_length=2000)


class EvaluationQuerySummary(BaseModel):
    id: str
    album_id: str
    query_text: str
    notes: str | None
    judgment_count: int
    relevant_count: int
    created_at: str
    updated_at: str


class EvaluationQueryListResponse(BaseModel):
    items: list[EvaluationQuerySummary]
    total: int


class RelevanceJudgmentInput(BaseModel):
    photo_id: str = Field(min_length=1)
    relevance: int = Field(ge=0, le=3)


class RelevanceJudgmentBatchRequest(BaseModel):
    judgments: list[RelevanceJudgmentInput] = Field(min_length=1, max_length=500)
    annotator: str = Field(default="local", min_length=1, max_length=100)


class RelevanceJudgmentBatchResponse(BaseModel):
    query_id: str
    upserted_count: int
    judgment_count: int
    relevant_count: int


class EvaluationCandidate(BaseModel):
    rank: int
    photo_id: str
    filename: str
    thumbnail_url: str
    score: float
    relevance: int | None
    annotator: str | None


class EvaluationCandidateResponse(BaseModel):
    query: EvaluationQuerySummary
    provider: str
    items: list[EvaluationCandidate]


class EvaluationRunRequest(BaseModel):
    query_ids: list[str] | None = None
    cutoffs: list[int] = Field(default_factory=lambda: [1, 5, 10])


class EvaluationQueryMetrics(BaseModel):
    query_id: str
    query_text: str
    judgment_count: int
    relevant_count: int
    ranked_photo_ids: list[str]
    relevance_by_photo: dict[str, int]
    reciprocal_rank: float
    precision_at: dict[str, float]
    recall_at: dict[str, float]
    ndcg_at: dict[str, float]


class EvaluationRunResponse(BaseModel):
    run_id: str
    album_id: str
    provider: str
    cutoffs: list[int]
    query_count: int
    skipped_query_count: int
    macro_mrr: float
    macro_precision_at: dict[str, float]
    macro_recall_at: dict[str, float]
    macro_ndcg_at: dict[str, float]
    queries: list[EvaluationQueryMetrics]
    created_at: str


class CacheGcRequest(BaseModel):
    dry_run: bool = True
    min_age_seconds: int = Field(default=3600, ge=0, le=31_536_000)


class CacheGcResponse(BaseModel):
    dry_run: bool
    min_age_seconds: int
    scanned_files: int
    referenced_files: int
    orphan_files: int
    orphan_bytes: int
    deleted_files: int
    deleted_bytes: int
    young_orphan_files: int
    skipped_unsafe_files: int
    failed_files: int
    orphan_samples: list[str]
    errors: list[str]


class CacheCategoryUsage(BaseModel):
    files: int
    bytes: int


class CacheUsageResponse(BaseModel):
    data_dir: str
    categories: dict[str, CacheCategoryUsage]
    generated_files: int
    generated_bytes: int
    model_files: int
    model_bytes: int
    database_bytes: int
    total_state_bytes: int
    budget_bytes: int | None
    over_budget: bool
    over_budget_bytes: int


class CacheQuotaRequest(BaseModel):
    budget_bytes: int | None = Field(default=None, ge=1)
    dry_run: bool = True
    min_age_seconds: int = Field(default=3600, ge=0, le=31_536_000)


class CacheQuotaResponse(BaseModel):
    dry_run: bool
    budget_bytes: int
    usage_before: CacheUsageResponse
    collection: CacheGcResponse
    usage_after: CacheUsageResponse
    projected_total_state_bytes: int
    projected_satisfied: bool
    satisfied: bool
    warnings: list[str]


class MaintenanceRunSummary(BaseModel):
    id: str
    operation: str
    status: str
    dry_run: bool
    request: dict[str, object]
    result: dict[str, object] | None
    error: str | None
    created_at: str
    finished_at: str | None


class MaintenanceRunListResponse(BaseModel):
    items: list[MaintenanceRunSummary]
    total: int
    limit: int
    offset: int


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
    computed_count: int
    reused_count: int
    provider: str
    duration_ms: int
    clusters: list[PersonClusterSummary]


class PrepareJobRequest(BaseModel):
    folder: str = Field(min_length=1)
    name: str | None = Field(default=None, max_length=200)
    include_quality: bool = True
    include_embeddings: bool = True
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
    user_id: str = Field(default="local", min_length=1, max_length=100)


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


class CandidateUniverseSummary(BaseModel):
    album_photo_count: int = 0
    subset_photo_count: int = 0
    eligible_photo_count: int = 0
    excluded_reject_count: int = 0
    excluded_quality_count: int = 0
    excluded_group_count: int = 0
    candidate_ids_sha256: str = ""
    source_snapshot_sha256: str = ""
    decision_feature_snapshot_version: str | None = None
    decision_feature_snapshot_sha256: str = ""
    candidate_photo_ids: list[str] = Field(default_factory=list)


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
    user_id: str = "local"
    query_text: str | None = None
    provider_fingerprint: str | None = None
    preference_model_id: str | None = None
    preference_comparisons: int = 0
    algorithm: str = "legacy-fixed-weight-selection-v1"
    feature_schema: str | None = None
    projection_id: str | None = None
    candidate_universe: CandidateUniverseSummary | None = None


class PairwiseFeedbackRequest(BaseModel):
    album_id: str = Field(min_length=1)
    preferred_photo_id: str = Field(min_length=1)
    rejected_photo_id: str = Field(min_length=1)
    selection_id: str | None = Field(default=None, min_length=1)
    user_id: str = Field(default="local", min_length=1, max_length=100)
    choice: Literal["preferred", "tie", "skip", "both_bad"] = "preferred"
    suggestion_id: str | None = Field(default=None, min_length=1)


class PreferenceModelResponse(BaseModel):
    feedback_id: str
    user_id: str
    comparisons: int
    probability_before: float
    feature_difference: dict[str, float]
    weights: dict[str, float]
    choice: Literal["preferred", "tie", "skip", "both_bad"] = "preferred"
    algorithm: str | None = None
    contextual_event_id: str | None = None
    contextual_model_id: str | None = None
    provider_fingerprint: str | None = None
    feature_schema: str | None = None
    contextual_comparisons: int | None = None
    contextual_probability_before: float | None = None
    contextual_trained: bool | None = None
    contextual_diagnostics: dict[str, object] | None = None
    legacy_audit_persisted: bool = True


class PreferenceStateResponse(BaseModel):
    user_id: str
    comparisons: int
    weights: dict[str, float]
    algorithm: str | None = None
    contextual_model_id: str | None = None
    provider_fingerprint: str | None = None
    feature_schema: str | None = None
    contextual_comparisons: int | None = None
    contextual_diagnostics: dict[str, object] | None = None


class PreferencePairSuggestionRequest(BaseModel):
    posterior_samples: int = Field(default=64, ge=2, le=4096)
    shortlist_size: int = Field(default=16, ge=1, le=512)
    exhaustive: bool = False
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    exclude_previous: bool = True


class PreferencePairPhoto(BaseModel):
    photo_id: str
    filename: str
    thumbnail_url: str


class PreferencePairSuggestionResponse(BaseModel):
    suggestion_id: str
    selection_id: str
    album_id: str
    user_id: str
    query_text: str
    left: PreferencePairPhoto
    right: PreferencePairPhoto
    model_id_at_display: str | None
    algorithm: str
    provider_fingerprint: str
    feature_schema: str
    projection_id: str
    acquisition_version: str
    constraint_solver: str
    constraint_violation_count: int = 0
    mode: Literal["shortlist", "exhaustive"]
    current_photo_ids: list[str]
    current_bayes_regret: float
    probability_left_preferred: float
    predictive_entropy: float
    membership_variance: float
    shortlist_score: float
    pdrr: float
    raw_pdrr_estimate: float
    regret_if_left_preferred: float
    regret_if_right_preferred: float
    effective_sample_size_left: float
    effective_sample_size_right: float
    laplace_fallback_left: bool
    laplace_fallback_right: bool
    laplace_fallback_used: bool
    voi_invariant_ok: bool
    eligible_pair_count: int
    evaluated_pair_count: int
    candidate_count: int
    candidate_digest: str
    candidate_source_digest: str
    candidate_feature_digest: str
    requested_posterior_samples: int
    posterior_samples: int
    retry_count: int
    shortlist_size: int
    seed: int
    created_at: str | None = None


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
