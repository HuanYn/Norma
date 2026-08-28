<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type Workspace = "Library" | "AI Selection" | "About";

interface WorkerStatus {
  running: boolean;
  healthy: boolean;
  url: string;
  message: string;
  face_provider: string;
  schema_version?: number;
}

interface PhotoSummary {
  id: string;
  filename: string;
  thumbnail_url: string;
  quality_score: number | null;
  similarity_group: string | null;
  auto_reject: boolean;
  reject_reason: string | null;
}

interface AlbumIndexResponse {
  album_id: string;
  name: string;
  source_path: string;
  total: number;
  computed_count: number;
  reused_count: number;
  rejected: number;
  similar_groups: number;
  duration_ms: number;
  provider: string;
  photos: PhotoSummary[];
  errors: string[];
}

interface AlbumCatalogSummary {
  id: string;
  name: string;
  source_path: string;
  photo_count: number;
  rejected_count: number;
  quality_count: number;
  similar_group_count: number;
  embedded_count: number;
  embedding_provider: string | null;
  face_count: number;
  people_processed_count: number;
  people_provider: string | null;
  selection_count: number;
}

interface AlbumWorkspace extends AlbumIndexResponse {
  quality_count: number;
  embedded_count: number;
  embedding_provider: string | null;
  face_count: number;
  people_processed_count: number;
  people_provider: string | null;
}

interface AlbumEmbeddingResponse {
  album_id: string;
  count: number;
  computed_count: number;
  reused_count: number;
  provider: string;
  dimension: number;
  duration_ms: number;
}

interface SearchMatch {
  photo_id: string;
  filename: string;
  thumbnail_url: string;
  score: number;
  quality_score: number | null;
  auto_reject: boolean;
  similarity_group: string | null;
}

interface AlbumSearchResponse {
  album_id: string;
  mode: "text" | "image";
  provider: string;
  matches: SearchMatch[];
}

interface FaceSummary {
  face_id: string;
  photo_id: string;
  box: number[];
  thumbnail_url: string;
}

interface PersonClusterSummary {
  cluster_id: string;
  label: string;
  faces: FaceSummary[];
}

interface PeopleIndexResponse {
  album_id: string;
  total_faces: number;
  cluster_count: number;
  computed_count: number;
  reused_count: number;
  provider: string;
  duration_ms: number;
  clusters: PersonClusterSummary[];
}

interface PreparePeopleSummary {
  album_id: string;
  total_faces: number;
  cluster_count: number;
  computed_count: number;
  reused_count: number;
  provider: string;
  duration_ms: number;
}

interface PrepareJobResult {
  album?: Omit<AlbumIndexResponse, "photos">;
  embedding?: AlbumEmbeddingResponse;
  people?: PreparePeopleSummary | null;
  indexing_progress?: { completed: number; total: number };
  embedding_progress?: { completed: number; total: number };
  people_progress?: { completed: number; total: number };
}

interface PrepareJobResponse {
  id: string;
  job_type: "prepare_album";
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  stage: string;
  progress: number;
  payload: {
    folder: string;
    name: string | null;
    include_quality?: boolean;
    include_embeddings?: boolean;
    include_people?: boolean;
  };
  result: PrepareJobResult | null;
  error: string | null;
  cancel_requested: boolean;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
}

interface JobListResponse {
  items: PrepareJobResponse[];
  total: number;
  limit: number;
  offset: number;
}

interface AlbumPhotoListResponse {
  items: PhotoSummary[];
  total: number;
  limit: number;
  offset: number;
}

interface SelectionConstraints {
  target_count: number;
  min_quality: number;
  exclude_rejects: boolean;
  max_per_similarity_group: number;
}

interface SelectedPhoto {
  photo_id: string;
  filename: string;
  thumbnail_url: string;
  total_score: number;
  semantic_score: number;
  quality_score: number;
  similarity_group: string | null;
  reasons: string[];
}

interface SelectionResponse {
  selection_id: string;
  album_id: string;
  prompt: string;
  constraints: SelectionConstraints;
  feasible: boolean;
  candidate_count: number;
  solver: string;
  solver_status: string;
  duration_ms: number;
  selected: SelectedPhoto[];
  warnings: string[];
}

interface SelectionReplacementResponse {
  feasible: boolean;
  replacement_selection_id: string | null;
  replacement: SelectedPhoto | null;
  updated_selection: SelectionResponse | null;
  explanation: string[];
}

interface PreferenceModelResponse {
  comparisons: number;
  probability_before: number;
  weights: Record<string, number>;
}

const workspaces: Workspace[] = ["Library", "AI Selection", "About"];
const activeWorkspace = ref<Workspace>("Library");
const developerMode = ref(false);
const selectedFolder = ref("");
const worker = ref<WorkerStatus>({
  running: false,
  healthy: false,
  url: "http://127.0.0.1:8765",
  message: "Checking local AI worker…",
  face_provider: "",
});
const command = ref("");
const album = ref<AlbumWorkspace | null>(null);
const embedding = ref<AlbumEmbeddingResponse | null>(null);
const people = ref<PeopleIndexResponse | null>(null);
const peopleSummary = ref<PreparePeopleSummary | null>(null);
const searchResult = ref<AlbumSearchResponse | null>(null);
const selectionResult = ref<SelectionResponse | null>(null);
const indexing = ref(false);
const prepareJob = ref<PrepareJobResponse | null>(null);
const cancellingPrepare = ref(false);
const loadingPhotos = ref(false);
const photoTotal = ref(0);
const searching = ref(false);
const indexingError = ref<string | null>(null);
const searchError = ref<string | null>(null);
const feedbackBusy = ref(false);
const interactionMessage = ref<string | null>(null);
const compareMode = ref(false);
const compareChampionId = ref<string | null>(null);
const compareCandidateIndex = ref(1);
const compareCompleted = ref(false);
const learnedComparisonCount = ref<number | null>(null);

type AnalysisKind = "import" | "quality" | "embedding" | "people";

const PREPARE_JOB_STORAGE_KEY = "norma.activePrepareJob";
const PHOTO_PAGE_SIZE = 300;
let preparePollTimer: ReturnType<typeof setTimeout> | null = null;
let preparePollGeneration = 0;
let photoLoadGeneration = 0;

const statusLabel = computed(() =>
  worker.value.healthy ? "AI worker ready" : "AI worker unavailable",
);

const preparePercent = computed(() =>
  Math.max(0, Math.min(100, Math.round((prepareJob.value?.progress ?? 0) * 100))),
);

const activeJobKind = computed<AnalysisKind | null>(() =>
  prepareJob.value && !isTerminalJob(prepareJob.value) ? jobKind(prepareJob.value) : null,
);

const qualityReady = computed(
  () => Boolean(album.value?.total) && album.value?.quality_count === album.value?.total,
);

const embeddingReady = computed(
  () => Boolean(album.value?.total) && album.value?.embedded_count === album.value?.total,
);

const peopleProviderMatches = computed(
  () =>
    Boolean(album.value?.people_provider && worker.value.face_provider) &&
    album.value?.people_provider === worker.value.face_provider,
);

const peopleReady = computed(
  () =>
    Boolean(album.value?.total) &&
    album.value?.people_processed_count === album.value?.total &&
    peopleProviderMatches.value,
);

const prepareStageLabel = computed(() => {
  const kind = prepareJob.value ? jobKind(prepareJob.value) : "import";
  const labels: Record<string, string> = {
    queued: "Waiting for the local worker",
    indexing: kind === "quality" ? "Analyzing quality and near-duplicates" : "Reading photos and building previews",
    embedding: "Building semantic search",
    people: "Detecting and grouping people",
    completed: "Album ready",
    cancelled: "Preparation cancelled",
    failed: "Preparation failed",
    interrupted: "Preparation interrupted",
  };
  return labels[prepareJob.value?.stage ?? "queued"] ?? "Preparing locally";
});

const prepareProgressDetail = computed(() => {
  if (prepareJob.value?.cancel_requested) return "Cancel requested · finishing the current photo";
  const result = prepareJob.value?.result;
  const progress =
    prepareJob.value?.stage === "indexing"
      ? result?.indexing_progress
      : prepareJob.value?.stage === "embedding"
        ? result?.embedding_progress
        : prepareJob.value?.stage === "people"
          ? result?.people_progress
          : undefined;
  if (progress?.total) {
    return `${progress.completed.toLocaleString()} / ${progress.total.toLocaleString()} photos`;
  }
  if (prepareJob.value?.stage === "queued") return "The task is safely queued in the background";
  return "Original photos remain untouched";
});

const prepareButtonLabel = computed(() => {
  if (!indexing.value || activeJobKind.value !== "import") return "Open local folder";
  if (prepareJob.value?.cancel_requested) return "Stopping…";
  return `${preparePercent.value}% · ${prepareJob.value?.stage ?? "queued"}`;
});

function analysisCompleted(kind: Exclude<AnalysisKind, "import">) {
  if (!album.value) return 0;
  if (kind === "quality") return album.value.quality_count;
  if (kind === "embedding") return album.value.embedded_count;
  return album.value.people_processed_count;
}

function analysisPercent(kind: Exclude<AnalysisKind, "import">) {
  if (activeJobKind.value === kind) return preparePercent.value;
  if (kind === "people" && !peopleProviderMatches.value) return 0;
  const total = album.value?.total ?? 0;
  return total ? Math.round((analysisCompleted(kind) / total) * 100) : 0;
}

function analysisActionLabel(kind: Exclude<AnalysisKind, "import">) {
  if (activeJobKind.value === kind) {
    if (prepareJob.value?.cancel_requested) return "停止中";
    if (prepareJob.value?.status === "queued") return "排队";
    return `${preparePercent.value}%`;
  }
  const completed = analysisCompleted(kind);
  if (kind === "people" && completed && !peopleProviderMatches.value) return "重新运行";
  if (album.value?.total && completed === album.value.total) return "已完成";
  return completed ? "继续" : "开始";
}

function analysisDetail(kind: Exclude<AnalysisKind, "import">) {
  const total = album.value?.total ?? 0;
  const completed = analysisCompleted(kind);
  if (!total) return "打开相册后可用";
  if (kind === "people" && completed && !peopleProviderMatches.value) {
    return "现有结果与当前人脸模型不一致 · 需重新分析";
  }
  if (completed === total) {
    if (kind === "quality") return `${total.toLocaleString()} 张已评分`;
    if (kind === "embedding") return `${total.toLocaleString()} 张可搜索`;
    return `${total.toLocaleString()} 张已检查 · ${album.value?.face_count ?? 0} 张脸`;
  }
  return completed
    ? `${completed.toLocaleString()} / ${total.toLocaleString()}`
    : kind === "quality"
      ? "质量评分 + 相似归组"
      : kind === "embedding"
        ? "建立语义向量"
        : "检测人脸并聚类";
}

const hasMorePhotos = computed(
  () => Boolean(album.value) && (album.value?.photos.length ?? 0) < photoTotal.value,
);

async function refreshWorker() {
  try {
    const health = await api<{ status: string; schema_version: number; face_provider: string }>(
      "/health",
    );
    worker.value = {
      running: true,
      healthy: health.status === "ok",
      url: window.location.origin,
      message: "Local Python service and SQLite are ready",
      face_provider: health.face_provider,
      schema_version: health.schema_version,
    };
    const albumId = album.value?.album_id;
    if (albumId && peopleReady.value) {
      await loadPeopleGroups(albumId);
    } else if (albumId) {
      people.value = null;
    }
  } catch (error) {
    worker.value = {
      ...worker.value,
      healthy: false,
      message: String(error),
    };
  }
}

async function indexFolder() {
  const folder = selectedFolder.value.trim();
  if (!folder || indexing.value || searching.value || feedbackBusy.value) return;
  clearPreparePoll();
  indexing.value = true;
  indexingError.value = null;
  prepareJob.value = null;
  album.value = null;
  photoLoadGeneration += 1;
  loadingPhotos.value = false;
  photoTotal.value = 0;
  embedding.value = null;
  people.value = null;
  peopleSummary.value = null;
  searchResult.value = null;
  selectionResult.value = null;
  resetPreferenceCompare();
  interactionMessage.value = null;
  try {
    let job: PrepareJobResponse;
    try {
      job = await api<PrepareJobResponse>("/jobs/prepare", {
        method: "POST",
        body: JSON.stringify({
          folder,
          include_quality: false,
          include_embeddings: false,
          include_people: false,
        }),
      });
    } catch (createError) {
      const existing = await findActivePrepareJob(folder).catch(() => null);
      if (!existing) throw createError;
      job = existing;
    }
    beginPreparePolling(job);
  } catch (error) {
    indexing.value = false;
    indexingError.value = String(error);
  }
}

async function startAnalysis(kind: Exclude<AnalysisKind, "import">) {
  if (!album.value || indexing.value || searching.value || feedbackBusy.value) return;
  const flags = {
    include_quality: kind === "quality",
    include_embeddings: kind === "embedding",
    include_people: kind === "people",
  };
  indexing.value = true;
  indexingError.value = null;
  prepareJob.value = null;
  searchResult.value = null;
  selectionResult.value = null;
  resetPreferenceCompare();
  interactionMessage.value = null;
  try {
    let job: PrepareJobResponse;
    try {
      job = await api<PrepareJobResponse>("/jobs/prepare", {
        method: "POST",
        body: JSON.stringify({
          folder: album.value.source_path,
          name: album.value.name,
          ...flags,
        }),
      });
    } catch (createError) {
      const existing = await findActivePrepareJob(album.value.source_path).catch(() => null);
      if (!existing) throw createError;
      job = existing;
    }
    beginPreparePolling(job);
  } catch (error) {
    indexing.value = false;
    indexingError.value = String(error);
  }
}

function beginPreparePolling(job: PrepareJobResponse) {
  clearPreparePoll();
  const generation = preparePollGeneration;
  indexing.value = !isTerminalJob(job);
  prepareJob.value = job;
  rememberPrepareJob(job.id);
  void applyPrepareJob(job, generation).then(() => {
    if (generation !== preparePollGeneration || isTerminalJob(job)) return;
    schedulePreparePoll(job.id, generation);
  }).catch((error) => {
    if (generation !== preparePollGeneration) return;
    indexingError.value = `Task state could not be restored: ${String(error)}`;
    if (isTerminalJob(job)) {
      indexing.value = false;
      forgetPrepareJob();
    } else {
      schedulePreparePoll(job.id, generation);
    }
  });
}

function schedulePreparePoll(jobId: string, generation: number) {
  if (generation !== preparePollGeneration) return;
  preparePollTimer = setTimeout(() => void pollPrepareJob(jobId, generation), 650);
}

async function pollPrepareJob(jobId: string, generation: number) {
  if (generation !== preparePollGeneration) return;
  try {
    const job = await api<PrepareJobResponse>(`/jobs/${jobId}`);
    if (generation !== preparePollGeneration) return;
    indexingError.value = null;
    await applyPrepareJob(job, generation);
    if (generation !== preparePollGeneration || isTerminalJob(job)) return;
  } catch (error) {
    if (generation !== preparePollGeneration) return;
    indexingError.value = `Progress connection interrupted: ${String(error)}`;
  }
  schedulePreparePoll(jobId, generation);
}

async function applyPrepareJob(job: PrepareJobResponse, generation: number) {
  if (generation !== preparePollGeneration) return;
  prepareJob.value = job;
  let loadedCurrentPhotos = false;
  const preparedAlbum = job.result?.album;
  if (preparedAlbum) {
    const switchedAlbum = album.value?.album_id !== preparedAlbum.album_id;
    album.value = switchedAlbum
      ? {
          ...preparedAlbum,
          photos: [],
          quality_count: 0,
          embedded_count: 0,
          embedding_provider: null,
          face_count: 0,
          people_processed_count: 0,
          people_provider: null,
        }
      : { ...album.value!, ...preparedAlbum, photos: album.value?.photos ?? [] };
    photoTotal.value = preparedAlbum.total;
    if (switchedAlbum) {
      try {
        await refreshAlbumSummary(preparedAlbum.album_id);
      } catch (error) {
        indexingError.value = `Album status could not be refreshed: ${String(error)}`;
      }
      if (generation !== preparePollGeneration) return;
      await loadAlbumPhotos(true, preparedAlbum.album_id);
      if (generation !== preparePollGeneration) return;
      loadedCurrentPhotos = true;
    }
  }
  if (job.result?.embedding) embedding.value = job.result.embedding;
  if (job.payload.include_people && job.result?.people) peopleSummary.value = job.result.people;
  if (!isTerminalJob(job)) return;

  if (job.status === "completed" && album.value) {
    try {
      await refreshAlbumSummary(album.value.album_id);
    } catch (error) {
      indexingError.value = `Task completed, but album status could not be refreshed: ${String(error)}`;
    }
    if (generation !== preparePollGeneration) return;
    if (!loadedCurrentPhotos) {
      await loadAlbumPhotos(true, album.value.album_id);
    }
    if (generation !== preparePollGeneration) return;
  }
  if (prepareJob.value?.id !== job.id) return;
  indexing.value = false;
  forgetPrepareJob();
  if (preparePollTimer !== null) {
    clearTimeout(preparePollTimer);
    preparePollTimer = null;
  }
  if (job.status === "failed") {
    indexingError.value = job.error ?? "Local task failed.";
  } else if (job.status === "cancelled") {
    indexingError.value = album.value
      ? "Task cancelled. Previously completed results are still available."
      : "Import cancelled before the album was saved.";
  }
}

async function refreshAlbumSummary(albumId = album.value?.album_id) {
  if (!albumId) return;
  const summary = await api<AlbumCatalogSummary>(`/albums/${albumId}`);
  if (album.value?.album_id !== albumId) return;
  album.value = {
    ...album.value,
    name: summary.name,
    source_path: summary.source_path,
    total: summary.photo_count,
    rejected: summary.rejected_count,
    similar_groups: summary.similar_group_count,
    quality_count: summary.quality_count,
    embedded_count: summary.embedded_count,
    embedding_provider: summary.embedding_provider,
    face_count: summary.face_count,
    people_processed_count: summary.people_processed_count,
    people_provider: summary.people_provider,
  };
  photoTotal.value = summary.photo_count;
  if (summary.photo_count > 0 && summary.embedded_count === summary.photo_count) {
    if (!embedding.value || embedding.value.album_id !== albumId) {
      embedding.value = {
        album_id: albumId,
        count: summary.embedded_count,
        computed_count: 0,
        reused_count: summary.embedded_count,
        provider: summary.embedding_provider ?? "cached",
        dimension: 0,
        duration_ms: 0,
      };
    }
  } else {
    embedding.value = null;
  }
  if (peopleReady.value) {
    await loadPeopleGroups(albumId);
  } else {
    people.value = null;
  }
}

async function loadPeopleGroups(albumId = album.value?.album_id) {
  if (!albumId) return;
  if (album.value?.album_id !== albumId || !peopleReady.value) {
    if (album.value?.album_id === albumId) people.value = null;
    return;
  }
  try {
    const result = await api<PeopleIndexResponse>(`/albums/${albumId}/people`);
    if (album.value?.album_id === albumId && result.provider === worker.value.face_provider) {
      people.value = result;
    } else if (album.value?.album_id === albumId) {
      people.value = null;
    }
  } catch (error) {
    if (album.value?.album_id === albumId) {
      indexingError.value = `People analysis is ready, but groups could not be loaded: ${String(error)}`;
    }
  }
}

async function loadAlbumPhotos(replace = false, albumId = album.value?.album_id) {
  if (!albumId || (!replace && loadingPhotos.value)) return;
  const generation = replace ? ++photoLoadGeneration : photoLoadGeneration;
  const offset = replace ? 0 : (album.value?.photos.length ?? 0);
  loadingPhotos.value = true;
  try {
    const page = await api<AlbumPhotoListResponse>(
      `/albums/${albumId}/photos?limit=${PHOTO_PAGE_SIZE}&offset=${offset}&include_rejects=true&sort=path`,
    );
    if (generation !== photoLoadGeneration || album.value?.album_id !== albumId) return;
    const photos = replace ? page.items : [...album.value.photos, ...page.items];
    album.value = { ...album.value, photos };
    photoTotal.value = page.total;
  } catch (error) {
    if (generation === photoLoadGeneration) {
      indexingError.value = `Album is ready, but previews could not be loaded: ${String(error)}`;
    }
  } finally {
    if (generation === photoLoadGeneration) loadingPhotos.value = false;
  }
}

async function cancelPrepareJob() {
  const job = prepareJob.value;
  if (!job || isTerminalJob(job) || cancellingPrepare.value) return;
  cancellingPrepare.value = true;
  try {
    prepareJob.value = await api<PrepareJobResponse>(`/jobs/${job.id}/cancel`, {
      method: "POST",
    });
  } catch (error) {
    indexingError.value = String(error);
  } finally {
    cancellingPrepare.value = false;
  }
}

async function findActivePrepareJob(folder: string) {
  const jobs = await api<JobListResponse>("/jobs?limit=200&offset=0");
  const key = normalizeFolder(folder);
  return (
    jobs.items.find(
      (job) =>
        !isTerminalJob(job) && normalizeFolder(String(job.payload.folder ?? "")) === key,
    ) ?? null
  );
}

function normalizeFolder(folder: string) {
  return folder.trim().replaceAll("/", "\\").replace(/\\+$/, "").toLocaleLowerCase();
}

function isTerminalJob(job: PrepareJobResponse) {
  return ["completed", "failed", "cancelled"].includes(job.status);
}

function jobKind(job: PrepareJobResponse): AnalysisKind {
  const quality = job.payload.include_quality ?? true;
  const embeddings = job.payload.include_embeddings ?? true;
  const faces = job.payload.include_people ?? true;
  if (!quality && !embeddings && !faces) return "import";
  if (quality && !embeddings && !faces) return "quality";
  if (!quality && embeddings && !faces) return "embedding";
  if (!quality && !embeddings && faces) return "people";
  return "import";
}

function clearPreparePoll() {
  preparePollGeneration += 1;
  if (preparePollTimer !== null) clearTimeout(preparePollTimer);
  preparePollTimer = null;
}

function rememberPrepareJob(jobId: string) {
  try {
    window.localStorage.setItem(PREPARE_JOB_STORAGE_KEY, jobId);
  } catch {
    // The task remains persisted server-side when browser storage is unavailable.
  }
}

function forgetPrepareJob() {
  try {
    window.localStorage.removeItem(PREPARE_JOB_STORAGE_KEY);
  } catch {
    // Nothing else to clean up.
  }
}

async function restorePrepareJob() {
  let jobId: string | null = null;
  try {
    jobId = window.localStorage.getItem(PREPARE_JOB_STORAGE_KEY);
  } catch {
    return;
  }
  if (!jobId) return;
  try {
    const job = await api<PrepareJobResponse>(`/jobs/${jobId}`);
    selectedFolder.value = job.payload.folder;
    beginPreparePolling(job);
  } catch {
    forgetPrepareJob();
  }
}

async function runSearch() {
  const query = command.value.trim();
  if (!album.value || !embeddingReady.value || !query || searching.value || indexing.value) return;
  searching.value = true;
  searchError.value = null;
  try {
    if (looksLikeSelection(query)) {
      if (!qualityReady.value) {
        throw new Error("智能选片需要质量与相似度数据；请先在 Library 点击“质量与相似”。");
      }
      selectionResult.value = await api<SelectionResponse>("/selections", {
        method: "POST",
        body: JSON.stringify({ album_id: album.value.album_id, prompt: query }),
      });
      searchResult.value = null;
      resetPreferenceCompare();
      interactionMessage.value = null;
    } else {
      searchResult.value = await api<AlbumSearchResponse>("/albums/search", {
        method: "POST",
        body: JSON.stringify({ album_id: album.value.album_id, query, limit: 20 }),
      });
      selectionResult.value = null;
      resetPreferenceCompare();
    }
  } catch (error) {
    searchError.value = String(error);
  } finally {
    searching.value = false;
  }
}

async function replacePhoto(photo: SelectedPhoto) {
  if (!selectionResult.value || feedbackBusy.value || indexing.value) return;
  feedbackBusy.value = true;
  searchError.value = null;
  interactionMessage.value = null;
  try {
    const result = await api<SelectionReplacementResponse>(
      `/selections/${selectionResult.value.selection_id}/replace`,
      {
        method: "POST",
        body: JSON.stringify({ remove_photo_id: photo.photo_id }),
      },
    );
    if (result.feasible && result.updated_selection && result.replacement) {
      selectionResult.value = result.updated_selection;
      resetPreferenceCompare();
      interactionMessage.value = `Replaced ${photo.filename} with ${result.replacement.filename}.`;
    } else {
      interactionMessage.value = result.explanation[0] ?? "No valid replacement found.";
    }
  } catch (error) {
    searchError.value = String(error);
  } finally {
    feedbackBusy.value = false;
  }
}

function resetPreferenceCompare() {
  compareMode.value = false;
  compareChampionId.value = null;
  compareCandidateIndex.value = 1;
  compareCompleted.value = false;
}

function startPreferenceCompare() {
  const selected = selectionResult.value?.selected ?? [];
  if (selected.length < 2 || feedbackBusy.value || indexing.value) return;
  compareChampionId.value = selected[0].photo_id;
  compareCandidateIndex.value = 1;
  compareCompleted.value = false;
  compareMode.value = true;
  interactionMessage.value = null;
}

function stopPreferenceCompare() {
  compareMode.value = false;
}

const compareChampion = computed<SelectedPhoto | null>(() => {
  const selected = selectionResult.value?.selected ?? [];
  return selected.find((photo) => photo.photo_id === compareChampionId.value) ?? null;
});

const compareChallenger = computed<SelectedPhoto | null>(() =>
  selectionResult.value?.selected[compareCandidateIndex.value] ?? null,
);

const compareLeft = computed<SelectedPhoto | null>(() =>
  compareCandidateIndex.value % 2 === 1 ? compareChampion.value : compareChallenger.value,
);

const compareRight = computed<SelectedPhoto | null>(() =>
  compareCandidateIndex.value % 2 === 1 ? compareChallenger.value : compareChampion.value,
);

const compareRoundTotal = computed(() =>
  Math.max(0, (selectionResult.value?.selected.length ?? 0) - 1),
);

async function choosePreference(preferred: SelectedPhoto, rejected: SelectedPhoto) {
  if (!selectionResult.value || !compareMode.value || feedbackBusy.value || indexing.value) return;
  feedbackBusy.value = true;
  searchError.value = null;
  try {
    const result = await api<PreferenceModelResponse>("/feedback/pairwise", {
      method: "POST",
      body: JSON.stringify({
        album_id: selectionResult.value.album_id,
        preferred_photo_id: preferred.photo_id,
        rejected_photo_id: rejected.photo_id,
        selection_id: selectionResult.value.selection_id,
      }),
    });
    learnedComparisonCount.value = result.comparisons;
    compareChampionId.value = preferred.photo_id;
    if (compareCandidateIndex.value >= selectionResult.value.selected.length - 1) {
      compareMode.value = false;
      compareCompleted.value = true;
      interactionMessage.value = `${preferred.filename} wins this preference round · ${result.comparisons} comparisons learned.`;
    } else {
      compareCandidateIndex.value += 1;
    }
  } catch (error) {
    searchError.value = String(error);
  } finally {
    feedbackBusy.value = false;
  }
}

function handleCompareKeydown(event: KeyboardEvent) {
  if (!compareMode.value || feedbackBusy.value || indexing.value) return;
  const target = event.target as HTMLElement | null;
  if (target?.matches("input, textarea, select")) return;
  if (event.key === "Escape") {
    event.preventDefault();
    stopPreferenceCompare();
    return;
  }
  if (event.key === "ArrowLeft" && compareLeft.value && compareRight.value) {
    event.preventDefault();
    void choosePreference(compareLeft.value, compareRight.value);
  }
  if (event.key === "ArrowRight" && compareLeft.value && compareRight.value) {
    event.preventDefault();
    void choosePreference(compareRight.value, compareLeft.value);
  }
}

function looksLikeSelection(query: string) {
  return /(?:选|挑|找|保留)\s*\d+\s*(?:张|幅)|\b\d+\s*(?:photos?|images?|shots?)\b/i.test(query);
}

function thumbnailUrl(photo: PhotoSummary) {
  return photo.thumbnail_url;
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init.body ? { "Content-Type": "application/json", ...init.headers } : init.headers,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail ?? payload.error ?? `Request failed: ${response.status}`);
  }
  return payload as T;
}

const visiblePhotos = computed(() => album.value?.photos.filter((photo) => !photo.auto_reject) ?? []);
const rejectedPhotos = computed(() => album.value?.photos.filter((photo) => photo.auto_reject) ?? []);

onMounted(() => {
  void refreshWorker();
  void restorePrepareJob();
  window.addEventListener("keydown", handleCompareKeydown);
});

onBeforeUnmount(() => {
  clearPreparePoll();
  window.removeEventListener("keydown", handleCompareKeydown);
});
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand-lockup">
        <span class="brand-mark">N</span>
        <div>
          <h1>Norma</h1>
          <small>Local photo intelligence</small>
        </div>
      </div>

      <nav aria-label="Workspaces">
        <button
          v-for="workspace in workspaces"
          :key="workspace"
          class="nav-item"
          :class="{ active: activeWorkspace === workspace }"
          @click="activeWorkspace = workspace"
        >
          <span>{{ workspace }}</span>
        </button>
      </nav>

      <div class="sidebar-footer">
        <label class="toggle-row" title="Developer mode">
          <span>DEV</span>
          <input v-model="developerMode" type="checkbox" />
        </label>
        <button class="status-card" :title="worker.message" @click="refreshWorker">
          <span class="status-dot" :class="{ healthy: worker.healthy }" />
          <strong>{{ statusLabel }}</strong>
        </button>
      </div>
    </aside>

    <main>
      <section
        v-if="activeWorkspace === 'Library'"
        class="workspace library"
        :class="{ 'has-album': album }"
      >
        <div class="library-toolbar">
          <div class="toolbar-title">
            <p class="section-kicker">{{ album ? album.name : "Local library" }}</p>
            <h3>{{ album ? `${album.total} photos` : "Open a photo folder" }}</h3>
            <p v-if="!album">Originals stay untouched. Analysis and previews remain on this computer.</p>
          </div>
          <div class="folder-input">
            <input
              v-model="selectedFolder"
              aria-label="Local JPG folder path"
              placeholder="C:\Users\you\Pictures\Trip"
              :disabled="indexing || searching || feedbackBusy"
              @keyup.enter="indexFolder"
            />
            <button class="primary-button" :disabled="indexing || searching || feedbackBusy || !selectedFolder.trim()" @click="indexFolder">
              {{ prepareButtonLabel }}
            </button>
          </div>
          <div
            v-if="indexing && prepareJob && (!album || activeJobKind === 'import')"
            class="prepare-progress"
            role="status"
            aria-live="polite"
          >
            <div class="prepare-progress-heading">
              <strong>{{ prepareStageLabel }}</strong>
              <span>{{ preparePercent }}%</span>
            </div>
            <div
              class="prepare-progress-track"
              role="progressbar"
              aria-label="Album preparation progress"
              aria-valuemin="0"
              aria-valuemax="100"
              :aria-valuenow="preparePercent"
            >
              <span :style="{ width: `${preparePercent}%` }" />
            </div>
            <div class="prepare-progress-footer">
              <small>{{ prepareProgressDetail }}</small>
              <button
                :disabled="prepareJob.cancel_requested || cancellingPrepare"
                @click="cancelPrepareJob"
              >{{ prepareJob.cancel_requested ? "Stopping…" : "Cancel" }}</button>
            </div>
          </div>
          <p v-if="indexingError" class="error-message" role="alert">{{ indexingError }}</p>
          <div v-if="album" class="analysis-strip" aria-label="按需分析" aria-live="polite">
            <div class="analysis-item" :class="{ running: activeJobKind === 'quality', ready: qualityReady }">
              <button
                class="analysis-main"
                :disabled="indexing || searching || feedbackBusy || !worker.healthy"
                @click="startAnalysis('quality')"
              >
                <span><strong>质量与相似</strong><small>{{ analysisDetail('quality') }}</small></span>
                <b>{{ analysisActionLabel('quality') }}</b>
                <span
                  class="analysis-progress"
                  role="progressbar"
                  aria-label="质量与相似度分析进度"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  :aria-valuenow="analysisPercent('quality')"
                ><i :style="{ width: `${analysisPercent('quality')}%` }" /></span>
              </button>
              <button
                v-if="activeJobKind === 'quality'"
                class="analysis-cancel"
                :disabled="prepareJob?.cancel_requested || cancellingPrepare"
                aria-label="取消质量分析"
                @click="cancelPrepareJob"
              >×</button>
            </div>

            <div class="analysis-item" :class="{ running: activeJobKind === 'embedding', ready: embeddingReady }">
              <button
                class="analysis-main"
                :disabled="indexing || searching || feedbackBusy || !worker.healthy"
                @click="startAnalysis('embedding')"
              >
                <span><strong>语义索引</strong><small>{{ analysisDetail('embedding') }}</small></span>
                <b>{{ analysisActionLabel('embedding') }}</b>
                <span
                  class="analysis-progress"
                  role="progressbar"
                  aria-label="语义索引进度"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  :aria-valuenow="analysisPercent('embedding')"
                ><i :style="{ width: `${analysisPercent('embedding')}%` }" /></span>
              </button>
              <button
                v-if="activeJobKind === 'embedding'"
                class="analysis-cancel"
                :disabled="prepareJob?.cancel_requested || cancellingPrepare"
                aria-label="取消语义索引"
                @click="cancelPrepareJob"
              >×</button>
            </div>

            <div class="analysis-item" :class="{ running: activeJobKind === 'people', ready: peopleReady }">
              <button
                class="analysis-main"
                :disabled="indexing || searching || feedbackBusy || !worker.healthy"
                @click="startAnalysis('people')"
              >
                <span><strong>人脸分组</strong><small>{{ analysisDetail('people') }}</small></span>
                <b>{{ analysisActionLabel('people') }}</b>
                <span
                  class="analysis-progress"
                  role="progressbar"
                  aria-label="人脸分组进度"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  :aria-valuenow="analysisPercent('people')"
                ><i :style="{ width: `${analysisPercent('people')}%` }" /></span>
              </button>
              <button
                v-if="activeJobKind === 'people'"
                class="analysis-cancel"
                :disabled="prepareJob?.cancel_requested || cancellingPrepare"
                aria-label="取消人脸分组"
                @click="cancelPrepareJob"
              >×</button>
            </div>
          </div>
        </div>

        <div v-if="album" class="library-results">
          <div class="photo-grid library-grid" aria-label="Indexed photos">
            <figure v-for="photo in visiblePhotos" :key="photo.id" class="photo-card">
              <img :src="thumbnailUrl(photo)" :alt="photo.filename" />
              <figcaption>
                <span>{{ photo.filename }}</span>
                <small>Q {{ photo.quality_score === null ? "—" : photo.quality_score.toFixed(0) }}</small>
              </figcaption>
            </figure>
          </div>
          <details v-if="qualityReady && rejectedPhotos.length" class="reject-fold">
            <summary>
              <span>Review</span>
              AI suggested exclusions · {{ rejectedPhotos.length }} loaded / {{ album.rejected }} total
            </summary>
            <div class="photo-grid compact">
              <figure v-for="photo in rejectedPhotos" :key="photo.id" class="photo-card rejected">
                <img :src="thumbnailUrl(photo)" :alt="photo.filename" />
                <figcaption><span>{{ photo.filename }}</span><small>{{ photo.reject_reason }}</small></figcaption>
              </figure>
            </div>
          </details>
          <div v-if="hasMorePhotos" class="load-more-row">
            <button :disabled="loadingPhotos" @click="loadAlbumPhotos(false)">
              {{ loadingPhotos ? "Loading previews…" : `Load ${Math.min(PHOTO_PAGE_SIZE, photoTotal - album.photos.length)} more` }}
            </button>
            <small>{{ album.photos.length.toLocaleString() }} / {{ photoTotal.toLocaleString() }} loaded</small>
          </div>
        </div>

        <div v-else class="empty-grid" aria-label="Photo library placeholder">
          <div v-for="tile in 8" :key="tile" class="photo-placeholder">
            <span>{{ String(tile).padStart(2, "0") }}</span>
          </div>
        </div>
      </section>

      <section v-else-if="activeWorkspace === 'AI Selection'" class="workspace selection">
        <div class="selection-toolbar">
          <div class="toolbar-title">
            <p class="section-kicker">AI selection</p>
            <h3>{{ album?.name ?? "No album open" }}</h3>
          </div>
          <div class="command-bar">
            <input
              v-model="command"
              aria-label="Ask anything about this album"
              :disabled="!embeddingReady || searching || indexing"
              :placeholder="embeddingReady ? '搜索照片，或输入：选 12 张夜景，质量至少 45…' : '先在 Library 点击“语义索引”'"
              @keyup.enter="runSearch"
            />
            <button
              aria-label="Run command"
              :disabled="!embeddingReady || !command.trim() || searching || indexing"
              @click="runSearch"
            >{{ searching ? "…" : "Search" }}</button>
          </div>
        </div>
        <p v-if="searchError" class="error-message">{{ searchError }}</p>
        <div v-if="selectionResult" class="search-results">
          <div class="search-summary">
            <p>
              {{ selectionResult.feasible ? `${selectionResult.selected.length} selected photos` : "Hard constraints are infeasible" }}
            </p>
            <div class="selection-summary-actions">
              <small>{{ selectionResult.solver }} · {{ selectionResult.solver_status }} · {{ selectionResult.duration_ms }} ms</small>
              <button
                v-if="selectionResult.selected.length > 1 && !compareMode"
                class="compare-launch"
                :disabled="feedbackBusy || indexing"
                @click="startPreferenceCompare"
              >{{ compareCompleted ? "Compare again" : "A/B preference" }}</button>
            </div>
          </div>
          <div v-if="!compareMode" class="constraint-row">
            <span>count = {{ selectionResult.constraints.target_count }}</span>
            <span>quality ≥ {{ selectionResult.constraints.min_quality }}</span>
            <span>similar group ≤ {{ selectionResult.constraints.max_per_similarity_group }}</span>
            <span>{{ selectionResult.constraints.exclude_rejects ? "rejects excluded" : "rejects allowed" }}</span>
            <span v-if="learnedComparisonCount !== null">{{ learnedComparisonCount }} preferences learned</span>
          </div>
          <p v-for="warning in selectionResult.warnings" :key="warning" class="selection-warning">{{ warning }}</p>
          <p v-if="interactionMessage" class="interaction-message" role="status" aria-live="polite">{{ interactionMessage }}</p>

          <section
            v-if="compareMode && compareLeft && compareRight"
            class="preference-arena"
            aria-label="A/B photo preference comparison"
          >
            <header class="preference-arena-header">
              <div>
                <span>Preference arena</span>
                <strong>Round {{ compareCandidateIndex }} / {{ compareRoundTotal }}</strong>
              </div>
              <p>Choose the photo you prefer. The winner meets the next challenger.</p>
              <button class="arena-exit" :disabled="feedbackBusy" @click="stopPreferenceCompare">Exit</button>
            </header>

            <div class="preference-stage" :aria-busy="feedbackBusy">
              <button
                class="preference-choice left"
                :disabled="feedbackBusy"
                :aria-label="`Prefer ${compareLeft.filename}`"
                @click="choosePreference(compareLeft, compareRight)"
              >
                <img :src="compareLeft.thumbnail_url" :alt="compareLeft.filename" />
                <span class="choice-key">←</span>
                <span class="choice-caption">
                  <strong>{{ compareLeft.filename }}</strong>
                  <small>Q {{ compareLeft.quality_score.toFixed(0) }} · score {{ compareLeft.total_score.toFixed(3) }}</small>
                </span>
              </button>
              <button
                class="preference-choice right"
                :disabled="feedbackBusy"
                :aria-label="`Prefer ${compareRight.filename}`"
                @click="choosePreference(compareRight, compareLeft)"
              >
                <img :src="compareRight.thumbnail_url" :alt="compareRight.filename" />
                <span class="choice-key">→</span>
                <span class="choice-caption">
                  <strong>{{ compareRight.filename }}</strong>
                  <small>Q {{ compareRight.quality_score.toFixed(0) }} · score {{ compareRight.total_score.toFixed(3) }}</small>
                </span>
              </button>
            </div>

            <footer class="preference-arena-footer">
              <span><kbd>←</kbd> prefer left</span>
              <span>{{ feedbackBusy ? "Saving preference…" : "Click a photo or use the arrow keys" }}</span>
              <span><kbd>→</kbd> prefer right</span>
            </footer>
          </section>

          <div v-else-if="selectionResult.selected.length" class="photo-grid" aria-label="Optimized collection">
            <figure v-for="photo in selectionResult.selected" :key="photo.photo_id" class="photo-card">
              <img :src="photo.thumbnail_url" :alt="photo.filename" />
              <figcaption>
                <span>{{ photo.filename }}</span>
                <small>{{ photo.total_score.toFixed(3) }}</small>
              </figcaption>
              <div class="photo-actions">
                <button :disabled="feedbackBusy || indexing" @click="replacePhoto(photo)">Replace</button>
              </div>
              <details class="photo-reasons">
                <summary>Why this photo</summary>
                <ul><li v-for="reason in photo.reasons" :key="reason">{{ reason }}</li></ul>
              </details>
            </figure>
          </div>
        </div>
        <div v-else-if="searchResult" class="search-results">
          <div class="search-summary">
            <p>{{ searchResult.matches.length }} semantic matches</p>
            <small>{{ searchResult.provider }} · cosine similarity</small>
          </div>
          <div class="photo-grid" aria-label="Semantic search results">
            <figure v-for="match in searchResult.matches" :key="match.photo_id" class="photo-card">
              <img :src="match.thumbnail_url" :alt="match.filename" />
              <figcaption>
                <span>{{ match.filename }}</span>
                <small>{{ match.score.toFixed(3) }}</small>
              </figcaption>
            </figure>
          </div>
        </div>
        <div v-else class="result-surface">
          <p>{{ embeddingReady ? "Search, select, compare." : album ? "先建立语义索引" : "Open an album to begin." }}</p>
          <small>{{ embeddingReady ? "Try “夜景”, or “选 12 张人像，每个相似组最多 1 张”." : album ? "返回 Library，点击“语义索引”；需要智能选片时也请先运行质量分析。" : "Your local photo index will appear here." }}</small>
        </div>
        <section v-if="people?.clusters.length" class="people-surface" aria-label="People groups">
          <div class="search-summary">
            <p>People in this album</p>
            <small>
              {{ people.provider }} · {{ people.total_faces }} faces ·
              {{ people.computed_count }} new / {{ people.reused_count }} reused
            </small>
          </div>
          <div class="people-grid">
            <article v-for="cluster in people.clusters" :key="cluster.cluster_id" class="person-card">
              <img
                :src="cluster.faces[0].thumbnail_url"
                :alt="cluster.label"
              />
              <span>{{ cluster.label }}</span>
              <small>{{ cluster.faces.length }} photo{{ cluster.faces.length === 1 ? "" : "s" }}</small>
            </article>
          </div>
        </section>
      </section>

      <section v-else class="workspace create">
        <div class="selection-heading">
          <p class="section-kicker">Local by design</p>
          <h3>A private photo workspace in your browser.</h3>
        </div>
        <div class="create-options">
          <article><span>01</span><h4>Local originals</h4><p>Norma reads your folder without moving, deleting, or uploading original photos.</p></article>
          <article><span>02</span><h4>Grounded AI</h4><p>Search and selection run against the indexed album, with visible constraints and scores.</p></article>
          <article><span>03</span><h4>Learn your taste</h4><p>Pairwise choices are stored locally and refine later selections.</p></article>
        </div>
      </section>

      <aside v-if="developerMode" class="developer-panel">
        <span>DEV</span>
        <code>{{ JSON.stringify({ worker, prepareJob, album, embedding, peopleSummary, people, searchResult, selectionResult }, null, 2) }}</code>
      </aside>
    </main>
  </div>
</template>
