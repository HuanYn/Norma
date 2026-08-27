<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type Workspace = "Library" | "AI Selection" | "About";

interface WorkerStatus {
  running: boolean;
  healthy: boolean;
  url: string;
  message: string;
  schema_version?: number;
}

interface PhotoSummary {
  id: string;
  filename: string;
  thumbnail_url: string;
  quality_score: number;
  similarity_group: string | null;
  auto_reject: boolean;
  reject_reason: string | null;
}

interface AlbumIndexResponse {
  album_id: string;
  name: string;
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
  quality_score: number;
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
});
const command = ref("");
const album = ref<AlbumIndexResponse | null>(null);
const embedding = ref<AlbumEmbeddingResponse | null>(null);
const people = ref<PeopleIndexResponse | null>(null);
const searchResult = ref<AlbumSearchResponse | null>(null);
const selectionResult = ref<SelectionResponse | null>(null);
const indexing = ref(false);
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

const statusLabel = computed(() =>
  worker.value.healthy ? "AI worker ready" : "AI worker unavailable",
);

async function refreshWorker() {
  try {
    const health = await api<{ status: string; schema_version: number }>("/health");
    worker.value = {
      running: true,
      healthy: health.status === "ok",
      url: window.location.origin,
      message: "Local Python service and SQLite are ready",
      schema_version: health.schema_version,
    };
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
  if (!folder) return;
  indexing.value = true;
  indexingError.value = null;
  embedding.value = null;
  people.value = null;
  searchResult.value = null;
  selectionResult.value = null;
  resetPreferenceCompare();
  interactionMessage.value = null;
  try {
    album.value = await api<AlbumIndexResponse>("/albums/index", {
      method: "POST",
      body: JSON.stringify({ folder }),
    });
    embedding.value = await api<AlbumEmbeddingResponse>(`/albums/${album.value.album_id}/embed`, {
      method: "POST",
    });
    people.value = await api<PeopleIndexResponse>(`/albums/${album.value.album_id}/people/index`, {
      method: "POST",
    });
  } catch (error) {
    indexingError.value = String(error);
  } finally {
    indexing.value = false;
  }
}

async function runSearch() {
  const query = command.value.trim();
  if (!album.value || !embedding.value || !query || searching.value) return;
  searching.value = true;
  searchError.value = null;
  try {
    if (looksLikeSelection(query)) {
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
  if (!selectionResult.value || feedbackBusy.value) return;
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
  if (selected.length < 2 || feedbackBusy.value) return;
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
  if (!selectionResult.value || !compareMode.value || feedbackBusy.value) return;
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
  if (!compareMode.value || feedbackBusy.value) return;
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
  window.addEventListener("keydown", handleCompareKeydown);
});

onBeforeUnmount(() => window.removeEventListener("keydown", handleCompareKeydown));
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
              :disabled="indexing"
              @keyup.enter="indexFolder"
            />
            <button class="primary-button" :disabled="indexing || !selectedFolder.trim()" @click="indexFolder">
              {{ indexing ? "Preparing locally…" : "Open local folder" }}
            </button>
          </div>
          <p v-if="indexingError" class="error-message">{{ indexingError }}</p>
          <div v-if="album" class="album-stats">
            <span>{{ visiblePhotos.length }} keep</span>
            <span v-if="rejectedPhotos.length">{{ rejectedPhotos.length }} review</span>
            <span>{{ album.similar_groups }} groups</span>
            <span v-if="people">{{ people.cluster_count }} people</span>
            <span v-if="embedding">{{ embedding.provider }}</span>
          </div>
        </div>

        <div v-if="album" class="library-results">
          <div class="photo-grid library-grid" aria-label="Indexed photos">
            <figure v-for="photo in visiblePhotos" :key="photo.id" class="photo-card">
              <img :src="thumbnailUrl(photo)" :alt="photo.filename" />
              <figcaption>
                <span>{{ photo.filename }}</span>
                <small>Q {{ photo.quality_score.toFixed(0) }}</small>
              </figcaption>
            </figure>
          </div>
          <details v-if="rejectedPhotos.length" class="reject-fold">
            <summary><span>Review</span> AI suggested exclusions · {{ rejectedPhotos.length }}</summary>
            <div class="photo-grid compact">
              <figure v-for="photo in rejectedPhotos" :key="photo.id" class="photo-card rejected">
                <img :src="thumbnailUrl(photo)" :alt="photo.filename" />
                <figcaption><span>{{ photo.filename }}</span><small>{{ photo.reject_reason }}</small></figcaption>
              </figure>
            </div>
          </details>
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
              :disabled="!embedding || searching"
              :placeholder="embedding ? '搜索照片，或输入：选 12 张夜景，质量至少 45…' : '先在 Library 打开相册'"
              @keyup.enter="runSearch"
            />
            <button
              aria-label="Run command"
              :disabled="!embedding || !command.trim() || searching"
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
                :disabled="feedbackBusy"
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
                <button :disabled="feedbackBusy" @click="replacePhoto(photo)">Replace</button>
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
          <p>{{ embedding ? "Search, select, compare." : "Open an album to begin." }}</p>
          <small>{{ embedding ? "Try “夜景”, or “选 12 张人像，每个相似组最多 1 张”." : "Your local photo index will appear here." }}</small>
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
        <code>{{ JSON.stringify({ worker, album, embedding, people, searchResult, selectionResult }, null, 2) }}</code>
      </aside>
    </main>
  </div>
</template>
