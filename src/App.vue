<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";

type Workspace = "Library" | "AI Selection" | "Create";

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

const workspaces: Workspace[] = ["Library", "AI Selection", "Create"];
const activeWorkspace = ref<Workspace>("Library");
const developerMode = ref(false);
const selectedFolder = ref<string | null>(null);
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

const statusLabel = computed(() =>
  worker.value.healthy ? "AI worker ready" : "AI worker unavailable",
);

async function refreshWorker() {
  try {
    worker.value = await invoke<WorkerStatus>("worker_health");
  } catch (error) {
    worker.value = {
      ...worker.value,
      healthy: false,
      message: String(error),
    };
  }
}

async function chooseFolder() {
  selectedFolder.value = await invoke<string | null>("pick_photo_folder");
  if (!selectedFolder.value) return;
  indexing.value = true;
  indexingError.value = null;
  embedding.value = null;
  people.value = null;
  searchResult.value = null;
  selectionResult.value = null;
  try {
    album.value = await invoke<AlbumIndexResponse>("index_album", {
      folder: selectedFolder.value,
    });
    embedding.value = await invoke<AlbumEmbeddingResponse>("embed_album", {
      albumId: album.value.album_id,
    });
    people.value = await invoke<PeopleIndexResponse>("index_people", {
      albumId: album.value.album_id,
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
      selectionResult.value = await invoke<SelectionResponse>("create_selection", {
        albumId: album.value.album_id,
        prompt: query,
      });
      searchResult.value = null;
    } else {
      searchResult.value = await invoke<AlbumSearchResponse>("search_album", {
        albumId: album.value.album_id,
        query,
        limit: 20,
      });
      selectionResult.value = null;
    }
  } catch (error) {
    searchError.value = String(error);
  } finally {
    searching.value = false;
  }
}

function looksLikeSelection(query: string) {
  return /(?:选|挑|找|保留)\s*\d+\s*(?:张|幅)|\b\d+\s*(?:photos?|images?|shots?)\b/i.test(query);
}

function thumbnailUrl(photo: PhotoSummary) {
  return `${worker.value.url}${photo.thumbnail_url}`;
}

const visiblePhotos = computed(() => album.value?.photos.filter((photo) => !photo.auto_reject) ?? []);
const rejectedPhotos = computed(() => album.value?.photos.filter((photo) => photo.auto_reject) ?? []);

onMounted(refreshWorker);
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div>
        <p class="eyebrow">Personal photo intelligence</p>
        <h1>Norma</h1>
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
          <span class="nav-index">0{{ workspaces.indexOf(workspace) + 1 }}</span>
        </button>
      </nav>

      <div class="sidebar-footer">
        <label class="toggle-row">
          <span>Developer Mode</span>
          <input v-model="developerMode" type="checkbox" />
        </label>
        <button class="status-card" @click="refreshWorker">
          <span class="status-dot" :class="{ healthy: worker.healthy }" />
          <span>
            <strong>{{ statusLabel }}</strong>
            <small>{{ worker.message }}</small>
          </span>
        </button>
      </div>
    </aside>

    <main>
      <header class="topbar">
        <div>
          <p class="eyebrow">Workspace</p>
          <h2>{{ activeWorkspace }}</h2>
        </div>
        <button class="ghost-button">Me <span class="avatar">N</span></button>
      </header>

      <section v-if="activeWorkspace === 'Library'" class="workspace library">
        <div class="hero-copy">
          <p class="section-kicker">Start with a place you remember</p>
          <h3>Your photos stay where they are.</h3>
          <p>
            Norma reads JPG metadata and creates local thumbnails. Originals are
            never modified, moved, or deleted.
          </p>
          <button class="primary-button" :disabled="indexing" @click="chooseFolder">
            {{ indexing ? "Indexing locally…" : "Choose JPG folder" }}
          </button>
          <p v-if="selectedFolder" class="path-chip">{{ selectedFolder }}</p>
          <p v-if="indexingError" class="error-message">{{ indexingError }}</p>
          <div v-if="album" class="album-stats">
            <span>{{ album.total }} JPGs</span>
            <span>{{ album.similar_groups }} similar groups</span>
            <span>{{ album.duration_ms }} ms</span>
            <span v-if="embedding">semantic index {{ embedding.duration_ms }} ms</span>
            <span v-if="people">{{ people.total_faces }} faces · {{ people.cluster_count }} people groups</span>
          </div>
        </div>

        <div v-if="album" class="library-results">
          <div class="photo-grid" aria-label="Indexed photos">
            <figure v-for="photo in visiblePhotos" :key="photo.id" class="photo-card">
              <img :src="thumbnailUrl(photo)" :alt="photo.filename" />
              <figcaption>
                <span>{{ photo.filename }}</span>
                <small>Q {{ photo.quality_score.toFixed(0) }}</small>
              </figcaption>
            </figure>
          </div>
          <details v-if="rejectedPhotos.length" class="reject-fold">
            <summary>AI suggested exclusions · {{ rejectedPhotos.length }}</summary>
            <div class="photo-grid compact">
              <figure v-for="photo in rejectedPhotos" :key="photo.id" class="photo-card rejected">
                <img :src="thumbnailUrl(photo)" :alt="photo.filename" />
                <figcaption><span>{{ photo.filename }}</span><small>{{ photo.reject_reason }}</small></figcaption>
              </figure>
            </div>
          </details>
        </div>

        <div v-else class="empty-grid" aria-label="Photo library placeholder">
          <div v-for="tile in 6" :key="tile" class="photo-placeholder">
            <span>{{ String(tile).padStart(2, "0") }}</span>
          </div>
        </div>
      </section>

      <section v-else-if="activeWorkspace === 'AI Selection'" class="workspace selection">
        <div class="selection-heading">
          <p class="section-kicker">Grounded selection</p>
          <h3>Describe the collection, not the clicks.</h3>
          <p>Hard constraints remain explicit. Soft taste stays adjustable.</p>
        </div>
        <div class="command-bar">
          <input
            v-model="command"
            aria-label="Ask anything about this album"
            :disabled="!embedding || searching"
            :placeholder="embedding ? 'Try: 选 12 张夜景，质量至少 45…' : 'Import an album first…'"
            @keyup.enter="runSearch"
          />
          <button
            aria-label="Run command"
            :disabled="!embedding || !command.trim() || searching"
            @click="runSearch"
          >{{ searching ? "…" : "↗" }}</button>
        </div>
        <p v-if="searchError" class="error-message">{{ searchError }}</p>
        <div v-if="selectionResult" class="search-results">
          <div class="search-summary">
            <p>
              {{ selectionResult.feasible ? `${selectionResult.selected.length} selected photos` : "Hard constraints are infeasible" }}
            </p>
            <small>{{ selectionResult.solver }} · {{ selectionResult.solver_status }} · {{ selectionResult.duration_ms }} ms</small>
          </div>
          <div class="constraint-row">
            <span>count = {{ selectionResult.constraints.target_count }}</span>
            <span>quality ≥ {{ selectionResult.constraints.min_quality }}</span>
            <span>similar group ≤ {{ selectionResult.constraints.max_per_similarity_group }}</span>
            <span>{{ selectionResult.constraints.exclude_rejects ? "rejects excluded" : "rejects allowed" }}</span>
          </div>
          <p v-for="warning in selectionResult.warnings" :key="warning" class="selection-warning">{{ warning }}</p>
          <div v-if="selectionResult.selected.length" class="photo-grid" aria-label="Optimized collection">
            <figure v-for="photo in selectionResult.selected" :key="photo.photo_id" class="photo-card">
              <img :src="`${worker.url}${photo.thumbnail_url}`" :alt="photo.filename" />
              <figcaption>
                <span>{{ photo.filename }}</span>
                <small>{{ photo.total_score.toFixed(3) }}</small>
              </figcaption>
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
              <img :src="`${worker.url}${match.thumbnail_url}`" :alt="match.filename" />
              <figcaption>
                <span>{{ match.filename }}</span>
                <small>{{ match.score.toFixed(3) }}</small>
              </figcaption>
            </figure>
          </div>
        </div>
        <div v-else class="result-surface">
          <p>{{ embedding ? "Ready for a grounded search" : "No semantic index yet" }}</p>
          <small>{{ embedding ? "Search uses only the current local album." : "Import an album, then ask Norma to find photos." }}</small>
        </div>
        <section v-if="people?.clusters.length" class="people-surface" aria-label="People groups">
          <div class="search-summary">
            <p>People in this album</p>
            <small>{{ people.provider }} · {{ people.total_faces }} faces</small>
          </div>
          <div class="people-grid">
            <article v-for="cluster in people.clusters" :key="cluster.cluster_id" class="person-card">
              <img
                :src="`${worker.url}${cluster.faces[0].thumbnail_url}`"
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
          <p class="section-kicker">After the selection</p>
          <h3>Turn still memories into a short experience.</h3>
        </div>
        <div class="create-options">
          <article><span>01</span><h4>Create 15s Video</h4><p>Directed motion, transitions, and licensed music.</p></article>
          <article><span>02</span><h4>Enter Photo</h4><p>A finite, controlled exploration from one image.</p></article>
          <article><span>03</span><h4>Share</h4><p>A minimal page for the chosen set and generated media.</p></article>
        </div>
      </section>

      <aside v-if="developerMode" class="developer-panel">
        <span>DEV</span>
        <code>{{ JSON.stringify({ worker, album, embedding, people, searchResult, selectionResult }, null, 2) }}</code>
      </aside>
    </main>
  </div>
</template>
