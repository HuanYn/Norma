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
}

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
          <button class="primary-button" @click="chooseFolder">
            Choose JPG folder
          </button>
          <p v-if="selectedFolder" class="path-chip">{{ selectedFolder }}</p>
        </div>

        <div class="empty-grid" aria-label="Photo library placeholder">
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
            placeholder="Ask anything about this album…"
          />
          <button aria-label="Run command">↗</button>
        </div>
        <div class="result-surface">
          <p>No selection yet</p>
          <small>Import an album, then ask Norma to find or select photos.</small>
        </div>
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
        <code>{{ JSON.stringify(worker, null, 2) }}</code>
      </aside>
    </main>
  </div>
</template>

