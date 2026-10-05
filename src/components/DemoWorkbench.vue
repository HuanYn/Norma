<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";

type Photo = { photo_id: string; filename: string; thumbnail_url: string };
type Job = {
  id: string; status: string; stage: string; error?: string | null;
  progress?: number; step_percent?: number | null; step?: number | null;
  total_steps?: number | null; cancel_requested?: boolean;
};
type Score = { photo_id: string; technical_quality: number; aesthetic_quality: number };
type RemoteStatus = { configured: boolean; reachable: boolean; error?: string; worker?: unknown };
type Lane = "quality" | "video";
type SavedJob = { id: string; savedAt: number; filename?: string; contextId?: string };
type HistoricalVideo = { job: Job; filename: string; savedAt: number };
class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

const props = defineProps<{ albumId?: string; selectedPhotos: Photo[]; busy: boolean; selectionContextId?: string }>();
const expanded = ref(false);
const pickedId = ref("");
const consent = ref(false);
const prompt = ref("Slow cinematic camera push in. Preserve the subject, composition and natural lighting.");
const qualityJob = ref<Job | null>(null);
const videoJob = ref<Job | null>(null);
const scores = ref<Score[]>([]);
const qualityError = ref("");
const videoError = ref("");
const qualityPending = ref(false);
const videoPending = ref(false);
const qualityCancelling = ref(false);
const videoCancelling = ref(false);
const remote = ref<RemoteStatus | null>(null);
const checking = ref(false);
const videoFilename = ref("");
const historicalVideos = ref<HistoricalVideo[]>([]);
const historyPending = ref<Record<string, boolean>>({});
const historyError = ref("");
const pollPaused = ref({ quality: false, video: false });

let generation = 0;
const controllers = new Set<AbortController>();
const timers: Partial<Record<Lane, ReturnType<typeof setTimeout>>> = {};
const deadlines = { quality: 0, video: 0 };
const failures = { quality: 0, video: 0 };
const pollCycles = { quality: 0, video: 0 };
const terminal = new Set(["completed", "failed", "cancelled", "interrupted"]);
const active = (job: Job | null) => !!job && !terminal.has(job.status);
const chosen = computed(() => props.selectedPhotos.find(photo => photo.photo_id === pickedId.value));
const qualityActive = computed(() => active(qualityJob.value));
const videoActive = computed(() => active(videoJob.value));
const visibleScores = computed(() => {
  const selected = new Set(props.selectedPhotos.map(photo => photo.photo_id));
  return scores.value.filter(score => selected.has(score.photo_id));
});
const qualityPercent = computed(() => Math.round(Math.max(0, Math.min(1, qualityJob.value?.progress ?? 0)) * 100));
const canGenerate = computed(() => !!chosen.value && !!props.albumId && consent.value
  && !!prompt.value.trim() && !props.busy && !videoPending.value && !videoActive.value
  && remote.value?.reachable === true);
const videoUrl = computed(() => videoJob.value?.status === "completed"
  ? `/demo/video/jobs/${encodeURIComponent(videoJob.value.id)}/artifact` : "");

const stageNames: Record<string, string> = {
  queued: "排队中", running: "处理中", starting: "准备任务", loading_model: "加载模型",
  encoding: "编码图片与提示词", denoising: "生成画面", decoding: "解码视频",
  exporting: "导出视频", completed: "完成", failed: "失败", cancelled: "已取消",
  interrupted: "任务中断", scoring: "逐张评估", loading_models: "加载评估模型",
  validating_models_and_sources: "检查模型与照片", analyzing_aesthetics: "逐张评估",
};
const stageLabel = (job: Job | null) => !job ? "" : job.stage.startsWith("scoring_images:")
  ? `逐张评估 ${job.stage.slice("scoring_images:".length)}` : stageNames[job.stage] ?? job.stage;
const scoreMean = (field: "technical_quality" | "aesthetic_quality") => {
  const values = visibleScores.value.map(score => score[field]).filter(Number.isFinite);
  return values.length ? (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(2) : "—";
};

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  controllers.add(controller);
  const timeout = setTimeout(() => controller.abort(), 35000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const body = await response.json();
    if (!response.ok) throw new ApiError(typeof body.detail === "string"
      ? body.detail : `请求失败（HTTP ${response.status}）`, response.status);
    return body as T;
  } finally {
    clearTimeout(timeout);
    controllers.delete(controller);
  }
}
const post = <T>(path: string, body?: unknown) => request<T>(path, {
  method: "POST", headers: { "Content-Type": "application/json" },
  ...(body === undefined ? {} : { body: JSON.stringify(body) }),
});
const errorText = (failure: unknown) => failure instanceof Error
  ? failure.name === "AbortError" ? "请求已暂停或超时，请刷新状态；服务端任务不一定已取消。" : failure.message
  : "请求失败，请稍后重试。";
const key = (lane: Lane) => `norma-demo-${lane}:${props.albumId}`;
const historyKey = () => `norma-demo-video-history:${props.albumId}`;

function rememberHistory() {
  try { sessionStorage.setItem(historyKey(), JSON.stringify(historicalVideos.value)); }
  catch { /* Keep the current page's history even when storage is unavailable. */ }
}

function archiveVideo(job: Job, filename: string) {
  const item = { job, filename, savedAt: Date.now() };
  const existing = historicalVideos.value.findIndex(value => value.job.id === job.id);
  if (existing >= 0) historicalVideos.value[existing] = item;
  else historicalVideos.value.unshift(item);
  rememberHistory();
}

function detachPreviousVideo() {
  stopPolling("video");
  if (videoJob.value) archiveVideo(videoJob.value, videoFilename.value || "此前提交的照片");
  videoJob.value = null;
  videoFilename.value = "";
  videoError.value = "";
  consent.value = false;
  pollPaused.value.video = false;
  try { sessionStorage.removeItem(key("video")); } catch { /* Optional storage. */ }
}

async function updateHistoricalVideo(id: string, cancelRequested = false) {
  const entry = historicalVideos.value.find(value => value.job.id === id);
  if (!entry || historyPending.value[id]) return;
  const current = generation;
  historyPending.value[id] = true;
  historyError.value = "";
  try {
    const endpoint = `/demo/video/jobs/${encodeURIComponent(id)}`;
    const job = cancelRequested ? await post<Job>(`${endpoint}/cancel`) : await request<Job>(endpoint);
    if (current === generation) archiveVideo(job, entry.filename);
  } catch (failure) {
    if (current === generation) {
      historyError.value = errorText(failure);
      if (failure instanceof ApiError && failure.status === 404) {
        archiveVideo({ id, status: "interrupted", stage: "interrupted", error: "此前的任务已无法找到。" }, entry.filename);
      }
    }
  } finally { if (current === generation) historyPending.value[id] = false; }
}

function remember(lane: Lane, job: Job) {
  try {
    sessionStorage.setItem(key(lane), JSON.stringify({ id: job.id, savedAt: Date.now(),
      ...(lane === "video" ? { filename: videoFilename.value, contextId: props.selectionContextId } : {}) }));
  } catch { /* Storage may be disabled; the backend still owns the task. */ }
}

function stopPolling(lane: Lane) {
  pollCycles[lane]++;
  if (timers[lane]) clearTimeout(timers[lane]);
  delete timers[lane];
}

async function refreshScores(current = generation) {
  if (!props.albumId) return;
  const result = await request<{ items: Score[] }>(`/albums/${encodeURIComponent(props.albumId)}/aesthetics`);
  if (current === generation) scores.value = result.items ?? [];
}

function acceptJob(lane: Lane, job: Job, current: number) {
  if (current !== generation) return;
  if (lane === "quality") qualityJob.value = job;
  else videoJob.value = job;
  remember(lane, job);
  if (terminal.has(job.status)) {
    stopPolling(lane);
    if (lane === "quality") {
      void refreshScores(current).catch(failure => { if (current === generation) qualityError.value = errorText(failure); });
    }
  }
}

async function poll(lane: Lane, id: string, current: number, cycle: number) {
  if (current !== generation || cycle !== pollCycles[lane]) return;
  if (Date.now() > deadlines[lane]) {
    pollPaused.value[lane] = true;
    return;
  }
  try {
    const path = lane === "quality" ? `/aesthetics/jobs/${encodeURIComponent(id)}`
      : `/demo/video/jobs/${encodeURIComponent(id)}`;
    const job = await request<Job>(path);
    if (current !== generation || cycle !== pollCycles[lane]) return;
    failures[lane] = 0;
    acceptJob(lane, job, current);
    if (!terminal.has(job.status)) timers[lane] = setTimeout(() => void poll(lane, id, current, cycle), 2000);
  } catch (failure) {
    if (current !== generation || cycle !== pollCycles[lane]) return;
    if (failure instanceof ApiError && failure.status === 404) {
      acceptJob(lane, { id, status: "interrupted", stage: "interrupted",
        error: "找不到此前的任务。请检查服务端是否更换；需要生成时重新提交。" }, current);
      try { sessionStorage.removeItem(key(lane)); } catch { /* Optional storage. */ }
      return;
    }
    failures[lane]++;
    if (failures[lane] >= 3) {
      pollPaused.value[lane] = true;
      if (lane === "quality") qualityError.value = errorText(failure);
      else videoError.value = errorText(failure);
    } else timers[lane] = setTimeout(() => void poll(lane, id, current, cycle), failures[lane] * 4000);
  }
}

function resumePolling(lane: Lane, id?: string) {
  const jobId = id ?? (lane === "quality" ? qualityJob.value?.id : videoJob.value?.id);
  if (!jobId) return;
  stopPolling(lane);
  deadlines[lane] = Date.now() + 2 * 60 * 60 * 1000;
  failures[lane] = 0;
  pollPaused.value[lane] = false;
  if (lane === "quality") qualityError.value = "";
  else videoError.value = "";
  void poll(lane, jobId, generation, pollCycles[lane]);
}

async function checkRemote() {
  if (checking.value) return;
  const current = generation;
  checking.value = true;
  try {
    const result = await request<RemoteStatus>("/demo/video/status");
    if (current === generation) remote.value = result;
  } catch (failure) {
    if (current === generation) remote.value = { configured: false, reachable: false, error: errorText(failure) };
  } finally { if (current === generation) checking.value = false; }
}

async function analyzeQuality() {
  if (!props.albumId || props.busy || qualityPending.value || qualityActive.value) return;
  const current = generation;
  qualityPending.value = true;
  qualityError.value = "";
  try {
    const job = await post<Job>(`/albums/${encodeURIComponent(props.albumId)}/aesthetics/jobs`, { force: false });
    if (current === generation) { acceptJob("quality", job, current); if (active(job)) resumePolling("quality", job.id); }
  } catch (failure) { if (current === generation) qualityError.value = errorText(failure); }
  finally { if (current === generation) qualityPending.value = false; }
}

async function generateVideo() {
  if (!canGenerate.value || !chosen.value) return;
  const current = generation;
  const submittedContext = props.selectionContextId;
  const photo = chosen.value;
  videoPending.value = true;
  videoError.value = "";
  try {
    const job = await post<Job>("/demo/video/jobs", {
      album_id: props.albumId, photo_id: photo.photo_id, upload_confirmed: true,
      options: { prompt: prompt.value.trim() },
    });
    if (current === generation) {
      if (submittedContext !== props.selectionContextId) {
        archiveVideo(job, photo.filename);
        return;
      }
      videoFilename.value = photo.filename;
      consent.value = false;
      acceptJob("video", job, current);
      if (active(job)) resumePolling("video", job.id);
    }
  } catch (failure) {
    if (current === generation) {
      if (submittedContext !== props.selectionContextId) historyError.value = errorText(failure);
      else videoError.value = errorText(failure);
    }
  }
  finally { if (current === generation) videoPending.value = false; }
}

async function cancel(lane: Lane) {
  const job = lane === "quality" ? qualityJob.value : videoJob.value;
  if (!job || !active(job)) return;
  const current = generation;
  const submittedContext = props.selectionContextId;
  const submittedFilename = videoFilename.value;
  const pending = lane === "quality" ? qualityCancelling : videoCancelling;
  if (pending.value) return;
  pending.value = true;
  stopPolling(lane);
  try {
    const prefix = lane === "quality" ? "/aesthetics/jobs" : "/demo/video/jobs";
    const result = await post<Job>(`${prefix}/${encodeURIComponent(job.id)}/cancel`);
    if (current === generation) {
      if (lane === "video" && submittedContext !== props.selectionContextId) {
        archiveVideo(result, submittedFilename);
        return;
      }
      acceptJob(lane, result, current);
      if (active(result)) resumePolling(lane, result.id);
    }
  } catch (failure) {
    if (current === generation) {
      if (lane === "video" && submittedContext !== props.selectionContextId) {
        historyError.value = errorText(failure);
        return;
      }
      resumePolling(lane, job.id);
      if (lane === "quality") qualityError.value = errorText(failure);
      else videoError.value = errorText(failure);
    }
  } finally { if (current === generation) pending.value = false; }
}

function clear() {
  generation++;
  controllers.forEach(controller => controller.abort());
  controllers.clear();
  stopPolling("quality"); stopPolling("video");
  qualityJob.value = null; videoJob.value = null; scores.value = [];
  qualityError.value = ""; videoError.value = ""; remote.value = null;
  qualityPending.value = false; videoPending.value = false; checking.value = false;
  qualityCancelling.value = false; videoCancelling.value = false;
  consent.value = false; videoFilename.value = "";
  historicalVideos.value = []; historyPending.value = {}; historyError.value = "";
  pollPaused.value = { quality: false, video: false };
}

watch(() => props.albumId, () => {
  clear();
  if (!props.albumId) return;
  try {
    const savedHistory: HistoricalVideo[] = JSON.parse(sessionStorage.getItem(historyKey()) ?? "[]");
    if (Array.isArray(savedHistory)) historicalVideos.value = savedHistory.filter(entry =>
      entry?.job && /^[0-9a-f]{32}$/.test(entry.job.id) && typeof entry.filename === "string"
      && Number.isFinite(entry.savedAt) && Date.now() - entry.savedAt < 86400000,
    ).map(entry => ({ ...entry, job: { id: entry.job.id, status: "queued", stage: "历史任务，状态待刷新" } }));
  } catch { /* Invalid history never changes the current selection. */ }
  for (const lane of ["quality", "video"] as const) {
    try {
      const saved: SavedJob = JSON.parse(sessionStorage.getItem(key(lane)) ?? "null");
      if (saved && /^[a-zA-Z0-9-]{16,80}$/.test(saved.id) && Date.now() - saved.savedAt < 86400000) {
        const restored: Job = { id: saved.id, status: "queued", stage: "恢复任务状态" };
        if (lane === "video" && (!saved.contextId || saved.contextId !== props.selectionContextId)) {
          archiveVideo({ ...restored, stage: "历史任务，状态待刷新" }, saved.filename ?? "此前提交的照片");
          try { sessionStorage.removeItem(key("video")); } catch { /* Optional storage. */ }
          continue;
        }
        if (lane === "quality") qualityJob.value = restored;
        else { videoJob.value = restored; videoFilename.value = saved.filename ?? "已提交照片"; }
        resumePolling(lane, saved.id);
      }
    } catch { /* Ignore stale or inaccessible session storage. */ }
  }
  if (expanded.value) { void checkRemote(); void refreshScores().catch(() => {}); }
}, { immediate: true });
watch(() => props.selectionContextId, detachPreviousVideo, { flush: "sync" });
watch(() => props.selectedPhotos.map(photo => photo.photo_id), ids => {
  if (!ids.includes(pickedId.value)) pickedId.value = ids[0] ?? "";
}, { immediate: true });
watch(pickedId, () => { consent.value = false; });
onBeforeUnmount(clear);

function onToggle(event: Event) {
  expanded.value = (event.target as HTMLDetailsElement).open;
  if (expanded.value) {
    void checkRemote();
    const current = generation;
    void refreshScores(current).catch(failure => { if (current === generation) qualityError.value = errorText(failure); });
  }
}
</script>

<template>
  <details class="demo-workbench" @toggle="onToggle">
    <summary>模型实验 · 美学评估 / 图生视频 <span v-if="qualityActive || videoActive">有任务进行中</span></summary>
    <div class="demo-panel">
      <section class="demo-quality" aria-label="学习型美学评估">
        <div class="demo-actions">
          <button :disabled="!albumId || busy || qualityPending || qualityActive" @click="analyzeQuality">
            {{ qualityPending ? "提交中…" : "分析相册质量与美学" }}
          </button>
          <small>点击才运行；复用已完成的评分，不修改原图。</small>
        </div>
        <div v-if="qualityJob" class="demo-progress" role="status">
          <span>{{ stageLabel(qualityJob) }} · {{ qualityPercent }}%</span>
          <progress :value="qualityPercent" max="100" aria-label="美学分析进度" />
          <button v-if="qualityActive" :disabled="qualityCancelling || qualityJob.cancel_requested" @click="cancel('quality')">
            {{ qualityJob.cancel_requested ? "等待安全停止…" : "取消" }}
          </button>
        </div>
        <p v-if="scores.length" class="demo-note">
          已有 {{ scores.length }} 张评分；当前选中 {{ visibleScores.length }} 张有结果。
          <span v-if="visibleScores.length">平均质量 {{ scoreMean('technical_quality') }} / 约100 · 美学 {{ scoreMean('aesthetic_quality') }} / 10</span>
        </p>
        <p v-if="qualityError || qualityJob?.error" class="demo-error" role="alert">{{ qualityError || qualityJob?.error }}</p>
        <button v-if="pollPaused.quality" @click="resumePolling('quality')">自动查询已暂停，刷新任务状态</button>
      </section>

      <section class="demo-video" aria-label="远端图生视频">
        <div class="demo-actions">
          <strong>单镜头预览</strong>
          <small>{{ checking ? "检查连接…" : remote?.reachable ? "远端 Worker 已连接（不代表模型已就绪）" : "远端 Worker 未连接" }}</small>
          <button :disabled="checking" @click="checkRemote">检查连接</button>
        </div>
        <p v-if="remote?.error" class="demo-note">{{ remote.error }}</p>
        <p v-if="!selectedPhotos.length" class="demo-note">先在照片区域选择一张或多张图片，再选择本次生成的照片。</p>
        <div v-else class="demo-compose">
          <figure v-if="chosen">
            <img :src="chosen.thumbnail_url" :alt="chosen.filename" />
            <figcaption>640 × 384 居中裁剪预览</figcaption>
          </figure>
          <div class="demo-fields">
            <label>生成照片
              <select v-model="pickedId" :disabled="videoPending">
                <option v-for="photo in selectedPhotos" :key="photo.photo_id" :value="photo.photo_id">{{ photo.filename }}</option>
              </select>
            </label>
            <label>镜头与运动描述
              <textarea v-model="prompt" rows="2" maxlength="2000" :disabled="videoPending" />
            </label>
            <label class="demo-consent"><input v-model="consent" type="checkbox" :disabled="videoPending" />
              我确认把这张处理后的照片和提示词发送到自己的服务器；图片与任务会留存在服务器。
            </label>
            <div class="demo-actions">
              <button :disabled="!canGenerate" @click="generateVideo">{{ videoPending ? "提交中…" : "生成约2秒预览" }}</button>
              <small>49帧 · 24fps · 20步；真实模型生成，首次加载可能较慢。</small>
            </div>
          </div>
        </div>
        <div v-if="videoJob" class="demo-result" aria-live="polite">
          <p class="demo-note">{{ videoFilename }} · {{ stageLabel(videoJob) }}</p>
          <div v-if="videoActive" class="demo-progress" role="status">
            <template v-if="videoJob.stage === 'denoising' && videoJob.step_percent != null">
              <progress :value="videoJob.step_percent" max="100" aria-label="视频去噪步骤进度" />
              <span>去噪 {{ videoJob.step }}/{{ videoJob.total_steps }} · {{ videoJob.step_percent }}%（不含加载和导出）</span>
            </template>
            <template v-else><progress aria-label="当前阶段处理中" /><span>此阶段无法准确估计百分比。</span></template>
            <button :disabled="videoCancelling || videoJob.cancel_requested" @click="cancel('video')">
              {{ videoJob.cancel_requested ? "等待安全停止…" : "取消生成" }}
            </button>
          </div>
          <template v-if="videoUrl">
            <video :src="videoUrl" controls preload="metadata" playsinline />
            <a :href="videoUrl" download>下载生成视频</a>
          </template>
        </div>
        <p v-if="videoError || videoJob?.error" class="demo-error" role="alert">{{ videoError || videoJob?.error }}</p>
        <button v-if="pollPaused.video" @click="resumePolling('video')">自动查询已暂停，刷新任务状态</button>
        <details v-if="historicalVideos.length" class="demo-history">
          <summary>此前请求的视频任务（{{ historicalVideos.length }}）· 不是本次选片结果</summary>
          <p class="demo-note">新选片不会取消或重新上传旧任务；可在这里手动刷新、查看或取消。</p>
          <div v-for="entry in historicalVideos" :key="entry.job.id" class="demo-result">
            <p class="demo-note">此前请求 · {{ entry.filename }} · {{ stageLabel(entry.job) }}</p>
            <div class="demo-actions">
              <button :disabled="historyPending[entry.job.id]" @click="updateHistoricalVideo(entry.job.id)">刷新旧任务</button>
              <button v-if="active(entry.job)" :disabled="historyPending[entry.job.id] || entry.job.cancel_requested" @click="updateHistoricalVideo(entry.job.id, true)">
                {{ entry.job.cancel_requested ? "已请求取消" : "取消旧任务" }}
              </button>
              <a v-if="entry.job.status === 'completed'" :href="`/demo/video/jobs/${encodeURIComponent(entry.job.id)}/artifact`" target="_blank" rel="noopener">查看此前生成的视频</a>
            </div>
            <p v-if="entry.job.error" class="demo-error">{{ entry.job.error }}</p>
          </div>
        </details>
        <p v-if="historyError" class="demo-error" role="alert">此前请求：{{ historyError }}</p>
        <small v-if="videoActive || qualityActive">切换相册或关闭页面不会取消服务端任务；请使用取消按钮。</small>
      </section>
    </div>
  </details>
</template>

<style scoped>
.demo-workbench { margin: 10px 0; font-size: 12px; border: 1px solid #8884; border-radius: 8px; }
summary { cursor: pointer; padding: 10px 12px; }
summary span { margin-left: 10px; opacity: .65; font-weight: normal; }
.demo-panel { padding: 0 12px 12px; }
.demo-quality { padding: 6px 0 12px; border-bottom: 1px solid #8883; }
.demo-video { padding-top: 12px; }
.demo-actions, .demo-progress { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
button, select, textarea { color: inherit; border: 1px solid #8886; border-radius: 5px; background: transparent; font: inherit; }
button { padding: 6px 10px; cursor: pointer; }
button:disabled { opacity: .45; cursor: default; }
select, textarea { padding: 6px; width: 100%; min-width: 0; box-sizing: border-box; }
select option { background: #171918; color: #eee; }
textarea { resize: vertical; }
small, .demo-note, figcaption { opacity: .7; line-height: 1.6; }
.demo-note { margin: 8px 0; }
.demo-compose { display: flex; gap: 14px; margin-top: 12px; align-items: flex-start; }
figure { width: 200px; flex: 0 0 200px; margin: 0; }
figure img { display: block; width: 100%; aspect-ratio: 5 / 3; object-fit: cover; object-position: center; border-radius: 5px; }
figcaption { padding-top: 4px; font-size: 10px; }
.demo-fields { display: grid; gap: 9px; flex: 1; min-width: 0; }
.demo-fields label { display: grid; gap: 4px; }
.demo-fields .demo-consent { display: flex; align-items: flex-start; line-height: 1.5; }
.demo-consent input { margin: 2px 4px 0 0; flex: 0 0 auto; }
.demo-progress { margin-top: 9px; }
progress { width: 160px; height: 8px; accent-color: #cb805e; }
.demo-error { color: #df967a; overflow-wrap: anywhere; }
.demo-result video { display: block; width: 100%; max-width: 480px; max-height: 300px; border-radius: 5px; margin: 8px 0; }
.demo-result a { color: inherit; }
@media (max-width: 640px) { .demo-compose { flex-direction: column; } figure { flex-basis: auto; width: 160px; } .demo-fields { width: 100%; } }
</style>
