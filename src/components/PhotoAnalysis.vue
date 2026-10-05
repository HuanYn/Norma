<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";

const props = defineProps<{
  albumId?: string;
  query: string;
  ready: boolean;
  busy: boolean;
  cloud: boolean;
  configured: boolean;
}>();

interface Analysis {
  answer: string;
  claims: { claim_id: string; text: string }[];
  citations: { claim_id: string; photo_id: string }[];
  retrieval: { matches: { photo_id: string; thumbnail_url: string; filename: string }[] };
}

const pending = ref(false);
const error = ref("");
const result = ref<Analysis | null>(null);
const askedQuery = ref("");
let generation = 0;
let controller: AbortController | null = null;
const canAnalyze = computed(() => !!props.albumId && !!props.query.trim()
  && props.ready && props.configured && !props.busy && !pending.value);

function clearResult() {
  generation++;
  controller?.abort();
  controller = null;
  pending.value = false;
  result.value = null;
  error.value = "";
}

watch(() => [props.albumId, props.cloud, props.configured], clearResult);
onBeforeUnmount(clearResult);

async function analyze() {
  if (!canAnalyze.value) return;
  const current = ++generation;
  const albumId = props.albumId;
  const query = props.query.trim();
  controller = new AbortController();
  pending.value = true;
  error.value = "";
  result.value = null;
  askedQuery.value = query;
  try {
    const response = await fetch(`/albums/${encodeURIComponent(albumId!)}/rag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 3, user_id: "local" }),
      signal: controller.signal,
    });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === "string"
      ? body.detail : `分析失败（HTTP ${response.status}）`);
    if (current === generation) result.value = body;
  } catch (failure) {
    if (current === generation) error.value = failure instanceof Error
      ? failure.message : "分析失败，请检查模型配置。";
  } finally {
    if (current === generation) { pending.value = false; controller = null; }
  }
}

function evidenceFor(claimId: string) {
  const ids = new Set(result.value?.citations.filter(c => c.claim_id === claimId)
    .map(c => c.photo_id));
  return result.value?.retrieval.matches.filter(photo => ids.has(photo.photo_id)) ?? [];
}
</script>

<template>
  <section class="photo-analysis" aria-label="照片分析" :aria-busy="pending">
    <div class="analysis-action">
      <button :disabled="!canAnalyze" @click="analyze">
        {{ pending ? "正在检索和分析…" : cloud ? "云端看图分析" : "本地看图分析" }}
      </button>
      <span v-if="!configured">请先在本机配置云端接口地址、视觉模型和 API Key。</span>
      <span v-else-if="cloud">点击后，本地检索的前 3 张候选压缩图和问题将发送至已配置的云端服务，可能产生调用费用。</span>
      <span v-else>对当前问题本地检索并分析前 3 张候选图片。</span>
    </div>
    <p v-if="pending" role="status">模型正在处理，请稍候。切换页面不会保证取消已经发出的模型请求。</p>
    <p v-if="error" role="alert" class="analysis-error">{{ error }}</p>
    <div v-if="result" class="analysis-result" aria-live="polite">
      <p class="analysis-query">{{ askedQuery }}</p>
      <article v-for="claim in result.claims" :key="claim.claim_id">
        <p>{{ claim.text }}</p>
        <div class="analysis-evidence">
          <a v-for="photo in evidenceFor(claim.claim_id)" :key="photo.photo_id"
            :href="photo.thumbnail_url" target="_blank" rel="noopener noreferrer">
            <img :src="photo.thumbnail_url" :alt="photo.filename" loading="lazy" />
            <span>{{ photo.filename }}</span>
          </a>
        </div>
      </article>
      <small>已校验引用对应本次图片；具体判断仍需结合图片核对。</small>
    </div>
  </section>
</template>

<style scoped>
.photo-analysis { margin: 12px 0; font-size: 12px; }
.analysis-action { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.analysis-action button { border: 1px solid currentColor; border-radius: 6px; padding: 9px 14px; background: transparent; color: inherit; cursor: pointer; }
.analysis-action button:disabled { opacity: .45; cursor: default; }
.analysis-action span, .analysis-result small { opacity: .7; line-height: 1.6; }
.analysis-error { color: #c87055; }
.analysis-result { margin-top: 14px; padding: 14px; border: 1px solid #8884; border-radius: 8px; }
.analysis-result p { line-height: 1.7; white-space: pre-wrap; }
.analysis-query { opacity: .65; }
.analysis-evidence { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 16px; }
.analysis-evidence a { display: grid; max-width: 150px; gap: 4px; color: inherit; text-decoration: none; }
.analysis-evidence img { width: 130px; height: 90px; object-fit: contain; background: #8881; border-radius: 4px; }
.analysis-evidence span { overflow-wrap: anywhere; font-size: 10px; }
</style>
