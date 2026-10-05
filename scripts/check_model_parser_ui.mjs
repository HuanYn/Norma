// Offline behavioral checks compiled from the real Vue setup scripts.
// No browser network, model calls, server processes or production build.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { parse, compileScript } from "vue/compiler-sfc";
import ts from "typescript";
import * as Vue from "vue";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const storage = new Map();
globalThis.window = { addEventListener() {}, removeEventListener() {}, location: { origin: "http://127.0.0.1" } };
globalThis.sessionStorage = {
  getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, String(value)),
  removeItem: key => storage.delete(key),
};
let requests = [];
let respond = () => { throw new Error("Unexpected request in an offline check"); };
globalThis.fetch = async (path, options) => {
  requests.push({ path, options });
  return respond(path, options);
};
const response = (body, ok = true) => ({ ok, status: ok ? 200 : 422, json: async () => body });

function loadSetup(relative, initialProps = {}) {
  const source = readFileSync(resolve(root, relative), "utf8");
  const { descriptor } = parse(source, { filename: relative });
  const compiled = compileScript(descriptor, { id: "offline-ui-check" });
  const code = ts.transpileModule(compiled.content, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  const cleanup = [];
  const testVue = { ...Vue, onMounted: () => {}, onBeforeUnmount: fn => cleanup.push(fn) };
  const module = { exports: {} };
  const require = id => {
    if (id === "vue") return testVue;
    if (id.endsWith(".vue")) return {};
    throw new Error(`Unexpected setup import: ${id}`);
  };
  new Function("require", "module", "exports", code)(require, module, module.exports);
  const props = Vue.reactive(initialProps);
  const scope = Vue.effectScope();
  const state = scope.run(() => module.exports.default.setup(props, { expose() {} }));
  return { state, props, template: descriptor.template.content, dispose() {
    cleanup.forEach(fn => fn()); scope.stop();
  } };
}

function readyApp() {
  const fixture = loadSetup("src/App.vue");
  const vm = fixture.state;
  assert.equal(vm.useModelParser.value, false);
  assert.equal(vm.allowCloudParsing.value, false);
  vm.worker.value = { ...vm.worker.value, healthy: true, embedding_provider: "fake",
    vlm_provider: "openai-compatible", vlm_configured: true };
  vm.album.value = { album_id: "album-a", total: 3, quality_count: 3,
    embedded_count: 3, embedding_provider: "fake", photos: [] };
  vm.embeddingProviderStatus.value = { model_backed: false };
  vm.command.value = "选2张建筑";
  requests = [];
  return fixture;
}

let passed = 0;
async function check(name, operation) {
  storage.clear(); requests = [];
  await operation();
  passed++;
  console.log(`PASS ${name}`);
}

await check("model parser defaults off; provider change revokes both opt-ins", async () => {
  const fixture = readyApp(); const vm = fixture.state;
  try {
    vm.worker.value = { ...vm.worker.value, vlm_provider: "local" };
    vm.useModelParser.value = true; vm.allowCloudParsing.value = true;
    vm.worker.value = { ...vm.worker.value, vlm_provider: "openai-compatible" };
    assert.equal(vm.useModelParser.value, false);
    assert.equal(vm.allowCloudParsing.value, false);
    assert.match(fixture.template, /实验性模型解析/);
    assert.match(fixture.template, /两次实测均未通过/);
  } finally { fixture.dispose(); }
});

await check("cloud model selection without separate consent sends nothing", async () => {
  const fixture = readyApp(); const vm = fixture.state;
  try {
    vm.useModelParser.value = true;
    await vm.runSearch();
    assert.equal(requests.length, 0);
    assert.match(vm.searchError.value, /明确同意/);
    assert.equal(vm.selectionResult.value, null);
  } finally { fixture.dispose(); }
});

await check("cloud consent is exact-text and single-submission", async () => {
  const fixture = readyApp(); const vm = fixture.state;
  try {
    vm.useModelParser.value = true; vm.allowCloudParsing.value = true;
    vm.command.value = "选2张夜景";
    assert.equal(vm.allowCloudParsing.value, false);
    vm.allowCloudParsing.value = true;
    respond = () => response({ selection_id: "new", selected: [] });
    await vm.runSearch();
    assert.equal(requests.length, 1);
    assert.equal(requests[0].path, "/selections/structured");
    assert.equal(JSON.parse(requests[0].options.body).allow_cloud, true);
    assert.equal(vm.allowCloudParsing.value, false);
    await vm.runSearch();
    assert.equal(requests.length, 1);
    assert.match(vm.searchError.value, /明确同意/);
  } finally { fixture.dispose(); }
});

await check("failed model parse clears previous results before awaiting and never falls back", async () => {
  const fixture = readyApp(); const vm = fixture.state;
  try {
    vm.useModelParser.value = true; vm.allowCloudParsing.value = true;
    vm.selectionResult.value = { selection_id: "old", selected: [{ photo_id: "old-photo" }] };
    vm.searchResult.value = { photos: [{ photo_id: "old-search" }] };
    const oldContext = vm.selectionContextId.value;
    let resolveRequest;
    respond = () => new Promise(resolve => { resolveRequest = resolve; });
    const pending = vm.runSearch();
    assert.equal(vm.selectionResult.value, null);
    assert.equal(vm.searchResult.value, null);
    assert.notEqual(vm.selectionContextId.value, oldContext);
    resolveRequest(response({ detail: "strict intent validation failed" }, false));
    await pending;
    assert.equal(requests.length, 1);
    assert.equal(requests[0].path, "/selections/structured");
    assert.equal(vm.selectionResult.value, null);
    assert.match(vm.searchError.value, /strict intent validation/);
    assert.match(fixture.template, /:selection-context-id="selectionContextId"/);
  } finally { fixture.dispose(); }
});

await check("a late parsed selection cannot populate another album", async () => {
  const fixture = readyApp(); const vm = fixture.state;
  try {
    vm.useModelParser.value = true; vm.allowCloudParsing.value = true;
    let resolveRequest;
    respond = () => new Promise(resolve => { resolveRequest = resolve; });
    const pending = vm.runSearch();
    vm.album.value = { ...vm.album.value, album_id: "album-b" };
    resolveRequest(response({ selection_id: "old-album-result", selected: [{ photo_id: "old" }] }));
    await pending;
    assert.equal(vm.selectionResult.value, null);
    assert.equal(requests.length, 1);
    assert.equal(JSON.parse(requests[0].options.body).album_id, "album-a");
  } finally { fixture.dispose(); }
});

const photo = { photo_id: "photo-a", filename: "a.jpg", thumbnail_url: "/preview/a.jpg" };
function workbench() {
  return loadSetup("src/components/DemoWorkbench.vue", {
    albumId: "album-a", selectedPhotos: [photo], busy: false, selectionContextId: "page:1",
  });
}

await check("a completed MP4 moves to explicit history when a new selection starts", async () => {
  const fixture = workbench(); const vm = fixture.state;
  try {
    vm.videoFilename.value = "previous.jpg";
    vm.videoJob.value = { id: "a".repeat(32), status: "completed", stage: "completed" };
    vm.consent.value = true;
    assert.notEqual(vm.videoUrl.value, "");
    fixture.props.selectionContextId = "page:2";
    assert.equal(vm.videoJob.value, null);
    assert.equal(vm.videoUrl.value, "");
    assert.equal(vm.consent.value, false);
    assert.equal(vm.historicalVideos.value[0].job.id, "a".repeat(32));
    assert.equal(vm.historicalVideos.value[0].filename, "previous.jpg");
    assert.match(fixture.template, /不是本次选片结果/);
    assert.equal(requests.length, 0);
  } finally { fixture.dispose(); }
});

await check("old active jobs remain cancellable without blocking a new confirmed submission", async () => {
  const fixture = workbench(); const vm = fixture.state;
  try {
    const id = "b".repeat(32);
    vm.videoJob.value = { id, status: "running", stage: "denoising" };
    vm.videoFilename.value = "previous.jpg";
    fixture.props.selectionContextId = "page:2";
    vm.remote.value = { configured: true, reachable: true };
    vm.consent.value = true;
    assert.equal(vm.canGenerate.value, true);
    respond = () => response({ id, status: "running", stage: "denoising", cancel_requested: true });
    await vm.updateHistoricalVideo(id, true);
    assert.equal(requests.length, 1);
    assert.equal(requests[0].path, `/demo/video/jobs/${id}/cancel`);
    assert.equal(vm.historicalVideos.value[0].job.cancel_requested, true);
    assert.equal(vm.videoJob.value, null);
  } finally { fixture.dispose(); }
});

await check("an in-flight upload response from an old selection cannot populate the new result", async () => {
  const fixture = workbench(); const vm = fixture.state;
  try {
    vm.remote.value = { configured: true, reachable: true };
    vm.consent.value = true;
    let resolveRequest;
    respond = () => new Promise(resolve => { resolveRequest = resolve; });
    const pending = vm.generateVideo();
    assert.equal(requests.length, 1);
    fixture.props.selectionContextId = "page:2";
    resolveRequest(response({ id: "c".repeat(32), status: "queued", stage: "queued" }));
    await pending;
    assert.equal(vm.videoJob.value, null);
    assert.equal(vm.videoUrl.value, "");
    assert.equal(vm.historicalVideos.value[0].job.id, "c".repeat(32));
    assert.equal(requests.length, 1); // No repeat upload and no hidden status polling.
  } finally { fixture.dispose(); }
});

await check("restoring another page's saved video cannot restore it as a current MP4", async () => {
  storage.set("norma-demo-video:album-a", JSON.stringify({ id: "d".repeat(32),
    savedAt: Date.now(), filename: "saved.jpg", contextId: "older-page:1" }));
  const fixture = workbench(); const vm = fixture.state;
  try {
    assert.equal(vm.videoJob.value, null);
    assert.equal(vm.videoUrl.value, "");
    assert.equal(vm.historicalVideos.value[0].filename, "saved.jpg");
    assert.equal(vm.historicalVideos.value[0].job.status, "queued");
    assert.equal(requests.length, 0);
    assert.equal(storage.has("norma-demo-video:album-a"), false);
  } finally { fixture.dispose(); }
});

await check("an old upload failure is labelled historical rather than failing the new request", async () => {
  const fixture = workbench(); const vm = fixture.state;
  try {
    vm.remote.value = { configured: true, reachable: true };
    vm.consent.value = true;
    let rejectRequest;
    respond = () => new Promise((_, reject) => { rejectRequest = reject; });
    const pending = vm.generateVideo();
    fixture.props.selectionContextId = "page:2";
    rejectRequest(new Error("old upload timed out"));
    await pending;
    assert.equal(vm.videoError.value, "");
    assert.match(vm.historyError.value, /old upload timed out/);
    assert.equal(vm.videoJob.value, null);
    assert.equal(requests.length, 1);
  } finally { fixture.dispose(); }
});

console.log(`${passed} offline UI checks passed.`);
