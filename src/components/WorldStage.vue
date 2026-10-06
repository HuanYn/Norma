<script setup lang="ts">
import { computed, ref } from 'vue';

const props = defineProps<{
  src: string; poster: string; filename: string; enabled: boolean;
  generating: boolean; pendingAction?: string;
  history: { duration: number; action: string }[];
}>();
const emit = defineEmits<{ action: [action: string]; activate: [] }>();
const stage = ref<HTMLElement | null>(null), time = ref(0), fullscreenError = ref('');
const pads = [
  { name: '移动', keys: [['forward','W','前进'],['left','A','左移'],['backward','S','后退'],['right','D','右移']] },
  { name: '视角', keys: [['look_up','↑','向上看'],['look_left','←','向左看'],['look_down','↓','向下看'],['look_right','→','向右看']] },
];
const replayAction = computed(() => props.history.find(step => time.value < step.duration)?.action ?? props.history.at(-1)?.action);
const litAction = computed(() => props.generating ? props.pendingAction : replayAction.value);
function updateTime(event: Event) { time.value = (event.target as HTMLVideoElement).currentTime; }
async function fullscreen() {
  emit('activate'); fullscreenError.value = '';
  try {
    if (document.fullscreenElement === stage.value) await document.exitFullscreen();
    else await stage.value?.requestFullscreen();
  } catch { fullscreenError.value = '当前浏览器不支持场景全屏'; }
}
</script>

<template>
  <div ref="stage" class="world-stage" @click="emit('activate')">
    <video v-if="src" :key="src" :src="src" :poster="poster" controls controlslist="nofullscreen" playsinline
      @timeupdate="updateTime" @seeked="updateTime" @loadedmetadata="updateTime" />
    <img v-else :src="poster" :alt="filename" />
    <button class="stage-fullscreen" :aria-label="`${filename} 场景全屏`" @click.stop="fullscreen">⛶ 全屏</button>
    <div class="stage-hud" :data-highlight="litAction || ''">
      <div v-for="pad in pads" :key="pad.name" class="key-pad" :aria-label="pad.name">
        <span class="pad-name">{{ pad.name }}</span>
        <button v-for="[action,key,label] in pad.keys" :key="action" :data-action="action"
          :class="{ lit: litAction===action }" :title="label" :aria-label="`${filename} ${label}`"
          :disabled="!enabled" @click.stop="emit('activate');emit('action',action)"><kbd>{{key}}</kbd></button>
      </div>
    </div>
    <span class="stage-mode">{{generating?'已提交方向 · 等待生成':src?(enabled?'亮键为回放动作 · 点击生成下一步':'亮键为当前回放动作'):'WASD 移动 · 方向键转头'}}</span>
    <span v-if="fullscreenError" role="status" class="fullscreen-error">{{fullscreenError}}</span>
  </div>
</template>

<style scoped>
.world-stage{position:relative;background:#090c08;isolation:isolate;--key-size:38px;--key-gap:5px}
.world-stage>video,.world-stage>img{display:block;width:100%;height:500px;object-fit:contain}
.stage-hud{position:absolute;inset:auto 16px 58px;display:flex;justify-content:space-between;pointer-events:none}
.key-pad{display:grid;grid-template-columns:repeat(3,var(--key-size));grid-template-rows:16px repeat(2,var(--key-size));gap:var(--key-gap);padding:8px;background:#0c1521a8;border-radius:9px;pointer-events:auto}
.pad-name{grid-area:1/1/2/4;color:#d6e3ee;font:10px sans-serif;letter-spacing:2px;text-align:center}
.key-pad button{display:grid;place-items:center;width:var(--key-size);height:var(--key-size);padding:0;border:1px solid #b7cddd30;border-radius:5px;background:#263343dd;color:#e6edf5;cursor:pointer;box-shadow:0 2px 0 #070c14}
.key-pad button:nth-of-type(1){grid-area:2/2}.key-pad button:nth-of-type(2){grid-area:3/1}.key-pad button:nth-of-type(3){grid-area:3/2}.key-pad button:nth-of-type(4){grid-area:3/3}
.key-pad kbd{font:24px system-ui,sans-serif}.key-pad button:disabled{cursor:default;opacity:.8}.key-pad button.lit{background:#39ac90;color:white;border-color:#97ffda;box-shadow:0 0 13px #39ac9080;opacity:1}
.key-pad button:focus-visible,.stage-fullscreen:focus-visible{outline:2px solid white;outline-offset:3px}.key-pad button:not(:disabled):hover{border-color:#97ffda}
.stage-fullscreen{position:absolute;right:12px;top:12px;border:1px solid #ffffff35;border-radius:6px;padding:7px 10px;background:#0c1521ba;color:white;cursor:pointer}
.stage-mode{position:absolute;top:14px;left:12px;max-width:70%;font:11px sans-serif;color:#e3eaf1;background:#0c1521ba;padding:6px 8px;border-radius:5px;pointer-events:none}
.fullscreen-error{position:absolute;top:50px;left:12px;color:#ffb19d}
.world-stage:fullscreen{width:100%;height:100%;--key-size:54px;--key-gap:7px}.world-stage:fullscreen>video,.world-stage:fullscreen>img{height:100%;width:100%}.world-stage:fullscreen .stage-hud{inset:auto 30px 65px}
@media(max-width:750px){.world-stage>video,.world-stage>img{height:420px}.world-stage{--key-size:32px;--key-gap:4px}.stage-hud{left:8px;right:8px}.key-pad{padding:6px}.key-pad kbd{font-size:21px}.stage-mode{font-size:10px;max-width:65%}}
</style>
