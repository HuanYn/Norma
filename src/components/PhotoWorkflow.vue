<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref, watch } from 'vue';

type Photo = { photo_id: string; filename: string; thumbnail_url: string; aesthetic_quality?: number; technical_quality?: number; reasons?: string[] };
type Curation = { id: string; album_id: string; total: number; kept: Photo[]; hidden: Photo[]; group_count: number; thresholds: {min_aesthetic:number; min_technical:number} };
type Selection = { selection_id: string; selected: Photo[]; feasible: boolean; warnings: string[]; intent_provenance?: { effective_prompt?: string }; algorithm: string };
type Job = { id: string; status: string; stage: string; progress: number; result?: { album?: { album_id: string } }; error?: string };
type WorldState = { id: string; status: string; phase: string; next_sequence: number; max_steps: number; completed_chunks?: number; total_chunks?: number; history: { artifact: string; duration: number; action: string }[] };
type Card = { photo: Photo; selectionId: string; state: WorldState | null; error: string; busy: boolean; autoStart: boolean };
const storageKey = 'norma.photo-workflow.v1';
const folder = ref(''), albumId = ref(''), albums = ref<{ id: string; name: string; photo_count: number; source_path: string }[]>([]);
const curation = ref<Curation | null>(null), selection = ref<Selection | null>(null), chosen = ref<string[]>([]);
const importedPhotos = ref<Photo[]>([]), importedTotal = ref(0);
const prompt = ref('挑4张雪山照片'), minAesthetic = ref(5), minTechnical = ref(40);
const memory = ref(false), languageModel = ref(false), cloudConsent = ref(false), uploadConsent = ref(false);
const vlm = ref('local'), vlmConfigured = ref(false);
const busy = ref(false), error = ref(''), status = ref('先导入照片，再点击美学评估。');
const job = ref<Job | null>(null), lane = ref<'import' | 'prepare' | 'aesthetics' | ''>('');
const showHidden = ref(false), cards = ref<Card[]>([]), activeCard = ref(0);
let alive = true, jobTimer: ReturnType<typeof setTimeout> | null = null;
const worldTimers = new Map<number, ReturnType<typeof setTimeout>>();
const displayed = computed(() => selection.value?.selected ?? (showHidden.value ? curation.value?.hidden : curation.value?.kept) ?? importedPhotos.value);
const thresholdsChanged = computed(()=>!!curation.value&&(curation.value.thresholds.min_aesthetic!==minAesthetic.value||curation.value.thresholds.min_technical!==minTechnical.value));
const worldsActive = computed(() => cards.value.some(c => c.busy || (c.state && !['closed', 'failed', 'interrupted'].includes(c.state.status))));
const controls = [ ['forward','W','前进'], ['left','A','左移'], ['backward','S','后退'], ['right','D','右移'],
  ['look_up','↑','向上看'], ['look_left','←','向左看'], ['look_down','↓','向下看'], ['look_right','→','向右看'] ];
const phaseNames: Record<string,string> = { queued:'等待执行', loading_model:'加载模型（首次需要等待）', encoding_input:'理解输入照片',
  generating:'生成下一段画面', decoding:'解码连续画面', saving:'保存视频', ready:'可以继续探索', closing:'正在结束', closed:'已结束', failed:'生成失败', interrupted:'服务已重启' };
async function api<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, body === undefined ? {} : { method:'POST', headers:{'Content-Type':'application/json','X-Norma-World':'1'}, body:JSON.stringify(body) });
  const data = await r.json(); if (!r.ok) throw Object.assign(Error(typeof data.detail === 'string' ? data.detail : '请求失败，请核对参数或模型状态'), {status:r.status}); return data;
}
function persist() { localStorage.setItem(storageKey, JSON.stringify({ folder:folder.value, albumId:albumId.value, curation:curation.value, selection:selection.value,
  chosen:chosen.value, prompt:prompt.value, minAesthetic:minAesthetic.value, minTechnical:minTechnical.value,
  job:job.value, lane:lane.value, cards:cards.value.map(c=>({...c,busy:false})) })); }
async function refreshAlbums() { albums.value = (await api<{items: typeof albums.value}>('/albums?limit=200')).items; }
async function previewAlbum() {
  const id=albumId.value;if(!id)return;
  const data=await api<{items: (Photo & {id:string})[];total:number}>(`/albums/${id}/photos?limit=500&include_rejects=true`);
  if(id!==albumId.value||!alive)return;
  importedPhotos.value=data.items.map(p=>({...p,photo_id:p.id}));importedTotal.value=data.total;
}
function clearResults() { curation.value=null; selection.value=null; chosen.value=[]; showHidden.value=false; uploadConsent.value=false; }
async function loadAlbum() {
  if (busy.value || worldsActive.value) return;
  clearResults(); const found=albums.value.find(a=>a.id===albumId.value); if(found) folder.value=found.source_path;
  importedPhotos.value=[];importedTotal.value=0;
  status.value='已选择相册，点击美学评估。'; persist();
  try{await previewAlbum();}catch(e){error.value=String(e);}
}
async function submitPrepare(kind: 'import'|'prepare') {
  busy.value=true; error.value=''; lane.value=kind;
  try {
    job.value=await api<Job>('/jobs/prepare',{ folder:folder.value, include_quality:kind==='prepare', include_embeddings:kind==='prepare', include_people:false });
    persist(); pollJob();
  } catch(e) { error.value=String(e); busy.value=false; lane.value=''; persist(); }
}
async function importPhotos() { if(!folder.value.trim()||worldsActive.value)return; clearResults();importedPhotos.value=[];importedTotal.value=0; await submitPrepare('import'); }
async function assess() { if(!albumId.value||worldsActive.value)return; clearResults(); await submitPrepare('prepare'); }
async function pollJob() {
  if(jobTimer)clearTimeout(jobTimer); if(!alive||!job.value||!lane.value)return;
  try {
    const url=lane.value==='aesthetics'?'/aesthetics/jobs/':'/jobs/';
    job.value=await api<Job>(url+job.value.id); persist();
    const names={import:'导入照片',prepare:'准备质量、相似组和语义模型',aesthetics:'MUSIQ美学与质量评估'};
    status.value=`${names[lane.value]} · ${Math.round(job.value.progress*100)}% · ${job.value.stage}`;
    if(job.value.status==='completed') {
      if(job.value.result?.album?.album_id)albumId.value=job.value.result.album.album_id;
      if(lane.value==='prepare') {
        lane.value='aesthetics'; job.value=await api<Job>(`/albums/${albumId.value}/aesthetics/jobs`,{force:false}); persist();
      } else if(lane.value==='aesthetics') {
        await curate(); lane.value=''; busy.value=false; await refreshAlbums(); persist(); return;
      } else { lane.value='';busy.value=false;status.value='导入完成。原图未修改，点击美学评估开始筛选。';await refreshAlbums();await previewAlbum();persist();return; }
    } else if(['failed','cancelled'].includes(job.value.status)) { throw Error(job.value.error||'操作已取消；已完成的缓存保留，可重新点击评估'); }
    jobTimer=setTimeout(pollJob,1200);
  } catch(e) {error.value=String(e);busy.value=false;status.value='操作停止；不会自动重试昂贵任务。';persist();}
}
async function cancelJob() { if(!job.value)return; try {await api((lane.value==='aesthetics'?'/aesthetics/jobs/':'/jobs/')+job.value.id+'/cancel',{});}catch(e){error.value=String(e);} }
async function curate() {
  curation.value=await api<Curation>(`/workflow/albums/${albumId.value}/curations`,{min_aesthetic:minAesthetic.value,min_technical:minTechnical.value});
  selection.value=null;chosen.value=[];showHidden.value=false;
  status.value=`${curation.value.total} 张 → 保留 ${curation.value.kept.length} 张；其余 ${curation.value.hidden.length} 张仅折叠，原图全部保留。`;
}
async function reFilter() { if(busy.value||worldsActive.value)return;busy.value=true;error.value='';try{await curate();persist();}catch(e){error.value=String(e);}finally{busy.value=false;} }
async function pick() {
  if(!curation.value||busy.value||worldsActive.value)return;
  if(thresholdsChanged.value){error.value='门槛已修改，请先点击重新筛选。';return;}
  if(languageModel.value&&vlm.value==='openai-compatible'&&!cloudConsent.value){error.value='请先确认本次文字发送到云端。';return;}
  busy.value=true;error.value='';chosen.value=[];uploadConsent.value=false;selection.value=null;status.value='模型正在理解文字并从保留照片中选片…';
  try {
    selection.value=await api<Selection>(`/workflow/curations/${curation.value.id}/select`,{prompt:prompt.value,use_preference_memory:memory.value,
      use_language_model:languageModel.value,allow_cloud:cloudConsent.value,default_count:9});
    status.value=selection.value.feasible?`已挑出 ${selection.value.selected.length} 张；勾选其中两张生成交互视频。`:'当前照片不足以满足要求，请调整输入。';
  }catch(e){error.value=String(e);}finally{busy.value=false;cloudConsent.value=false;persist();}
}
function choose(id:string){if(worldsActive.value)return;chosen.value=chosen.value.includes(id)?chosen.value.filter(x=>x!==id):chosen.value.length<2?[...chosen.value,id]:chosen.value;uploadConsent.value=false;persist();}
function artifact(c:Card) { const last=c.state?.history.at(-1);return last?`/demo/world/api/sessions/${c.state!.id}/artifacts/${last.artifact}`:''; }
function canAct(c:Card) {return !c.busy&&c.state?.status==='ready'&&c.state.next_sequence<c.state.max_steps;}
async function pollWorld(index:number){
  const c=cards.value[index];if(!alive||!c?.state?.id)return;
  try {
    c.state=await api<WorldState>(`/demo/world/api/sessions/${c.state.id}`);persist();
    if(c.state.status==='ready'&&c.autoStart&&c.state.next_sequence===0){c.autoStart=false;persist();await act(index,'forward');return;}
    if(['initializing','generating','closing'].includes(c.state.status)) worldTimers.set(index,setTimeout(()=>pollWorld(index),1600));
  }catch(e){c.error=String(e)+'；请刷新核对状态，不自动重发操作。';if([401,404].includes((e as {status?:number}).status||0)&&c.state){c.state.status='interrupted';c.state.phase='interrupted';c.autoStart=false;}c.busy=false;persist();}
}
async function beginVideos(){
  if(chosen.value.length!==2||!selection.value||!uploadConsent.value||worldsActive.value)return;
  error.value='';try{const bootstrap=await fetch('/demo/world/');if(!bootstrap.ok)throw Error('交互视频入口暂不可用');}catch(e){error.value=String(e);return;}
  cards.value=chosen.value.map(id=>({photo:selection.value!.selected.find(p=>p.photo_id===id)!,selectionId:selection.value!.selection_id,
    state:null,error:'',busy:true,autoStart:true}));activeCard.value=0;persist();
  for(let index=0;index<cards.value.length;index++){
    const c=cards.value[index];
    try{
      const r=await fetch(`/workflow/selections/${c.selectionId}/images/${c.photo.photo_id}`);
      if(!r.ok){const d=await r.json();throw Error(d.detail||'输入照片不可用');}
      const blob=await r.blob();
      const image=await new Promise<string>((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=reject;reader.readAsDataURL(blob);});
      c.state=await api<WorldState>('/demo/world/api/sessions',{image_base64:image,upload_confirmed:true,
        prompt:'Photorealistic scene. Preserve the source scene structure, subjects and lighting. Camera motion follows the supplied controls.'});
      c.busy=false;persist();pollWorld(index);
    }catch(e){c.error=String(e);c.busy=false;c.autoStart=false;persist();}
  }
  uploadConsent.value=false;
}
async function act(index:number,action:string){const c=cards.value[index];if(!canAct(c))return;c.busy=true;c.error='';activeCard.value=index;
  try{c.state=await api<WorldState>(`/demo/world/api/sessions/${c.state!.id}/steps`,{sequence:c.state!.next_sequence,action});pollWorld(index);}
  catch(e){c.error=String(e);c.autoStart=false;}finally{c.busy=false;persist();}}
async function close(index:number){const c=cards.value[index];if(!c.state||c.busy)return;c.autoStart=false;c.busy=true;
  try{c.state=await api<WorldState>(`/demo/world/api/sessions/${c.state.id}/close`,{});pollWorld(index);}catch(e){c.error=String(e);}finally{c.busy=false;persist();}}
function keydown(e:KeyboardEvent){
  const target=e.target as HTMLElement;if(e.repeat||e.ctrlKey||e.metaKey||e.altKey||target.isContentEditable||['INPUT','TEXTAREA','SELECT'].includes(target.tagName))return;
  const mapping:Record<string,string>={w:'forward',a:'left',s:'backward',d:'right',ArrowUp:'look_up',ArrowLeft:'look_left',ArrowDown:'look_down',ArrowRight:'look_right'};
  const action=mapping[e.key]||mapping[e.key.toLowerCase()];const c=cards.value[activeCard.value];if(action&&c&&canAct(c)){e.preventDefault();act(activeCard.value,action);}
}
watch(prompt,()=>{cloudConsent.value=false;});
onMounted(async()=>{
  document.addEventListener('keydown',keydown);
  try{
    const health=await api<{vlm_provider:string;vlm_configured:boolean}>('/health');vlm.value=health.vlm_provider;vlmConfigured.value=health.vlm_configured;
    await refreshAlbums();await fetch('/demo/world/');
    const raw=localStorage.getItem(storageKey);
    if(raw){const saved=JSON.parse(raw);folder.value=saved.folder||'';albumId.value=saved.albumId||'';prompt.value=saved.prompt||prompt.value;
      minAesthetic.value=saved.minAesthetic??5;minTechnical.value=saved.minTechnical??40;selection.value=saved.selection||null;chosen.value=saved.chosen||[];
      curation.value=saved.curation||null;cards.value=(saved.cards||[]).slice(0,2);job.value=saved.job||null;lane.value=saved.lane||'';
      if(lane.value&&job.value){busy.value=true;pollJob();}
      else if(curation.value){try{curation.value=await api<Curation>(`/workflow/curations/${curation.value.id}`);status.value='已恢复上次筛选。';}catch{clearResults();status.value='旧筛选已失效，请重新评估。';}}
      cards.value.forEach((c,i)=>{c.busy=false;if(c.state?.id)pollWorld(i);});
      if(albumId.value&&!curation.value)await previewAlbum();
    }
  }catch(e){error.value=String(e);}
});
onBeforeUnmount(()=>{alive=false;if(jobTimer)clearTimeout(jobTimer);worldTimers.forEach(clearTimeout);document.removeEventListener('keydown',keydown);persist();});
</script>

<template>
<section class="photo-workflow" aria-label="照片到交互视频 Demo">
  <div class="flow-heading"><div><small>NORMA / PHOTO TO WORLD</small><h2>从一组照片，到可探索的瞬间</h2></div><span>01 导入 · 02 美学精选 · 03 文字选片 · 04 探索</span></div>
  <div class="flow-toolbar">
    <input v-model="folder" aria-label="照片文件夹路径" placeholder="输入本地照片文件夹，例如 F:\摄影\旅行" :disabled="busy||worldsActive" />
    <button @click="importPhotos" :disabled="busy||worldsActive||!folder.trim()">导入照片</button>
    <select v-model="albumId" aria-label="选择已导入相册" @change="loadAlbum" :disabled="busy||worldsActive"><option value="">已有相册</option><option v-for="a in albums" :value="a.id" :key="a.id">{{a.name}} · {{a.photo_count}} 张</option></select>
    <button class="accent" @click="assess" :disabled="busy||worldsActive||!albumId">美学评估 · 相似只留最佳</button>
  </div>
  <div class="flow-options"><label>美学 ≥ <input v-model.number="minAesthetic" type="number" min="1" max="10" step="0.1" :disabled="busy||worldsActive" /></label><label>技术质量 ≥ <input v-model.number="minTechnical" type="number" min="0" max="100" :disabled="busy||worldsActive" /></label>
    <button v-if="curation" @click="reFilter" :disabled="busy||worldsActive">重新筛选（复用评分）</button><small>相似组按模型综合分留一张；其余折叠，不删除原图。</small></div>
  <div class="flow-status" role="status">{{status}} <button v-if="busy&&lane&&job" @click="cancelJob">取消任务</button></div>
  <progress v-if="busy&&lane&&job" :value="job.progress" max="1" :aria-label="status" />
  <p v-if="error" class="flow-error" role="alert">{{error}}</p>
  <p v-if="thresholdsChanged" class="flow-notice">门槛已修改，点击“重新筛选”后才会应用到选片。</p>
  <p v-if="!curation&&importedTotal" class="flow-notice">已导入 {{importedTotal}} 张。{{importedTotal>500?'此处预览前500张，评估仍处理整个相册。':'点击美学评估后，再按模型分数筛选。'}}</p>
  <div v-if="curation" class="flow-prompt"><input v-model="prompt" aria-label="模型选片要求" placeholder="挑4张雪山照片 / 挑一组适合发旅游朋友圈的照片" @keyup.enter="pick" :disabled="busy||worldsActive" /><button @click="pick" class="accent" :disabled="busy||worldsActive||!curation.kept.length||!prompt.trim()">{{busy&&!lane?'模型选片中…':'模型选片'}}</button></div>
  <div v-if="curation" class="flow-options"><small>本地 OpenCLIP 理解图文相似度 + MUSIQ 美学排序；未写数量默认最多9张，不调用规则分数冒充模型。</small><label><input type="checkbox" v-model="memory" :disabled="busy" />参考我的偏好</label>
    <details><summary>文字大模型解析（可选）</summary><label><input type="checkbox" v-model="languageModel" :disabled="busy||!vlmConfigured" />额外调用已配置大模型解析文字</label><p v-if="vlm==='local'">当前本地解析模型尚未稳定通过验证；开启后失败会停止，不会自动切换。</p><label v-else><input type="checkbox" v-model="cloudConsent" :disabled="busy" />同意发送本次文字到云端，不发送相册图片</label></details></div>
  <div v-if="curation" class="flow-results-head"><strong>{{selection?`模型选出 ${selection.selected.length} 张`:`美学保留 ${curation.kept.length} / ${curation.total} 张`}}</strong>
    <button v-if="selection" @click="selection=null;chosen=[];uploadConsent=false;persist()" :disabled="worldsActive">返回美学筛选</button>
    <button v-else @click="showHidden=!showHidden">{{showHidden?'查看保留照片':`查看折叠 ${curation.hidden.length} 张`}}</button>
    <small v-if="selection?.intent_provenance?.effective_prompt">本次要求：{{selection.intent_provenance.effective_prompt}}</small></div>
  <p v-for="warning in selection?.warnings||[]" :key="warning" class="flow-error">{{warning}}</p>
  <div class="flow-grid"><article v-for="p in displayed" :key="p.photo_id" :class="{picked:chosen.includes(p.photo_id)}">
    <button v-if="selection" class="photo-pick" :aria-label="`选择 ${p.filename} 生成视频`" :aria-pressed="chosen.includes(p.photo_id)" @click="choose(p.photo_id)" :disabled="worldsActive||(!chosen.includes(p.photo_id)&&chosen.length>=2)"><img :src="p.thumbnail_url" :alt="p.filename" loading="lazy" /><span>{{chosen.includes(p.photo_id)?'✓ 已选':'＋ 生成视频'}}</span></button>
    <img v-else :src="p.thumbnail_url" :alt="p.filename" loading="lazy" />
    <footer><small>{{p.filename}}</small><small v-if="p.aesthetic_quality!=null">美学 {{p.aesthetic_quality.toFixed(2)}} · 质量 {{p.technical_quality?.toFixed(1)}}</small><details v-if="p.reasons?.length"><summary>为什么选／折叠</summary><p v-for="r in p.reasons" :key="r">{{r}}</p></details></footer>
  </article></div>
  <div v-if="selection" class="flow-video-start"><strong>已选 {{chosen.length}} / 2 张</strong><label><input type="checkbox" v-model="uploadConsent" :disabled="worldsActive||chosen.length!==2" />将这两张照片发送到我的3090服务器</label><button class="accent" @click="beginVideos" :disabled="worldsActive||chosen.length!==2||!uploadConsent">生成两张交互视频</button><small>不拼接照片；每张独立探索，首版最多7步 / 10.31秒。</small></div>
  <div class="world-pair"><article v-for="(c,i) in cards" :key="c.photo.photo_id" :class="{active:activeCard===i}" tabindex="0" @click="activeCard=i" @focus="activeCard=i" :aria-label="`探索 ${c.photo.filename}`">
    <header><strong>{{c.photo.filename}}</strong><span>{{activeCard===i?'键盘控制此场景':'点击切换控制'}}</span></header>
    <video v-if="artifact(c)" :src="artifact(c)" controls playsinline :key="artifact(c)" /><img v-else :src="c.photo.thumbnail_url" :alt="c.photo.filename" />
    <div class="world-under"><p>{{c.state?(c.state.status==='ready'&&c.state.next_sequence>=c.state.max_steps?'本轮探索完成，可下载或结束':phaseNames[c.state.phase]||c.state.status):c.busy?'准备并发送照片…':'未开始'}} · {{c.state?.next_sequence||0}} / 7 步</p>
      <progress v-if="c.state&&['initializing','generating','closing'].includes(c.state.status)" :value="c.state.phase==='generating'&&c.state.total_chunks?c.state.completed_chunks:undefined" :max="c.state.total_chunks||1" />
      <p v-if="c.error" class="flow-error">{{c.error}}</p><div class="world-keyboard"><button v-for="[action,key,label] in controls" :key="action" :title="label" :aria-label="`${c.photo.filename} ${label}`" :disabled="!canAct(c)" @click.stop="act(i,action)"><kbd>{{key}}</kbd><small>{{label}}</small></button></div>
      <div class="world-actions"><button :disabled="!canAct(c)" @click="act(i,'hold')">停留</button><button :disabled="!c.state||c.busy||['closed','failed','interrupted'].includes(c.state.status)" @click="close(i)">结束／取消</button><a v-if="artifact(c)" :href="artifact(c)" download>下载当前视频</a></div>
      <small>WASD 移动 · 方向键转头；生成完成后再按下一步。画面外内容由模型补全，并非真实三维重建。</small>
    </div></article></div>
</section>
</template>

<style scoped>
.photo-workflow{padding:24px 28px;color:#e7e9e3}.flow-heading,.flow-toolbar,.flow-options,.flow-results-head,.flow-video-start,.world-pair header{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.flow-heading{justify-content:space-between;margin-bottom:20px}.flow-heading h2{font-size:25px;font-weight:500;margin:6px 0}.flow-heading small,.flow-heading>span,small{color:#a6afa0}.flow-heading>span{font-size:12px}.photo-workflow input,.photo-workflow select,.photo-workflow button{font:inherit;color:inherit;background:#22281f;border:1px solid #3d4937;border-radius:7px;padding:10px}.photo-workflow button{cursor:pointer}.photo-workflow button:disabled{opacity:.4;cursor:default}.flow-toolbar>input{flex:1;min-width:240px}.photo-workflow .accent{background:#ba6540;color:#fff;border-color:#ba6540}.flow-options{font-size:12px;margin:12px 0;gap:14px}.flow-options input[type=number]{width:68px;padding:5px}.flow-options label,.flow-video-start label{display:flex;gap:6px;align-items:center}.flow-options details{max-width:520px}.flow-options details p{line-height:1.5}.flow-status{font-size:13px;color:#bbc7b2;margin:14px 0}.flow-error{color:#f2b09e;font-size:13px}progress{width:100%;height:8px;accent-color:#aabd91}.flow-prompt{display:flex;gap:10px;margin-top:22px}.flow-prompt input{flex:1;min-width:0}.flow-results-head{margin:20px 0 12px}.flow-results-head button{font-size:12px;padding:6px}.flow-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}.flow-grid article{border:2px solid transparent;border-radius:8px;overflow:hidden;background:#171c15}.flow-grid article.picked{border-color:#cb8158}.flow-grid img{width:100%;height:230px;object-fit:cover;display:block}.flow-grid footer{padding:8px;display:flex;flex-direction:column;gap:4px;font-size:11px}.flow-grid footer>small{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.photo-workflow .photo-pick{padding:0;width:100%;position:relative;border:0;border-radius:0;background:none}.photo-pick span{position:absolute;top:8px;right:8px;padding:5px;background:#111b;border-radius:4px;font-size:12px}.flow-video-start{position:sticky;bottom:0;padding:16px;background:#1a221bf0;border:1px solid #3d4937;border-radius:8px;margin:18px 0;font-size:13px;z-index:3}.world-pair{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:24px}.world-pair>article{border:1px solid #394532;border-radius:10px;overflow:hidden;background:#141912;min-width:0}.world-pair>article.active{border-color:#d79365}.world-pair header{justify-content:space-between;padding:12px;font-size:12px}.world-pair>article>video,.world-pair>article>img{display:block;width:100%;height:500px;object-fit:contain;background:#090c08}.world-under{padding:12px}.world-under>p{margin:4px 0 12px;font-size:13px}.world-keyboard{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.world-keyboard button{display:flex;flex-direction:column;align-items:center;gap:4px;padding:8px}.world-keyboard kbd{font:18px monospace}.world-actions{display:flex;align-items:center;gap:12px;margin:12px 0;font-size:12px}.world-actions a{color:#bcd2a8}.world-under>small{font-size:11px;line-height:1.5;display:block}@media(max-width:750px){.photo-workflow{padding:16px}.world-pair{grid-template-columns:1fr}.flow-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.flow-grid img{height:180px}.world-pair>article>video,.world-pair>article>img{height:420px}}
</style>
<style scoped>
.flow-notice{font-size:12px;color:#c1cbb7}.world-keyboard{grid-template-columns:repeat(6,minmax(0,1fr));grid-template-rows:repeat(2,auto)}
.world-keyboard button:nth-child(1){grid-area:1/2}.world-keyboard button:nth-child(2){grid-area:2/1}.world-keyboard button:nth-child(3){grid-area:2/2}.world-keyboard button:nth-child(4){grid-area:2/3}
.world-keyboard button:nth-child(5){grid-area:1/5}.world-keyboard button:nth-child(6){grid-area:2/4}.world-keyboard button:nth-child(7){grid-area:2/5}.world-keyboard button:nth-child(8){grid-area:2/6}
</style>
