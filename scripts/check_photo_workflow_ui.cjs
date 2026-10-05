// Real development album + local learned scores + two independent GPU worlds.
// No cloud request, no fake user feedback. Creates local demo selections only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require('C:/Users/25438/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const origin='http://127.0.0.1:8767', key='norma.photo-workflow.v1';
const output=path.resolve(process.argv[2]||'.norma/photo-workflow-browser-20261005');
const report={scope:'104 development photos; real pipeline integration, NOT independent model-quality evaluation',errors:[],worlds:[],stages:[]};
async function main(){
 fs.mkdirSync(output,{recursive:false});
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 let page; const owned=new Set();
 const save=()=>fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(report,null,2));
 try{
  page=await browser.newPage({viewport:{width:1440,height:1100}});
  page.on('pageerror',e=>report.errors.push(String(e)));
  page.on('response',async r=>{if(r.request().method()==='POST'&&r.url()===origin+'/demo/world/api/sessions'&&r.status()===202){const value=await r.json();owned.add(value.id);}});
  await page.route('**/*',route=>{const r=route.request(),u=new URL(r.url());if(u.origin!==origin){report.errors.push('Unexpected external request: '+u.origin);return route.abort();}return route.continue();});
  await page.goto(origin,{waitUntil:'domcontentloaded'});
  await page.getByLabel('照片文件夹路径').waitFor();
  await page.getByLabel('照片文件夹路径').fill(path.resolve('.norma/workflow-demo-104'));
  await page.screenshot({path:path.join(output,'01-before-import.png'),fullPage:true});
  await page.getByRole('button',{name:'导入照片',exact:true}).click();
  const state=()=>page.evaluate(k=>JSON.parse(localStorage.getItem(k)||'{}'),key);
  async function waitState(predicate,limit=1200000){
    const end=Date.now()+limit;let last='';
    while(Date.now()<end){const s=await state();if(predicate(s))return s;
      const error=await page.locator('.photo-workflow > .flow-error, .world-under .flow-error').allTextContents();if(error.length)throw Error(error.join('\n'));
      if(s.cards?.some(c=>['failed','interrupted'].includes(c.state?.status)))throw Error('World failed: '+JSON.stringify(s.cards));
      const phase=s.lane+':'+s.job?.stage;if(phase!==last){report.stages.push({at:new Date().toISOString(),phase});save();last=phase;console.log(phase);}
      await page.waitForTimeout(1800);
    }throw Error('Workflow timeout');
  }
  let s=await waitState(s=>s.albumId&&!s.lane&&s.job?.status==='completed');report.album_id=s.albumId;
  const album=await (await page.request.get(origin+'/albums/'+s.albumId)).json();assert.equal(album.photo_count,104);report.imported=album.photo_count;
  await page.screenshot({path:path.join(output,'02-imported.png'),fullPage:true});
  await page.getByRole('button',{name:'美学评估 · 相似只留最佳',exact:true}).click();
  s=await waitState(s=>s.curation&&!s.lane);report.curation=s.curation;
  assert.equal(s.curation.total,104);assert.ok(s.curation.kept.length>=4);
  assert.equal(new Set(s.curation.kept.map(p=>p.group)).size,s.curation.kept.length);
  assert.equal(s.curation.kept.length+s.curation.hidden.length,104);
  await page.screenshot({path:path.join(output,'03-after-aesthetics.png'),fullPage:true});save();
  async function select(text,name){
    await page.getByLabel('模型选片要求').fill(text);
    const response=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/select'),{timeout:180000});
    await page.getByRole('button',{name:'模型选片',exact:true}).click();
    const r=await response;const data=await r.json();assert.equal(r.status(),200,JSON.stringify(data));
    await waitState(s=>s.selection?.selection_id===data.selection_id);
    assert.ok(data.feasible);assert.ok(data.selected.every(p=>report.curation.kept.some(q=>q.photo_id===p.photo_id)));
    report[name]=data;save();await page.screenshot({path:path.join(output,name+'.png'),fullPage:true});return data;
  }
  assert.equal((await select('挑4长雪山照片','04-snow-query')).selected.length,4);
  await select('挑一组适合发旅游朋友圈的照片','05-travel-query');
  if(process.argv.includes('--selection-only')){
    report.scope+='; selection-only, no GPU sessions';
    const thresholds=page.locator('.flow-options input[type=number]');
    await thresholds.nth(0).fill('5.5');
    await page.getByRole('button',{name:'模型选片',exact:true}).click();
    assert.match(await page.locator('.photo-workflow > .flow-error').innerText(),/门槛已修改/);
    report.threshold_change_guard=true;report.status='passed';return;
  }
  await page.locator('.flow-grid .photo-pick').nth(0).click();await page.locator('.flow-grid .photo-pick').nth(1).click();
  assert.ok(await page.getByRole('button',{name:'生成两张交互视频',exact:true}).isDisabled());
  await page.getByLabel('将这两张照片发送到我的3090服务器').check();
  await page.getByRole('button',{name:'生成两张交互视频',exact:true}).click();
  async function worlds(counts){return waitState(s=>s.cards?.length===2&&s.cards.every((c,i)=>c.state?.status==='ready'&&c.state.next_sequence===counts[i]&&!c.busy),900000);}
  s=await worlds([1,1]);assert.notEqual(s.cards[0].state.id,s.cards[1].state.id);
  report.session_ids=s.cards.map(c=>c.state.id);save();
  const cards=page.locator('.world-pair > article');
  await cards.nth(0).locator('header').click();await page.keyboard.press('w');await worlds([2,1]);
  await page.keyboard.press('ArrowRight');await worlds([3,1]);
  await cards.nth(1).locator('header').click();await page.keyboard.press('ArrowLeft');await worlds([3,2]);
  report.keyboard_isolation=true;
  await page.screenshot({path:path.join(output,'06-two-worlds-controls.png'),fullPage:true});
  await page.reload({waitUntil:'domcontentloaded'});await worlds([3,2]);report.refresh_preserved=true;
  for(let i=0;i<2;i++){
    let current=(await state()).cards[i].state.next_sequence;
    while(current<7){const c=(await state()).cards[i];await page.getByRole('button',{name:`${c.photo.filename} 前进`,exact:true}).click();
      const expected=(await state()).cards.map(c=>c.state.next_sequence);expected[i]=current+1;await worlds(expected);current++;}
  }
  s=await worlds([7,7]);
  for(let i=0;i<2;i++){
    const c=s.cards[i];report.worlds.push(c.state);
    const video=page.locator('.world-pair > article').nth(i).locator('video');
    await video.evaluate(async e=>{e.muted=true;e.currentTime=0;await e.play();});
    await page.waitForFunction(i=>document.querySelectorAll('.world-pair video')[i]?.ended,i,{timeout:25000});
    const playback=await video.evaluate(e=>({duration:e.duration,rate:e.playbackRate,error:e.error?.code||null}));
    assert.equal(playback.duration,10.3125);assert.equal(playback.rate,1);assert.equal(playback.error,null);
    const item=c.state.history.at(-1);const r=await page.request.get(`${origin}/demo/world/api/sessions/${c.state.id}/artifacts/${item.artifact}`);
    assert.equal(r.status(),200);const bytes=await r.body();fs.writeFileSync(path.join(output,`world-${i+1}.mp4`),bytes);
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),item.sha256);
  }
  await page.screenshot({path:path.join(output,'07-completed.png'),fullPage:true});assert.deepEqual(report.errors,[]);report.status='passed';
 }catch(e){report.status='failed';report.failure=String(e);if(page)await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw e;}
 finally{if(page)for(const sid of owned)await page.request.post(`${origin}/demo/world/api/sessions/${sid}/close`,{data:{},headers:{'X-Norma-World':'1'}}).catch(()=>{});save();await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
