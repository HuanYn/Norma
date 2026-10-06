// Real CPU model regression. Video APIs are forbidden; no GPU worker startup.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const origin='http://127.0.0.1:8767';
(async()=>{
 const output=path.resolve(process.argv[2]);fs.mkdirSync(output,{recursive:false});
 const curationId=process.argv[3];assert.ok(curationId);
 const report={scope:'real CPU models on trip; category predictions require separate visual inspection, not an independent benchmark',errors:[]};
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage({viewport:{width:1700,height:1050}});
 try{
  page.on('pageerror',e=>report.errors.push(String(e)));
  await page.route('**/*',route=>new URL(route.request().url()).origin===origin?route.continue():route.abort());
  await page.route('**/demo/world/api/**',()=>{throw Error('World API forbidden');});
  const response=await page.request.get(origin+'/workflow/curations/'+curationId);assert.equal(response.status(),200);
  const curation=await response.json();
  await page.addInitScript(state=>localStorage.setItem('norma.photo-workflow.v1',JSON.stringify(state)),{
   folder:'D:\\Photos\\Trip',albumId:curation.album_id,curation,chosen:[],cards:[],minAesthetic:5,minTechnical:40,preferenceProfile:'assistant'});
  await page.goto(origin,{waitUntil:'domcontentloaded'});
  await page.getByText('文字大模型解析（可选）',{exact:true}).click();
  await page.getByLabel('额外调用已配置大模型解析文字').check();
  const prompt='挑出最有代表性的6张照片，一张人5张风景';
  await page.getByLabel('模型选片要求').fill(prompt);
  const started=Date.now(),pending=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/select'),{timeout:360000});
  await page.getByRole('button',{name:'模型选片',exact:true}).click();
  const selected=await pending;report.http=selected.status();report.result=await selected.json();report.seconds=(Date.now()-started)/1000;
  fs.writeFileSync(path.join(output,'response.json'),JSON.stringify(report,null,2));
  assert.equal(selected.status(),200,JSON.stringify(report.result));assert.equal(report.result.selected.length,6);
  assert.deepEqual(report.result.constraints.category_counts,{portrait:1,landscape:5});
  assert.equal(report.result.intent_provenance.default_count_applied,false);
  assert.equal(report.result.selected.filter(p=>p.category_evidence.label==='portrait').length,1);
  assert.equal(report.result.selected.filter(p=>p.category_evidence.label==='landscape').length,5);
  assert.ok(report.result.selected.every(p=>curation.kept.some(k=>k.photo_id===p.photo_id)));
  const ids=new Set(report.result.selected.map(p=>p.photo_id));
  assert.ok(report.result.intent_provenance.collection_plan.pair_conflicts.every(([a,b])=>!ids.has(a)||!ids.has(b)));
  await page.locator('.flow-grid .photo-pick').first().waitFor();
  await page.locator('.flow-grid img').evaluateAll(imgs=>Promise.all(imgs.map(i=>i.decode().catch(()=>{}))));
  await page.screenshot({path:path.join(output,'after-selection.png'),fullPage:true});
  for(const photo of report.result.selected){
   const r=await page.request.get(origin+photo.thumbnail_url);
   assert.equal(r.status(),200);fs.writeFileSync(path.join(output,photo.filename+'.thumb.jpg'),await r.body());
  }
  // Actual backend negative test, no mocked responses, fails before model load.
  const conflict=await page.request.post(origin+'/workflow/curations/'+curationId+'/select',{data:{prompt:'选6张，2张人5张风景'}});
  assert.equal(conflict.status(),409);report.conflict=await conflict.json();
  const warmStart=Date.now();
  const warm=await page.request.post(origin+'/workflow/curations/'+curationId+'/select',{
    data:{prompt,use_language_model:true,use_preference_memory:true,allow_proxy_memory:true},timeout:180000});
  assert.equal(warm.status(),200);const warmResult=await warm.json();
  assert.deepEqual(warmResult.selected.map(p=>p.photo_id),report.result.selected.map(p=>p.photo_id));
  report.warm_repeat={seconds:(Date.now()-warmStart)/1000,selection_id:warmResult.selection_id,same_selected_ids:true};
  assert.deepEqual(report.errors,[]);report.status='passed';
 }catch(e){report.status='failed';report.failure=String(e);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw e;}
 finally{fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(report,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
