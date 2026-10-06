// Real trip-album text-selection regression. No video or GPU worker requests.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const origin='http://127.0.0.1:8767';
(async()=>{
 const output=path.resolve(process.argv[2]);fs.mkdirSync(output,{recursive:false});
 const curationId=process.argv[3];assert.ok(curationId);
 const report={scope:'real local CPU LLM + OpenCLIP + existing MUSIQ on user trip album; not an accuracy benchmark',errors:[]};
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage({viewport:{width:1440,height:1050}});
 try{
  page.on('pageerror',e=>report.errors.push(String(e)));
  await page.route('**/*',route=>new URL(route.request().url()).origin===origin?route.continue():route.abort());
  await page.route('**/demo/world/api/**',()=>{throw Error('No world API is allowed in this test');});
  const response=await page.request.get(origin+'/workflow/curations/'+curationId);assert.equal(response.status(),200);
  const curation=await response.json();assert.equal(curation.album_id,'998eaeb56c485567aa46f7753fbe6fc2');
  await page.addInitScript(state=>localStorage.setItem('norma.photo-workflow.v1',JSON.stringify(state)),{
   folder:'D:\\Photos\\Trip',albumId:curation.album_id,curation,chosen:[],cards:[],minAesthetic:5,minTechnical:40,preferenceProfile:'assistant'});
  await page.goto(origin,{waitUntil:'domcontentloaded'});
  await page.getByText('文字大模型解析（可选）',{exact:true}).click();
  await page.getByLabel('额外调用已配置大模型解析文字').check();
  const prompt='挑出最适合这趟旅游展示的9张照片';
  await page.getByLabel('模型选片要求').fill(prompt);
  const started=Date.now(),pending=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/select'),{timeout:360000});
  await page.getByRole('button',{name:'模型选片',exact:true}).click();
  const selected=await pending;report.http=selected.status();report.result=await selected.json();report.seconds=(Date.now()-started)/1000;
  assert.equal(selected.status(),200,JSON.stringify(report.result));assert.equal(report.result.selected.length,9);
  assert.equal(report.result.intent_provenance.text_interpretation,'language-model');
  assert.equal(report.result.intent_provenance.language_model.contract_version,'norma-compact-semantics-v4');
  assert.equal(report.result.intent_provenance.default_count_applied,false);
  assert.ok(report.result.selected.every(p=>curation.kept.some(k=>k.photo_id===p.photo_id)));
  await page.locator('.flow-grid .photo-pick').first().waitFor();
  await page.screenshot({path:path.join(output,'after-selection.png'),fullPage:true});
  // Explicit failure fixture checks only the UI recovery, not actual inference.
  await page.route('**/workflow/curations/*/select',route=>route.fulfill({status:409,contentType:'application/json',body:JSON.stringify({detail:'Requirement issue is not grounded in the request'})}));
  await page.getByRole('button',{name:'模型选片',exact:true}).click();
  await page.getByText('选片未完成，已停止；请调整要求后重试。',{exact:true}).waitFor();
  assert.match(await page.locator('.photo-workflow > .flow-error').innerText(),/模型添加了你没有提出的条件/);
  assert.ok(await page.getByRole('button',{name:'模型选片',exact:true}).isEnabled());
  report.error_recovery_fixture='passed';
  await page.screenshot({path:path.join(output,'error-recovery-fixture.png'),fullPage:true});
  assert.deepEqual(report.errors,[]);report.status='passed';
 }catch(e){report.status='failed';report.failure=String(e);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw e;}
 finally{fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(report,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
