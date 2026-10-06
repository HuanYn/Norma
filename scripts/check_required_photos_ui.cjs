// Verifies pins with real model retrieval; never claims two identities were verified.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const base='http://127.0.0.1:8767',album='998eaeb56c485567aa46f7753fbe6fc2';
(async()=>{
 const output=path.resolve(process.argv[2]);fs.mkdirSync(output,{recursive:false});
 const report={scope:'real image inclusion only; manual person references remain awaiting user',status:'running',errors:[]};
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage({viewport:{width:1700,height:1050}});
 try{
  await page.route('**/demo/world/api/**',()=>{throw Error('No GPU calls');});
  await page.route('**/people/**',()=>{throw Error('No face recognition');});
  page.on('pageerror',e=>report.errors.push(String(e)));
  await page.addInitScript(state=>localStorage.setItem('norma.photo-workflow.v1',JSON.stringify(state)),{albumId:album,folder:'D:\\Photos\\Trip',cards:[]});
  await page.goto(base,{waitUntil:'domcontentloaded'});
  await page.getByText('本次必留 2 张：',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('模型选片要求').inputValue(),'挑出最有代表性的9张照片，两个不同的人最少分别有一张');
  await page.getByRole('button',{name:'模型选片',exact:true}).click();
  await page.getByText('请在下面照片上各点一张“人物A代表照”和“人物B代表照”，确认它们分别代表两个人。',{exact:true}).waitFor();
  await page.screenshot({path:path.join(output,'pending-representatives.png')});
  const draft=await (await page.request.get(base+`/workflow/albums/${album}/selection-draft`)).json();
  const blocked=await page.request.post(base+`/workflow/curations/${draft.curation_id}/select`,{data:{prompt:draft.prompt,required_photo_ids:draft.required_photo_ids},timeout:180000});
  assert.equal(blocked.status(),409);report.missing_representatives=await blocked.json();
  // Deliberately tests only the 9-photo + attached-images part. Not the full user request.
  const started=Date.now();
  const picked=await page.request.post(base+`/workflow/curations/${draft.curation_id}/select`,{data:{prompt:'挑出最有代表性的9张照片',required_photo_ids:draft.required_photo_ids},timeout:180000});
  assert.equal(picked.status(),200);report.inclusion_only=await picked.json();report.seconds=(Date.now()-started)/1000;
  assert.equal(report.inclusion_only.selected.length,9);
  assert.ok(draft.required_photo_ids.every(id=>report.inclusion_only.selected.some(p=>p.photo_id===id)));
  assert.equal(report.inclusion_only.intent_provenance.manual_representatives,null);
  report.full_user_request_completed=false;
  report.status='passed';assert.deepEqual(report.errors,[]);
 }catch(e){report.status='failed';report.error=String(e);await page.screenshot({path:path.join(output,'failure.png')}).catch(()=>{});throw e;}
 finally{fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(report,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
