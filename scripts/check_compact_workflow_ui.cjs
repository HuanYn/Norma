// Real local Qwen + existing development gallery. No human labels or cloud calls.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(process.env.NORMA_PLAYWRIGHT_PATH || 'playwright');
async function main(){
 const output=path.resolve(process.argv[2]);fs.mkdirSync(output,{recursive:false});
 const before=JSON.parse(fs.readFileSync('.norma/photo-workflow-browser-20261005-03/result.json','utf8'));
 const report={scope:'real local optional parser + development-gallery integration, NOT independent retrieval quality',errors:[]};
 const browser=await chromium.launch({headless:true,channel:'msedge'});const page=await browser.newPage({viewport:{width:1440,height:1100}});
 try{
  page.on('pageerror',e=>report.errors.push(String(e)));
  await page.route('**/*',r=>new URL(r.request().url()).origin==='http://127.0.0.1:8767'?r.continue():r.abort());
  await page.addInitScript(s=>localStorage.setItem('norma.photo-workflow.v1',JSON.stringify(s)),{folder:path.resolve('.norma/workflow-demo-104'),albumId:before.album_id,curation:before.curation,chosen:[],cards:[],minAesthetic:5,minTechnical:40});
  await page.goto('http://127.0.0.1:8767/',{waitUntil:'domcontentloaded'});
  await page.getByText('文字大模型解析（可选）',{exact:true}).click();
  await page.getByLabel('额外调用已配置大模型解析文字').check();
  await page.getByLabel('模型选片要求').fill('选3张建筑，偏暖色电影感');
  await page.screenshot({path:path.join(output,'before-request.png'),fullPage:true});
  const started=Date.now(),response=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/select'),{timeout:360000});
  await page.getByRole('button',{name:'模型选片',exact:true}).click();
  const r=await response;report.status_code=r.status();report.selection=await r.json();report.request_seconds=(Date.now()-started)/1000;
  assert.equal(r.status(),200,JSON.stringify(report.selection));assert.equal(report.selection.selected.length,3);
  assert.match(JSON.stringify(report.selection.intent_provenance),/norma-compact-semantics-v2/);
  await page.locator('.flow-grid .photo-pick').first().waitFor();
  await page.screenshot({path:path.join(output,'after-selection.png'),fullPage:true});
  assert.deepEqual(report.errors,[]);report.status='passed';
 }catch(e){report.status='failed';report.error=String(e);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw e;}
 finally{fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(report,null,2));await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
