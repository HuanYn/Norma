// UI-only contract checks. Mock session transport; NEVER submit GPU work.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require('playwright');
const base='http://127.0.0.1:8767', key='norma.photo-workflow.v1';
const out=path.resolve(process.argv[2]||'.norma/world-hud-ui-20261005-v1');
const source=JSON.parse(fs.readFileSync('.norma/demo-final-20261005-v2/result.json','utf8'));
const worlds=source.worlds.map(w=>({...w,status:'ready',phase:'ready',next_sequence:3,max_steps:14,history:w.history.slice(0,3)}));
const initial=()=>({albumId:source.album_id,curation:source.curation,selection:source['05-travel-query'],
  cards:worlds.map((w,i)=>({photo:source['05-travel-query'].selected[i],selectionId:source['05-travel-query'].selection_id,state:w,error:'',busy:false,autoStart:false}))});
const result={scope:'UI-only mocked session transport with real recorded video; no new GPU inference',checks:[],errors:[]};
(async()=>{
 fs.mkdirSync(out,{recursive:false});const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage({viewport:{width:1440,height:1100}});let submitted=[];
 try{
  page.on('pageerror',e=>result.errors.push(String(e)));
  await page.route('**/demo/world/api/**',async route=>{
   const request=route.request(),url=new URL(request.url());const i=worlds.findIndex(w=>url.pathname.includes(w.id));
   if(i<0)throw Error('Unexpected world API request: '+url.pathname);
   if(url.pathname.includes('/artifacts/')){
    const bytes=fs.readFileSync(`.norma/demo-final-20261005-v2/world-${i+1}.mp4`),range=request.headers().range;
    if(range){const match=/bytes=(\d+)-(\d*)/.exec(range);assert.ok(match);const start=Number(match[1]),end=match[2]?Math.min(Number(match[2]),bytes.length-1):bytes.length-1;
     return route.fulfill({status:206,contentType:'video/mp4',headers:{'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${bytes.length}`},body:bytes.subarray(start,end+1)});}
    return route.fulfill({status:200,contentType:'video/mp4',headers:{'Accept-Ranges':'bytes'},body:bytes});
   }
   if(request.method()==='POST'){
    assert.ok(url.pathname.endsWith('/steps'));const body=request.postDataJSON();submitted.push({i,...body});
    worlds[i]={...worlds[i],status:'generating',phase:'generating'};
   }
   return route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(worlds[i])});
  });
  await page.addInitScript(({key,state})=>localStorage.setItem(key,JSON.stringify(state)),{key,state:initial()});
  await page.goto(base,{waitUntil:'domcontentloaded'});
  const stages=page.locator('.world-stage');await stages.first().waitFor();
  const button=()=>stages.first().getByRole('button',{name:'user-mountain.jpg 前进',exact:true});
  await button().waitFor();await page.waitForFunction(()=>!document.querySelector('.key-pad button')?.disabled);
  assert.equal(await page.locator('.key-pad button').count(),16);
  await page.waitForFunction(()=>[...document.querySelectorAll('.world-stage video')].every(v=>v.readyState>=2));
  for(const [i,t,action] of [[0,0.5,'forward'],[0,3.2,'look_right'],[1,1.5,'look_left']]){
   await stages.nth(i).locator('video').evaluate((v,t)=>{v.currentTime=t;},t);
   await page.waitForFunction(({i,action})=>document.querySelectorAll('.stage-hud')[i].dataset.highlight===action,{i,action});
   await stages.nth(i).locator('video').evaluate(async v=>{await v.play();await new Promise(resolve=>v.requestVideoFrameCallback(resolve));v.pause();});
  }
  result.checks.push('two 8-key overlays; replay highlights W / right / left at receipt times');
  await stages.first().screenshot({path:path.join(out,'desktop-hud.png')});
  await stages.first().getByRole('button',{name:'user-mountain.jpg 场景全屏'}).click();
  await page.waitForFunction(()=>document.fullscreenElement?.classList.contains('world-stage'));
  assert.ok(await stages.first().locator('.stage-hud').isVisible());
  await page.screenshot({path:path.join(out,'fullscreen-hud.png')});
  await page.evaluate(()=>document.exitFullscreen());result.checks.push('fullscreen contains both video and controls');
  const mappings=[['w','forward'],['a','left'],['s','backward'],['d','right'],['ArrowUp','look_up'],['ArrowLeft','look_left'],['ArrowDown','look_down'],['ArrowRight','look_right']];
  for(const [keyName,action] of mappings){
   worlds[0].status='ready';worlds[0].phase='ready';await page.reload({waitUntil:'domcontentloaded'});
   await page.waitForFunction(()=>!document.querySelector('.key-pad button')?.disabled);
   await page.locator('.world-pair > article').first().locator('header').click();
   const count=submitted.length;await page.keyboard.press(keyName);
   await page.waitForFunction(()=>document.querySelector('.key-pad button')?.disabled);
   assert.equal(submitted.length,count+1);assert.equal(submitted.at(-1).action,action);assert.equal(submitted.at(-1).i,0);
   assert.equal(await stages.first().locator('.stage-hud').getAttribute('data-highlight'),action);
   await page.keyboard.press(keyName);assert.equal(submitted.length,count+1);
  }
  result.checks.push('8 keyboard mappings; active-card isolation; pending highlight and repeat guard');
  worlds[0].status='ready';worlds[0].phase='ready';await page.reload({waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>!document.querySelector('.key-pad button')?.disabled);
  await stages.nth(1).getByRole('button',{name:'user-temple.jpg 向左看',exact:true}).click();
  await page.waitForFunction(()=>document.querySelectorAll('.world-stage')[1].querySelector('.key-pad button').disabled);
  assert.equal(submitted.at(-1).i,1);assert.equal(submitted.at(-1).action,'look_left');
  result.checks.push('mouse overlay click routes to second card');
  await page.setViewportSize({width:390,height:844});await stages.first().scrollIntoViewIfNeeded();
  const sizes=await stages.first().evaluate(e=>{const s=e.getBoundingClientRect();return [...e.querySelectorAll('.key-pad')].map(p=>{const b=p.getBoundingClientRect();return b.left>=s.left&&b.right<=s.right;});});
  assert.deepEqual(sizes,[true,true]);await stages.first().screenshot({path:path.join(out,'mobile-hud.png')});
  result.checks.push('390px mobile: both pads inside scene');assert.deepEqual(result.errors,[]);result.status='passed';
 }catch(e){result.status='failed';result.failure=String(e);await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});throw e;}
 finally{fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
