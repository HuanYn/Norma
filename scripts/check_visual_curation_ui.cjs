const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const output=path.resolve(process.argv[2]),report=JSON.parse(fs.readFileSync(path.join(output,'result.json'),'utf8'));
 const browser=await chromium.launch({headless:true,channel:'msedge'}),page=await browser.newPage({viewport:{width:1700,height:1050}});
 try{
  await page.route('**/demo/world/api/**',()=>{throw Error('No video API allowed');});
  await page.addInitScript(state=>localStorage.setItem('norma.photo-workflow.v1',JSON.stringify(state)),{albumId:report.after.album_id,folder:'D:\\Photos\\Trip',cards:[]});
  await page.goto('http://127.0.0.1:8767/',{waitUntil:'domcontentloaded'});
  await page.getByText(`额外折叠相近画面 ${report.after.visual_duplicates.folded_count} 张`,{exact:true}).waitFor({timeout:60000});
  await page.locator('.flow-grid img').first().waitFor();
  assert.equal(await page.locator('.flow-grid article').count(),report.after.kept.length);
  await page.locator('.flow-grid img').evaluateAll(imgs=>Promise.all(imgs.slice(0,18).map(i=>i.decode().catch(()=>{}))));
  await page.screenshot({path:path.join(output,'after.png')});
  await page.getByRole('button',{name:`查看折叠 ${report.after.hidden.length} 张`,exact:true}).click();
  assert.equal(await page.locator('.flow-grid article').count(),report.after.hidden.length);
  const example=page.locator('.flow-grid article').filter({has:page.getByText('IMG_2133.JPG',{exact:true})});
  await example.getByText('为什么选／折叠',{exact:true}).click();await example.scrollIntoViewIfNeeded();
  assert.match(await example.innerText(),/IMG_2134.JPG/);
  await page.screenshot({path:path.join(output,'folded-reason.png')});
  fs.writeFileSync(path.join(output,'ui.json'),JSON.stringify({status:'passed',kept:report.after.kept.length,hidden:report.after.hidden.length,reason_checked:'IMG_2133 -> IMG_2134'}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
