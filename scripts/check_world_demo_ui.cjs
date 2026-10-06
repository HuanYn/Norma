// Read-only validation of two already generated PRIVATE local demos; no submissions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const output = path.join(root, '.norma/user-world-demo-20261005/browser');
const origin = 'http://127.0.0.1:8767';
const report = { date: new Date().toISOString(), scope: 'Offline gallery playback/download only; NOT interactive world validation', errors: [], videos: {} };
async function main() {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
    page.on('pageerror', error => report.errors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') report.errors.push(message.text()); });
    page.on('response', response => { if (response.status() >= 400) report.errors.push(`HTTP ${response.status()} ${response.url()}`); });
    await page.route('**/*', async route => {
      const request = route.request();
      if (new URL(request.url()).origin !== origin || !['GET', 'HEAD'].includes(request.method())) {
        report.errors.push('Unexpected request ' + request.method() + ' ' + request.url());
        return route.abort();
      }
      return route.continue();
    });
    await page.goto(origin + '/world-demo/', { waitUntil: 'networkidle' });
    const manifest = await (await page.request.get(origin + '/world-demo/manifest.json')).json();
    await page.locator('article[data-name=mountain] button').click();
    await page.locator('article[data-name=temple] button').click();
    assert.equal(await page.locator('.original:visible').count(), 2);
    await page.screenshot({ path: path.join(output, 'input-comparison.png'), fullPage: true });
    for (const name of ['mountain', 'temple']) {
      const article = page.locator(`article[data-name=${name}]`);
      await article.locator('button').click();
      const video = article.locator('video');
      await video.evaluate(async element => { element.muted = true; await element.play(); });
      await page.waitForFunction(name => document.querySelector(`article[data-name=${name}] video`).ended, name, { timeout: 25000 });
      const state = await video.evaluate(element => ({ duration: element.duration, time: element.currentTime, width: element.videoWidth,
        height: element.videoHeight, ended: element.ended, error: element.error?.code || null, playbackRate: element.playbackRate, loop: element.loop }));
      assert.ok(state.duration >= 10 && state.ended && state.time >= 10);
      assert.equal(state.playbackRate, 1); assert.equal(state.loop, false); assert.equal(state.error, null);
      assert.equal(state.width, manifest[name].width); assert.equal(state.height, manifest[name].height);
      const downloadPromise = page.waitForEvent('download');
      await article.locator('a.download').click();
      const download = await downloadPromise;
      const destination = path.join(output, `${name}-download.mp4`);
      await download.saveAs(destination);
      const sha256 = crypto.createHash('sha256').update(fs.readFileSync(destination)).digest('hex');
      assert.equal(sha256, manifest[name].sha256);
      report.videos[name] = { ...state, download_sha256: sha256 };
    }
    await page.screenshot({ path: path.join(output, 'completed-demo.png'), fullPage: true });
    assert.deepEqual(report.errors, []);
    report.status = 'passed';
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(report, null, 2));
  }
  console.log(JSON.stringify(report, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
