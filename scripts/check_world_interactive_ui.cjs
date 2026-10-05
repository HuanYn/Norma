// REAL GPU generation via the local browser gateway. Explicit invocation only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require('C:/Users/25438/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const output = path.resolve(process.argv[2] || '.norma/world-interactive-browser-20261005');
const example = process.argv[3] || 'temple';
const origin = 'http://127.0.0.1:8768';
const report = { scope: 'Actual sequential interactive generation; not an independent quality benchmark', example,
  started: new Date().toISOString(), errors: [], steps: [] };
async function main() {
  fs.mkdirSync(output, { recursive: false });
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  let page;
  const save = () => fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(report, null, 2));
  try {
    page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
    page.on('pageerror', e => report.errors.push(String(e)));
    await page.goto(origin);
    await page.locator(`[data-example=${example}]`).click();
    await page.waitForFunction(() => document.querySelector('#preview').complete && document.querySelector('#preview').naturalWidth > 0);
    assert.ok(await page.locator('#start').isDisabled());
    await page.screenshot({ path: path.join(output, '01-input-before.png'), fullPage: true });
    await page.locator('#consent').check();
    await page.locator('#start').click();
    const ready = async (steps) => {
      const deadline = Date.now() + 900000;
      while (Date.now() < deadline) {
        const response = await page.request.get(origin + '/api/current');
        assert.equal(response.status(), 200); const s = await response.json();
        if (s.id) report.session_id = s.id;
        if (['failed', 'closed', 'interrupted'].includes(s.status)) throw Error('Session failed: ' + JSON.stringify(s));
        if (s.status === 'ready' && s.next_sequence === steps) return s;
        await page.waitForTimeout(2000);
      }
      throw Error('GPU operation timed out');
    };
    let state = await ready(0); report.session_id = state.id; report.initialization = state.initialization; save();
    const actions = ['forward', 'right', 'look_left', 'forward', 'left', 'look_right', 'hold'];
    for (let n = 0; n < actions.length; n++) {
      // The next action is only sent after the preceding output is downloaded/playable.
      await page.locator(`[data-action=${actions[n]}]`).click();
      state = await ready(n + 1); const step = state.history.at(-1);
      report.steps.push(step); save(); console.log(`step ${n + 1}: ${step.frames} frames, ${step.seconds.toFixed(2)}s`);
      await page.waitForFunction(name => document.querySelector('#video').src.endsWith(name) && document.querySelector('#video').readyState >= 2,
        step.artifact, { timeout: 60000 });
      const response = await page.request.get(origin + `/api/sessions/${state.id}/artifacts/${step.artifact}`);
      assert.equal(response.status(), 200);
      assert.equal(crypto.createHash('sha256').update(await response.body()).digest('hex'), step.sha256);
      if (n > 0) { assert.equal(step.kv_cache_identity, report.steps[0].kv_cache_identity); assert.ok(step.prefix_max_abs_error <= 2 / 255); }
      if (n === 1) {
        await page.reload(); await ready(2);
        report.refresh_preserved_session = (await page.evaluate(async () => (await (await fetch('/api/current')).json()).id)) === state.id;
        assert.ok(report.refresh_preserved_session);
      }
      await page.screenshot({ path: path.join(output, `step-${n + 1}.png`), fullPage: true });
    }
    const video = page.locator('#video');
    await video.evaluate(async e => { e.currentTime = 0; e.muted = true; await e.play(); });
    await page.waitForFunction(() => document.querySelector('#video').ended, null, { timeout: 25000 });
    report.playback = await video.evaluate(e => ({ duration: e.duration, width: e.videoWidth, height: e.videoHeight,
      time: e.currentTime, loop: e.loop, rate: e.playbackRate, error: e.error?.code || null }));
    assert.equal(report.playback.duration, 10.3125); assert.equal(report.playback.rate, 1); assert.equal(report.playback.loop, false);
    const downloading = page.waitForEvent('download'); await page.locator('#download').click();
    await (await downloading).saveAs(path.join(output, 'interactive-10s.mp4'));
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(output, 'interactive-10s.mp4'))).digest('hex'), report.steps.at(-1).sha256);
    assert.ok(await page.locator('[data-action=forward]').isDisabled());
    await page.locator('#close').click();
    await page.waitForFunction(() => !document.querySelector('[data-example=mountain]').disabled);
    await page.locator('[data-example=mountain]').click();
    await page.waitForFunction(() => !document.querySelector('#preview').hidden && document.querySelector('#video').hidden);
    assert.deepEqual(report.errors, []); report.status = 'passed';
  } catch (e) { report.status = 'failed'; report.failure = String(e); throw e; }
  finally {
    if (page && report.session_id) await page.request.post(origin + `/api/sessions/${report.session_id}/close`,
      { data: {}, headers: { 'X-Norma-World': '1' } }).catch(() => {});
    save(); await browser.close();
  }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
