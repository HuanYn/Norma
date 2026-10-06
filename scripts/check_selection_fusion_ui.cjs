// Real UI selections only. No fabricated user feedback or cloud requests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const origin = 'http://127.0.0.1:8767';
const output = path.resolve(process.argv[2] || '.norma/selection-fusion-browser-20261005');
const report = { scope: 'Same development album/query, actual baseline vs learned score fusion; no independent quality labels', errors: [], runs: {} };
async function main() {
  fs.mkdirSync(output, { recursive: false });
  const jobs = await (await fetch(origin + '/jobs?limit=100&offset=0')).json();
  const job = jobs.items.find(j => j.job_type === 'prepare_album' && j.status === 'completed' && j.result?.album?.album_id === '54a2e3cf1d605ab8a5682cbeab1b2885');
  assert.ok(job);
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
    await context.addInitScript(id => localStorage.setItem('norma.activePrepareJob', id), job.id);
    const page = await context.newPage();
    page.on('pageerror', e => report.errors.push(String(e)));
    await page.route('**/*', route => {
      const r = route.request(), u = new URL(r.url());
      if (u.origin !== origin || (!['GET', 'HEAD'].includes(r.method()) && !(r.method() === 'POST' && ['/selections', '/providers/embedding/warmup'].includes(u.pathname)))) {
        report.errors.push('Unexpected request ' + r.method() + ' ' + u.pathname); return route.abort();
      }
      return route.continue();
    });
    await page.goto(origin, { waitUntil: 'networkidle' });
    await page.getByText('8 photos', { exact: true }).waitFor();
    // Explicitly warm the already provisioned local model after a server restart.
    const status = await (await page.request.get(origin + '/providers/embedding/status')).json();
    if (!status.loaded) {
      const response = await page.request.post(origin + '/providers/embedding/warmup');
      assert.ok(response.ok());
      const deadline = Date.now() + 180000;
      while (Date.now() < deadline) {
        const state = await (await page.request.get(origin + '/providers/embedding/status')).json();
        if (state.loaded) break;
        if (state.warmup_state === 'failed') throw Error('Local embedding warmup failed');
        await page.waitForTimeout(1500);
      }
      await page.reload({ waitUntil: 'networkidle' });
    }
    await page.getByRole('button', { name: 'AI Selection', exact: true }).click();
    const input = page.getByRole('textbox', { name: 'Ask anything about this album' });
    await input.fill('选 4 张旅行照片');
    for (const learned of [false, true]) {
      await page.getByLabel(/使用模型美学与质量排序/).setChecked(learned);
      const response = page.waitForResponse(r => r.url() === origin + '/selections' && r.request().method() === 'POST', { timeout: 180000 });
      await page.getByRole('button', { name: 'Run command', exact: true }).click();
      const r = await response; const result = await r.json();
      assert.equal(r.status(), 200, JSON.stringify(result)); assert.equal(result.feasible, true);
      assert.equal(Boolean(result.learned_quality), learned); assert.equal(result.selected.length, 4);
      await page.getByText('4 selected photos', { exact: true }).waitFor();
      await page.waitForFunction(() => [...document.querySelectorAll('.photo-grid img')].every(e => e.complete && e.naturalWidth > 0));
      const name = learned ? 'after-learned-fusion' : 'before-baseline';
      report.runs[name] = result;
      await page.screenshot({ path: path.join(output, name + '.png'), fullPage: true });
    }
    assert.deepEqual(report.errors, []); report.status = 'passed';
  } catch (e) { report.status = 'failed'; report.failure = String(e); throw e; }
  finally { await browser.close(); fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(report, null, 2)); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
