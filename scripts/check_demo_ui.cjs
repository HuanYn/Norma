// Staging-only browser smoke: real public album, real API, no VLM/video submission.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const origin = 'http://127.0.0.1:8767';
const albumId = '54a2e3cf1d605ab8a5682cbeab1b2885';
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'docs', 'benchmarks', 'assets', 'demo-sprint-20261005');
const report = {
  date: new Date().toISOString(), origin, album_id: albumId,
  context: 'Real staging UI and real public-image model scores. before-open.png and after.png show the SAME implementation with its panel collapsed/expanded; not an old-version or accuracy improvement comparison.',
  viewport: { width: 1440, height: 1100 },
  checks: {}, console_errors: [], page_errors: [], failed_responses: [], blocked_requests: [],
};

async function main() {
  const album = await (await fetch(`${origin}/albums/${albumId}`)).json();
  assert.equal(album.photo_count, 8);
  assert.equal(album.embedded_count, 8);
  const jobs = await (await fetch(`${origin}/jobs?limit=100&offset=0`)).json();
  const prepared = jobs.items.find(job => job.job_type === 'prepare_album'
    && job.status === 'completed' && job.result?.album?.album_id === albumId);
  assert.ok(prepared, 'Need a genuine completed prepare job for the public test album');
  fs.mkdirSync(output, { recursive: true });
  const publicFolder = path.join(root, '.norma', 'sprint-album');
  assert.equal(path.resolve(album.source_path), publicFolder);
  const attribution = JSON.parse(fs.readFileSync(path.join(publicFolder, 'ATTRIBUTION.json'), 'utf8'));
  const publicFiles = new Set(fs.readdirSync(publicFolder).filter(filename => /\.jpe?g$/i.test(filename)));
  const filteredAttribution = {
    ...attribution,
    images: attribution.images.filter(item => publicFiles.has(item.file)),
    screenshot_note: 'Public photos are shown as resized/cropped UI thumbnails in this smoke. Source authors, source pages, and individual licenses are retained below. No claim of model accuracy or improvement is made.',
  };
  assert.equal(filteredAttribution.images.length, 8);
  fs.writeFileSync(path.join(output, 'ATTRIBUTION.json'), JSON.stringify(filteredAttribution, null, 2) + '\n', 'utf8');
  const installedBrowser = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  ].find(candidate => fs.existsSync(candidate));
  const browser = await chromium.launch({ headless: true,
    ...(installedBrowser ? { executablePath: installedBrowser } : {}) });
  try {
    const context = await browser.newContext({ viewport: report.viewport, deviceScaleFactor: 1 });
    // Restore the real completed job, using the same persistence contract as reload.
    await context.addInitScript(id => localStorage.setItem('norma.activePrepareJob', id), prepared.id);
    const page = await context.newPage();
    page.on('console', message => { if (message.type() === 'error') report.console_errors.push(message.text()); });
    page.on('pageerror', error => report.page_errors.push(String(error)));
    page.on('response', response => {
      if (response.status() >= 400) report.failed_responses.push({ url: new URL(response.url()).pathname, status: response.status() });
    });
    // Defense in depth: the smoke may only write one ordinary rule-baseline selection.
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const allowedWrite = request.method() === 'POST' && url.origin === origin && url.pathname === '/selections';
      if ((url.origin !== origin && url.protocol !== 'data:')
          || (!['GET', 'HEAD'].includes(request.method()) && !allowedWrite)) {
        report.blocked_requests.push({ method: request.method(), path: url.pathname });
        await route.abort('blockedbyclient');
        return;
      }
      await route.continue();
    });
    await page.goto(origin, { waitUntil: 'networkidle', timeout: 30000 });
    await page.getByText('8 photos', { exact: true }).waitFor({ timeout: 15000 });
    report.checks.real_album_restored = true;
    await page.getByRole('button', { name: 'AI Selection', exact: true }).click();
    const query = page.getByRole('textbox', { name: 'Ask anything about this album' });
    await query.waitFor();
    try {
      await page.waitForFunction(() => !document.querySelector('input[aria-label="Ask anything about this album"]')?.disabled,
        undefined, { timeout: 15000 });
    } catch (error) {
      report.checks.readiness_status = await (await fetch(`${origin}/providers/embedding/status`)).json();
      // Never overwrite the captured historical defect with a different later
      // failure (for example a freshly restarted, genuinely unloaded provider).
      const failureStem = `readiness-failure-${Date.now()}`;
      await page.screenshot({ path: path.join(output, `${failureStem}.png`), fullPage: true });
      fs.writeFileSync(path.join(output, `${failureStem}.json`), JSON.stringify({ ...report,
        observed_failure: 'Query remained disabled; inspect the actual provider status before attributing a cause.' }, null, 2) + '\n');
      throw error;
    }
    assert.equal(await query.isEnabled(), true, 'Semantic query input must be ready');
    const llmToggle = page.getByLabel(/实验性模型解析/);
    const memoryToggle = page.getByLabel(/参考我的偏好记录/);
    assert.equal(await llmToggle.isChecked(), false);
    assert.equal(await memoryToggle.isChecked(), false);
    await query.fill('选3张建筑');
    const selectedPromise = page.waitForResponse(response =>
      response.url() === `${origin}/selections` && response.request().method() === 'POST', { timeout: 60000 });
    await page.getByRole('button', { name: 'Run command', exact: true }).click();
    const selectedResponse = await selectedPromise;
    const selection = await selectedResponse.json();
    assert.equal(selectedResponse.status(), 200, JSON.stringify(selection));
    assert.equal(selection.feasible, true);
    assert.equal(selection.selected.length, 3);
    await page.getByText('3 selected photos', { exact: true }).waitFor();
    report.checks.rule_baseline_selection = {
      query: '选3张建筑', selected_count: selection.selected.length,
      provider: selection.provider_fingerprint, model_parser: false,
      preference_memory: false, photo_ids: selection.selected.map(photo => photo.photo_id),
    };
    await page.evaluate(async () => {
      await Promise.all([...document.images].map(image => image.complete ? Promise.resolve() : new Promise(resolve => { image.onload = resolve; image.onerror = resolve; })));
    });
    await page.screenshot({ path: path.join(output, 'before-open.png'), fullPage: true });
    const summary = page.locator('details.demo-workbench > summary');
    await summary.click();
    const quality = page.getByRole('region', { name: '学习型美学评估' });
    await quality.getByText(/已有 8 张评分/).waitFor({ timeout: 15000 });
    const actual = await (await fetch(`${origin}/albums/${albumId}/aesthetics`)).json();
    const selectedScores = actual.items.filter(item => selection.selected.some(photo => photo.photo_id === item.photo_id));
    assert.equal(selectedScores.length, 3);
    const mean = field => (selectedScores.reduce((sum, item) => sum + item[field], 0) / selectedScores.length).toFixed(2);
    await quality.getByText(new RegExp(`平均质量 ${mean('technical_quality')} / 约100`)).waitFor();
    await quality.getByText(new RegExp(`美学 ${mean('aesthetic_quality')} / 10`)).waitFor();
    report.checks.real_learned_scores = {
      total: actual.scored_count, selected_scored: selectedScores.length,
      selected_mean_technical: mean('technical_quality'), selected_mean_aesthetic: mean('aesthetic_quality'),
      accuracy_claim: false,
    };
    const consent = page.getByLabel(/我确认把这张处理后的照片/);
    assert.equal(await consent.isChecked(), false);
    assert.equal(await page.getByRole('button', { name: '生成约2秒预览', exact: true }).isDisabled(), true);
    report.checks.video_requires_explicit_consent = true;
    report.checks.no_video_or_vlm_submission = report.blocked_requests.length === 0;
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(output, 'after.png'), fullPage: true });
    report.checks.image_load_failures = await page.locator('img').evaluateAll(images => images.filter(image => !image.complete || image.naturalWidth === 0).map(image => image.getAttribute('alt')));
    report.checks.horizontal_overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    assert.equal(report.checks.image_load_failures.length, 0);
    assert.equal(report.checks.horizontal_overflow, false);
    assert.equal(report.page_errors.length, 0);
    assert.equal(report.blocked_requests.length, 0);
    report.passed = true;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(output, 'browser-smoke.json'), JSON.stringify(report, null, 2) + '\n', 'utf8');
  }
  console.log(JSON.stringify(report, null, 2));
}

main().catch(error => {
  report.passed = false;
  report.failure = String(error);
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, 'browser-smoke.json'), JSON.stringify(report, null, 2) + '\n', 'utf8');
  console.error(error);
  process.exitCode = 1;
});
