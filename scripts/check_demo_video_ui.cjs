// One authorized public-image video submission through the real staging UI.
// A persisted attempt marker prevents reruns from accidentally submitting twice.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const origin = 'http://127.0.0.1:8767';
const albumId = '54a2e3cf1d605ab8a5682cbeab1b2885';
const root = path.resolve(__dirname, '..');
const output = path.join(root, 'docs', 'benchmarks', 'assets', 'demo-sprint-20261005');
const marker = path.join(output, 'video-ui-submission.json');
const reportPath = path.join(output, 'video-ui-smoke.json');
const report = {
  schema: 'norma-real-video-browser-smoke-v1', date: new Date().toISOString(),
  origin, album_id: albumId, context: 'One public-image job submitted via explicit UI consent to the user-owned server. No VLM or paid API calls. Engineering generation/playback smoke, not a video-quality improvement study.',
  viewport: { width: 1440, height: 1100 }, checks: {}, observed_states: [],
  console_errors: [], page_errors: [], failed_responses: [], blocked_requests: [],
  video_post_count: 0,
};
const save = () => fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n', 'utf8');
async function get(url) {
  const response = await fetch(origin + url);
  assert.equal(response.status, 200, `Unexpected HTTP ${response.status} on ${url}`);
  return response.json();
}

async function main() {
  fs.mkdirSync(output, { recursive: true });
  if (fs.existsSync(marker)) {
    const previous = JSON.parse(fs.readFileSync(marker, 'utf8'));
    console.log(JSON.stringify({ already_attempted: true, no_new_submission: true, ...previous }));
    if (fs.existsSync(reportPath)) console.log(fs.readFileSync(reportPath, 'utf8'));
    return;
  }
  const provider = await get('/providers/embedding/status');
  assert.ok(provider.loaded && !provider.error && ['ready', 'idle'].includes(provider.warmup_state), 'Wait for the existing semantic model to become ready; do not submit another warmup here');
  const album = await get(`/albums/${albumId}`);
  assert.equal(path.resolve(album.source_path), path.join(root, '.norma', 'sprint-album'));
  const catalog = await get(`/albums/${albumId}/photos?limit=20&offset=0&include_rejects=true&sort=path`);
  const jobs = await get('/jobs?limit=100&offset=0');
  const prepared = jobs.items.find(job => job.job_type === 'prepare_album' && job.status === 'completed' && job.result?.album?.album_id === albumId);
  assert.ok(prepared);
  const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' });
  let page;
  try {
    const context = await browser.newContext({ viewport: report.viewport, acceptDownloads: true });
    await context.addInitScript(id => localStorage.setItem('norma.activePrepareJob', id), prepared.id);
    page = await context.newPage();
    page.on('console', message => { if (message.type() === 'error') report.console_errors.push(message.text()); });
    page.on('pageerror', error => report.page_errors.push(String(error)));
    page.on('response', async response => {
      const pathname = new URL(response.url()).pathname;
      if (response.status() >= 400) report.failed_responses.push({ path: pathname, status: response.status() });
      if (response.request().method() === 'GET' && /^\/demo\/video\/jobs\/[a-f0-9]{32}$/.test(pathname)) {
        try {
          const job = await response.json();
          const state = { at: new Date().toISOString(), status: job.status, stage: job.stage, step: job.step, step_percent: job.step_percent };
          const last = report.observed_states.at(-1);
          if (!last || last.stage !== state.stage || last.step !== state.step || last.status !== state.status) {
            report.observed_states.push(state);
            save();
          }
        } catch { /* Main assertions inspect the authoritative final job. */ }
      }
    });
    await page.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      let allowed = url.origin === origin && ['GET', 'HEAD'].includes(request.method());
      if (request.method() === 'POST' && url.origin === origin && url.pathname === '/selections') allowed = true;
      if (request.method() === 'POST' && url.origin === origin && url.pathname === '/demo/video/jobs') {
        report.video_post_count++;
        const body = request.postDataJSON();
        allowed = report.video_post_count === 1 && body.album_id === albumId && body.photo_id === report.photo?.id && body.upload_confirmed === true;
        report.submitted_request = body;
        save();
      }
      if (!allowed) {
        report.blocked_requests.push({ method: request.method(), path: url.pathname });
        await route.abort('blockedbyclient');
        return;
      }
      await route.continue();
    });
    await page.goto(origin, { waitUntil: 'networkidle' });
    await page.getByText('8 photos', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'AI Selection', exact: true }).click();
    await page.waitForFunction(() => !document.querySelector('input[aria-label="Ask anything about this album"]')?.disabled, undefined, { timeout: 15000 });
    assert.equal(await page.getByLabel(/实验性模型解析|使用大模型理解选片要求/).isChecked(), false);
    assert.equal(await page.getByLabel(/参考我的偏好记录/).isChecked(), false);
    await page.getByRole('textbox', { name: 'Ask anything about this album' }).fill('选3张建筑');
    const selectionPromise = page.waitForResponse(response => response.url() === `${origin}/selections` && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Run command', exact: true }).click();
    const selection = await (await selectionPromise).json();
    assert.equal(selection.feasible, true);
    assert.equal(selection.selected.length, 3);
    const choices = catalog.items.filter(photo => selection.selected.some(item => item.photo_id === photo.id) && photo.width / photo.height > 1.2 && photo.width / photo.height < 2.2)
      .sort((a, b) => Math.abs(a.width / a.height - 640 / 384) - Math.abs(b.width / b.height - 640 / 384));
    assert.ok(choices.length, 'Do not generate the extreme panoramic banner');
    const photo = choices[0];
    report.photo = { id: photo.id, filename: photo.filename, width: photo.width, height: photo.height };
    report.selection_id = selection.selection_id;
    await page.locator('details.demo-workbench > summary').click();
    const panel = page.getByRole('region', { name: '远端图生视频' });
    await panel.getByRole('combobox').selectOption(photo.id);
    await panel.getByLabel('镜头与运动描述', { exact: true }).fill('A slow gentle cinematic camera push toward this historic street. Preserve the building geometry and original colors, subtle parallax, natural daylight, no cuts.');
    await panel.getByText(/远端 Worker 已连接/).waitFor({ timeout: 30000 });
    const consent = panel.getByLabel(/我确认把这张处理后的照片/);
    const submit = panel.getByRole('button', { name: '生成约2秒预览', exact: true });
    assert.equal(await consent.isChecked(), false);
    assert.equal(await submit.isDisabled(), true);
    report.checks.unchecked_consent_blocks_submit = true;
    await consent.check();
    assert.equal(await submit.isEnabled(), true);
    await page.screenshot({ path: path.join(output, 'video-ui-before-submit.png'), fullPage: true });
    // Exclusive marker is written before the only click. If an HTTP response is
    // lost, rerunning will NOT cause a second possibly duplicate remote job.
    fs.writeFileSync(marker, JSON.stringify({ attempted_at: new Date().toISOString(), photo: report.photo, note: 'One authorized attempt; never automatically retry submission.' }, null, 2), { encoding: 'utf8', flag: 'wx' });
    const submittedAt = Date.now();
    const responsePromise = page.waitForResponse(response => response.url() === `${origin}/demo/video/jobs` && response.request().method() === 'POST', { timeout: 40000 });
    await submit.click();
    const response = await responsePromise;
    const job = await response.json();
    assert.ok([200, 202].includes(response.status()), JSON.stringify(job));
    assert.match(job.id, /^[a-f0-9]{32}$/);
    report.job_id = job.id;
    fs.writeFileSync(marker, JSON.stringify({ attempted_at: new Date(submittedAt).toISOString(), job_id: job.id, photo: report.photo, no_automatic_resubmit: true }, null, 2) + '\n');
    save();
    console.log(JSON.stringify({ submitted_job_id: job.id, photo: report.photo, video_post_count: report.video_post_count }));
    const deadline = submittedAt + 300000;
    let finalJob = job;
    while (!['completed', 'failed', 'cancelled', 'interrupted'].includes(finalJob.status) && Date.now() < deadline) {
      await page.waitForTimeout(4000);
      finalJob = await get(`/demo/video/jobs/${job.id}`);
    }
    report.final_job = finalJob;
    report.submit_to_terminal_seconds = (Date.now() - submittedAt) / 1000;
    save();
    assert.equal(finalJob.status, 'completed', JSON.stringify(finalJob));
    const video = panel.locator('video').first();
    await video.waitFor({ state: 'visible', timeout: 10000 });
    await page.waitForFunction(() => {
      const video = document.querySelector('.demo-video video');
      return video && video.readyState >= 2 && video.videoWidth > 0 && Number.isFinite(video.duration);
    }, undefined, { timeout: 30000 });
    const before = await video.evaluate(video => ({ duration: video.duration, width: video.videoWidth, height: video.videoHeight, readyState: video.readyState, currentTime: video.currentTime, error: video.error?.message ?? null }));
    assert.equal(before.width, 640);
    assert.equal(before.height, 384);
    assert.ok(before.duration > 1.9 && before.duration < 2.2);
    await video.evaluate(async video => { video.muted = true; video.currentTime = 0; await video.play(); });
    await page.waitForTimeout(800);
    const played = await video.evaluate(video => { const time = video.currentTime; video.pause(); return { currentTime: time, paused: video.paused, error: video.error?.message ?? null }; });
    assert.ok(played.currentTime > 0.3, 'Decoded HTML video must actually advance during playback');
    assert.equal(played.error, null);
    report.checks.html_video = { metadata: before, playback: played };
    const downloadLink = panel.getByRole('link', { name: '下载生成视频', exact: true });
    assert.equal(await downloadLink.getAttribute('href'), `/demo/video/jobs/${job.id}/artifact`);
    const downloadPromise = page.waitForEvent('download', { timeout: 30000 });
    await downloadLink.click();
    const download = await downloadPromise;
    assert.equal(await download.failure(), null);
    const artifact = path.join(output, 'video', `${job.id}-ui-download.mp4`);
    fs.mkdirSync(path.dirname(artifact), { recursive: true });
    await download.saveAs(artifact);
    const bytes = fs.readFileSync(artifact);
    assert.ok(bytes.length > 1000);
    assert.equal(bytes.subarray(4, 8).toString('ascii'), 'ftyp');
    report.checks.download = { file: path.relative(root, artifact).replaceAll('\\', '/'), bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') };
    assert.equal(report.video_post_count, 1);
    assert.equal(report.page_errors.length, 0);
    assert.equal(report.blocked_requests.length, 0);
    assert.equal(report.failed_responses.length, 0);
    await page.screenshot({ path: path.join(output, 'video-ui-completed.png'), fullPage: true });
    report.passed = true;
    save();
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    report.passed = false;
    report.failure = String(error);
    if (page) await page.screenshot({ path: path.join(output, `video-ui-failure-${Date.now()}.png`), fullPage: true }).catch(() => {});
    save();
    throw error;
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
