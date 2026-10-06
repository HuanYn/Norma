// Read-only release smoke: opens the homepage and checks its primary controls.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:8767';
  const output = path.resolve(process.argv[3] || '.norma/release-ui');
  fs.mkdirSync(output, { recursive: false });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.NORMA_BROWSER_CHANNEL ? { channel: process.env.NORMA_BROWSER_CHANNEL } : {}),
  });
  const report = { status: 'running', page_errors: [], scope: 'homepage controls; read-only' };
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.on('pageerror', error => report.page_errors.push(String(error)));
    await page.route('**/demo/world/api/**', route => route.abort());
    const response = await page.goto(base, { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200);
    await page.getByRole('heading', { name: '从一组照片，到可探索的瞬间' }).waitFor();
    await page.getByText('AI worker ready', { exact: true }).waitFor();
    for (const name of ['导入照片', '美学评估 · 相似只留最佳']) {
      assert.equal(await page.getByRole('button', { name, exact: true }).count(), 1, name);
    }
    // Selection is revealed only after the user has assessed an album.
    assert.equal(await page.getByRole('button', { name: '模型选片', exact: true }).count(), 0);
    assert.deepEqual(report.page_errors, []);
    await page.screenshot({ path: path.join(output, 'homepage.png'), fullPage: true });
    report.status = 'passed';
  } catch (error) {
    report.status = 'failed';
    report.error = String(error);
    process.exitCode = 1;
  } finally {
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(report, null, 2));
    await browser.close();
    console.log(JSON.stringify(report));
  }
})();
