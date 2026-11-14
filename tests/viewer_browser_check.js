// Loads a viewer page in headless Chromium and reports what tests/test_volume_viewer.py asserts on.
// Usage: node viewer_browser_check.js <viewer.html> '<[x, y, z] marker voxel in the cropped volume>'
const { chromium } = require('playwright');

const [, , file, markerJson] = process.argv;
const marker = JSON.parse(markerJson);

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1100, height: 900 }, reducedMotion: 'reduce' });
  const page = await context.newPage();
  const problems = [];
  page.on('console', (message) => { if (message.type() === 'error') problems.push('console: ' + message.text()); });
  page.on('pageerror', (error) => problems.push('pageerror: ' + error.message));
  page.on('request', (request) => {
    const url = request.url();
    if (!url.startsWith('file:') && !url.startsWith('data:')) problems.push('request: ' + url);
  });
  await page.goto('file://' + file);
  await page.waitForFunction(() => window.__vv && window.__vv.anatomy, null, { timeout: 60000 });
  const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await settle();

  const report = await page.evaluate(() => ({
    checksums: window.__vv.checksums,
    status_hidden: document.getElementById('vv-status').hidden,
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    peak_idx: window.__vv.state.idx.slice(),
    peak_readout: document.getElementById('vv-readout').textContent,
    edges: Object.fromEntries(window.__vv.planes.map((plane) => [plane.id, plane.edges])),
    chart_marks: ['vv-share', 'vv-prob'].map((id) => document.querySelectorAll('#' + id + ' .vv-bar-mark, #' + id + ' .vv-dot').length),
    table_rows: document.querySelectorAll('#vv-table tbody tr').length,
  }));

  // Hide the map, put the crosshair on the marker voxel and find the brightest pixel of each slice view.
  const setSlider = async (id, value) => {
    await page.evaluate(([id, value]) => {
      const slider = document.getElementById(id);
      slider.value = value;
      slider.dispatchEvent(new Event('input'));
    }, [id, value]);
  };
  const grab3d = () => page.evaluate(() => document.getElementById('vv-3d').toDataURL());
  const before = await page.evaluate(() => Array.from(window.__vv.planes[0].image.data.slice(0, 40000)).reduce((a, b) => a + b, 0));
  await page.evaluate(() => { const box = document.getElementById('vv-show'); box.checked = false; box.dispatchEvent(new Event('change')); });
  await settle();
  const after = await page.evaluate(() => Array.from(window.__vv.planes[0].image.data.slice(0, 40000)).reduce((a, b) => a + b, 0));
  report.map_changes_pixels = before !== after;
  await setSlider('vv-x', marker[0]);
  await setSlider('vv-y', marker[1]);
  await setSlider('vv-z', marker[2]);
  await settle();
  report.readout = await page.evaluate(() => document.getElementById('vv-readout').textContent);
  report.brightest = await page.evaluate(() => Object.fromEntries(window.__vv.planes.map((plane) => {
    const pixels = plane.image.data;
    let best = -1, at = 0;
    for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > best) { best = pixels[i]; at = i / 4; }
    return [plane.id, [at % plane.nu, Math.floor(at / plane.nu)]];
  })));

  // Below the slab the readout must say there is no evidence.
  await setSlider('vv-z', 0);
  await settle();
  report.outside_readout = await page.evaluate(() => document.getElementById('vv-readout').textContent);

  // Turning the 3D view changes its picture.
  await page.evaluate(() => { const box = document.getElementById('vv-show'); box.checked = true; box.dispatchEvent(new Event('change')); });
  await settle();
  const first = await grab3d();
  await page.focus('#vv-3d');
  for (let i = 0; i < 6; i++) await page.keyboard.press('ArrowRight');
  await settle();
  report.rotation_changes_3d = first !== (await grab3d());

  report.problems = problems;
  console.log(JSON.stringify(report));
  await browser.close();
})().catch((error) => { console.error(error); process.exit(1); });
