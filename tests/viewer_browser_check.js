// Loads viewer pages in headless Chromium and reports what tests/test_volume_viewer.py asserts on.
// Usage: node viewer_browser_check.js <spec.json>   (the spec names the pages; one JSON report is printed)
const fs = require('fs');
const { pathToFileURL } = require('url');
const { chromium } = require('playwright');

const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));

// Runs in the page before its own script: records every piece of text drawn on a canvas, with its position,
// so the report says what is on screen and not what the page's data structures hold.
const RECORD_CANVAS_TEXT = () => {
  window.__drawn = [];
  const original = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function (text, x, y, ...rest) {
    window.__drawn.push({ id: this.canvas.id, text: String(text), x, y });
    return original.call(this, text, x, y, ...rest);
  };
};

async function open(browser, file, options = {}) {
  const context = await browser.newContext({
    viewport: options.viewport || { width: 1100, height: 900 },
    reducedMotion: options.motion ? 'no-preference' : 'reduce',
    hasTouch: Boolean(options.touch),
  });
  const page = await context.newPage();
  const problems = [];
  page.on('console', (message) => { if (message.type() === 'error') problems.push('console: ' + message.text()); });
  page.on('pageerror', (error) => problems.push('pageerror: ' + error.message));
  page.on('request', (request) => { if (!request.url().startsWith('file:')) problems.push('request: ' + request.url()); });
  page.on('dialog', (dialog) => { problems.push('dialog: ' + dialog.message()); dialog.dismiss().catch(() => {}); });
  await page.addInitScript(RECORD_CANVAS_TEXT);
  if (options.init) await page.addInitScript(options.init);
  await page.goto(pathToFileURL(file).href);
  return { context, page, problems };
}

// Waits until the page shows the scan. Returns false at once, without waiting out a time limit, when the
// page has refused to draw or its script has thrown: the report then says why instead of the run timing out.
async function started(page, problems) {
  for (let waited = 0; waited < 60000; waited += 100) {
    const condition = await page.evaluate(() => {
      if (window.__vv && window.__vv.anatomy) return 'ready';
      const status = document.getElementById('vv-status');
      return status && status.getAttribute('role') === 'alert' ? 'refused' : 'waiting';
    });
    if (condition === 'ready') return true;
    if (condition === 'refused' || problems.some((problem) => problem.startsWith('pageerror'))) return false;
    await page.waitForTimeout(100);
  }
  return false;
}
const notStarted = async (context, page, problems) => {
  const report = { started: false, status: await text(page, '#vv-status'), problems };
  await context.close();
  return report;
};

const settle = (page) => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const pause = (page, ms) => page.waitForTimeout(ms);
const text = (page, selector) => page.evaluate((selector) => {
  const node = document.querySelector(selector);
  return node ? node.textContent.replace(/\s+/g, ' ').trim() : null;
}, selector);
const setSlider = (page, id, value) => page.evaluate(([id, value]) => {
  const slider = document.getElementById(id);
  slider.value = value;
  slider.dispatchEvent(new Event('input'));
}, [id, value]);
const setCrosshair = async (page, index) => {
  await setSlider(page, 'vv-x', index[0]);
  await setSlider(page, 'vv-y', index[1]);
  await setSlider(page, 'vv-z', index[2]);
  await settle(page);
};
const showMap = async (page, on) => {
  await page.evaluate((on) => { const box = document.getElementById('vv-show'); box.checked = on; box.dispatchEvent(new Event('change')); }, on);
  await settle(page);
};
// Pointer coordinates are relative to the window, so a view is brought on screen before it is pointed at.
const bringIntoView = (page, id) => page.evaluate((id) => document.getElementById(id).scrollIntoView({ block: 'center' }), id);
const state = (page) => page.evaluate(() => ({ idx: window.__vv.state.idx.slice(), yaw: window.__vv.state.yaw, pitch: window.__vv.state.pitch, spin: window.__vv.state.spin }));
// Number of lit pixels of a visible canvas: zero means nothing was drawn on it.
const drawn = (page, id) => page.evaluate((id) => {
  const canvas = document.getElementById(id);
  const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  let lit = 0;
  for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > 40) lit++;
  return lit;
}, id);
// The colour on screen at a voxel of a slice view, and at the mirror images of that place.
const onScreen = (page, id, u, v) => page.evaluate(([id, u, v]) => {
  const plane = window.__vv.planes.find((candidate) => candidate.id === id);
  const { x, y, w, h } = plane.rect;
  const context = plane.canvas.getContext('2d');
  const place = (a, b) => [x + ((a + 0.5) / plane.nu) * w, y + ((plane.nv - 1 - b + 0.5) / plane.nv) * h];
  const colour = (a, b) => { const [px, py] = place(a, b); return Array.from(context.getImageData(Math.round(px), Math.round(py), 1, 1).data.slice(0, 3)); };
  const [px, py] = place(u, v);
  return {
    here: colour(u, v),
    mirrored_across: colour(plane.nu - 1 - u, v),
    mirrored_up_down: colour(u, plane.nv - 1 - v),
    side: [px < plane.canvas.width / 2 ? 'left' : 'right', py < plane.canvas.height / 2 ? 'top' : 'bottom'],
  };
}, [id, u, v]);
const PLANES = [['vv-axial', 0, 1, 2], ['vv-coronal', 0, 2, 1], ['vv-sagittal', 1, 2, 0]];
// A crosshair on the slice that holds `voxel` but ten voxels away from it in the plane, so its lines do not cover it.
const beside = (voxel, shape, u, v, w) => {
  const away = (value, size) => (value < size / 2 ? Math.min(size - 1, value + 10) : Math.max(0, value - 10));
  const index = [0, 0, 0];
  index[w] = voxel[w];
  index[u] = away(voxel[u], shape[u]);
  index[v] = away(voxel[v], shape[v]);
  return index;
};
// Mean brightness of the left and right thirds of the 3D picture (its middle band, away from the corner marker),
// and where the orientation letters were last drawn.
const picture3d = (page) => page.evaluate(() => {
  const canvas = document.getElementById('vv-3d');
  const W = canvas.width, H = canvas.height;
  const context = canvas.getContext('2d');
  const mean = (x0, x1) => {
    const pixels = context.getImageData(Math.round(x0), Math.round(H * 0.25), Math.round(x1 - x0), Math.round(H * 0.4)).data;
    let sum = 0;
    for (let i = 0; i < pixels.length; i += 4) sum += pixels[i];
    return sum / (pixels.length / 4);
  };
  const letters = {};
  for (const entry of window.__drawn) if (entry.id === 'vv-3d' && /^[RAS]$/.test(entry.text)) letters[entry.text] = [entry.x, entry.y];
  return { left: mean(W * 0.2, W * 0.45), right: mean(W * 0.55, W * 0.8), letters };
});

async function mainPage(browser) {
  const { context, page, problems } = await open(browser, spec.pages.main);
  if (!(await started(page, problems))) return notStarted(context, page, problems);
  await settle(page);
  const report = await page.evaluate(() => ({
    started: true,
    checksums: window.__vv.checksums,
    status_hidden: document.getElementById('vv-status').hidden,
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    start_idx: window.__vv.state.idx.slice(),
    title: document.querySelector('.vv-title').textContent,
    heading_levels: [document.querySelector('.vv-title').tagName, ...Array.from(document.querySelectorAll('.vv-sub'), (node) => node.tagName)],
    touch_action: getComputedStyle(document.getElementById('vv-axial')).touchAction,
    spin_label: document.getElementById('vv-spin').textContent,
    table_rows: Array.from(document.querySelectorAll('#vv-table tbody tr'), (row) => Array.from(row.cells, (cell) => cell.textContent)),
    table_head: Array.from(document.querySelectorAll('#vv-table thead th'), (cell) => cell.textContent),
    bar_heights: Array.from(document.querySelectorAll('#vv-share .vv-bar-mark'), (bar) => Number(bar.getAttribute('height'))),
    dot_heights: Array.from(document.querySelectorAll('#vv-prob .vv-dot'), (dot) => Number(dot.getAttribute('cy'))),
    scored_dots: Array.from(document.querySelectorAll('#vv-prob .vv-dot'), (dot, i) => (dot.classList.contains('vv-scored') ? i : -1)).filter((i) => i >= 0),
    slider_valuetext: document.getElementById('vv-z').getAttribute('aria-valuetext'),
    axial_label: document.getElementById('vv-axial').getAttribute('aria-label'),
  }));
  report.facts = await text(page, '#vv-facts');
  report.start_readout = await text(page, '#vv-readout');
  report.read = await text(page, '#vv-read');
  report.prob_text = await text(page, '#vv-prob-text');
  report.start_now = await text(page, '#vv-prob-now');
  const shape = await page.evaluate(() => window.__vv.data.shape);

  // The page opens with the crosshair on the strongest voxel, which sits in the gap of the crosshair lines.
  // Its colour on screen is the map's blue; it fades with the opacity slider and is plain grey with the map off.
  const colourAtCrosshair = () => page.evaluate(() => {
    const plane = window.__vv.planes[0];
    const { x, y, w, h } = plane.rect;
    const idx = window.__vv.state.idx;
    const px = x + ((idx[plane.u] + 0.5) / plane.nu) * w, py = y + ((plane.nv - 1 - idx[plane.v] + 0.5) / plane.nv) * h;
    return Array.from(plane.canvas.getContext('2d').getImageData(Math.round(px), Math.round(py), 1, 1).data.slice(0, 3));
  });
  report.colour_with_map = await colourAtCrosshair();
  await setSlider(page, 'vv-opacity', 20);
  await settle(page);
  report.colour_at_low_opacity = await colourAtCrosshair();
  await setSlider(page, 'vv-opacity', 85);
  await showMap(page, false);
  report.colour_without_map = await colourAtCrosshair();

  // The scan on screen. With the map hidden, each slice view is asked for the slice that holds the marker voxel.
  report.screen = {};
  for (const [id, u, v, w] of PLANES) {
    await setCrosshair(page, beside(spec.marker, shape, u, v, w));
    const seen = await onScreen(page, id, spec.marker[u], spec.marker[v]);
    const letters = await page.evaluate((id) => {
      const drawn = window.__drawn.filter((entry) => entry.id === id && /^[A-Z]$/.test(entry.text)).slice(-4);
      const pick = (better) => drawn.reduce((best, entry) => (better(entry, best) ? entry : best)).text;
      return { left: pick((a, b) => a.x < b.x), right: pick((a, b) => a.x > b.x), top: pick((a, b) => a.y < b.y), bottom: pick((a, b) => a.y > b.y) };
    }, id);
    report.screen[id] = {
      marker: seen.here[0], mirrored_across: seen.mirrored_across[0], mirrored_up_down: seen.mirrored_up_down[0],
      marker_side: seen.side, letters,
    };
  }

  // The map on screen. A lone voxel of attention lies in air away from the head: it is tinted where it belongs
  // and nowhere at the mirror images of that place, and the tint goes when the floor is raised above its value.
  await showMap(page, true);
  report.map_screen = {};
  for (const [id, u, v, w] of PLANES) {
    await setCrosshair(page, beside(spec.stray, shape, u, v, w));
    report.map_screen[id] = await onScreen(page, id, spec.stray[u], spec.stray[v]);
  }
  await setSlider(page, 'vv-floor', 60);
  await settle(page);
  report.stray_above_floor = (await onScreen(page, 'vv-sagittal', spec.stray[1], spec.stray[2])).here;
  await setSlider(page, 'vv-floor', 25);

  // The two marked levels of the slab, read from the pixels of the coronal view near its right edge, where
  // there is only air; the crosshair's own line is left out.
  await setCrosshair(page, [Math.floor(shape[0] / 2), Math.floor(shape[1] / 2), spec.slab[0] + 4]);
  report.slab_rows = await page.evaluate(() => {
    const plane = window.__vv.planes[1];
    const { x, y, w, h } = plane.rect;
    const nz = window.__vv.data.shape[2], slab = window.__vv.data.slab;
    const column = plane.canvas.getContext('2d').getImageData(Math.round(x + w - 3), 0, 1, plane.canvas.height).data;
    const crosshair = y + ((plane.nv - 1 - window.__vv.state.idx[2] + 0.5) / plane.nv) * h;
    const found = [];
    for (let row = 0; row < plane.canvas.height; row++) {
      if (column[row * 4] > 60 && Math.abs(row - crosshair) > 2 && (!found.length || row - found[found.length - 1] > 2)) found.push(row);
    }
    return { found, expected: [y + ((nz - 1 - slab[1]) / nz) * h, y + ((nz - slab[0]) / nz) * h] };
  });

  // The crosshair on the marker voxel: position in millimetres, slice numbers, and the note for a slice with no map.
  await setCrosshair(page, spec.marker);
  report.marker_readout = await text(page, '#vv-readout');
  report.marker_slices = [await text(page, '#vv-sagittal-info'), await text(page, '#vv-coronal-info'), await text(page, '#vv-axial-info')];
  report.marker_now = await text(page, '#vv-prob-now');
  await setSlider(page, 'vv-z', 0);
  await settle(page);
  report.outside_readout = await text(page, '#vv-readout');

  // Controls. A bar of the chart goes to its slice; the button goes to the strongest voxel.
  await page.click('#vv-share .vv-hit >> nth=3');
  await settle(page);
  report.bar_click_slice = await page.evaluate(() => window.__vv.state.idx[2] + window.__vv.data.offset[2]);
  await page.click('#vv-peak');
  await settle(page);
  report.peak_idx = (await state(page)).idx;
  report.peak_readout = await text(page, '#vv-readout');

  // Tooltips stay inside what they belong to, and never make the page scroll sideways.
  report.chart_tips = [];
  for (const which of ['nth=0', 'nth=-1']) {
    await page.hover('#vv-prob .vv-hit >> ' + which);
    report.chart_tips.push(await page.evaluate(() => {
      const tip = document.getElementById('vv-prob-tip').getBoundingClientRect();
      const box = document.querySelector('#vv-prob').parentNode.getBoundingClientRect();
      return {
        shown: !document.getElementById('vv-prob-tip').hidden,
        inside: tip.left >= box.left - 0.5 && tip.right <= box.right + 0.5,
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        text: document.getElementById('vv-prob-tip').textContent,
      };
    }));
  }
  await bringIntoView(page, 'vv-coronal');
  const edge = await page.evaluate(() => {
    const plane = window.__vv.planes[1];
    const box = plane.canvas.getBoundingClientRect();
    const k = box.width / plane.canvas.width;
    return { x: box.left + (plane.rect.x + plane.rect.w - 2) * k, y: box.top + (plane.rect.y + plane.rect.h / 2) * k };
  });
  await page.mouse.move(edge.x, edge.y);
  report.view_tip = await page.evaluate(() => {
    const view = document.getElementById('vv-coronal').parentNode;
    const tip = view.querySelector('.vv-tip');
    const a = tip.getBoundingClientRect(), b = view.getBoundingClientRect();
    return { shown: !tip.hidden, inside: a.left >= b.left - 0.5 && a.right <= b.right + 0.5 && a.top >= b.top - 0.5, text: tip.textContent };
  });

  // Pointer behaviour on a slice view: the secondary button does nothing, and a drag that leaves the image
  // keeps following the pointer to the last voxel.
  await bringIntoView(page, 'vv-axial');
  const axial = await page.evaluate(() => {
    const plane = window.__vv.planes[0];
    const box = plane.canvas.getBoundingClientRect();
    const k = box.width / plane.canvas.width;
    return {
      x: box.left + (plane.rect.x + plane.rect.w * 0.3) * k, y: box.top + (plane.rect.y + plane.rect.h * 0.3) * k,
      right: box.right, nu: plane.nu,
    };
  });
  const beforeRight = (await state(page)).idx;
  await page.mouse.click(axial.x, axial.y, { button: 'right' });
  await settle(page);
  report.right_button_moves = JSON.stringify((await state(page)).idx) !== JSON.stringify(beforeRight);
  await page.mouse.move(axial.x, axial.y);
  await page.mouse.down();
  await page.mouse.move(axial.right + 40, axial.y, { steps: 5 });
  await page.mouse.up();
  await settle(page);
  report.drag_past_edge = [(await state(page)).idx[0], axial.nu - 1];

  // The 3D view. Seen from behind, the patient's right is on the right of the picture and the letter R points
  // right; seen from the front both are on the left. The scan's right half is the brighter one.
  const face = async (yaw) => {
    await page.evaluate((yaw) => { window.__vv.state.yaw = yaw; window.__vv.state.pitch = 0; }, yaw);
    await setSlider(page, 'vv-opacity', 84);
    await settle(page);
    return picture3d(page);
  };
  await showMap(page, false);
  report.view3d = { from_behind: await face(0), from_the_front: await face(Math.PI) };
  await showMap(page, true);
  await setSlider(page, 'vv-opacity', 85);
  await bringIntoView(page, 'vv-3d');
  const lit3d = () => page.evaluate(() => {
    const canvas = document.getElementById('vv-3d');
    // The middle of the view, away from the orientation marker in its corner.
    const x0 = Math.round(canvas.width * 0.25), y0 = Math.round(canvas.height * 0.2);
    const w = Math.round(canvas.width * 0.5), h = Math.round(canvas.height * 0.5);
    const pixels = canvas.getContext('2d').getImageData(x0, y0, w, h).data;
    let lit = 0, signature = 0;
    for (let i = 0; i < pixels.length; i += 4) { if (pixels[i] > 40) lit++; signature += pixels[i] * (((i / 4) % 97) + 1); }
    return { share: lit / (pixels.length / 4), signature };
  });
  const first = await lit3d();
  const box3d = await page.evaluate(() => { const box = document.getElementById('vv-3d').getBoundingClientRect(); return { x: box.left + box.width / 2, y: box.top + box.height / 2 }; });
  const start = await state(page);
  await page.mouse.move(box3d.x, box3d.y);
  await page.mouse.down();
  await page.mouse.move(box3d.x + 40, box3d.y + 30, { steps: 5 });
  await page.mouse.up();
  await settle(page);
  const turned = await state(page);
  const second = await lit3d();
  Object.assign(report.view3d, {
    head_share: first.share,
    picture_changes: first.signature !== second.signature,
    drag_right_yaw: turned.yaw - start.yaw,
    drag_down_pitch: turned.pitch - start.pitch,
  });
  await page.focus('#vv-3d');
  await page.keyboard.press('ArrowRight');
  await settle(page);
  report.view3d.arrow_right_yaw = (await state(page)).yaw - turned.yaw;
  await page.evaluate(() => { window.__vv.state.yaw = -0.005; window.__vv.state.pitch = -0.3; });
  await setSlider(page, 'vv-opacity', 80);
  await settle(page);
  report.view3d.caption_near_full_turn = await text(page, '#vv-3d-info');

  report.problems = problems;
  await context.close();
  return report;
}

// Touches are sent through the browser's debugging interface, which is how a finger reaches the page.
async function touchPage(browser) {
  const { context, page, problems } = await open(browser, spec.pages.main, { touch: true, motion: true, viewport: { width: 700, height: 900 } });
  if (!(await started(page, problems))) return notStarted(context, page, problems);
  await settle(page);
  const client = await context.newCDPSession(page);
  const touch = (type, points) => client.send('Input.dispatchTouchEvent', { type, touchPoints: points.map(([x, y], id) => ({ x, y, id })) });
  const report = { started: true };

  // A tap on a slice view places the crosshair on the voxel under the finger.
  await bringIntoView(page, 'vv-axial');
  const target = await page.evaluate(() => {
    const plane = window.__vv.planes[0];
    const box = plane.canvas.getBoundingClientRect();
    const k = box.width / plane.canvas.width;
    const u = 5, v = 7;
    return {
      x: box.left + (plane.rect.x + ((u + 0.5) / plane.nu) * plane.rect.w) * k,
      y: box.top + (plane.rect.y + ((plane.nv - 1 - v + 0.5) / plane.nv) * plane.rect.h) * k,
      u, v, cx: box.left + box.width / 2, cy: box.top + box.height / 2,
    };
  });
  await page.touchscreen.tap(target.x, target.y);
  await settle(page);
  const tapped = (await state(page)).idx;
  report.tap_places_crosshair = tapped[0] === target.u && tapped[1] === target.v;

  // A finger that rests on the turning 3D head and then drags starts from where the head is by then,
  // and a tap on the head stops the turning.
  await bringIntoView(page, 'vv-3d');
  await page.evaluate(() => { if (!window.__vv.state.spin) document.getElementById('vv-spin').click(); });
  const head = await page.evaluate(() => { const box = document.getElementById('vv-3d').getBoundingClientRect(); return { x: box.left + box.width / 2, y: box.top + box.height / 2 }; });
  await touch('touchStart', [[head.x, head.y]]);
  await pause(page, 700);
  const resting = await state(page);
  await touch('touchMove', [[head.x + 8, head.y]]);
  await settle(page);
  const grabbed = await state(page);
  await touch('touchMove', [[head.x + 16, head.y]]);
  await settle(page);
  const dragged = await state(page);
  await touch('touchEnd', []);
  report.rest_then_drag = {
    turned_while_resting: resting.spin,
    jump_on_grab: grabbed.yaw - resting.yaw,
    turn_for_8_px: dragged.yaw - grabbed.yaw,
    still_turning: dragged.spin,
  };
  await page.evaluate(() => { if (!window.__vv.state.spin) document.getElementById('vv-spin').click(); });
  await pause(page, 150);
  await page.touchscreen.tap(head.x, head.y);
  await settle(page);
  report.tap_stops_turning = !(await state(page)).spin;

  // Two fingers on a view are a pinch for the browser: the crosshair stays where it is.
  await bringIntoView(page, 'vv-axial');
  const pinch = await page.evaluate(() => { const box = document.getElementById('vv-axial').getBoundingClientRect(); return { cx: box.left + box.width / 2, cy: box.top + box.height / 2 }; });
  const beforePinch = (await state(page)).idx;
  await touch('touchStart', [[pinch.cx - 10, pinch.cy], [pinch.cx + 10, pinch.cy]]);
  for (let spread = 14; spread <= 50; spread += 12) await touch('touchMove', [[pinch.cx - spread, pinch.cy], [pinch.cx + spread, pinch.cy]]);
  await touch('touchEnd', []);
  await settle(page);
  report.two_fingers_move_crosshair = JSON.stringify((await state(page)).idx) !== JSON.stringify(beforePinch);

  report.problems = problems;
  await context.close();
  return report;
}

async function motionPage(browser) {
  const report = { started: true, problems: [] };
  {
    const { context, page, problems } = await open(browser, spec.pages.main, { motion: true });
    if (!(await started(page, problems))) return notStarted(context, page, problems);
    await pause(page, 500);
    report.turning_after_load = await page.evaluate(() => document.getElementById('vv-spin').getAttribute('aria-pressed'));
    const a = (await state(page)).yaw;
    await pause(page, 400);
    const b = (await state(page)).yaw;
    report.turns_while_visible = b !== a;
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    await pause(page, 400);
    const c = (await state(page)).yaw;
    await pause(page, 500);
    const d = (await state(page)).yaw;
    report.off_screen_fraction = await page.evaluate(() => {
      const box = document.getElementById('vv-3d').getBoundingClientRect();
      return Math.max(0, Math.min(box.bottom, window.innerHeight) - Math.max(box.top, 0)) / box.height;
    });
    report.turns_while_off_screen = d !== c;
    await page.evaluate(() => window.scrollTo(0, 0));
    await pause(page, 600);
    report.turns_again_when_back = (await state(page)).yaw !== d;
    report.problems = problems;
    await context.close();
  }
  {
    // A window so short that little of the 3D view shows at first: the opening turn waits until most of it does.
    const { context, page, problems } = await open(browser, spec.pages.main, { motion: true, viewport: { width: 1100, height: 240 } });
    if (!(await started(page, problems))) return notStarted(context, page, problems);
    await pause(page, 500);
    report.short_window = await page.evaluate(() => {
      const box = document.getElementById('vv-3d').getBoundingClientRect();
      return {
        fraction: Math.max(0, Math.min(box.bottom, window.innerHeight) - Math.max(box.top, 0)) / box.height,
        pressed: document.getElementById('vv-spin').getAttribute('aria-pressed'),
      };
    });
    await page.evaluate(() => document.getElementById('vv-3d').scrollIntoView({ block: 'center' }));
    await pause(page, 700);
    report.short_window.pressed_once_in_view = await page.evaluate(() => document.getElementById('vv-spin').getAttribute('aria-pressed'));
    report.problems = report.problems.concat(problems);
    await context.close();
  }
  return report;
}

// Layout at several window sizes: the four captions are the same height, nothing overflows sideways, and the
// views share a screen with the slider that drives them.
async function layoutPage(browser) {
  const { context, page, problems } = await open(browser, spec.pages.main);
  if (!(await started(page, problems))) return notStarted(context, page, problems);
  const report = { started: true, sizes: [] };
  for (const [width, height] of [[360, 780], [640, 768], [660, 768], [700, 768], [860, 900], [1000, 768], [1100, 900]]) {
    await page.setViewportSize({ width, height });
    await settle(page);
    report.sizes.push(await page.evaluate(([width, height]) => ({
      width, height,
      captions: Array.from(document.querySelectorAll('.vv-caption'), (caption) => Math.round(caption.getBoundingClientRect().height)),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      views_to_slider: Math.round(document.getElementById('vv-z').getBoundingClientRect().bottom - document.querySelector('.vv-views').getBoundingClientRect().top),
    }), [width, height]));
  }
  report.problems = problems;
  await context.close();
  return report;
}

async function plainPage(browser) {
  const { context, page, problems } = await open(browser, spec.pages.nomap);
  if (!(await started(page, problems))) return notStarted(context, page, problems);
  await settle(page);
  const report = await page.evaluate(() => ({
    started: true,
    status_hidden: document.getElementById('vv-status').hidden,
    title: document.querySelector('.vv-title').textContent,
    document_title: document.title,
    attention_controls_shown: Array.from(document.querySelectorAll('.vv-attention')).filter((node) => !node.hidden && node.getClientRects().length).length,
    labels: ['vv-axial', 'vv-coronal', 'vv-sagittal', 'vv-3d'].map((id) => document.getElementById(id).getAttribute('aria-label')),
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    notice: document.getElementById('vv-notice').textContent,
  }));
  report.readout = await text(page, '#vv-readout');
  report.read = await text(page, '#vv-read');
  report.facts = await text(page, '#vv-facts');
  report.scan_drawn = (await drawn(page, 'vv-axial')) > 0 && (await drawn(page, 'vv-3d')) > 0;
  await page.click('#vv-reset');
  await settle(page);
  report.problems = problems;
  await context.close();
  return report;
}

async function hostilePage(browser) {
  const { context, page, problems } = await open(browser, spec.pages.hostile);
  if (!(await started(page, problems))) return notStarted(context, page, problems);
  await settle(page);
  await page.hover('#vv-prob .vv-hit >> nth=0');
  const report = await page.evaluate(() => ({
    started: true,
    pwned: Object.keys(window).filter((key) => key.startsWith('pwned')),
    scripts: document.scripts.length,
    injected_elements: document.querySelectorAll('img, iframe, a, base, object, embed, form, link, meta[http-equiv]').length,
    handler_attributes: Array.from(document.querySelectorAll('*')).filter((node) => Array.from(node.attributes).some((attribute) => attribute.name.startsWith('on'))).length,
    notice: document.getElementById('vv-notice').textContent,
    warnings: Array.from(document.querySelectorAll('#vv-warnings li'), (item) => item.textContent),
    title: document.querySelector('.vv-title').textContent,
    facts: document.getElementById('vv-facts').textContent,
    tip: document.getElementById('vv-prob-tip').textContent,
    table_head: Array.from(document.querySelectorAll('#vv-table thead th'), (cell) => cell.textContent),
  }));
  report.problems = problems;
  await context.close();
  return report;
}

async function sparsePage(browser) {
  const { context, page, problems } = await open(browser, spec.pages.sparse);
  if (!(await started(page, problems))) return notStarted(context, page, problems);
  await settle(page);
  const report = { started: true, steps: [], problems };
  for (const local of spec.sparse_steps) {
    await setSlider(page, 'vv-z', local);
    await settle(page);
    report.steps.push({
      slice: await text(page, '#vv-axial-info'),
      note: await text(page, '#vv-readout .vv-val'),
      now: await text(page, '#vv-share-now'),
    });
  }
  report.facts = await text(page, '#vv-facts');
  report.read = await text(page, '#vv-read');
  await context.close();
  return report;
}

// A page whose embedded scan cannot be shown must say so, draw nothing and offer nothing to operate,
// whatever is done to it afterwards.
async function refusedPage(browser, file, options) {
  const { context, page, problems } = await open(browser, file, options);
  const drew = await started(page, problems);
  // Poke the controls the way a script or a stray key press could, although they are no longer on screen.
  await page.evaluate(() => {
    for (const [id, value] of [['vv-z', 3], ['vv-opacity', 40]]) {
      const slider = document.getElementById(id);
      slider.value = value;
      slider.dispatchEvent(new Event('input'));
    }
    for (const id of ['vv-reset', 'vv-spin', 'vv-peak']) document.getElementById(id).click();
    document.getElementById('vv-3d').dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
  });
  await settle(page);
  const report = await page.evaluate(() => ({
    status: document.getElementById('vv-status').textContent,
    status_hidden: document.getElementById('vv-status').hidden,
    status_marked: document.getElementById('vv-status').classList.contains('vv-failed') && document.getElementById('vv-status').getAttribute('role') === 'alert',
    exposed: Boolean(window.__vv),
    left_on_screen: Array.from(document.querySelectorAll('#vv canvas, #vv input, #vv button, #vv svg, #vv table')).filter((node) => node.getClientRects().length).length,
    spinning: document.getElementById('vv-spin').getAttribute('aria-pressed'),
  }));
  report.drew = drew;
  report.drawn = (await drawn(page, 'vv-axial')) + (await drawn(page, 'vv-3d'));
  report.problems = problems;
  await context.close();
  return report;
}

(async () => {
  const browser = await chromium.launch();
  const report = {
    main: await mainPage(browser),
    touch: await touchPage(browser),
    motion: await motionPage(browser),
    layout: await layoutPage(browser),
    plain: await plainPage(browser),
    hostile: await hostilePage(browser),
    sparse: await sparsePage(browser),
    truncated: await refusedPage(browser, spec.pages.truncated),
    altered: await refusedPage(browser, spec.pages.altered),
    altered_map: await refusedPage(browser, spec.pages.altered_map),
    mirrored_map: await refusedPage(browser, spec.pages.mirrored_map),
    old_browser: await refusedPage(browser, spec.pages.main, { init: () => { delete window.DecompressionStream; window.DecompressionStream = undefined; } }),
  };
  console.log(JSON.stringify(report));
  await browser.close();
})().catch((error) => { console.error(error); process.exit(1); });
