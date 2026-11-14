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

const ready = (page) => page.waitForFunction(() => window.__vv && window.__vv.anatomy, null, { timeout: 60000 });
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
// Pointer coordinates are relative to the window, so a view is brought on screen before it is pointed at.
const bringIntoView = (page, id) => page.evaluate((id) => document.getElementById(id).scrollIntoView({ block: 'center' }), id);
const state = (page) => page.evaluate(() => ({ idx: window.__vv.state.idx.slice(), yaw: window.__vv.state.yaw, pitch: window.__vv.state.pitch }));
// Sum of the pixels of a visible canvas: zero means nothing was drawn on it.
const drawn = (page, id) => page.evaluate((id) => {
  const canvas = document.getElementById(id);
  const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  let lit = 0;
  for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > 40) lit++;
  return lit;
}, id);

async function mainPage(browser) {
  const { context, page, problems } = await open(browser, spec.pages.main);
  await ready(page);
  await settle(page);
  const report = await page.evaluate(() => ({
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

  // What is on screen. With the map hidden, each slice view is asked for the slice that holds the marker
  // voxel while the crosshair sits ten voxels away from it, so the crosshair lines do not cover it.
  // The page opens with the crosshair on the strongest voxel, which sits in the gap of the crosshair lines:
  // its colour on screen is the map's blue, and plain grey once the map is switched off.
  const colourAtCrosshair = () => page.evaluate(() => {
    const plane = window.__vv.planes[0];
    const { x, y, w, h } = plane.rect;
    const idx = window.__vv.state.idx;
    const px = x + ((idx[plane.u] + 0.5) / plane.nu) * w, py = y + ((plane.nv - 1 - idx[plane.v] + 0.5) / plane.nv) * h;
    return Array.from(plane.canvas.getContext('2d').getImageData(Math.round(px), Math.round(py), 1, 1).data.slice(0, 3));
  });
  report.colour_with_map = await colourAtCrosshair();
  await page.evaluate(() => { const box = document.getElementById('vv-show'); box.checked = false; box.dispatchEvent(new Event('change')); });
  await settle(page);
  report.colour_without_map = await colourAtCrosshair();
  const shape = await page.evaluate(() => window.__vv.data.shape);
  const away = (value, size) => (value < size / 2 ? Math.min(size - 1, value + 10) : Math.max(0, value - 10));
  report.screen = {};
  for (const [id, u, v, w] of [['vv-axial', 0, 1, 2], ['vv-coronal', 0, 2, 1], ['vv-sagittal', 1, 2, 0]]) {
    const index = [0, 0, 0];
    index[w] = spec.marker[w];
    index[u] = away(spec.marker[u], shape[u]);
    index[v] = away(spec.marker[v], shape[v]);
    await setCrosshair(page, index);
    report.screen[id] = await page.evaluate(([id, mu, mv]) => {
      const plane = window.__vv.planes.find((candidate) => candidate.id === id);
      const { x, y, w, h } = plane.rect;
      const context = plane.canvas.getContext('2d');
      const place = (u, v) => [x + ((u + 0.5) / plane.nu) * w, y + ((plane.nv - 1 - v + 0.5) / plane.nv) * h];
      const level = (u, v) => { const [px, py] = place(u, v); return context.getImageData(Math.round(px), Math.round(py), 1, 1).data[0]; };
      const [px, py] = place(mu, mv);
      const letters = window.__drawn.filter((entry) => entry.id === id && /^[A-Z]$/.test(entry.text)).slice(-4);
      const pick = (better) => letters.reduce((best, entry) => (better(entry, best) ? entry : best)).text;
      return {
        marker: level(mu, mv),
        mirrored_across: level(plane.nu - 1 - mu, mv),
        mirrored_up_down: level(mu, plane.nv - 1 - mv),
        marker_side: [px < plane.canvas.width / 2 ? 'left' : 'right', py < plane.canvas.height / 2 ? 'top' : 'bottom'],
        letters: {
          left: pick((a, b) => a.x < b.x), right: pick((a, b) => a.x > b.x),
          top: pick((a, b) => a.y < b.y), bottom: pick((a, b) => a.y > b.y),
        },
      };
    }, [id, spec.marker[u], spec.marker[v]]);
  }

  // The crosshair on the marker voxel: position in millimetres, slice numbers, and the note for a slice with no map.
  await setCrosshair(page, spec.marker);
  report.marker_readout = await text(page, '#vv-readout');
  report.marker_slices = [await text(page, '#vv-sagittal-info'), await text(page, '#vv-coronal-info'), await text(page, '#vv-axial-info')];
  report.marker_now = await text(page, '#vv-prob-now');
  await setSlider(page, 'vv-z', 0);
  await settle(page);
  report.outside_readout = await text(page, '#vv-readout');

  // Controls. A bar of the chart goes to its slice; the button goes to the strongest voxel.
  await page.evaluate(() => { const box = document.getElementById('vv-show'); box.checked = true; box.dispatchEvent(new Event('change')); });
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

  // The 3D view shows the head, turns with the pointer, and its caption never reads 360 degrees.
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
  await bringIntoView(page, 'vv-3d');
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
  report.view3d = {
    head_share: first.share,
    picture_changes: first.signature !== second.signature,
    drag_right_yaw: turned.yaw - start.yaw,
    drag_down_pitch: turned.pitch - start.pitch,
  };
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

async function touchPage(browser) {
  const { context, page, problems } = await open(browser, spec.pages.main, { touch: true });
  await ready(page);
  await settle(page);
  await bringIntoView(page, 'vv-axial');
  const target = await page.evaluate(() => {
    const plane = window.__vv.planes[0];
    const box = plane.canvas.getBoundingClientRect();
    const k = box.width / plane.canvas.width;
    const u = 5, v = 7;
    return {
      x: box.left + (plane.rect.x + ((u + 0.5) / plane.nu) * plane.rect.w) * k,
      y: box.top + (plane.rect.y + ((plane.nv - 1 - v + 0.5) / plane.nv) * plane.rect.h) * k,
      u, v,
    };
  });
  await page.touchscreen.tap(target.x, target.y);
  await settle(page);
  const after = (await state(page)).idx;
  await context.close();
  return { tap_places_crosshair: after[0] === target.u && after[1] === target.v, problems };
}

async function motionPage(browser) {
  const report = {};
  {
    const { context, page, problems } = await open(browser, spec.pages.main, { motion: true });
    await ready(page);
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
    await ready(page);
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

async function plainPage(browser) {
  const { context, page, problems } = await open(browser, spec.pages.nomap);
  await ready(page);
  await settle(page);
  const report = await page.evaluate(() => ({
    status_hidden: document.getElementById('vv-status').hidden,
    title: document.querySelector('.vv-title').textContent,
    document_title: document.title,
    attention_controls_shown: Array.from(document.querySelectorAll('.vv-attention')).filter((node) => !node.hidden && node.getClientRects().length).length,
    labels: ['vv-axial', 'vv-coronal', 'vv-sagittal', 'vv-3d'].map((id) => document.getElementById(id).getAttribute('aria-label')),
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
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
  await ready(page);
  await settle(page);
  await page.hover('#vv-prob .vv-hit >> nth=0');
  const report = await page.evaluate(() => ({
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
  await ready(page);
  await settle(page);
  const report = { steps: [], problems };
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

// A page whose embedded scan cannot be shown must say so and draw nothing, whatever is done to it afterwards.
async function refusedPage(browser, file, options) {
  const { context, page, problems } = await open(browser, file, options);
  await page.waitForFunction(() => document.getElementById('vv-status').getAttribute('role') === 'alert', null, { timeout: 60000 });
  await setSlider(page, 'vv-z', 3);
  await setSlider(page, 'vv-opacity', 40);
  await page.click('#vv-reset');
  await page.click('#vv-spin');
  await settle(page);
  const report = await page.evaluate(() => ({
    status: document.getElementById('vv-status').textContent,
    status_hidden: document.getElementById('vv-status').hidden,
    exposed: Boolean(window.__vv),
  }));
  report.drawn = (await drawn(page, 'vv-axial')) + (await drawn(page, 'vv-3d'));
  report.problems = problems.filter((problem) => !problem.startsWith('console: Failed to load resource'));
  await context.close();
  return report;
}

(async () => {
  const browser = await chromium.launch();
  const report = {
    main: await mainPage(browser),
    touch: await touchPage(browser),
    motion: await motionPage(browser),
    plain: await plainPage(browser),
    hostile: await hostilePage(browser),
    sparse: await sparsePage(browser),
    truncated: await refusedPage(browser, spec.pages.truncated),
    altered: await refusedPage(browser, spec.pages.altered),
    old_browser: await refusedPage(browser, spec.pages.main, { init: () => { delete window.DecompressionStream; window.DecompressionStream = undefined; } }),
  };
  console.log(JSON.stringify(report));
  await browser.close();
})().catch((error) => { console.error(error); process.exit(1); });
