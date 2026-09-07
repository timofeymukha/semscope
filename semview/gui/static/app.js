// semview GUI application: data loading, view control, overlays, probes.
import { Renderer } from './renderer.js';
import { Spectral } from './spectral.js';

const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, d = 4) => {
  if (v === null || v === undefined || Number.isNaN(v)) return 'nan';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(d - 1);
  return Number(v.toPrecision(d + 1)).toString();
};

// ----------------------------------------------------------------- state
const S = {
  meta: null,
  step: 0,
  field: null,
  cmap: 'viridis',
  cmaps: null,
  colormaps: {},
  range: { auto: true, sym: false, lo: 0, hi: 1 },
  invert: false,
  mode: 0,
  edges: true, edgeWidth: 1,
  contours: false, nContours: 12,
  nodes: false, nodeSize: 3,
  axes: true, colorbar: true,
  pxPerCell: 8,
  bg: 'dark',
  view: { cx: 0, cy: 0, scale: 1 },
  playing: false, timer: null,
  fieldCache: new Map(),   // key step:name -> {values, min, max, time}
  mesh: null,              // {x, y, elmap}
  spectral: null,
  probePinned: null,       // {x, y}
  lines: [],               // line probes, see the line probe section
  snapAngle: false,
  drag: null,
  hover: null,
  needsRender: true,
  viewFitted: false,
  lineMode: false,
};

const BG = { dark: '#0d0f14', light: '#f4f6fa', black: '#000000', white: '#ffffff' };
const isLight = () => S.bg === 'light' || S.bg === 'white';

// ----------------------------------------------------------------- canvas / renderer
const glCanvas = $('gl');
const overlay = $('overlay');
const octx = overlay.getContext('2d');
let R;
try {
  R = new Renderer(glCanvas);
} catch (err) {
  toast('WebGL2 initialisation failed: ' + err.message, true);
  throw err;
}

function resize() {
  const view = $('view');
  const w = view.clientWidth, h = view.clientHeight, dpr = window.devicePixelRatio || 1;
  if (w === 0 || h === 0) return;           // hidden tab / not laid out yet
  const first = R.width <= 1 || !S.viewFitted;
  R.resize(w, h, dpr);
  overlay.width = Math.round(w * dpr); overlay.height = Math.round(h * dpr);
  if (first && S.meta && S.meta.open) fitView();
  requestRender();
}
new ResizeObserver(resize).observe($('view'));

function requestRender() { S.needsRender = true; }

function renderOptions() {
  const light = isLight();
  return {
    background: BG[S.bg],
    mode: S.mode,
    range: [S.range.lo, S.range.hi],
    invert: S.invert,
    contours: S.contours, nContours: S.nContours, contourWidth: 1.2,
    contourColor: light ? '#111111' : '#ffffff', contourAlpha: 0.85,
    edges: S.edges, edgeWidth: S.edgeWidth,
    edgeColor: light ? '#1b2130' : '#ffffff', edgeAlpha: light ? 0.55 : 0.45,
    nodes: S.nodes, nodeSize: S.nodeSize, nodeColor: light ? '#0b3d91' : '#5cc8ff',
    nanColor: light ? '#d9dee8' : '#1c2130',
    pxPerCell: S.pxPerCell,
  };
}

let lastFrame = performance.now(), fps = 0;
function frame() {
  if (S.needsRender) {
    S.needsRender = false;
    R.setView(S.view);
    const t0 = performance.now();
    R.render(renderOptions());
    drawOverlay();
    const dt = performance.now() - t0;
    fps = 0.9 * fps + 0.1 * (1000 / Math.max(performance.now() - lastFrame, 1));
    lastFrame = performance.now();
    updateStatus(dt);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function updateStatus(dt) {
  if (!S.meta) { $('status').textContent = ''; return; }
  const st = R.stats;
  $('status').textContent = `${st.visible.toLocaleString()} / ${S.meta.nelv.toLocaleString()} elements · ${(st.triangles / 1000).toFixed(0)}k tris · ${dt.toFixed(1)} ms` + (S.meta.mpi_ranks > 1 ? ` · ${S.meta.mpi_ranks} MPI ranks` : '');
}

// ----------------------------------------------------------------- coordinate transforms
const W = () => R.width, H = () => R.height;
const dataToScreen = (x, y) => [(x - S.view.cx) * S.view.scale + W() / 2, H() / 2 - (y - S.view.cy) * S.view.scale];
const screenToData = (px, py) => [(px - W() / 2) / S.view.scale + S.view.cx, (H() / 2 - py) / S.view.scale + S.view.cy];

function fitView() {
  if (!S.meta || !S.meta.open) return;
  const [x0, x1, y0, y1] = S.meta.bounds;
  const m = 0.06;
  if (W() <= 1 || H() <= 1) { S.viewFitted = false; return; }   // fit again once we have a size
  const sx = W() / ((x1 - x0) * (1 + 2 * m) || 1), sy = H() / ((y1 - y0) * (1 + 2 * m) || 1);
  S.view = { cx: 0.5 * (x0 + x1), cy: 0.5 * (y0 + y1), scale: Math.min(sx, sy) };
  S.viewFitted = true;
  requestRender();
}

// ----------------------------------------------------------------- data loading
async function api(route, params = {}) {
  const q = new URLSearchParams(params).toString();
  const res = await fetch(`/api/${route}${q ? '?' + q : ''}`);
  const ctype = res.headers.get('Content-Type') || '';
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  if (ctype.includes('json')) return res.json();
  const buf = await res.arrayBuffer();
  return { buf, headers: res.headers };
}

async function loadState() {
  S.meta = await api('state');
  if (!S.meta.open) { $('file-info').textContent = 'No dataset loaded — use Open…'; openDialog(); return; }
  await loadMesh();
  populateFields();
  setupTime();
  fitView();
  await loadField();
}

async function loadMesh() {
  const m = S.meta;
  const { buf } = await api('mesh');
  const all = new Float32Array(buf);
  const N = m.nelv * m.n * m.n;
  const x = all.subarray(0, N), y = all.subarray(N, 2 * N);
  const el = await api('elmap');
  S.mesh = { x, y, elmap: new Int32Array(el.buf) };
  S.spectral = new Spectral(m.gll);
  R.setMesh({ x, y, n: m.n, nelv: m.nelv, nodes: m.gll, bary: m.bary });
  S.fieldCache.clear();
  const file = m.path.split('/').pop();
  $('file-info').innerHTML = `<b>${file}</b><br>${m.nelv.toLocaleString()} elements · order N = ${m.order} (${m.n}×${m.n} GLL)<br>${m.nsteps} step${m.nsteps > 1 ? 's' : ''}` + (m.mpi_ranks > 1 ? ` · read with ${m.mpi_ranks} MPI ranks` : '');
  document.title = `semview — ${m.name}`;
}

function populateFields() {
  const sel = $('field');
  const prev = S.field;
  sel.innerHTML = '';
  const groups = [['stored', S.meta.fields], ['derived', S.meta.available.filter(f => !S.meta.fields.includes(f))], ['resolution', S.meta.fields.map(f => `decay:${f}`)]];
  for (const [label, names] of groups) {
    if (!names.length) continue;
    const og = document.createElement('optgroup');
    og.label = label;
    for (const nm of names) {
      const o = document.createElement('option');
      o.value = nm;
      o.textContent = nm.startsWith('decay:') ? `log10 spectral decay of ${nm.slice(6)}` : (S.meta.derived[nm] ? `${nm}  (${S.meta.derived[nm]})` : nm);
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  S.field = (prev && S.meta.available.includes(prev)) ? prev : S.meta.fields[0];
  sel.value = S.field;
}

function setupTime() {
  const n = S.meta.nsteps;
  $('time-block').hidden = n <= 1;
  $('step-slider').max = n - 1;
  $('step-max').textContent = n - 1;
  S.step = clamp(S.step, 0, n - 1);
  $('step-slider').value = S.step;
}

async function fetchField(step, name) {
  const key = `${step}:${name}`;
  if (S.fieldCache.has(key)) return S.fieldCache.get(key);
  const { buf, headers } = await api('field', { name, step });
  const entry = { values: new Float32Array(buf), min: parseFloat(headers.get('X-Min')), max: parseFloat(headers.get('X-Max')), time: parseFloat(headers.get('X-Time')) };
  S.fieldCache.set(key, entry);
  if (S.fieldCache.size > 48) S.fieldCache.delete(S.fieldCache.keys().next().value);
  return entry;
}

let loadToken = 0;
async function loadField() {
  if (!S.meta || !S.meta.open) return;
  const token = ++loadToken;
  try {
    const f = await fetchField(S.step, S.field);
    if (token !== loadToken) return;
    S.current = f;
    R.setField(f.values);
    if (S.range.auto) setRange(f.min, f.max, false);
    $('step-label').textContent = S.step;
    $('time-label').textContent = `t = ${fmt(f.time, 6)}`;
    requestRender();
    updateProbe();
    refreshLines();
    if (S.playing) prefetch();
  } catch (err) { toast(err.message, true); }
}

function prefetch() {
  for (let k = 1; k <= 3; k++) {
    const s = (S.step + k) % S.meta.nsteps;
    fetchField(s, S.field).catch(() => {});
  }
}

function setRange(lo, hi, fromUser) {
  if (S.range.sym) { const a = Math.max(Math.abs(lo), Math.abs(hi)); lo = -a; hi = a; }
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  S.range.lo = lo; S.range.hi = hi;
  if (!fromUser) { $('vmin').value = fmt(lo, 5); $('vmax').value = fmt(hi, 5); }
  requestRender();
}

async function loadColormaps() {
  S.colormaps = await api('colormaps');
  const sel = $('cmap');
  sel.innerHTML = '';
  for (const nm of Object.keys(S.colormaps)) {
    const o = document.createElement('option'); o.value = nm; o.textContent = nm; sel.appendChild(o);
  }
  sel.value = S.cmap;
  R.setColormap(S.colormaps[S.cmap]);
}

// ----------------------------------------------------------------- overlay (axes, colorbar, markers)
function tickFmt(v, step) {
  if (Math.abs(v) < 1e-12 * step) v = 0;
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || step < 1e-6 * Math.max(a, 1e-300))) return v.toExponential(Math.max(1, Math.min(8, Math.ceil(Math.log10(Math.max(a, 1e-300) / step)))));
  const decimals = Math.max(0, Math.min(12, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(decimals);
}

function niceStep(range, target) {
  const raw = range / target;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  const f = m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10;
  return f * p;
}

function drawOverlay() {
  const dpr = R.dpr, w = W(), h = H();
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, w, h);
  if (!S.meta || !S.meta.open) return;
  const light = isLight();
  const fg = light ? '#1b2130' : '#e6e9f0', dim = light ? 'rgba(27,33,48,.45)' : 'rgba(230,233,240,.45)';
  octx.font = '11px ui-monospace, Menlo, Consolas, monospace';
  if (S.axes) {
    const [xa, ya] = screenToData(0, h), [xb, yb] = screenToData(w, 0);
    const sx = niceStep(xb - xa, w / 110), sy = niceStep(yb - ya, h / 70);
    octx.strokeStyle = dim; octx.fillStyle = fg; octx.lineWidth = 1;
    octx.textAlign = 'center'; octx.textBaseline = 'bottom';
    for (let x = Math.ceil(xa / sx) * sx; x <= xb; x += sx) {
      const [px] = dataToScreen(x, 0);
      octx.beginPath(); octx.moveTo(px, h); octx.lineTo(px, h - 6); octx.stroke();
      octx.fillText(tickFmt(x, sx), px, h - 8);
    }
    octx.textAlign = 'left'; octx.textBaseline = 'middle';
    for (let y = Math.ceil(ya / sy) * sy; y <= yb; y += sy) {
      const [, py] = dataToScreen(0, y);
      octx.beginPath(); octx.moveTo(0, py); octx.lineTo(6, py); octx.stroke();
      octx.fillText(tickFmt(y, sy), 9, py);
    }
  }
  if (S.colorbar && S.current) {
    const n = 5;
    const labels = [];
    for (let k = 0; k <= n; k++) labels.push(fmt(S.range.lo + (S.range.hi - S.range.lo) * k / n, 4));
    const labelW = Math.max(...labels.map(t => octx.measureText(t).width));
    const cw = 14, ch = Math.min(h * 0.55, 420), cx = w - 16 - labelW - 7 - cw, cy = (h - ch) / 2;
    const tab = S.colormaps[S.cmap];
    for (let i = 0; i < ch; i++) {
      let t = 1 - i / ch; if (S.invert) t = 1 - t;
      const c = tab[Math.min(tab.length - 1, Math.floor(t * tab.length))];
      octx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      octx.fillRect(cx, cy + i, cw, 1.5);
    }
    octx.strokeStyle = dim; octx.strokeRect(cx + 0.5, cy + 0.5, cw, ch);
    octx.fillStyle = fg; octx.textAlign = 'left'; octx.textBaseline = 'middle';
    for (let k = 0; k <= n; k++) {
      const py = cy + ch - ch * k / n;
      octx.beginPath(); octx.moveTo(cx + cw, py); octx.lineTo(cx + cw + 4, py); octx.stroke();
      octx.fillText(labels[k], cx + cw + 7, py);
    }
    if (S.contours) {
      octx.strokeStyle = light ? 'rgba(0,0,0,.6)' : 'rgba(255,255,255,.6)';
      for (let k = 1; k < S.nContours; k++) { const py = cy + ch - ch * k / S.nContours; octx.beginPath(); octx.moveTo(cx, py); octx.lineTo(cx + cw, py); octx.stroke(); }
    }
    octx.font = '600 13px -apple-system, Segoe UI, Inter, Roboto, sans-serif';
    octx.textAlign = 'left'; octx.textBaseline = 'top';
    octx.fillText(S.field, 18, 14);
    octx.font = '11px ui-monospace, Menlo, Consolas, monospace';
    octx.fillStyle = dim;
    octx.fillText(`t = ${fmt(S.current.time, 6)}` + (S.meta.nsteps > 1 ? `   step ${S.step}` : '') + (S.mode === 1 ? '   [nodal / linear]' : ''), 18, 32);
  }
  // pinned probe
  if (S.probePinned) {
    const [px, py] = dataToScreen(S.probePinned.x, S.probePinned.y);
    octx.strokeStyle = '#ff7eb6'; octx.lineWidth = 1.5;
    octx.beginPath(); octx.arc(px, py, 6, 0, 2 * Math.PI); octx.stroke();
    octx.beginPath(); octx.moveTo(px - 10, py); octx.lineTo(px + 10, py); octx.moveTo(px, py - 10); octx.lineTo(px, py + 10); octx.stroke();
  }
  // line probes: solid segment with end-point handles; when the chart is zoomed
  // along the distance axis, a translucent halo marks the part it shows
  const T = S.lines.length ? themeColors() : null;
  for (const L of S.lines) {
    const lc = lineColor(L);
    const isActive = L.id === LP.active;
    const [ax, ay] = dataToScreen(L.x0, L.y0), [bx, by] = dataToScreen(L.x1, L.y1);
    octx.globalAlpha = L.visible ? 1 : 0.35;
    const Lo = LP.layout;
    const R = L.visible && Lo && LP.xview ? Lo.vis.find(v => v.L === L) : null;
    if (R && R.total > 0) {
      const sa = Math.max(Lo.dmin, R.D0), sb = Math.min(Lo.dmax, R.D1);
      if (sb > sa) {
        const ta = sa / R.total, tb = sb / R.total;
        octx.save(); octx.globalAlpha *= 0.35; octx.strokeStyle = lc; octx.lineWidth = 9; octx.lineCap = 'butt';
        octx.beginPath(); octx.moveTo(ax + (bx - ax) * ta, ay + (by - ay) * ta); octx.lineTo(ax + (bx - ax) * tb, ay + (by - ay) * tb); octx.stroke();
        octx.restore();
      }
    }
    octx.strokeStyle = lc; octx.lineWidth = isActive ? 1.8 : 1.3;
    octx.beginPath(); octx.moveTo(ax, ay); octx.lineTo(bx, by); octx.stroke();
    // end-point handles (draggable)
    for (const [hx, hy, part] of [[ax, ay, 0], [bx, by, 1]]) {
      const hot = LP.hoverHit && LP.hoverHit.line === L && LP.hoverHit.part === part;
      octx.fillStyle = lc; octx.strokeStyle = T.handleStroke; octx.lineWidth = 1.2;
      octx.beginPath(); octx.arc(hx, hy, hot ? 6 : 4, 0, 2 * Math.PI); octx.fill(); octx.stroke();
    }
    if (isActive) {   // label near the start point
      octx.fillStyle = lc; octx.font = '600 11px -apple-system, Segoe UI, Inter, Roboto, sans-serif'; octx.textAlign = 'left'; octx.textBaseline = 'bottom';
      octx.fillText(`L${L.id}`, ax + 7, ay - 6);
      octx.font = '11px ui-monospace, Menlo, Consolas, monospace';
    }
    if (L.hoverIdx != null && L.data && L.visible) {
      const [px, py] = dataToScreen(L.data.x[L.hoverIdx], L.data.y[L.hoverIdx]);
      octx.strokeStyle = lc; octx.lineWidth = 1.5; octx.beginPath(); octx.arc(px, py, 6, 0, 2 * Math.PI); octx.stroke();
    }
    octx.globalAlpha = 1;
  }
}

// ----------------------------------------------------------------- probe
function probeAt(px, py) {
  if (!S.mesh || !S.current) return null;
  const e = R.pickElement(px, py, S.pxPerCell);
  if (e < 0) return null;
  const [x, y] = screenToData(px, py);
  const bb = R.mesh.bbox;
  const size = Math.max(bb[4 * e + 1] - bb[4 * e], bb[4 * e + 3] - bb[4 * e + 2]);
  const inv = S.spectral.invert(S.mesh.x, S.mesh.y, e, x, y, size);
  const r = clamp(inv.r, -1, 1), s = clamp(inv.s, -1, 1);
  const v = S.spectral.evaluate(S.current.values, e, r, s);
  return { x, y, e, gid: S.mesh.elmap[e], r, s, v, ok: inv.ok };
}

function showProbe(p) {
  const el = $('probe');
  if (!p) { el.textContent = '—'; el.classList.add('muted'); return; }
  el.classList.remove('muted');
  el.textContent = `x = ${fmt(p.x, 6)}   y = ${fmt(p.y, 6)}\nelement ${p.gid}  (local ${p.e})\n(r, s) = (${fmt(p.r, 4)}, ${fmt(p.s, 4)})\n${S.field} = ${fmt(p.v, 7)}${p.ok ? '' : '   (inversion not converged)'}`;
}

function updateProbe() {
  if (S.probePinned) { const [px, py] = dataToScreen(S.probePinned.x, S.probePinned.y); showProbe(probeAt(px, py)); }
  else if (S.hover) showProbe(probeAt(S.hover[0], S.hover[1]));
}

// ----------------------------------------------------------------- undo / redo
// Snapshot history of the editable viewer state: line probes, the pinned
// probe and the view.  Settings (field, colormap, ...) are not part of it.
const HIST = { undo: [], redo: [], max: 200, lastViewPush: 0 };
const cloneLine = L => ({ ...L, hoverIdx: null });   // sampled data is immutable and shared
function snapshot() {
  return { lines: S.lines.map(cloneLine), active: LP.active, pinned: S.probePinned ? { ...S.probePinned } : null, view: { ...S.view } };
}
function pushHistory(label, snap = snapshot()) {
  HIST.undo.push({ label, snap });
  if (HIST.undo.length > HIST.max) HIST.undo.shift();
  HIST.redo = [];
  HIST.lastViewPush = 0;
}
/** View changes made in quick succession (wheel ticks) collapse into one step. */
function pushViewHistory(label, coalesceMs = 0) {
  const now = performance.now();
  const last = HIST.undo[HIST.undo.length - 1];
  if (coalesceMs && last && last.label === label && now - HIST.lastViewPush < coalesceMs) { HIST.lastViewPush = now; return; }
  pushHistory(label);
  HIST.lastViewPush = now;
}
function restore(snap) {
  S.lines = snap.lines.map(cloneLine);
  LP.active = snap.active;
  LP.nextId = Math.max(LP.nextId, ...S.lines.map(L => L.id + 1));
  S.probePinned = snap.pinned ? { ...snap.pinned } : null;
  S.view = { ...snap.view };
  if (S.drag && S.drag.kind !== 'pan') { S.drag = null; glCanvas.style.cursor = ''; }
  renderLegend();
  if (S.lines.length) showLinePanel(); else { $('line-panel').hidden = true; LP.layout = null; }
  drawLineChart();
  refreshLines();
  updateProbe();
  requestRender();
}
function undo() {
  if (!HIST.undo.length) { toast('Nothing to undo'); return; }
  const h = HIST.undo.pop();
  HIST.redo.push({ label: h.label, snap: snapshot() });
  restore(h.snap);
  toast(`Undo: ${h.label}`);
}
function redo() {
  if (!HIST.redo.length) { toast('Nothing to redo'); return; }
  const h = HIST.redo.pop();
  HIST.undo.push({ label: h.label, snap: snapshot() });
  HIST.lastViewPush = 0;
  restore(h.snap);
  toast(`Redo: ${h.label}`);
}
function clearHistory() { HIST.undo = []; HIST.redo = []; HIST.lastViewPush = 0; }

// ----------------------------------------------------------------- line probes
// Several segments can exist at once; each is sampled exactly on the server
// (spectral evaluation along the segment) and drawn in the shared chart.
const LINE_COLORS_DARK = ['#ffd166', '#5cc8ff', '#ff7eb6', '#4ade80', '#c084fc', '#fb923c', '#f87171', '#22d3ee'];
const LINE_COLORS_LIGHT = ['#c2410c', '#1d4ed8', '#be185d', '#15803d', '#7e22ce', '#b45309', '#b91c1c', '#0e7490'];
const LP = { layout: null, drag: null, panelDrag: null, xview: null, yview: null, hoverS: null, active: null, nextId: 1, hoverHit: null, showGrid: true, showElem: true };
const SNAP_DEG = 10;

const lineColor = L => (isLight() ? LINE_COLORS_LIGHT : LINE_COLORS_DARK)[L.colorIdx % LINE_COLORS_DARK.length];
const activeLine = () => S.lines.find(L => L.id === LP.active) || null;

function themeColors() {
  const cs = getComputedStyle(document.body);
  const light = isLight();
  return {
    text: cs.getPropertyValue('--text').trim() || (light ? '#1b2130' : '#e6e9f0'),
    muted: cs.getPropertyValue('--muted').trim() || '#8b93a7',
    grid: light ? 'rgba(27,33,48,.10)' : 'rgba(230,233,240,.10)',
    axis: light ? 'rgba(27,33,48,.5)' : 'rgba(139,147,167,.6)',
    elem: light ? 'rgba(11,61,145,.22)' : 'rgba(92,200,255,.28)',
    handleStroke: light ? '#ffffff' : '#0d0f14',
  };
}

function newLine(x, y) {
  const used = new Set(S.lines.map(L => L.colorIdx));
  let colorIdx = 0;
  while (used.has(colorIdx) && colorIdx < LINE_COLORS_DARK.length) colorIdx++;
  const L = { id: LP.nextId++, colorIdx, x0: x, y0: y, x1: x, y1: y, visible: true, data: null, field: null, step: null, hoverIdx: null, token: 0 };
  S.lines.push(L);
  LP.active = L.id;
  return L;
}

function removeLine(L, record = true) {
  if (record) pushHistory(`delete L${L.id}`);
  S.lines = S.lines.filter(o => o !== L);
  if (LP.active === L.id) LP.active = S.lines.length ? S.lines[S.lines.length - 1].id : null;
  if (!S.lines.length) { $('line-panel').hidden = true; LP.xview = null; LP.yview = null; }
  renderLegend(); drawLineChart(); requestRender();
}

function clearLines(record = true) {
  if (record && (S.lines.length || S.probePinned)) pushHistory('clear lines');
  S.lines = []; LP.active = null; LP.xview = null; LP.yview = null; LP.layout = null;
  $('line-panel').hidden = true;
  requestRender();
}

const lineLength = L => Math.hypot(L.x1 - L.x0, L.y1 - L.y0);
const lineAngle = L => Math.atan2(L.y1 - L.y0, L.x1 - L.x0) * 180 / Math.PI;

/** Snap the end point (x, y) so that the segment from (ox, oy) has an angle that is a multiple of SNAP_DEG. */
function snapEnd(ox, oy, x, y) {
  const dx = x - ox, dy = y - oy;
  if (dx === 0 && dy === 0) return [x, y];
  const step = SNAP_DEG * Math.PI / 180;
  const ang = Math.round(Math.atan2(dy, dx) / step) * step;
  const len = Math.abs(dx * Math.cos(ang) + dy * Math.sin(ang));   // projection onto the snapped direction
  return [ox + len * Math.cos(ang), oy + len * Math.sin(ang)];
}
const snapActive = ev => S.snapAngle || ev.ctrlKey;

/** What is under the cursor: an end point (part 0/1) or the body of a line. */
function hitLine(px, py) {
  let best = null, bestD = 9;
  const ordered = [...S.lines].sort((a, b) => (a.id === LP.active) - (b.id === LP.active));  // active line last = wins ties
  for (const L of ordered) {
    const [ax, ay] = dataToScreen(L.x0, L.y0), [bx, by] = dataToScreen(L.x1, L.y1);
    const d0 = Math.hypot(px - ax, py - ay), d1 = Math.hypot(px - bx, py - by);
    if (d0 <= bestD) { best = { line: L, part: 0 }; bestD = d0; }
    if (d1 <= bestD) { best = { line: L, part: 1 }; bestD = d1; }
  }
  if (best) return best;
  for (const L of ordered) {
    const [ax, ay] = dataToScreen(L.x0, L.y0), [bx, by] = dataToScreen(L.x1, L.y1);
    const len2 = (bx - ax) ** 2 + (by - ay) ** 2;
    if (len2 === 0) continue;
    const t = clamp(((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / len2, 0, 1);
    const d = Math.hypot(px - (ax + t * (bx - ax)), py - (ay + t * (by - ay)));
    if (d <= 6) best = { line: L, part: 'body' };
  }
  return best;
}

function showDragInfo(L) {
  $('probe').classList.remove('muted');
  $('probe').textContent = `line L${L.id}: (${fmt(L.x0, 5)}, ${fmt(L.y0, 5)}) → (${fmt(L.x1, 5)}, ${fmt(L.y1, 5)})\nlength ${fmt(lineLength(L), 5)}   angle ${lineAngle(L).toFixed(1)}°`;
}

async function runLine(L) {
  const field = S.field, step = S.step, token = ++L.token;
  try {
    const d = await api('line', { x0: L.x0, y0: L.y0, x1: L.x1, y1: L.y1, step, fields: field, n: 1000 });
    if (token !== L.token || !S.lines.includes(L)) return;
    L.data = d; L.field = field; L.step = step;
    showLinePanel();
    renderLegend();
    drawLineChart();
    requestRender();
  } catch (err) { toast(err.message, true); }
}
function refreshLines() { for (const L of S.lines) if (!L.data || L.field !== S.field || L.step !== S.step) runLine(L); }

function showLinePanel() {
  const p = $('line-panel');
  if (p.hidden) {
    p.hidden = false;
    if (!p.style.left) {   // first time: bottom-right corner of the view
      const v = $('view').getBoundingClientRect();
      p.style.left = Math.max(8, v.width - p.offsetWidth - 14) + 'px';
      p.style.top = Math.max(8, v.height - p.offsetHeight - 14) + 'px';
    }
    clampPanel();
  }
  const n = S.lines.filter(L => L.visible).length;
  $('lp-title').textContent = `${S.field} along ${S.lines.length} line${S.lines.length === 1 ? '' : 's'}` + (n !== S.lines.length ? ` (${n} shown)` : '');
}

function clampPanel() {
  const p = $('line-panel'), v = $('view').getBoundingClientRect();
  const left = clamp(parseFloat(p.style.left) || 0, 0, Math.max(0, v.width - p.offsetWidth));
  const top = clamp(parseFloat(p.style.top) || 0, 0, Math.max(0, v.height - p.offsetHeight));
  p.style.left = left + 'px'; p.style.top = top + 'px';
}

function renderLegend() {
  const box = $('lp-legend');
  box.innerHTML = '';
  for (const L of S.lines) {
    const item = document.createElement('span');
    item.className = 'lp-item' + (L.id === LP.active ? ' active' : '') + (L.visible ? '' : ' hidden-line');
    item.title = `(${fmt(L.x0)}, ${fmt(L.y0)}) → (${fmt(L.x1)}, ${fmt(L.y1)})  length ${fmt(lineLength(L))}, ${lineAngle(L).toFixed(1)}°\nclick: select · checkbox: show/hide · ✕: delete`;
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = L.visible;
    cb.onclick = ev => { ev.stopPropagation(); pushHistory(`${cb.checked ? 'show' : 'hide'} L${L.id}`); L.visible = cb.checked; renderLegend(); showLinePanel(); drawLineChart(); requestRender(); };
    const sw = document.createElement('i'); sw.className = 'swatch'; sw.style.background = lineColor(L);
    const nm = document.createElement('span'); nm.textContent = `L${L.id}`;
    const x = document.createElement('button'); x.className = 'x'; x.textContent = '✕'; x.title = 'delete this line';
    x.onclick = ev => { ev.stopPropagation(); removeLine(L); };
    item.append(cb, sw, nm, x);
    item.onclick = () => { LP.active = L.id; renderLegend(); drawLineChart(); requestRender(); };
    box.appendChild(item);
  }
}

/** Inside-the-domain range of one line for its current data. */
function lineRange(L) {
  const d = L.data; if (!d) return null;
  const vals = d.values[L.field]; if (!vals) return null;
  let first = -1, last = -1;
  for (let i = 0; i < vals.length; i++) if (vals[i] !== null) { if (first < 0) first = i; last = i; }
  if (first < 0) return null;
  return { vals, first, last, D0: d.distance[first], D1: d.distance[last], total: d.distance[d.distance.length - 1] };
}

/** Common chart ranges for all visible lines (distance axis shared, absolute along each line). */
function chartLayout() {
  const vis = [];
  for (const L of S.lines) { if (!L.visible) continue; const R = lineRange(L); if (R) vis.push({ L, ...R }); }
  if (!vis.length) return null;
  const D0 = Math.min(...vis.map(v => v.D0)), D1 = Math.max(...vis.map(v => v.D1));
  const span = Math.max(D1 - D0, 1e-300);
  let dmin = D0, dmax = D1;
  if (LP.xview) { dmin = clamp(LP.xview.dmin, D0, D1); dmax = clamp(LP.xview.dmax, D0, D1); if (dmax - dmin < span * 1e-9) { dmin = D0; dmax = D1; } }
  let lo = Infinity, hi = -Infinity;
  for (const v of vis) {
    const dist = v.L.data.distance;
    for (let i = v.first; i <= v.last; i++) {
      const y = v.vals[i]; if (y === null) continue;
      if (dist[i] < dmin || dist[i] > dmax) continue;
      if (y < lo) lo = y; if (y > hi) hi = y;
    }
  }
  if (!(lo <= hi)) { lo = -1; hi = 1; }
  else if (lo === hi) { lo -= 0.5; hi += 0.5; }
  else { const pad = 0.06 * (hi - lo); lo -= pad; hi += pad; }
  if (LP.yview) { lo = LP.yview.lo; hi = LP.yview.hi; }
  return { vis, D0, D1, span, dmin, dmax, lo, hi };
}

function nearestIndex(dist, s, first, last) {
  let a = first, b = last;
  while (b - a > 1) { const m = (a + b) >> 1; if (dist[m] < s) a = m; else b = m; }
  return (s - dist[a] < dist[b] - s) ? a : b;
}

function drawLineChart() {
  const c = $('lp-canvas'), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w || !h || $('line-panel').hidden) return;
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const T = themeColors();
  const Lo = chartLayout();
  ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
  if (!Lo) {
    LP.layout = null;
    ctx.fillStyle = T.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(S.lines.length ? 'no visible samples inside the domain' : 'no lines', w / 2, h / 2);
    $('lp-info').textContent = '';
    return;
  }
  const sy = niceStep(Lo.hi - Lo.lo, Math.max(2, (h - 40) / 38));
  const yTicks = [];
  for (let v = Math.ceil(Lo.lo / sy) * sy; v <= Lo.hi + 1e-9 * sy; v += sy) yTicks.push(v);
  const labelW = Math.max(30, ...yTicks.map(v => ctx.measureText(tickFmt(v, sy)).width));
  const pad = { l: 10 + labelW, r: 14, t: 10, b: 28 };
  const pw = Math.max(10, w - pad.l - pad.r), ph = Math.max(10, h - pad.t - pad.b);
  const X = sd => pad.l + pw * (sd - Lo.dmin) / (Lo.dmax - Lo.dmin);
  const Y = v => pad.t + ph * (1 - (v - Lo.lo) / (Lo.hi - Lo.lo));
  LP.layout = { ...Lo, pad, pw, ph, w, h };
  // grid and ticks
  const sx = niceStep(Lo.dmax - Lo.dmin, Math.max(2, pw / 90));
  ctx.lineWidth = 1;
  ctx.strokeStyle = T.grid; ctx.fillStyle = T.muted;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let sd = Math.ceil(Lo.dmin / sx) * sx; sd <= Lo.dmax + 1e-9 * sx; sd += sx) {
    const x = X(sd);
    ctx.beginPath(); ctx.moveTo(x, pad.t + ph); ctx.lineTo(x, pad.t + ph + 3); ctx.stroke();   // tick mark
    if (LP.showGrid) { ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke(); }
    ctx.fillText(tickFmt(sd, sx), x, pad.t + ph + 5);
  }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yTicks) {
    const y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l - 3, y); ctx.lineTo(pad.l, y); ctx.stroke();   // tick mark
    if (LP.showGrid) { ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke(); }
    ctx.fillText(tickFmt(v, sy), pad.l - 5, y);
  }
  ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
  ctx.fillText('distance along the line', pad.l + pw - 16, h - 1);   // keep clear of the resize grip
  // data, clipped to the plot area
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, pw, ph); ctx.clip();
  const act = activeLine();
  const actV = Lo.vis.find(v => v.L === act) || Lo.vis[Lo.vis.length - 1];
  if (actV && LP.showElem) {   // element boundaries of the selected line
    ctx.strokeStyle = T.elem;
    const d = actV.L.data;
    for (let i = actV.first + 1; i <= actV.last; i++) {
      if (d.elem[i] === d.elem[i - 1]) continue;
      const sd = 0.5 * (d.distance[i] + d.distance[i - 1]);
      if (sd < Lo.dmin || sd > Lo.dmax) continue;
      const x = X(sd); ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
    }
  }
  for (const v of Lo.vis) {
    const dist = v.L.data.distance;
    ctx.strokeStyle = lineColor(v.L); ctx.lineWidth = v.L === (actV && actV.L) ? 1.9 : 1.3; ctx.lineJoin = 'round';
    ctx.globalAlpha = Lo.vis.length > 1 && v.L !== (actV && actV.L) ? 0.85 : 1;
    ctx.beginPath();
    let pen = false;
    for (let i = v.first; i <= v.last; i++) {
      const y = v.vals[i];
      if (y === null) { pen = false; continue; }
      const x = X(dist[i]), yy = Y(y);
      if (!pen) { ctx.moveTo(x, yy); pen = true; } else ctx.lineTo(x, yy);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // hover cursor + markers
  const infos = [];
  if (LP.hoverS != null && LP.hoverS >= Lo.dmin && LP.hoverS <= Lo.dmax) {
    const x = X(LP.hoverS);
    ctx.strokeStyle = T.axis; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
    for (const v of Lo.vis) {
      const dist = v.L.data.distance;
      if (LP.hoverS < v.D0 - 1e-12 || LP.hoverS > v.D1 + 1e-12) { v.L.hoverIdx = null; continue; }
      const i = nearestIndex(dist, LP.hoverS, v.first, v.last);
      v.L.hoverIdx = i;
      const y = v.vals[i];
      if (y !== null) { ctx.fillStyle = lineColor(v.L); ctx.beginPath(); ctx.arc(X(dist[i]), Y(y), 3.2, 0, 2 * Math.PI); ctx.fill(); }
      infos.push(`L${v.L.id} = ${y === null ? 'outside' : fmt(y, 6)}`);
    }
  } else {
    for (const L of S.lines) L.hoverIdx = null;
  }
  ctx.restore();
  ctx.strokeStyle = T.axis; ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, pw, ph);
  let info = '';
  if (infos.length) info = `s = ${fmt(LP.hoverS, 5)}   ${infos.join(' · ')}`;
  else if (LP.xview || LP.yview) info = 'zoomed · double-click to reset';
  $('lp-info').textContent = info;
}

const lpc = $('lp-canvas');
function lpPos(ev) { const r = lpc.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; }
function lpDistance(mx) { const L = LP.layout; return L.dmin + (mx - L.pad.l) / L.pw * (L.dmax - L.dmin); }
lpc.addEventListener('mousemove', ev => {
  const L = LP.layout; if (!L || LP.drag) return;
  const [mx] = lpPos(ev);
  LP.hoverS = clamp(lpDistance(mx), L.dmin, L.dmax);
  drawLineChart(); requestRender();
});
lpc.addEventListener('mouseleave', () => { if (!LP.drag) { LP.hoverS = null; drawLineChart(); requestRender(); } });
lpc.addEventListener('wheel', ev => {
  ev.preventDefault();
  const L = LP.layout; if (!L) return;
  const [mx, my] = lpPos(ev);
  const f = Math.exp(ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015));
  if (ev.shiftKey) {
    const v = L.lo + (1 - (my - L.pad.t) / L.ph) * (L.hi - L.lo);
    LP.yview = { lo: v - (v - L.lo) * f, hi: v + (L.hi - v) * f };
  } else {
    const sd = lpDistance(mx);
    const a = Math.max(L.D0, sd - (sd - L.dmin) * f), b = Math.min(L.D1, sd + (L.dmax - sd) * f);
    if (b - a < L.span * 1e-6) return;
    LP.xview = (a <= L.D0 && b >= L.D1) ? null : { dmin: a, dmax: b };
  }
  drawLineChart(); requestRender();
}, { passive: false });
lpc.addEventListener('mousedown', ev => {
  const L = LP.layout; if (!L) return;
  const [mx, my] = lpPos(ev);
  LP.drag = { mx, my, dmin: L.dmin, dmax: L.dmax, yview: LP.yview ? { ...LP.yview } : null, L };
  ev.preventDefault();
});
function lpDrag(ev) {
  const D = LP.drag, L = D.L;
  const [mx, my] = lpPos(ev);
  const dx = mx - D.mx, dy = my - D.my;
  const width = D.dmax - D.dmin;
  let a = D.dmin - dx / L.pw * width;
  a = clamp(a, L.D0, L.D1 - width);
  LP.xview = (a <= L.D0 && a + width >= L.D1) ? null : { dmin: a, dmax: a + width };
  if (D.yview) {
    const dv = dy / L.ph * (D.yview.hi - D.yview.lo);
    LP.yview = { lo: D.yview.lo + dv, hi: D.yview.hi + dv };
  }
  drawLineChart(); requestRender();
}
$('lp-grid').onchange = ev => { LP.showGrid = ev.target.checked; drawLineChart(); };
$('lp-elem').onchange = ev => { LP.showElem = ev.target.checked; drawLineChart(); };
const resetChartZoom = () => { LP.xview = null; LP.yview = null; drawLineChart(); requestRender(); };
lpc.addEventListener('dblclick', resetChartZoom);
$('lp-reset').onclick = resetChartZoom;
$('lp-close').onclick = clearLines;
// move the panel by its header; resize with the grip in the corner (CSS resize)
$('lp-head').addEventListener('mousedown', ev => {
  if (ev.target.closest('button')) return;
  const p = $('line-panel');
  LP.panelDrag = { dx: ev.clientX - p.offsetLeft, dy: ev.clientY - p.offsetTop };
  ev.preventDefault();
});
new ResizeObserver(() => { if (!$('line-panel').hidden) { clampPanel(); drawLineChart(); } }).observe($('line-panel'));

// ----------------------------------------------------------------- mouse interaction
overlay.style.pointerEvents = 'none';
glCanvas.addEventListener('wheel', ev => {
  ev.preventDefault();
  const rect = glCanvas.getBoundingClientRect();
  const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  const [x, y] = screenToData(px, py);
  const f = Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015));
  const scale = clamp(S.view.scale * f, 1e-12, 1e18);
  pushViewHistory('zoom', 800);
  // keep the point under the cursor fixed
  S.view.cx = x - (px - W() / 2) / scale;
  S.view.cy = y + (py - H() / 2) / scale;
  S.view.scale = scale;
  requestRender();
  updateProbe();
}, { passive: false });

glCanvas.addEventListener('mousedown', ev => {
  if (ev.button !== 0) return;
  const rect = glCanvas.getBoundingClientRect();
  const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  if (S.drag && S.drag.kind === 'line' && S.drag.fromMenu) {   // finish a line started from the context menu
    const L = S.drag.line, before = S.drag.before; S.drag = null;
    glCanvas.style.cursor = '';
    LP.xview = null; LP.yview = null;
    if (lineLength(L) > 0) { pushHistory(`add line L${L.id}`, before); runLine(L); } else removeLine(L, false);
    ev.preventDefault();
    return;
  }
  const hit = ev.shiftKey ? null : hitLine(px, py);
  const before = snapshot();
  if (hit) {   // grab an end point or the body of an existing line
    const L = hit.line;
    S.drag = { kind: 'line-edit', line: L, part: hit.part, px, py, orig: { x0: L.x0, y0: L.y0, x1: L.x1, y1: L.y1 }, moved: false, before };
    LP.active = L.id; renderLegend(); drawLineChart(); requestRender();
  } else if (ev.shiftKey || S.lineMode) {
    const [x, y] = screenToData(px, py);
    S.drag = { kind: 'line', line: newLine(x, y), before };
    renderLegend();
  } else {
    S.drag = { kind: 'pan', px, py, cx: S.view.cx, cy: S.view.cy, moved: false, before };
  }
});
window.addEventListener('mousemove', ev => {
  if (LP.drag) { lpDrag(ev); return; }
  if (LP.panelDrag) {
    const p = $('line-panel');
    p.style.left = (ev.clientX - LP.panelDrag.dx) + 'px'; p.style.top = (ev.clientY - LP.panelDrag.dy) + 'px';
    clampPanel(); return;
  }
  const rect = glCanvas.getBoundingClientRect();
  const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  if (S.drag && S.drag.kind === 'pan') {
    const dx = px - S.drag.px, dy = py - S.drag.py;
    if (Math.abs(dx) + Math.abs(dy) > 2) S.drag.moved = true;
    S.view.cx = S.drag.cx - dx / S.view.scale;
    S.view.cy = S.drag.cy + dy / S.view.scale;
    requestRender();
  } else if (S.drag && S.drag.kind === 'line') {
    const L = S.drag.line;
    let [x, y] = screenToData(px, py);
    if (snapActive(ev)) [x, y] = snapEnd(L.x0, L.y0, x, y);
    L.x1 = x; L.y1 = y;
    showDragInfo(L);
    requestRender();
  } else if (S.drag && S.drag.kind === 'line-edit') {
    const D = S.drag, L = D.line;
    const [x, y] = screenToData(px, py);
    if (Math.abs(px - D.px) + Math.abs(py - D.py) > 2) D.moved = true;
    if (D.part === 'body') {
      const [ox, oy] = screenToData(D.px, D.py);
      L.x0 = D.orig.x0 + x - ox; L.y0 = D.orig.y0 + y - oy; L.x1 = D.orig.x1 + x - ox; L.y1 = D.orig.y1 + y - oy;
    } else {
      const other = D.part === 0 ? [L.x1, L.y1] : [L.x0, L.y0];
      let p = [x, y];
      if (snapActive(ev)) p = snapEnd(other[0], other[1], x, y);
      if (D.part === 0) { L.x0 = p[0]; L.y0 = p[1]; } else { L.x1 = p[0]; L.y1 = p[1]; }
    }
    showDragInfo(L);
    requestRender();
  } else if (ev.target === glCanvas) {
    S.hover = [px, py];
    LP.hoverHit = (ev.shiftKey) ? null : hitLine(px, py);
    glCanvas.style.cursor = LP.hoverHit ? (LP.hoverHit.part === 'body' ? 'move' : 'grab') : (S.lineMode ? 'crosshair' : '');
    if (!S.probePinned) showProbe(probeAt(px, py));
    requestRender();
  }
});
window.addEventListener('mouseup', ev => {
  LP.drag = null; LP.panelDrag = null;
  if (!S.drag || (S.drag.kind === 'line' && S.drag.fromMenu)) return;
  const d = S.drag; S.drag = null;
  if (d.kind === 'line') {
    LP.xview = null; LP.yview = null;
    if (lineLength(d.line) > 0) { pushHistory(`add line L${d.line.id}`, d.before); runLine(d.line); } else removeLine(d.line, false);
    setLineMode(false);
  } else if (d.kind === 'line-edit') {
    if (d.moved) { pushHistory(`move L${d.line.id}`, d.before); LP.xview = null; LP.yview = null; runLine(d.line); } else { renderLegend(); drawLineChart(); }
    requestRender();
  } else if (d.kind === 'pan' && d.moved) {
    pushHistory('pan', d.before);
  } else if (d.kind === 'pan' && !d.moved && ev.target === glCanvas) {
    const rect = glCanvas.getBoundingClientRect();
    const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    const p = probeAt(px, py);
    const was = S.probePinned;
    if (S.probePinned && p && Math.hypot(...dataToScreen(S.probePinned.x, S.probePinned.y).map((v, i) => v - [px, py][i])) < 8) { S.probePinned = null; }
    else if (p) S.probePinned = { x: p.x, y: p.y };
    else S.probePinned = null;
    if ((was && !S.probePinned) || (!was && S.probePinned) || (was && S.probePinned && (was.x !== S.probePinned.x || was.y !== S.probePinned.y))) pushHistory(S.probePinned ? 'pin probe' : 'unpin probe', d.before);
    updateProbe(); requestRender();
  }
});
glCanvas.addEventListener('mouseleave', () => { S.hover = null; LP.hoverHit = null; if (!S.probePinned) showProbe(null); });
function resetViewUser() { pushHistory('reset view'); fitView(); }
glCanvas.addEventListener('dblclick', ev => { if (!hitLine(ev.clientX - glCanvas.getBoundingClientRect().left, ev.clientY - glCanvas.getBoundingClientRect().top)) resetViewUser(); });

// ----------------------------------------------------------------- context menu
async function copyText(text, what = 'Copied') {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    }
    toast(`${what} to clipboard`);
  } catch (err) { toast('Clipboard unavailable: ' + err.message, true); }
}

function download(name, text, mime = 'text/plain') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function lineCSV(lines) {
  const rows = ['line,distance,x,y,element,' + (lines[0] ? lines[0].field : S.field)];
  for (const L of lines) {
    if (!L.data) continue;
    const d = L.data, vals = d.values[L.field] || [];
    for (let i = 0; i < d.distance.length; i++) {
      if (vals[i] === null) continue;
      const gid = d.elem[i] >= 0 && S.mesh ? S.mesh.elmap[d.elem[i]] : '';
      rows.push(`L${L.id},${d.distance[i]},${d.x[i]},${d.y[i]},${gid},${vals[i]}`);
    }
  }
  return rows.join('\n') + '\n';
}

/** A Python snippet that reproduces the current view with the scripting API. */
function pythonSnippet() {
  const [x0, y0] = screenToData(0, H()), [x1, y1] = screenToData(W(), 0);
  let cmap = S.cmap;
  if (S.invert) cmap = cmap.endsWith('_r') ? cmap.slice(0, -2) : cmap + '_r';
  const field = S.field.startsWith('decay:') ? S.field.slice(6) : S.field;
  const lines = [
    'import semview',
    '',
    `data = semview.load(${JSON.stringify(S.meta.path)}, step=${S.step})`,
    'pl = semview.Plotter(figsize=(9, 6))',
    `pl.add_field(data, ${JSON.stringify(field)}, cmap=${JSON.stringify(cmap)}, clim=(${fmt(S.range.lo, 6)}, ${fmt(S.range.hi, 6)})${S.mode === 1 ? ', method="nodal"' : ''})`,
  ];
  if (S.field.startsWith('decay:')) lines.push(`# GUI showed log10 of data.spectral_decay(${JSON.stringify(field)}) per element`);
  if (S.contours) lines.push(`pl.add_contours(data, ${JSON.stringify(field)}, levels=${S.nContours}, colors="w", linewidths=0.5)`);
  if (S.edges) lines.push('pl.add_mesh(data, color="k", linewidth=0.3)');
  if (S.nodes) lines.push('pl.add_nodes(data)');
  for (const L of S.lines) lines.push(`dist_L${L.id}, vals_L${L.id} = data.sample_line(${JSON.stringify(field)}, (${fmt(L.x0, 7)}, ${fmt(L.y0, 7)}), (${fmt(L.x1, 7)}, ${fmt(L.y1, 7)}), 1000)`);
  if (S.probePinned) lines.push(`probe = data.sample(${JSON.stringify(field)}, ${fmt(S.probePinned.x, 7)}, ${fmt(S.probePinned.y, 7)})`);
  lines.push(`pl.set_view((${fmt(x0, 7)}, ${fmt(x1, 7)}), (${fmt(y0, 7)}, ${fmt(y1, 7)}))`);
  lines.push(`pl.set_title(${JSON.stringify(`${S.meta.name}: ${S.field}`)})`);
  lines.push('pl.save("semview_view.png")');
  return lines.join('\n') + '\n';
}

function zoomToElement(e) {
  pushHistory('zoom to element');
  const bb = R.mesh.bbox;
  const a = bb[4 * e], b = bb[4 * e + 1], c = bb[4 * e + 2], d = bb[4 * e + 3];
  const m = 0.35;
  const sx = W() / ((b - a) * (1 + 2 * m) || 1e-300), sy = H() / ((d - c) * (1 + 2 * m) || 1e-300);
  S.view = { cx: 0.5 * (a + b), cy: 0.5 * (c + d), scale: Math.min(sx, sy) };
  requestRender(); updateProbe();
}

const ctxEl = $('ctx');
function closeCtx() { ctxEl.hidden = true; ctxEl.innerHTML = ''; }
function openCtx(items, clientX, clientY) {
  ctxEl.innerHTML = '';
  for (const it of items) {
    if (it === '-') { const d = document.createElement('div'); d.className = 'sep'; ctxEl.appendChild(d); continue; }
    if (it.header) { const d = document.createElement('div'); d.className = 'hd'; d.textContent = it.header; ctxEl.appendChild(d); continue; }
    const d = document.createElement('div');
    d.className = 'it' + (it.disabled ? ' disabled' : '');
    const label = document.createElement('span'); label.textContent = it.label; d.appendChild(label);
    if (it.key) { const k = document.createElement('span'); k.className = 'key'; k.textContent = it.key; d.appendChild(k); }
    d.onclick = () => { closeCtx(); it.action(); };
    ctxEl.appendChild(d);
  }
  ctxEl.hidden = false;
  const v = $('view').getBoundingClientRect();
  let left = clientX - v.left, top = clientY - v.top;
  left = Math.min(left, v.width - ctxEl.offsetWidth - 6);
  top = Math.min(top, v.height - ctxEl.offsetHeight - 6);
  ctxEl.style.left = Math.max(4, left) + 'px';
  ctxEl.style.top = Math.max(4, top) + 'px';
}
window.addEventListener('mousedown', ev => { if (!ctxEl.hidden && !ctxEl.contains(ev.target)) closeCtx(); }, true);
window.addEventListener('wheel', () => { if (!ctxEl.hidden) closeCtx(); }, { passive: true, capture: true });
window.addEventListener('blur', closeCtx);

glCanvas.addEventListener('contextmenu', ev => {
  ev.preventDefault();
  if (!S.meta || !S.meta.open) return;
  const rect = glCanvas.getBoundingClientRect();
  const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  const [x, y] = screenToData(px, py);
  const p = probeAt(px, py);
  const hit = hitLine(px, py);
  const items = [];
  items.push({ header: p ? `x = ${fmt(x, 6)}  y = ${fmt(y, 6)}\nelement ${p.gid}  (r, s) = (${fmt(p.r, 4)}, ${fmt(p.s, 4)})\n${S.field} = ${fmt(p.v, 7)}` : `x = ${fmt(x, 6)}  y = ${fmt(y, 6)}\noutside the domain` });
  items.push({ label: 'Copy coordinates', action: () => copyText(`${x}\t${y}`, 'Coordinates copied') });
  items.push({ label: 'Copy probe readout', disabled: !p, action: () => copyText(`x\ty\telement\tr\ts\t${S.field}\n${x}\t${y}\t${p.gid}\t${p.r}\t${p.s}\t${p.v}`, 'Probe copied') });
  const pinnedHere = S.probePinned && Math.hypot(...dataToScreen(S.probePinned.x, S.probePinned.y).map((v, i) => v - [px, py][i])) < 8;
  items.push({ label: pinnedHere ? 'Unpin probe' : 'Pin probe here', disabled: !p && !pinnedHere, action: () => { pushHistory(pinnedHere ? 'unpin probe' : 'pin probe'); S.probePinned = pinnedHere ? null : { x, y }; updateProbe(); requestRender(); } });
  items.push({ label: 'Line probe from here', key: 'click to finish', action: () => {
    const before = snapshot();
    const L = newLine(x, y); renderLegend();
    S.drag = { kind: 'line', line: L, fromMenu: true, before };
    glCanvas.style.cursor = 'crosshair';
    toast('Move the mouse and click to finish the line (Esc cancels)');
  } });
  if (hit) {
    const L = hit.line;
    items.push('-');
    items.push({ header: `line L${L.id}: (${fmt(L.x0, 5)}, ${fmt(L.y0, 5)}) → (${fmt(L.x1, 5)}, ${fmt(L.y1, 5)})` });
    items.push({ label: L.visible ? `Hide L${L.id} in the chart` : `Show L${L.id} in the chart`, action: () => { pushHistory(`${L.visible ? 'hide' : 'show'} L${L.id}`); L.visible = !L.visible; renderLegend(); showLinePanel(); drawLineChart(); requestRender(); } });
    items.push({ label: `Copy L${L.id} samples (CSV)`, disabled: !L.data, action: () => copyText(lineCSV([L]), `L${L.id} samples copied`) });
    items.push({ label: `Download L${L.id} samples (CSV)`, disabled: !L.data, action: () => download(`semview_${S.meta.name}_${S.field}_L${L.id}.csv`, lineCSV([L]), 'text/csv') });
    items.push({ label: `Copy end points of L${L.id}`, action: () => copyText(`${L.x0}\t${L.y0}\n${L.x1}\t${L.y1}`, 'End points copied') });
    items.push({ label: `Delete L${L.id}`, key: 'Del', action: () => removeLine(L) });
  }
  items.push('-');
  items.push({ label: 'Zoom to this element', disabled: !p, action: () => zoomToElement(p.e) });
  items.push({ label: 'Center view here', action: () => { pushHistory('center view'); S.view.cx = x; S.view.cy = y; requestRender(); updateProbe(); } });
  items.push({ label: 'Reset view', key: 'r', action: resetViewUser });
  items.push('-');
  const lastUndo = HIST.undo[HIST.undo.length - 1], lastRedo = HIST.redo[HIST.redo.length - 1];
  items.push({ label: lastUndo ? `Undo ${lastUndo.label}` : 'Undo', key: 'Ctrl+Z', disabled: !lastUndo, action: undo });
  items.push({ label: lastRedo ? `Redo ${lastRedo.label}` : 'Redo', key: 'Ctrl+Shift+Z', disabled: !lastRedo, action: redo });
  items.push('-');
  items.push({ label: 'Copy Python snippet of this view', action: () => copyText(pythonSnippet(), 'Python snippet copied') });
  items.push({ label: 'Copy view limits', action: () => { const [x0, y0] = screenToData(0, H()), [x1, y1] = screenToData(W(), 0); copyText(`xlim = (${x0}, ${x1})\nylim = (${y0}, ${y1})`, 'View limits copied'); } });
  items.push({ label: 'Save screenshot (PNG)', key: 's', action: screenshot });
  openCtx(items, ev.clientX, ev.clientY);
});

lpc.addEventListener('contextmenu', ev => {
  ev.preventDefault();
  const vis = S.lines.filter(L => L.visible && L.data);
  const items = [
    { header: `${S.field} along ${vis.length} visible line${vis.length === 1 ? '' : 's'}` },
    { label: 'Copy samples of visible lines (CSV)', disabled: !vis.length, action: () => copyText(lineCSV(vis), 'Samples copied') },
    { label: 'Download samples of visible lines (CSV)', disabled: !vis.length, action: () => download(`semview_${S.meta.name}_${S.field}_lines.csv`, lineCSV(vis), 'text/csv') },
    '-',
    { label: 'Reset zoom', key: 'double-click', action: resetChartZoom },
    { label: 'Close and clear all lines', key: 'Esc', action: clearLines },
  ];
  openCtx(items, ev.clientX, ev.clientY);
});

// ----------------------------------------------------------------- playback
function setPlaying(on) {
  S.playing = on;
  $('btn-play').textContent = on ? '❚❚' : '▶';
  if (S.timer) { clearInterval(S.timer); S.timer = null; }
  if (on) {
    prefetch();
    const fpsv = parseFloat($('fps').value);
    S.timer = setInterval(() => { if (!S.meta) return; setStep((S.step + 1) % S.meta.nsteps); }, 1000 / fpsv);
  }
}
function setStep(i) {
  if (!S.meta) return;
  S.step = clamp(i, 0, S.meta.nsteps - 1);
  $('step-slider').value = S.step;
  loadField();
}

// ----------------------------------------------------------------- controls wiring
$('field').onchange = ev => { S.field = ev.target.value; loadField(); };
$('cmap').onchange = ev => { S.cmap = ev.target.value; R.setColormap(S.colormaps[S.cmap]); requestRender(); };
$('cmap-invert').onchange = ev => { S.invert = ev.target.checked; requestRender(); };
$('range-auto').onchange = ev => { S.range.auto = ev.target.checked; if (S.range.auto && S.current) setRange(S.current.min, S.current.max, false); };
$('range-sym').onchange = ev => { S.range.sym = ev.target.checked; if (S.current) setRange(S.range.auto ? S.current.min : S.range.lo, S.range.auto ? S.current.max : S.range.hi, false); };
const manualRange = () => { const lo = parseFloat($('vmin').value), hi = parseFloat($('vmax').value); if (Number.isFinite(lo) && Number.isFinite(hi)) { S.range.auto = false; $('range-auto').checked = false; setRange(lo, hi, true); } };
$('vmin').onchange = manualRange; $('vmax').onchange = manualRange;
for (const b of $('mode').querySelectorAll('button')) b.onclick = () => { S.mode = +b.dataset.mode; for (const o of $('mode').querySelectorAll('button')) o.classList.toggle('on', o === b); requestRender(); };
$('show-edges').onchange = ev => { S.edges = ev.target.checked; requestRender(); };
$('edge-width').oninput = ev => { S.edgeWidth = +ev.target.value; requestRender(); };
$('show-contours').onchange = ev => { S.contours = ev.target.checked; requestRender(); };
$('n-contours').oninput = ev => { S.nContours = +ev.target.value; requestRender(); };
$('show-nodes').onchange = ev => { S.nodes = ev.target.checked; requestRender(); };
$('node-size').oninput = ev => { S.nodeSize = +ev.target.value; requestRender(); };
$('show-axes').onchange = ev => { S.axes = ev.target.checked; requestRender(); };
$('show-colorbar').onchange = ev => { S.colorbar = ev.target.checked; requestRender(); };
$('quality').onchange = ev => { S.pxPerCell = +ev.target.value; requestRender(); };
$('bg').onchange = ev => { S.bg = ev.target.value; document.body.classList.toggle('light', isLight()); if (S.lines.length) { renderLegend(); drawLineChart(); } requestRender(); };
$('step-slider').oninput = ev => setStep(+ev.target.value);
$('btn-play').onclick = () => setPlaying(!S.playing);
$('btn-first').onclick = () => setStep(0);
$('btn-last').onclick = () => setStep(S.meta.nsteps - 1);
$('btn-prev').onclick = () => setStep(S.step - 1);
$('btn-next').onclick = () => setStep(S.step + 1);
$('fps').onchange = () => { if (S.playing) setPlaying(true); };
function setLineMode(on) {
  S.lineMode = on;
  $('btn-line').classList.toggle('active', on);
  glCanvas.style.cursor = on ? 'crosshair' : '';
}
$('btn-line').onclick = () => setLineMode(!S.lineMode);
$('btn-clear-probe').onclick = () => { S.probePinned = null; clearLines(); showProbe(null); requestRender(); };
$('snap-angle').onchange = ev => { S.snapAngle = ev.target.checked; };
$('btn-reset').onclick = () => resetViewUser();
$('btn-reload').onclick = async () => { if (S.meta && S.meta.open) { await openPath(S.meta.path); } };
$('btn-shot').onclick = screenshot;
$('btn-open').onclick = openDialog;

window.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT') return;
  if (!$('dialog').hidden) { if (ev.key === 'Escape') closeDialog(); return; }
  switch (ev.key) {
    case ' ': ev.preventDefault(); setPlaying(!S.playing); break;
    case 'ArrowRight': setStep(S.step + 1); break;
    case 'ArrowLeft': setStep(S.step - 1); break;
    case 'Home': setStep(0); resetViewUser(); break;
    case 'End': setStep(S.meta ? S.meta.nsteps - 1 : 0); break;
    case 'm': $('show-edges').click(); break;
    case 'c': $('show-contours').click(); break;
    case 'n': $('show-nodes').click(); break;
    case 'r': resetViewUser(); break;
    case 'z': case 'Z': if (ev.ctrlKey || ev.metaKey) { ev.preventDefault(); if (ev.shiftKey) redo(); else undo(); } else return; break;
    case 'y': case 'Y': if (ev.ctrlKey || ev.metaKey) { ev.preventDefault(); redo(); } else return; break;
    case 's': screenshot(); break;
    case 'o': openDialog(); break;
    case 'l': setLineMode(!S.lineMode); break;
    case 'Escape':
      if (!ctxEl.hidden) { closeCtx(); break; }
      if (S.drag && S.drag.kind === 'line') { removeLine(S.drag.line, false); S.drag = null; glCanvas.style.cursor = ''; break; }
      S.probePinned = null; clearLines(); requestRender(); break;
    case 'Delete': case 'Backspace': { const L = activeLine(); if (L) removeLine(L); break; }
    case '1': case '2': $('mode').querySelector(`[data-mode="${+ev.key - 1}"]`).click(); break;
    default: return;
  }
});

// ----------------------------------------------------------------- screenshot
function screenshot() {
  R.setView(S.view);
  R.render(renderOptions());
  const out = document.createElement('canvas');
  out.width = glCanvas.width; out.height = glCanvas.height;
  const ctx = out.getContext('2d');
  ctx.drawImage(glCanvas, 0, 0);
  ctx.drawImage(overlay, 0, 0);
  out.toBlob(blob => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `semview_${S.meta ? S.meta.name : 'view'}_${S.field}_step${S.step}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }, 'image/png');
}

// ----------------------------------------------------------------- file dialog
async function browse(dir) {
  try {
    const d = await api('browse', { dir });
    $('dlg-path').value = d.dir;
    $('dlg-path').dataset.dir = d.dir;
    $('dlg-path').dataset.parent = d.parent || d.dir;
    const ul = $('dlg-list');
    ul.innerHTML = '';
    if (d.error) { const li = document.createElement('li'); li.textContent = d.error; ul.appendChild(li); }
    for (const e of d.entries) {
      const li = document.createElement('li');
      li.className = e.type;
      const tag = document.createElement('span'); tag.className = 'tag'; tag.textContent = e.type === 'dir' ? 'dir' : e.type === 'meta' ? 'series' : 'field';
      li.appendChild(tag);
      const nm = document.createElement('span'); nm.textContent = e.name; li.appendChild(nm);
      if (e.size !== undefined) { const sz = document.createElement('span'); sz.className = 'size'; sz.textContent = e.size > 1e6 ? (e.size / 1e6).toFixed(1) + ' MB' : (e.size / 1e3).toFixed(0) + ' kB'; li.appendChild(sz); }
      li.onclick = () => { const full = d.dir.replace(/\/$/, '') + '/' + e.name; if (e.type === 'dir') browse(full); else openPath(full); };
      ul.appendChild(li);
    }
  } catch (err) { toast(err.message, true); }
}
function openDialog() { $('dialog').hidden = false; browse(S.meta && S.meta.open ? S.meta.path.split('/').slice(0, -1).join('/') : (S.meta && S.meta.cwd) || ''); }
function closeDialog() { $('dialog').hidden = true; }
$('dlg-close').onclick = closeDialog;
$('dialog').addEventListener('click', ev => { if (ev.target === $('dialog')) closeDialog(); });
$('dlg-up').onclick = () => browse($('dlg-path').dataset.parent);
$('dlg-go').onclick = () => { const p = $('dlg-path').value.trim(); if (/\.nek5000$|\d\.f\d{5}$/.test(p)) openPath(p); else browse(p); };
$('dlg-path').addEventListener('keydown', ev => { if (ev.key === 'Enter') $('dlg-go').click(); });

async function openPath(path) {
  closeDialog();
  toast(`Opening ${path.split('/').pop()} …`);
  try {
    setPlaying(false);
    S.probePinned = null; clearLines(false); clearHistory();
    S.meta = await api('open', { path });
    S.fieldCache.clear();
    await loadMesh();
    populateFields();
    setupTime();
    fitView();
    await loadField();
    toast(`Loaded ${S.meta.name}: ${S.meta.nelv.toLocaleString()} elements, N = ${S.meta.order}`);
  } catch (err) { toast(err.message, true); }
}

// ----------------------------------------------------------------- misc
let toastTimer;
function toast(msg, error = false) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false; t.classList.toggle('error', error);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, error ? 6000 : 2500);
}

// ----------------------------------------------------------------- boot
window.semview = { S, R, api, requestRender, pythonSnippet, lineCSV };   // handy for debugging / scripting the GUI
(async () => {
  try {
    resize();
    await loadColormaps();
    await loadState();
  } catch (err) { toast(err.message, true); console.error(err); }
})();
