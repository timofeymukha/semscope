// semscope GUI application: data loading, view control, overlays, probes.
import { Renderer } from './renderer.js';
import { Spectral, basis } from './spectral.js';

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
  mode: 1,                 // 1 = nodal/linear (fast default), 0 = spectral (exact per pixel)
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
  boundaries: [],          // boundary definitions, see the boundaries section
  showBoundaries: true,
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
  if (!S.meta) { $('status').textContent = ''; $('topbar-title').textContent = ''; return; }
  $('topbar-title').textContent = S.meta.open ? `${S.meta.path.split('/').pop()} · ${S.field}` + (S.current ? ` · t = ${fmt(S.current.time, 6)}` : '') : '';
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

async function post(route, body) {
  const res = await fetch(`/api/${route}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const out = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(out.error || res.statusText);
  return out;
}

async function loadState() {
  S.meta = await api('state');
  if (!S.meta.open) { $('file-info').textContent = 'No dataset loaded — use Open…'; openDialog(); return; }
  await loadMesh();
  populateFields();
  setupTime();
  fitView();
  await loadField();
  if (S.meta.session) await applySession(S.meta.session);
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
  document.title = `semscope — ${m.name}`;
}

function populateFields() {
  const sel = $('field');
  const prev = S.field;
  sel.innerHTML = '';
  const defs = calcDefs();
  const groups = [['stored', S.meta.fields], ['calculator', defs.map(d => d.name).filter(n => !S.meta.fields.includes(n))]];
  for (const [label, names] of groups) {
    if (!names.length) continue;
    const og = document.createElement('optgroup');
    og.label = label;
    for (const nm of names) {
      const o = document.createElement('option');
      o.value = nm;
      const d = label === 'calculator' ? defs.find(c => c.name === nm) : null;
      o.textContent = d ? `${nm} = ${d.expr}` : nm;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  S.meta.available = [...S.meta.fields, ...groups[1][1]];
  S.field = (prev && S.meta.available.includes(prev)) ? prev : S.meta.fields[0];
  sel.value = S.field;
  renderCalcChips();
  renderCalcList();
  renderForceFields();
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
    if (CP.windowOpen) renderCalcList();
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
    octx.fillText(`t = ${fmt(S.current.time, 6)}` + (S.meta.nsteps > 1 ? `   step ${S.step}` : '') + (S.mode === 1 ? '   [nodal / linear]' : '   [spectral]'), 18, 32);
  }
  // pinned probe
  if (S.probePinned) {
    const [px, py] = dataToScreen(S.probePinned.x, S.probePinned.y);
    octx.strokeStyle = '#ff7eb6'; octx.lineWidth = 1.5;
    octx.beginPath(); octx.arc(px, py, 6, 0, 2 * Math.PI); octx.stroke();
    octx.beginPath(); octx.moveTo(px - 10, py); octx.lineTo(px + 10, py); octx.moveTo(px, py - 10); octx.lineTo(px, py + 10); octx.stroke();
  }
  drawBoundaries();
  drawNormalPreview();
  if (FC.hover && FC.windowOpen) {   // point of the wall hovered in the forces chart
    const [px, py] = dataToScreen(FC.hover.x, FC.hover.y);
    octx.strokeStyle = FC.hover.color; octx.lineWidth = 2;
    octx.beginPath(); octx.arc(px, py, 6, 0, 2 * Math.PI); octx.stroke();
    octx.beginPath(); octx.moveTo(px + FC.hover.nx * 8, py - FC.hover.ny * 8); octx.lineTo(px + FC.hover.nx * 22, py - FC.hover.ny * 22); octx.stroke();
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
      octx.fillText(`${lineName(L)}`, ax + 7, ay - 6);
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
  R.setView(S.view);   // the pick buffer must use the current view, not the last rendered one
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
  el.textContent = `x = ${fmt(p.x, 6)}   y = ${fmt(p.y, 6)}\nelement ${p.gid}  (local ${p.e})\n${S.field} = ${fmt(p.v, 7)}${p.ok ? '' : '   (inversion not converged)'}`;
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
const cloneBoundary = b => ({ ...b, edges: b.edges.slice() });
function snapshot() {
  return { lines: S.lines.map(cloneLine), active: LP.active, pinned: S.probePinned ? { ...S.probePinned } : null, view: { ...S.view }, boundaries: S.boundaries.map(cloneBoundary), editing: BD.editing ? BD.editing.id : null };
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
  S.boundaries = (snap.boundaries || []).map(cloneBoundary);
  BD.nextId = Math.max(BD.nextId, ...S.boundaries.map(b => b.id + 1));
  BD.editing = BD.pick ? boundaryById(snap.editing) : null;
  renderBoundaryList();
  if (S.drag && S.drag.kind !== 'pan') { S.drag = null; glCanvas.style.cursor = ''; }
  renderLegend();
  showLinePanel();
  if (!S.lines.length) LP.layout = null;
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
const LP = { layout: null, drag: null, xview: null, yview: null, hoverS: null, active: null, nextId: 1, hoverHit: null, showGrid: true, showElem: true, windowOpen: false };
const SNAP_DEG = 10;

const lineColor = L => (isLight() ? LINE_COLORS_LIGHT : LINE_COLORS_DARK)[L.colorIdx % LINE_COLORS_DARK.length];
const lineName = L => (L.kind === 'normal' ? 'N' : 'L') + L.id;
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
  const L = { id: LP.nextId++, colorIdx, kind: 'line', anchor: null, x0: x, y0: y, x1: x, y1: y, visible: true, data: null, field: null, step: null, hoverIdx: null, token: 0 };
  S.lines.push(L);
  LP.active = L.id;
  return L;
}

function removeLine(L, record = true) {
  if (record) pushHistory(`delete ${lineName(L)}`);
  S.lines = S.lines.filter(o => o !== L);
  if (LP.active === L.id) LP.active = S.lines.length ? S.lines[S.lines.length - 1].id : null;
  if (!S.lines.length) { LP.xview = null; LP.yview = null; }
  renderLegend(); showLinePanel(); drawLineChart(); requestRender();
}

function clearLines(record = true) {
  if (record && (S.lines.length || S.probePinned)) pushHistory('clear lines');
  S.lines = []; LP.active = null; LP.xview = null; LP.yview = null; LP.layout = null;
  renderLegend(); showLinePanel(); drawLineChart();
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
  renderLineList();
  $('probe').classList.remove('muted');
  $('probe').textContent = `line ${lineName(L)}: (${fmt(L.x0, 5)}, ${fmt(L.y0, 5)}) → (${fmt(L.x1, 5)}, ${fmt(L.y1, 5)})\nlength ${fmt(lineLength(L), 5)}   angle ${lineAngle(L).toFixed(1)}°`;
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

/** The line chart is a persistent window: closing it keeps the lines. */
function setLineWindow(open) {
  LP.windowOpen = !!open;
  const p = $('line-panel');
  p.hidden = !LP.windowOpen;
  if (LP.windowOpen) {
    placePanel(p, 'br');
    clampPanel(p);
    p.style.zIndex = ++FLOAT.z;
    renderLegend();
    showLinePanel();
    drawLineChart();
  }
  requestRender();
}
function showLinePanel() {
  const n = S.lines.filter(L => L.visible).length;
  $('lp-title').textContent = S.lines.length ? `${S.field} along ${S.lines.length} line${S.lines.length === 1 ? '' : 's'}` + (n !== S.lines.length ? ` (${n} shown)` : '') : 'line chart — no lines yet (Shift-drag on the plot)';
}

// Tool windows float over the plot: dragged by their header, resized from the
// corner (CSS resize), raised on click and kept inside the view.
const FLOAT = { drag: null, z: 5 };
function clampPanel(p = $('line-panel')) {
  const v = $('view').getBoundingClientRect();
  const left = clamp(parseFloat(p.style.left) || 0, 0, Math.max(0, v.width - p.offsetWidth));
  const top = clamp(parseFloat(p.style.top) || 0, 0, Math.max(0, v.height - p.offsetHeight));
  p.style.left = left + 'px'; p.style.top = top + 'px';
}
/** First-time placement in a corner of the view: 'br' (bottom-right) or 'tl' (top-left, below the field title). */
function placePanel(p, corner) {
  if (p.style.left) return;
  const v = $('view').getBoundingClientRect();
  const right = Math.max(8, v.width - p.offsetWidth - 14) + 'px', bottom = Math.max(8, v.height - p.offsetHeight - 14) + 'px';
  if (corner === 'tl') { p.style.left = '14px'; p.style.top = '56px'; }
  else if (corner === 'tr') { p.style.left = right; p.style.top = '40px'; }
  else { p.style.left = right; p.style.top = bottom; }
}
function floatingPanel(p, head, onLayout) {
  head.addEventListener('mousedown', ev => {
    if (ev.target.closest('button, input')) return;
    FLOAT.drag = { p, dx: ev.clientX - p.offsetLeft, dy: ev.clientY - p.offsetTop };
    ev.preventDefault();
  });
  p.addEventListener('mousedown', () => { p.style.zIndex = ++FLOAT.z; });
  new ResizeObserver(() => { if (!p.hidden) { clampPanel(p); if (onLayout) onLayout(); } }).observe(p);
}
const panelGeometry = p => ({ left: parseFloat(p.style.left) || null, top: parseFloat(p.style.top) || null, width: p.offsetWidth, height: p.offsetHeight });
function applyPanelGeometry(p, g) {
  if (!g) return;
  if (g.width) p.style.width = g.width + 'px';
  if (g.height) p.style.height = g.height + 'px';
  if (g.left != null) p.style.left = g.left + 'px';
  if (g.top != null) p.style.top = g.top + 'px';
}

/** Sidebar registry of the line probes: visibility, selection, editable end points. */
function renderLineList() {
  const box = $('line-list');
  box.innerHTML = '';
  for (const L of S.lines) {
    const item = document.createElement('div');
    item.className = 'line-item' + (L.id === LP.active ? ' active' : '') + (L.visible ? '' : ' hidden-line');
    const head = document.createElement('div'); head.className = 'head';
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = L.visible; cb.title = 'show / hide in the chart';
    cb.onchange = () => { pushHistory(`${cb.checked ? 'show' : 'hide'} ${lineName(L)}`); L.visible = cb.checked; renderLegend(); showLinePanel(); drawLineChart(); requestRender(); };
    const sw = document.createElement('i'); sw.className = 'swatch'; sw.style.background = lineColor(L);
    const nm = document.createElement('span'); nm.className = 'name'; nm.textContent = `${lineName(L)}`; nm.title = 'select';
    nm.onclick = () => { LP.active = L.id; renderLegend(); drawLineChart(); requestRender(); };
    const len = document.createElement('span'); len.className = 'len'; len.textContent = `${fmt(lineLength(L), 4)} · ${lineAngle(L).toFixed(1)}°`;
    if (L.kind === 'normal' && L.anchor) { len.title = `wall-normal from element ${S.mesh ? S.mesh.elmap[L.anchor.elem] : L.anchor.elem}, ${['bottom', 'right', 'top', 'left'][L.anchor.side]} side, ${L.anchor.node !== null && L.anchor.node !== undefined ? `node ${L.anchor.node}` : `t = ${fmt(L.anchor.t, 4)}`}`; len.textContent = `⊥ ${len.textContent}`; }
    const x = document.createElement('button'); x.className = 'x'; x.textContent = '✕'; x.title = 'delete';
    x.onclick = () => removeLine(L);
    head.append(cb, sw, nm, len, x);
    const grid = document.createElement('div'); grid.className = 'coords';
    const field = (key, value) => {
      const inp = document.createElement('input'); inp.type = 'text'; inp.className = 'mono'; inp.value = fmt(value, 6); inp.title = key;
      inp.onkeydown = ev => { ev.stopPropagation(); if (ev.key === 'Enter') inp.blur(); if (ev.key === 'Escape') { inp.value = fmt(L[key], 6); inp.blur(); } };
      inp.onchange = () => {
        const v = parseFloat(inp.value);
        if (!Number.isFinite(v)) { inp.value = fmt(L[key], 6); return; }
        if (v === L[key]) return;
        pushHistory(`move ${lineName(L)}`);
        if (L.kind === 'normal' && L.anchor && BD.ext) {
          const want = { x0: L.x0, y0: L.y0, x1: L.x1, y1: L.y1 }; want[key] = v;
          if (key === 'x1' || key === 'y1') setNormalLength(L, want.x1, want.y1);
          else { const h = boundaryPointAt(...dataToScreen(want.x0, want.y0), allowedEdgesFor(L), NL.snap, 1e9); if (h) setNormalOrigin(L, h); }
        } else L[key] = v;
        LP.xview = null; LP.yview = null;
        renderLineList(); runLine(L); requestRender();
      };
      return inp;
    };
    const lab = t => { const e = document.createElement('span'); e.textContent = t; return e; };
    grid.append(lab('from'), field('x0', L.x0), field('y0', L.y0), lab('to'), field('x1', L.x1), field('y1', L.y1));
    item.append(head, grid);
    box.appendChild(item);
  }
}

function renderLegend() {
  renderLineList();
  const box = $('lp-legend');
  box.innerHTML = '';
  for (const L of S.lines) {
    const item = document.createElement('span');
    item.className = 'lp-item' + (L.id === LP.active ? ' active' : '') + (L.visible ? '' : ' hidden-line');
    item.title = `(${fmt(L.x0)}, ${fmt(L.y0)}) → (${fmt(L.x1)}, ${fmt(L.y1)})  length ${fmt(lineLength(L))}, ${lineAngle(L).toFixed(1)}°\nclick: select · checkbox: show/hide · ✕: delete`;
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = L.visible;
    cb.onclick = ev => { ev.stopPropagation(); pushHistory(`${cb.checked ? 'show' : 'hide'} ${lineName(L)}`); L.visible = cb.checked; renderLegend(); showLinePanel(); drawLineChart(); requestRender(); };
    const sw = document.createElement('i'); sw.className = 'swatch'; sw.style.background = lineColor(L);
    const nm = document.createElement('span'); nm.textContent = `${lineName(L)}`;
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
      infos.push(`${lineName(v.L)} = ${y === null ? 'outside' : fmt(y, 6)}`);
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
$('lp-close').onclick = () => setLineWindow(false);
floatingPanel($('line-panel'), $('lp-head'), drawLineChart);

// ----------------------------------------------------------------- field calculator
// New fields are defined on the server as expressions of the stored ones
// (arithmetic, functions, dx()/dy() derivatives of any order).  They are
// evaluated at the GLL nodes of every time step and then behave like stored
// fields: they can be displayed, probed and sampled along lines.
const CP = { windowOpen: false, checkTimer: null, checkToken: 0 };
const calcDefs = () => (S.meta && S.meta.calc) || [];
const isStored = nm => !!(S.meta && S.meta.fields && S.meta.fields.includes(nm));

function setCalcWindow(open) {
  CP.windowOpen = !!open;
  const p = $('calc-panel');
  p.hidden = !CP.windowOpen;
  if (CP.windowOpen) {
    placePanel(p, 'tl');
    clampPanel(p);
    p.style.zIndex = ++FLOAT.z;
    renderCalcChips();
    renderCalcList();
    if (!$('cp-expr').value.trim()) setCalcStatus(calcIdle(), 'muted');
    $('cp-expr').focus();
  }
}
const calcIdle = () => (S.meta && S.meta.open) ? 'name = expression · Enter defines and shows the field · the chips insert at the caret' : 'open a dataset first';
function setCalcStatus(text, cls = 'muted') { const el = $('cp-status'); el.textContent = text; el.className = 'cp-status mono ' + cls; }

/** Insert into the expression at the caret; with a suffix the selection is wrapped, e.g. dx( … ). */
function insertExpr(prefix, suffix = null) {
  const inp = $('cp-expr');
  const a = inp.selectionStart ?? inp.value.length, b = inp.selectionEnd ?? a;
  const sel = inp.value.slice(a, b);
  const ins = suffix === null ? prefix : prefix + sel + suffix;
  const caret = suffix === null ? a + prefix.length : (sel ? a + ins.length : a + prefix.length);
  inp.value = inp.value.slice(0, a) + ins + inp.value.slice(b);
  inp.focus();
  inp.setSelectionRange(caret, caret);
  scheduleCheck();
}

function renderCalcChips() {
  const box = $('cp-chips');
  box.innerHTML = '';
  if (!S.meta || !S.meta.open) return;
  const sx = S.meta.calc_syntax || { functions: [], binary_functions: [], constants: [] };
  const row = (label, chips) => {
    const r = document.createElement('div'); r.className = 'cp-row';
    const l = document.createElement('span'); l.className = 'cp-label'; l.textContent = label; r.appendChild(l);
    for (const [text, action, title] of chips) {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip'; b.textContent = text; if (title) b.title = title;
      b.onmousedown = ev => ev.preventDefault();   // keep focus and caret in the expression input
      b.onclick = action;
      r.appendChild(b);
    }
    box.appendChild(r);
  };
  const names = [...S.meta.fields, ...calcDefs().map(d => d.name).filter(n => !isStored(n)), 'x', 'y'];
  row('fields', names.map(nm => [nm, () => insertExpr(nm), isStored(nm) ? 'stored field' : (nm === 'x' || nm === 'y' ? 'coordinate' : 'calculator field')]));
  row('operators', ['+', '-', '*', '/', '^', '(', ')'].map(op => [op, () => insertExpr('()'.includes(op) ? op : ` ${op} `), op === '^' ? 'power (also **)' : null]));
  row('derivatives', [
    ['dx( )', () => insertExpr('dx(', ')'), 'd/dx of the selection (spectrally exact within each element)'],
    ['dy( )', () => insertExpr('dy(', ')'), 'd/dy'],
    ['dx( , 2)', () => insertExpr('dx(', ', 2)'), 'second derivative in x — dx(f, n) gives order n'],
    ['dy( , 2)', () => insertExpr('dy(', ', 2)'), 'second derivative in y'],
    ['dx(dy( ))', () => insertExpr('dx(dy(', '))'), 'mixed derivative by nesting'],
  ]);
  row('functions', [...sx.functions.map(fn => [fn, () => insertExpr(`${fn}(`, ')'), `${fn}(f)`]), ...sx.binary_functions.map(fn => [fn, () => insertExpr(`${fn}(`, ', )'), `${fn}(a, b)`]), ...sx.constants.map(c => [c, () => insertExpr(c), 'constant'])]);
}

function renderCalcList() {
  const box = $('cp-list');
  box.innerHTML = '';
  const defs = calcDefs();
  $('cp-count').textContent = defs.length ? `${defs.length} field${defs.length === 1 ? '' : 's'}` : '';
  if (!defs.length) { const e = document.createElement('div'); e.className = 'cp-empty muted'; e.textContent = 'No calculator fields yet — e.g. vort = dx(v) - dy(u)'; box.appendChild(e); return; }
  for (const d of defs) {
    const item = document.createElement('div'); item.className = 'cp-item' + (S.field === d.name ? ' active' : '');
    const show = () => { S.field = d.name; $('field').value = d.name; loadField(); renderCalcList(); };
    const nm = document.createElement('span'); nm.className = 'name'; nm.textContent = d.name; nm.title = 'show this field'; nm.onclick = show;
    const ex = document.createElement('span'); ex.className = 'expr mono'; ex.textContent = `= ${d.expr}`; ex.title = d.expr; ex.onclick = show;
    const ed = document.createElement('button'); ed.className = 'edit'; ed.textContent = 'edit'; ed.title = 'load into the editor (Define replaces the definition)';
    ed.onclick = () => { $('cp-name').value = d.name; $('cp-expr').value = d.expr; $('cp-expr').focus(); scheduleCheck(); };
    const x = document.createElement('button'); x.className = 'x'; x.textContent = '✕'; x.title = 'remove this definition';
    x.onclick = () => removeCalcField(d.name);
    item.append(nm, ex, ed, x);
    box.appendChild(item);
  }
}

/** Live validation while typing (syntax, names, cycles); nothing is evaluated. */
function scheduleCheck() {
  clearTimeout(CP.checkTimer);
  const expr = $('cp-expr').value.trim();
  if (!expr) { setCalcStatus(calcIdle(), 'muted'); return; }
  if (!S.meta || !S.meta.open) return;
  CP.checkTimer = setTimeout(async () => {
    const token = ++CP.checkToken;
    try {
      const r = await api('calc/check', { expr, name: $('cp-name').value.trim() });
      if (token !== CP.checkToken) return;
      setCalcStatus(r.ok ? '✓ valid — Enter defines it' : '✗ ' + r.error, r.ok ? 'ok' : 'err');
    } catch (err) { if (token === CP.checkToken) setCalcStatus('✗ ' + err.message, 'err'); }
  }, 250);
}

/** Cached values and line samples of calculator fields are dropped when a definition changes. */
function invalidateCalc() {
  for (const key of [...S.fieldCache.keys()]) if (!isStored(key.slice(key.indexOf(':') + 1))) S.fieldCache.delete(key);
  for (const L of S.lines) if (L.field && !isStored(L.field)) L.data = null;
  if (FC.windowOpen && FC.sel.size) scheduleForces();
}

async function defineField() {
  const name = $('cp-name').value.trim(), expr = $('cp-expr').value.trim();
  if (!S.meta || !S.meta.open) { setCalcStatus('✗ open a dataset first', 'err'); return; }
  if (!name) { setCalcStatus('✗ give the new field a name', 'err'); $('cp-name').focus(); return; }
  if (!expr) { setCalcStatus('✗ enter an expression', 'err'); $('cp-expr').focus(); return; }
  try {
    const out = await post('calc/define', { name, expr, step: S.step });
    S.meta.calc = out.calc;
    invalidateCalc();
    populateFields();
    S.field = out.name; $('field').value = S.field;
    setCalcStatus(`${out.name} = ${out.expr}   ∈ [${fmt(out.min, 5)}, ${fmt(out.max, 5)}] at step ${S.step}`, 'ok');
    await loadField();
    renderCalcList();
  } catch (err) { setCalcStatus('✗ ' + err.message, 'err'); }
}

async function removeCalcField(name) {
  try {
    const out = await post('calc/remove', { name });
    S.meta.calc = out.calc;
    invalidateCalc();
    populateFields();
    setCalcStatus(`removed ${name}`, 'muted');
    await loadField();
    renderCalcList();
  } catch (err) { setCalcStatus('✗ ' + err.message, 'err'); }
}

/** Make the server definitions match a list (sessions): define missing ones, drop extras. */
async function syncCalcDefs(defs) {
  const want = new Map((defs || []).map(d => [d.name, d.expr]));
  let extra = calcDefs().filter(d => !want.has(d.name)).map(d => d.name);
  for (let pass = 0; extra.length && pass < extra.length + 1; pass++) {   // dependents block removal: retry in passes
    const left = [];
    for (const nm of extra) { try { S.meta.calc = (await post('calc/remove', { name: nm })).calc; } catch { left.push(nm); } }
    if (left.length === extra.length) break;
    extra = left;
  }
  for (const [name, expr] of want) {
    if (calcDefs().some(d => d.name === name && d.expr === expr)) continue;
    try { S.meta.calc = (await post('calc/define', { name, expr, step: S.step })).calc; }
    catch (err) { toast(`Calculator field ${name} not restored: ${err.message}`, true); }
  }
  invalidateCalc();
  populateFields();
  setCalcStatus(calcIdle(), 'muted');
}

$('cp-define').onclick = defineField;
$('cp-close').onclick = () => setCalcWindow(false);
$('btn-calc').onclick = () => setCalcWindow(!CP.windowOpen);
$('cp-expr').addEventListener('input', scheduleCheck);
$('cp-name').addEventListener('input', () => { if ($('cp-expr').value.trim()) scheduleCheck(); });
for (const id of ['cp-name', 'cp-expr']) $(id).addEventListener('keydown', ev => {
  if (ev.key === 'Enter') { ev.preventDefault(); defineField(); }
  else if (ev.key === 'Escape') ev.target.blur();
});
floatingPanel($('calc-panel'), $('cp-head'), null);

// ----------------------------------------------------------------- forces
// Forces on selected boundaries.  The server integrates the wall traction
// p n - mu (grad u + grad u^T) n (n pointing out of the fluid, i.e. the force
// on the body per unit depth) with the GLL quadrature of the boundary edges.
// The window shows pressure / viscous / total parts along configurable axes,
// the force coefficients, and Cp or Cf along the wall.
const FC = { windowOpen: false, sel: new Set(), result: null, token: 0, timer: null, hover: null, sig: '', plot: 'cp', layout: null, xview: null, yview: null, drag: null };
const FORCE_SELECTS = ['fp-u', 'fp-v', 'fp-w', 'fp-p', 'fp-rho-mode', 'fp-mu-mode'];

function setForceWindow(open) {
  FC.windowOpen = !!open;
  const p = $('force-panel');
  p.hidden = !FC.windowOpen;
  if (FC.windowOpen) {
    placePanel(p, 'tr');
    clampPanel(p);
    p.style.zIndex = ++FLOAT.z;
    renderForceFields(); renderForceBoundaries(); renderForceTable(); drawForceChart();
    if (FC.sel.size) scheduleForces(0); else setForceStatus(S.boundaries.length ? 'tick the boundaries to integrate over' : 'define boundaries first (Detect or New manual… in the sidebar)');
  } else { FC.hover = null; }
  requestRender();
}
function setForceStatus(text, cls = 'muted') { const el = $('fp-status'); el.textContent = text; el.className = 'cp-status mono ' + cls; }

/** (Re)fill a select with field names, keeping the user's choice when it still exists. */
function fillSelect(sel, names, fallback, noneLabel = null) {
  const prev = sel.dataset.populated === '1' ? sel.value : null;
  sel.innerHTML = '';
  if (noneLabel !== null) { const o = document.createElement('option'); o.value = ''; o.textContent = noneLabel; sel.appendChild(o); }
  for (const nm of names) { const o = document.createElement('option'); o.value = nm; o.textContent = nm; sel.appendChild(o); }
  let want;
  if (prev !== null && (names.includes(prev) || (prev === '' && noneLabel !== null))) want = prev;
  else if (fallback && names.includes(fallback)) want = fallback;
  else want = noneLabel !== null ? '' : (names[0] || '');
  sel.value = want;
  sel.dataset.populated = '1';
}
function renderForceFields() {
  if (!S.meta || !S.meta.open) return;
  const names = S.meta.available;
  fillSelect($('fp-u'), names, 'u'); fillSelect($('fp-v'), names, 'v'); fillSelect($('fp-w'), names, 'w', '—'); fillSelect($('fp-p'), names, 'p');
  fillSelect($('fp-rho-mode'), names, '', 'constant'); fillSelect($('fp-mu-mode'), names, '', 'constant');
  $('fp-rho').hidden = $('fp-rho-mode').value !== ''; $('fp-mu').hidden = $('fp-mu-mode').value !== '';
}
function resetForceWindow() {
  FC.sel.clear(); FC.result = null; FC.hover = null; FC.sig = ''; FC.xview = null; FC.yview = null;
  for (const id of FORCE_SELECTS) $(id).dataset.populated = '';
  renderForceTable();
  if (FC.windowOpen) drawForceChart();
}

const fnum = (id, def) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : def; };
/** The request / session description of the force settings, read from the inputs. */
function forceConfig() {
  const rhoMode = $('fp-rho-mode').value, muMode = $('fp-mu-mode').value;
  return {
    u: $('fp-u').value, v: $('fp-v').value, w: $('fp-w').value || null, p: $('fp-p').value,
    rho: rhoMode || fnum('fp-rho', 1), mu: muMode || fnum('fp-mu', 1),
    U_ref: fnum('fp-U', 1), L_ref: fnum('fp-L', 1), p_ref: fnum('fp-pref', 0),
    rho_ref: $('fp-rhoref').value.trim() === '' ? null : fnum('fp-rhoref', 1),
    axes: [
      { name: $('fp-a1').value.trim() || 'x', dir: [fnum('fp-a1x', 1), fnum('fp-a1y', 0)] },
      { name: $('fp-a2').value.trim() || 'y', dir: [fnum('fp-a2x', 0), fnum('fp-a2y', 1)] },
    ],
  };
}
function applyForceConfig(c) {
  if (!c) return;
  renderForceFields();
  const setSel = (id, v) => { const el = $(id); if (v === undefined || v === null) return; if ([...el.options].some(o => o.value === v)) el.value = v; };
  setSel('fp-u', c.u); setSel('fp-v', c.v); setSel('fp-w', c.w || ''); setSel('fp-p', c.p);
  for (const [k, mode, val] of [['rho', 'fp-rho-mode', 'fp-rho'], ['mu', 'fp-mu-mode', 'fp-mu']]) {
    if (c[k] === undefined || c[k] === null) continue;
    if (typeof c[k] === 'string') setSel(mode, c[k]); else { $(mode).value = ''; $(val).value = c[k]; }
    $(val).hidden = $(mode).value !== '';
  }
  for (const [k, id] of [['U_ref', 'fp-U'], ['L_ref', 'fp-L'], ['p_ref', 'fp-pref']]) if (c[k] !== undefined && c[k] !== null) $(id).value = c[k];
  $('fp-rhoref').value = c.rho_ref === undefined || c.rho_ref === null ? '' : c.rho_ref;
  if (c.axes && c.axes.length >= 2) {
    $('fp-a1').value = c.axes[0].name; $('fp-a1x').value = c.axes[0].dir[0]; $('fp-a1y').value = c.axes[0].dir[1];
    $('fp-a2').value = c.axes[1].name; $('fp-a2x').value = c.axes[1].dir[0]; $('fp-a2y').value = c.axes[1].dir[1];
  }
}

const forceSig = () => S.boundaries.filter(b => FC.sel.has(b.id)).map(b => `${b.id}:${b.name}:${b.edges.length}:${b.edges.join(',')}`).join('|');
/** Checkbox list of the boundaries; recomputes when a selected boundary changed (edited, renamed, undone). */
function renderForceBoundaries() {
  const box = $('fp-bd');
  const ids = new Set(S.boundaries.map(b => b.id));
  for (const id of [...FC.sel]) if (!ids.has(id)) FC.sel.delete(id);
  box.innerHTML = '';
  if (!S.boundaries.length) { const e = document.createElement('span'); e.className = 'muted'; e.textContent = 'none yet — Detect or New manual… in the Boundaries section'; box.appendChild(e); }
  for (const b of S.boundaries) {
    const lab = document.createElement('label'); lab.className = 'check';
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = FC.sel.has(b.id);
    cb.onchange = () => { if (cb.checked) FC.sel.add(b.id); else FC.sel.delete(b.id); FC.xview = null; FC.yview = null; scheduleForces(0); };
    const sw = document.createElement('i'); sw.className = 'swatch'; sw.style.background = bdColor(b);
    lab.append(cb, sw, document.createTextNode(`${b.name} (${b.edges.length})`));
    lab.title = `${b.edges.length} edge${b.edges.length === 1 ? '' : 's'}${b.source === 'auto' ? ', detected' : ', manual'}`;
    box.appendChild(lab);
  }
  if (FC.windowOpen && forceSig() !== FC.sig) scheduleForces();
}

function scheduleForces(delay = 150) {
  clearTimeout(FC.timer);
  FC.sig = forceSig();
  FC.timer = setTimeout(computeForces, delay);
}
async function computeForces() {
  if (!S.meta || !S.meta.open) return;
  const bds = S.boundaries.filter(b => FC.sel.has(b.id) && b.edges.length);
  if (!bds.length) {
    FC.result = null; FC.hover = null; renderForceTable(); drawForceChart();
    setForceStatus(S.boundaries.length ? 'tick the boundaries to integrate over' : 'define boundaries first (Detect or New manual… in the sidebar)');
    requestRender(); return;
  }
  const token = ++FC.token;
  try {
    const out = await post('forces', { step: S.step, boundaries: bds.map(b => ({ name: b.name, edges: b.edges })), ...forceConfig() });
    if (token !== FC.token) return;
    out.ids = bds.map(b => b.id);
    FC.result = out; FC.hover = null; $('fp-hover').textContent = '';
    renderForceTable(); drawForceChart();
    setForceStatus(`q = ½ ρ_ref U² = ${fmt(out.q, 5)} with ρ_ref = ${fmt(out.rho_ref, 5)}   ·   step ${out.step}, t = ${fmt(out.time, 6)}   ·   forces per unit depth, on the body`);
  } catch (err) { setForceStatus('✗ ' + err.message, 'err'); }
  requestRender();
}

const forceColor = (R, i) => bdColor(boundaryById(R.ids[i]) || { colorIdx: i });
function renderForceTable() {
  const t = $('fp-table'); t.innerHTML = '';
  const R = FC.result;
  $('fp-info').textContent = R ? `${R.boundaries.length} boundar${R.boundaries.length === 1 ? 'y' : 'ies'} · L = ${fmt(R.total.length, 5)}` : '';
  if (!R) return;
  const head = t.createTHead().insertRow();
  for (const h of ['boundary', 'axis', 'pressure', 'viscous', 'total', 'coefficient']) { const th = document.createElement('th'); th.textContent = h; head.appendChild(th); }
  const body = t.createTBody();
  const block = (res, label, color, cls) => {
    const rows = res.axes.map(a => ({ ...a }));
    if (res.components.length === 3) rows.push({ name: 'z', pressure: res.pressure[2], viscous: res.viscous[2], total: res.total[2], coefficient: res.total[2] / (res.q * res.L_ref) });
    rows.forEach((a, i) => {
      const tr = body.insertRow(); if (cls) tr.className = cls;
      const c0 = tr.insertCell();
      if (i === 0) {
        if (color) { const sw = document.createElement('i'); sw.className = 'swatch'; sw.style.background = color; c0.appendChild(sw); }
        c0.appendChild(document.createTextNode(label));
        c0.title = `${res.nedges} edges, length ${fmt(res.length, 6)}`;
      }
      tr.insertCell().textContent = a.name;
      for (const k of ['pressure', 'viscous', 'total']) tr.insertCell().textContent = fmt(a[k], 5);
      tr.insertCell().textContent = `C${a.name} = ${fmt(a.coefficient, 5)}`;
    });
  };
  R.boundaries.forEach((res, i) => block(res, res.name, forceColor(R, i), ''));
  if (R.boundaries.length > 1) block(R.total, `Σ ${R.boundaries.length}`, null, 'total');
}

const FP_LABELS = { cp: 'Cp', cf: 'Cf', p: 'p', tau: 'τw' };
function drawForceChart() {
  const c = $('fp-canvas'), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w || !h || $('force-panel').hidden) return;
  const Wd = Math.round(w * dpr), Hd = Math.round(h * dpr);
  if (c.width !== Wd || c.height !== Hd) { c.width = Wd; c.height = Hd; }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const light = isLight();
  const fg = light ? '#1b2130' : '#e6e9f0', dim = light ? 'rgba(27,33,48,.45)' : 'rgba(230,233,240,.45)', grid = light ? 'rgba(27,33,48,.1)' : 'rgba(230,233,240,.1)';
  ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
  FC.layout = null;
  const R = FC.result;
  if (!R) { ctx.fillStyle = dim; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('Cp and Cf along the selected boundaries appear here', w / 2, h / 2); return; }
  const key = FC.plot;
  const series = R.boundaries.map((res, i) => ({ res, color: forceColor(R, i), s: res.dist.s, v: res.dist[key], chain: res.dist.chain }));
  // full arc-length extent, then the zoomed window (wheel / drag), then the value range of what is visible
  let smax = 0;
  for (const q of series) for (let i = 0; i < q.s.length; i++) if (q.v[i] !== null && q.s[i] > smax) smax = q.s[i];
  if (!(smax > 0)) smax = 1;
  let s0 = 0, s1 = smax;
  if (FC.xview) { s0 = clamp(FC.xview.s0, 0, smax); s1 = clamp(FC.xview.s1, 0, smax); if (s1 - s0 < smax * 1e-9) { s0 = 0; s1 = smax; FC.xview = null; } }
  let lo = Infinity, hi = -Infinity;
  for (const q of series) for (let i = 0; i < q.s.length; i++) { const y = q.v[i]; if (y === null || q.s[i] < s0 || q.s[i] > s1) continue; if (y < lo) lo = y; if (y > hi) hi = y; }
  if (!(lo <= hi)) { lo = -1; hi = 1; } else if (lo === hi) { lo -= 0.5; hi += 0.5; } else { const pad = 0.06 * (hi - lo); lo -= pad; hi += pad; }
  if (FC.yview) { lo = FC.yview.lo; hi = FC.yview.hi; }
  const pad = { l: 58, r: 12, t: 8, b: 24 }, pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const X = s => pad.l + (s - s0) / (s1 - s0) * pw, Y = v => pad.t + (1 - (v - lo) / (hi - lo)) * ph;
  // grid and ticks
  const sx = niceStep(s1 - s0, pw / 90), sy = niceStep(hi - lo, ph / 40);
  ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.fillStyle = fg;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let s = Math.ceil(s0 / sx) * sx; s <= s1 + 1e-9 * smax; s += sx) { const px = X(s); ctx.beginPath(); ctx.moveTo(px, pad.t); ctx.lineTo(px, pad.t + ph); ctx.stroke(); ctx.fillText(tickFmt(s, sx), px, pad.t + ph + 4); }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let v = Math.ceil(lo / sy) * sy; v <= hi; v += sy) { const py = Y(v); ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + pw, py); ctx.stroke(); ctx.fillText(tickFmt(v, sy), pad.l - 5, py); }
  if (lo < 0 && hi > 0) { ctx.strokeStyle = dim; ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(pad.l + pw, Y(0)); ctx.stroke(); }
  ctx.strokeStyle = dim; ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, pw, ph);
  ctx.fillStyle = dim; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText((FC.xview || FC.yview) ? 'arc length s  (zoomed — double-click to reset)' : 'arc length s', pad.l + pw, pad.t + ph - 3);
  ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(FP_LABELS[key], pad.l + 4, pad.t + 3);
  // curves (one polyline per chain of connected edges)
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, pw, ph); ctx.clip();
  for (const q of series) {
    ctx.strokeStyle = q.color; ctx.lineWidth = 1.4; ctx.beginPath();
    let prev = -1;
    for (let i = 0; i < q.s.length; i++) {
      if (q.v[i] === null) { prev = -1; continue; }
      const px = X(q.s[i]), py = Y(q.v[i]);
      if (q.chain[i] !== prev) { ctx.moveTo(px, py); prev = q.chain[i]; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.restore();
  if (FC.hover && FC.hover.v !== null && FC.hover.key === key && FC.hover.s >= s0 && FC.hover.s <= s1) {
    ctx.strokeStyle = FC.hover.color; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(X(FC.hover.s), Y(FC.hover.v), 4, 0, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = dim; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(FC.hover.s), pad.t); ctx.lineTo(X(FC.hover.s), pad.t + ph); ctx.stroke(); ctx.setLineDash([]);
  }
  FC.layout = { pad, pw, ph, smax, s0, s1, lo, hi, series, X, Y };
}

const fpc = $('fp-canvas');
const fpPos = ev => { const r = fpc.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
const fpArc = mx => { const L = FC.layout; return L.s0 + (mx - L.pad.l) / L.pw * (L.s1 - L.s0); };
const resetForceZoom = () => { FC.xview = null; FC.yview = null; drawForceChart(); requestRender(); };
fpc.addEventListener('mousemove', ev => {
  const L = FC.layout; if (!L || FC.drag) return;
  const [mx, my] = fpPos(ev);
  let best = null, bd = Infinity;
  for (const q of L.series) for (let i = 0; i < q.s.length; i++) {
    if (q.v[i] === null || q.s[i] < L.s0 || q.s[i] > L.s1) continue;
    const dx = L.X(q.s[i]) - mx, dy = L.Y(q.v[i]) - my, d = dx * dx + 0.15 * dy * dy;   // mostly by arc length
    if (d < bd) { bd = d; best = { q, i }; }
  }
  if (!best) return;
  const { q, i } = best, d = q.res.dist;
  FC.hover = { key: FC.plot, x: d.x[i], y: d.y[i], nx: d.nx[i], ny: d.ny[i], s: d.s[i], v: q.v[i], color: q.color };
  $('fp-hover').textContent = `${q.res.name}: s = ${fmt(d.s[i], 5)}   (${fmt(d.x[i], 5)}, ${fmt(d.y[i], 5)})   Cp ${fmt(d.cp[i], 4)}   Cf ${fmt(d.cf[i], 4)}   p ${fmt(d.p[i], 4)}   τw ${fmt(d.tau[i], 4)}`;
  drawForceChart(); requestRender();
});
fpc.addEventListener('mouseleave', () => { if (FC.drag) return; FC.hover = null; $('fp-hover').textContent = ''; drawForceChart(); requestRender(); });
// zoom: wheel along the arc length, Shift+wheel along the value axis; drag pans; double-click resets
fpc.addEventListener('wheel', ev => {
  ev.preventDefault();
  const L = FC.layout; if (!L) return;
  const [mx, my] = fpPos(ev);
  const f = Math.exp(ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015));
  if (ev.shiftKey) {
    const v = L.lo + (1 - (my - L.pad.t) / L.ph) * (L.hi - L.lo);
    FC.yview = { lo: v - (v - L.lo) * f, hi: v + (L.hi - v) * f };
  } else {
    const sm = clamp(fpArc(mx), L.s0, L.s1);
    const a = Math.max(0, sm - (sm - L.s0) * f), b = Math.min(L.smax, sm + (L.s1 - sm) * f);
    if (b - a < L.smax * 1e-6) return;
    FC.xview = (a <= 0 && b >= L.smax) ? null : { s0: a, s1: b };
  }
  drawForceChart(); requestRender();
}, { passive: false });
fpc.addEventListener('mousedown', ev => {
  const L = FC.layout; if (!L || ev.button !== 0) return;
  const [mx, my] = fpPos(ev);
  FC.drag = { mx, my, s0: L.s0, s1: L.s1, yview: FC.yview ? { ...FC.yview } : { lo: L.lo, hi: L.hi }, L, moved: false };
  ev.preventDefault();
});
function fcDrag(ev) {
  const D = FC.drag, L = D.L;
  const [mx, my] = fpPos(ev);
  const dx = mx - D.mx, dy = my - D.my;
  if (Math.abs(dx) + Math.abs(dy) > 2) D.moved = true;
  if (!D.moved) return;
  const width = D.s1 - D.s0;
  const a = clamp(D.s0 - dx / L.pw * width, 0, Math.max(0, L.smax - width));
  FC.xview = (a <= 0 && a + width >= L.smax) ? null : { s0: a, s1: a + width };
  const dv = dy / L.ph * (D.yview.hi - D.yview.lo);
  FC.yview = { lo: D.yview.lo + dv, hi: D.yview.hi + dv };
  drawForceChart(); requestRender();
}
fpc.addEventListener('dblclick', resetForceZoom);
$('fp-reset').onclick = resetForceZoom;

/** Tab-separated force table (for the clipboard). */
function forceTableText() {
  const R = FC.result; if (!R) return '';
  const cfg = forceConfig();
  const rows = [`# forces on the body per unit depth, step ${R.step}, t = ${R.time}`, `# u=${cfg.u} v=${cfg.v}${cfg.w ? ` w=${cfg.w}` : ''} p=${cfg.p} rho=${cfg.rho} mu=${cfg.mu}  q = ${R.q} (rho_ref ${R.rho_ref}, U_ref ${cfg.U_ref}), L_ref ${cfg.L_ref}, p_ref ${cfg.p_ref}`, 'boundary\tlength\taxis\tdir_x\tdir_y\tpressure\tviscous\ttotal\tcoefficient'];
  const block = (res, label) => {
    for (const a of res.axes) rows.push(`${label}\t${res.length}\t${a.name}\t${a.dir[0]}\t${a.dir[1]}\t${a.pressure}\t${a.viscous}\t${a.total}\t${a.coefficient}`);
    if (res.components.length === 3) rows.push(`${label}\t${res.length}\tz\t0\t0\t${res.pressure[2]}\t${res.viscous[2]}\t${res.total[2]}\t${res.total[2] / (res.q * res.L_ref)}`);
  };
  for (const res of R.boundaries) block(res, res.name);
  if (R.boundaries.length > 1) block(R.total, 'total');
  return rows.join('\n') + '\n';
}
/** CSV of the wall distributions of every computed boundary. */
function forceCSV() {
  const R = FC.result; if (!R) return '';
  const rows = ['boundary,s,x,y,nx,ny,p,cp,tau_w,cf'];
  for (const res of R.boundaries) { const d = res.dist; for (let i = 0; i < d.s.length; i++) rows.push(`${res.name},${d.s[i]},${d.x[i]},${d.y[i]},${d.nx[i]},${d.ny[i]},${d.p[i]},${d.cp[i]},${d.tau[i]},${d.cf[i]}`); }
  return rows.join('\n') + '\n';
}

$('fp-close').onclick = () => setForceWindow(false);
$('fp-copy').onclick = () => { if (FC.result) copyText(forceTableText(), 'Force table copied'); else toast('Nothing computed yet', true); };
$('bd-forces').onclick = () => setForceWindow(!FC.windowOpen);
$('fp-plot').onchange = ev => { FC.plot = ev.target.value; FC.hover = null; drawForceChart(); requestRender(); };
$('fp-grid').addEventListener('change', ev => {
  const id = ev.target.id;
  if (id === 'fp-rho-mode') $('fp-rho').hidden = ev.target.value !== '';
  if (id === 'fp-mu-mode') $('fp-mu').hidden = ev.target.value !== '';
  if (id === 'fp-alpha') {   // rotate both axes
    const a = fnum('fp-alpha', 0) * Math.PI / 180, r = v => Math.round(v * 1e6) / 1e6;
    $('fp-a1x').value = r(Math.cos(a)); $('fp-a1y').value = r(Math.sin(a)); $('fp-a2x').value = r(-Math.sin(a)); $('fp-a2y').value = r(Math.cos(a));
  }
  if (ev.target.type === 'checkbox') return;   // boundary checkboxes schedule themselves
  scheduleForces(0);
});
$('fp-grid').addEventListener('keydown', ev => { if (ev.key === 'Enter' && ev.target.tagName === 'INPUT') { ev.preventDefault(); ev.target.blur(); } });
fpc.addEventListener('contextmenu', ev => {
  ev.preventDefault();
  const has = !!FC.result;
  openCtx([
    { header: has ? `${FC.result.boundaries.map(b => b.name).join(', ')} — ${FP_LABELS[FC.plot]} along the wall` : 'forces' },
    { label: 'Copy force table', disabled: !has, action: () => copyText(forceTableText(), 'Force table copied') },
    { label: 'Copy wall distributions (CSV)', disabled: !has, action: () => copyText(forceCSV(), 'Distributions copied') },
    { label: 'Download wall distributions (CSV)', disabled: !has, action: () => download(`semscope_${S.meta.name}_forces_step${S.step}.csv`, forceCSV(), 'text/csv') },
    '-',
    ...Object.entries(FP_LABELS).map(([k, lab]) => ({ label: `Plot ${lab}`, checked: FC.plot === k, action: () => { FC.plot = k; $('fp-plot').value = k; FC.hover = null; drawForceChart(); requestRender(); } })),
    '-',
    { label: 'Reset zoom', key: 'double-click', disabled: !FC.xview && !FC.yview, action: resetForceZoom },
    { label: 'Recompute', disabled: !FC.sel.size, action: () => scheduleForces(0) },
    { label: 'Hide this window (settings are kept)', key: 'f', action: () => setForceWindow(false) },
  ], ev.clientX, ev.clientY);
});
floatingPanel($('force-panel'), $('fp-head'), drawForceChart);

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
    if (lineLength(L) > 0) { pushHistory(`add line ${lineName(L)}`, before); setLineWindow(true); runLine(L); } else removeLine(L, false);
    ev.preventDefault();
    return;
  }
  const hit = (ev.shiftKey || BD.pick || NL.mode) ? null : hitLine(px, py);
  const before = snapshot();
  if (hit) {   // grab an end point or the body of an existing line
    const L = hit.line;
    S.drag = { kind: 'line-edit', line: L, part: hit.part, px, py, orig: { x0: L.x0, y0: L.y0, x1: L.x1, y1: L.y1 }, moved: false, before, allowed: null };
    if (L.kind === 'normal' && L.anchor) {
      const D = S.drag;
      D.allowed = allowedEdgesFor(L);
      if (!(BD.groups && BD.groups.angle === BD.angle)) ensureGroups().then(() => { if (S.drag === D) D.allowed = allowedEdgesFor(L); }).catch(() => {});
    }
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
  if (FC.drag) { fcDrag(ev); return; }
  if (FLOAT.drag) {
    const { p, dx, dy } = FLOAT.drag;
    p.style.left = (ev.clientX - dx) + 'px'; p.style.top = (ev.clientY - dy) + 'px';
    clampPanel(p); return;
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
    if (L.kind === 'normal' && L.anchor && BD.ext) {
      if (D.part === 1) setNormalLength(L, x, y);                    // far end: length only
      else { const h = boundaryPointAt(px, py, D.allowed, NL.snap, 60); if (h) setNormalOrigin(L, h); }   // origin: slide along its boundary
      showDragInfo(L);
      requestRender();
    } else if (D.part === 'body') {
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
  } else if (ev.target === glCanvas && NL.mode) {
    S.hover = [px, py];
    NL.hover = boundaryPointAt(px, py);
    $('probe').classList.toggle('muted', !NL.hover);
    $('probe').textContent = NL.hover ? `${NL.hover.node !== null ? `boundary node ${NL.hover.node}` : `boundary point t = ${fmt(NL.hover.t, 4)}`} of ${edgeLabel(NL.hover.edge)}\nnormal (${fmt(NL.hover.nx, 4)}, ${fmt(NL.hover.ny, 4)})\nclick: wall-normal line of length ${fmt(NL.length, 5)}` : (NL.snap ? 'hover a boundary GLL node' : 'hover the boundary');
    requestRender();
  } else if (ev.target === glCanvas && BD.pick) {
    S.hover = [px, py];
    const i = edgeAt(px, py);
    if (i !== BD.hoverEdge) { BD.hoverEdge = i; requestRender(); }
    $('probe').classList.toggle('muted', i < 0);
    $('probe').textContent = i >= 0 ? `external edge: ${edgeLabel(i)}
${BD.editing ? `click: add to / remove from ${BD.editing.name}` : 'click: start a new boundary'}
double-click: whole run up to the corners` : 'hover an external edge';
  } else if (ev.target === glCanvas) {
    S.hover = [px, py];
    LP.hoverHit = (ev.shiftKey) ? null : hitLine(px, py);
    glCanvas.style.cursor = LP.hoverHit ? (LP.hoverHit.part === 'body' ? 'move' : 'grab') : (S.lineMode ? 'crosshair' : '');
    if (!S.probePinned) showProbe(probeAt(px, py));
    requestRender();
  }
});
window.addEventListener('mouseup', ev => {
  LP.drag = null; FLOAT.drag = null; FC.drag = null;
  if (!S.drag || (S.drag.kind === 'line' && S.drag.fromMenu)) return;
  const d = S.drag; S.drag = null;
  if (d.kind === 'line') {
    LP.xview = null; LP.yview = null;
    if (lineLength(d.line) > 0) { pushHistory(`add line ${lineName(d.line)}`, d.before); setLineWindow(true); runLine(d.line); } else removeLine(d.line, false);
    setLineMode(false);
  } else if (d.kind === 'line-edit') {
    if (d.moved) { pushHistory(`move ${lineName(d.line)}`, d.before); LP.xview = null; LP.yview = null; runLine(d.line); } else { renderLegend(); drawLineChart(); }
    requestRender();
  } else if (d.kind === 'pan' && d.moved) {
    pushHistory('pan', d.before);
  } else if (d.kind === 'pan' && !d.moved && ev.target === glCanvas && NL.mode) {
    const rect = glCanvas.getBoundingClientRect();
    createNormalLine(boundaryPointAt(ev.clientX - rect.left, ev.clientY - rect.top));
  } else if (d.kind === 'pan' && !d.moved && ev.target === glCanvas && BD.pick) {
    const rect = glCanvas.getBoundingClientRect();
    pickClick(ev.clientX - rect.left, ev.clientY - rect.top, false);
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
glCanvas.addEventListener('dblclick', ev => {
  const rect = glCanvas.getBoundingClientRect();
  const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  if (BD.pick) { pickClick(px, py, true); return; }
  if (NL.mode) return;
  if (!hitLine(px, py)) resetViewUser();
});

// ----------------------------------------------------------------- boundaries
// External element edges come from the server once per dataset (as polylines);
// boundaries are sets of edge indices: auto-detected (chained + split at
// corners) or picked by hand on the plot.
const BD_COLORS = ['#f97316', '#22c55e', '#a855f7', '#ef4444', '#06b6d4', '#eab308', '#ec4899', '#84cc16'];
const BD = { ext: null, loading: null, angle: 90, groups: null, pick: false, editing: null, hoverEdge: -1, nextId: 1, clickTimer: null };
const bdColor = b => BD_COLORS[b.colorIdx % BD_COLORS.length];
const boundaryById = id => S.boundaries.find(b => b.id === id) || null;

async function loadBoundaryEdges() {
  if (BD.ext) return BD.ext;
  if (!BD.loading) {
    BD.loading = (async () => {
      const { buf } = await api('boundary/edges', { m: 16 });
      const head = new Int32Array(buf, 0, 3);
      const nb = head[0], m = head[1], n = head[2];
      let off = 12;
      const ids = new Int32Array(buf, off, nb * 2); off += nb * 8;
      const coords = new Float32Array(buf, off, nb * m * 2); off += nb * m * 8;
      const nodes = new Float32Array(buf, off, nb * n * 2); off += nb * n * 8;
      const normals = new Float32Array(buf, off, nb * n * 2);
      const index = new Map();
      const bbox = new Float32Array(nb * 4);
      for (let i = 0; i < nb; i++) {
        index.set(`${ids[2 * i]}:${ids[2 * i + 1]}`, i);
        let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
        for (let k = 0; k < m; k++) { const x = coords[(i * m + k) * 2], y = coords[(i * m + k) * 2 + 1]; if (x < a) a = x; if (x > b) b = x; if (y < c) c = y; if (y > d) d = y; }
        bbox[4 * i] = a; bbox[4 * i + 1] = b; bbox[4 * i + 2] = c; bbox[4 * i + 3] = d;
      }
      BD.ext = { nb, m, n, ids, coords, nodes, normals, index, bbox };
      return BD.ext;
    })();
  }
  try { return await BD.loading; } finally { BD.loading = null; }
}

/** Auto-detected grouping of the external edges at the current angle (used for double-click picking). */
async function ensureGroups() {
  await loadBoundaryEdges();
  if (BD.groups && BD.groups.angle === BD.angle) return BD.groups;
  const det = await api('boundary/detect', { angle: BD.angle });
  const edgeGroup = new Int32Array(BD.ext.nb).fill(-1);
  det.groups.forEach((g, gi) => { for (const e of g.edges) edgeGroup[e] = gi; });
  BD.groups = { angle: BD.angle, groups: det.groups, edgeGroup };
  return BD.groups;
}

function newBoundary(source, name = null) {
  const used = new Set(S.boundaries.map(b => b.colorIdx));
  let c = 0;
  while (used.has(c) && c < BD_COLORS.length) c++;
  const id = BD.nextId++;
  const b = { id, name: name || (source === 'manual' ? `M${id}` : `B${id}`), colorIdx: c, source, visible: true, edges: [], closed: false };
  S.boundaries.push(b);
  return b;
}

async function detectBoundaries() {
  if (!S.meta || !S.meta.open) return;
  try {
    const G = await ensureGroups();
    pushHistory('detect boundaries');
    S.boundaries = S.boundaries.filter(b => b.source !== 'auto');
    G.groups.forEach((g, gi) => { const b = newBoundary('auto', g.name); b.edges = g.edges.slice(); b.closed = g.closed; b.group = gi; });
    renderBoundaryList(); requestRender();
    toast(`${G.groups.length} boundar${G.groups.length === 1 ? 'y' : 'ies'} at a ${BD.angle}° feature angle`);
  } catch (err) { toast(err.message, true); }
}

function removeBoundary(b, record = true) {
  if (record) pushHistory(`delete boundary ${b.name}`);
  S.boundaries = S.boundaries.filter(o => o !== b);
  if (BD.editing === b) BD.editing = null;
  renderBoundaryList(); requestRender();
}

function setPickMode(on, boundary = null) {
  if (on && NL.mode) setNormalMode(false);
  BD.pick = on;
  BD.editing = on ? boundary : null;
  BD.hoverEdge = -1;
  document.body.classList.toggle('picking', on);
  $('bd-done').hidden = !on;
  $('bd-new').hidden = on;
  if (on) {
    loadBoundaryEdges().then(() => requestRender()).catch(err => toast(err.message, true));
    toast(boundary ? `Editing ${boundary.name}: click edges to add or remove them, double-click for a whole run, Done to finish` : 'Click external edges to build a boundary (double-click: whole run up to the corners); Done or Esc to finish');
  }
  renderBoundaryList(); requestRender();
}

const segDist = (px, py, ax, ay, bx, by) => {
  const l2 = (bx - ax) ** 2 + (by - ay) ** 2;
  const t = l2 ? clamp(((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2, 0, 1) : 0;
  return Math.hypot(px - (ax + t * (bx - ax)), py - (ay + t * (by - ay)));
};

/** Index of the external edge within maxDist screen px of the cursor, or -1. */
function edgeAt(px, py, maxDist = 8, allowed = null) {
  const E = BD.ext; if (!E) return -1;
  const [x, y] = screenToData(px, py);
  const tol = maxDist / S.view.scale;
  let best = -1, bestD = maxDist;
  for (let i = 0; i < E.nb; i++) {
    if (allowed && !allowed.has(i)) continue;
    if (x < E.bbox[4 * i] - tol || x > E.bbox[4 * i + 1] + tol || y < E.bbox[4 * i + 2] - tol || y > E.bbox[4 * i + 3] + tol) continue;
    for (let k = 0; k < E.m - 1; k++) {
      const o = (i * E.m + k) * 2;
      const [ax, ay] = dataToScreen(E.coords[o], E.coords[o + 1]);
      const [bx, by] = dataToScreen(E.coords[o + 2], E.coords[o + 3]);
      const d = segDist(px, py, ax, ay, bx, by);
      if (d < bestD) { bestD = d; best = i; }
    }
  }
  return best;
}

function boundaryAt(px, py) {
  if (!BD.ext || !S.showBoundaries) return null;
  const i = edgeAt(px, py, 7);
  if (i < 0) return null;
  const hits = S.boundaries.filter(b => b.visible && b.edges.includes(i));
  return hits.length ? hits[hits.length - 1] : null;
}

const edgeLabel = i => { const E = BD.ext; const e = E.ids[2 * i], sd = E.ids[2 * i + 1]; return `element ${S.mesh ? S.mesh.elmap[e] : e} (local ${e}), ${['bottom', 'right', 'top', 'left'][sd]} side`; };

async function toggleEdge(i, wholeRun = false) {
  if (i < 0 || !BD.ext) return;
  try {
    let idx = [i];
    if (wholeRun) { const G = await ensureGroups(); const gi = G.edgeGroup[i]; if (gi >= 0) idx = G.groups[gi].edges.slice(); }
    const before = snapshot();
    let b = BD.editing;
    if (!b) { b = newBoundary('manual'); BD.editing = b; }
    if (b.source === 'auto') b.source = 'manual';
    const set = new Set(b.edges);
    const adding = !set.has(i);
    for (const e of idx) { if (adding) set.add(e); else set.delete(e); }
    b.edges = [...set];
    pushHistory(`${adding ? 'add' : 'remove'} ${idx.length} edge${idx.length === 1 ? '' : 's'} (${b.name})`, before);
    renderBoundaryList(); requestRender();
  } catch (err) { toast(err.message, true); }
}

function pickClick(px, py, dbl) {
  // single clicks are delayed briefly so that a double-click toggles a whole run only once
  if (BD.clickTimer) { clearTimeout(BD.clickTimer); BD.clickTimer = null; }
  const i = edgeAt(px, py);
  if (dbl) { toggleEdge(i, true); return; }
  BD.clickTimer = setTimeout(() => { BD.clickTimer = null; toggleEdge(i, false); }, 260);
}

function renderBoundaryList() {
  const box = $('bd-list');
  box.innerHTML = '';
  for (const b of S.boundaries) {
    const row = document.createElement('div');
    row.className = 'bd-item' + (BD.editing === b ? ' active' : '') + (b.visible ? '' : ' hidden-b');
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = b.visible; cb.title = 'show / hide';
    cb.onchange = () => { pushHistory(`${cb.checked ? 'show' : 'hide'} boundary ${b.name}`); b.visible = cb.checked; renderBoundaryList(); requestRender(); };
    const sw = document.createElement('i'); sw.className = 'swatch'; sw.style.background = bdColor(b);
    const nm = document.createElement('input'); nm.className = 'name'; nm.value = b.name; nm.title = 'rename';
    nm.onchange = () => { const v = nm.value.trim(); if (v && v !== b.name) { pushHistory(`rename ${b.name}`); b.name = v; renderBoundaryList(); requestRender(); } else nm.value = b.name; };
    nm.onkeydown = ev => { if (ev.key === 'Enter') nm.blur(); ev.stopPropagation(); };
    const cnt = document.createElement('span'); cnt.className = 'count'; cnt.textContent = `${b.edges.length} edge${b.edges.length === 1 ? '' : 's'}${b.source === 'auto' ? '' : ' · manual'}`;
    const ed = document.createElement('button'); ed.className = 'edit'; ed.textContent = '✎'; ed.title = 'edit: pick edges on the plot';
    ed.onclick = () => setPickMode(!(BD.pick && BD.editing === b), b);
    const x = document.createElement('button'); x.className = 'x'; x.textContent = '✕'; x.title = 'delete';
    x.onclick = () => removeBoundary(b);
    row.append(cb, sw, nm, cnt, ed, x);
    box.appendChild(row);
  }
  renderForceBoundaries();
}

function drawBoundaries() {
  const E = BD.ext;
  if (!E) return;
  const light = isLight();
  const [vx0, vy0] = screenToData(0, H()), [vx1, vy1] = screenToData(W(), 0);
  const visible = i => !(E.bbox[4 * i + 1] < vx0 || E.bbox[4 * i] > vx1 || E.bbox[4 * i + 3] < vy0 || E.bbox[4 * i + 2] > vy1);
  const path = idxs => {
    octx.beginPath();
    for (const i of idxs) {
      if (!visible(i)) continue;
      for (let k = 0; k < E.m; k++) { const o = (i * E.m + k) * 2; const [sx, sy] = dataToScreen(E.coords[o], E.coords[o + 1]); if (k === 0) octx.moveTo(sx, sy); else octx.lineTo(sx, sy); }
    }
    octx.stroke();
  };
  octx.lineCap = 'round'; octx.lineJoin = 'round';
  if (BD.pick || NL.mode) {   // every external edge, faint, so that unassigned ones can be found
    octx.strokeStyle = light ? 'rgba(27,33,48,.35)' : 'rgba(255,255,255,.4)'; octx.lineWidth = 1.5;
    path(Array.from({ length: E.nb }, (_, i) => i));
  }
  if (S.showBoundaries) {
    for (const b of S.boundaries) {
      if (!b.visible || !b.edges.length) continue;
      octx.strokeStyle = bdColor(b); octx.lineWidth = BD.editing === b ? 4 : 2.5;
      path(b.edges);
      const mid = b.edges[Math.floor(b.edges.length / 2)];
      if (visible(mid)) {
        const o = (mid * E.m + Math.floor(E.m / 2)) * 2;
        const [lx, ly] = dataToScreen(E.coords[o], E.coords[o + 1]);
        octx.font = '600 11px -apple-system, Segoe UI, Inter, Roboto, sans-serif';
        octx.fillStyle = bdColor(b); octx.textAlign = 'left'; octx.textBaseline = 'bottom';
        octx.fillText(b.name, lx + 6, ly - 4);
        octx.font = '11px ui-monospace, Menlo, Consolas, monospace';
      }
    }
  }
  if (BD.pick && BD.hoverEdge >= 0) {
    octx.strokeStyle = light ? '#0b3d91' : '#5cc8ff'; octx.lineWidth = 5;
    path([BD.hoverEdge]);
  }
}

$('bd-detect').onclick = detectBoundaries;
$('bd-angle').onchange = ev => { BD.angle = clamp(parseFloat(ev.target.value) || 0, 0, 180); ev.target.value = BD.angle; };
$('bd-new').onclick = () => setPickMode(true, null);
$('bd-done').onclick = () => setPickMode(false);
$('bd-show').onchange = ev => { S.showBoundaries = ev.target.checked; requestRender(); };

// ----------------------------------------------------------------- wall-normal lines
// A line probe anchored at a boundary GLL node, following the inward normal.
const NL = { mode: false, hover: null, length: null, snap: false };

function defaultNormalLength() {
  if (!S.meta || !S.meta.open) return 1;
  const [x0, x1, y0, y1] = S.meta.bounds;
  const v = 0.1 * Math.min(x1 - x0, y1 - y0);
  return +v.toPrecision(3);
}
function setNormalMode(on) {
  if (on) { setPickMode(false); setLineMode(false); }
  NL.mode = on; NL.hover = null;
  document.body.classList.toggle('normal-mode', on);
  $('btn-normal').classList.toggle('active', on);
  if (on) {
    if (NL.length === null) { NL.length = defaultNormalLength(); $('nl-length').value = NL.length; }
    loadBoundaryEdges().then(() => requestRender()).catch(err => toast(err.message, true));
    toast('Hover a boundary node and click to create a wall-normal line; Esc or the button to finish');
  }
  requestRender();
}

/** Position and outward unit normal at parameter t (-1..1) along external edge i, from the spectral edge geometry. */
function edgePoint(i, t) {
  const E = BD.ext, n = E.n, sp = S.spectral;
  const Lb = new Float64Array(n), dLb = new Float64Array(n);
  basis(sp.nodes, sp.w, sp.D, t, Lb, dLb);
  let x = 0, y = 0, tx = 0, ty = 0;
  for (let k = 0; k < n; k++) { const o = (i * n + k) * 2; x += Lb[k] * E.nodes[o]; y += Lb[k] * E.nodes[o + 1]; tx += dLb[k] * E.nodes[o]; ty += dLb[k] * E.nodes[o + 1]; }
  let nx = ty, ny = -tx;
  const nn = Math.hypot(nx, ny) || 1; nx /= nn; ny /= nn;
  // orientation from the server-side normal at the nearest GLL node (handles mirrored elements)
  let k0 = 0; for (let k = 1; k < n; k++) if (Math.abs(sp.nodes[k] - t) < Math.abs(sp.nodes[k0] - t)) k0 = k;
  const o0 = (i * n + k0) * 2;
  if (nx * E.normals[o0] + ny * E.normals[o0 + 1] < 0) { nx = -nx; ny = -ny; }
  return { edge: i, t, node: null, x, y, nx, ny };
}

/** Closest point (in screen space) on external edge i to the cursor: coarse polyline search, then refinement in t. */
function closestOnEdge(i, px, py) {
  const E = BD.ext;
  let bestK = 0, bestD = Infinity;
  for (let k = 0; k < E.m; k++) { const o = (i * E.m + k) * 2; const [sx, sy] = dataToScreen(E.coords[o], E.coords[o + 1]); const d = Math.hypot(px - sx, py - sy); if (d < bestD) { bestD = d; bestK = k; } }
  let lo = -1 + 2 * Math.max(0, bestK - 1) / (E.m - 1), hi = -1 + 2 * Math.min(E.m - 1, bestK + 1) / (E.m - 1);
  let best = null, bestT = 0;
  for (let level = 0; level < 3; level++) {
    const N = 24; bestD = Infinity;
    for (let q = 0; q <= N; q++) {
      const t = lo + (hi - lo) * q / N;
      const pt = edgePoint(i, t);
      const [sx, sy] = dataToScreen(pt.x, pt.y);
      const d = Math.hypot(px - sx, py - sy);
      if (d < bestD) { bestD = d; bestT = t; best = pt; }
    }
    const h = (hi - lo) / N; lo = Math.max(-1, bestT - h); hi = Math.min(1, bestT + h);
  }
  return { ...best, d: bestD };
}

/** Nearest boundary GLL node within maxDist screen px (optionally restricted to a set of edges). */
function boundaryNodeAt(px, py, maxDist = 25, allowed = null) {
  const E = BD.ext; if (!E) return null;
  const [x, y] = screenToData(px, py);
  const tol = maxDist / S.view.scale;
  let best = null, bestD = maxDist;
  for (let i = 0; i < E.nb; i++) {
    if (allowed && !allowed.has(i)) continue;
    if (x < E.bbox[4 * i] - tol || x > E.bbox[4 * i + 1] + tol || y < E.bbox[4 * i + 2] - tol || y > E.bbox[4 * i + 3] + tol) continue;
    for (let k = 0; k < E.n; k++) {
      const o = (i * E.n + k) * 2;
      const [sx, sy] = dataToScreen(E.nodes[o], E.nodes[o + 1]);
      const d = Math.hypot(px - sx, py - sy);
      if (d < bestD) { bestD = d; best = { edge: i, node: k, t: S.spectral.nodes[k], x: E.nodes[o], y: E.nodes[o + 1], nx: E.normals[o], ny: E.normals[o + 1] }; }
    }
  }
  return best;
}

/** A point on the boundary near the cursor: a GLL node when snapping, otherwise anywhere on the edge curve. */
function boundaryPointAt(px, py, allowed = null, snap = NL.snap, maxDist = 25) {
  if (!BD.ext) return null;
  if (snap) return boundaryNodeAt(px, py, maxDist, allowed);
  const i = edgeAt(px, py, maxDist, allowed);
  return i < 0 ? null : closestOnEdge(i, px, py);
}

const edgeIndexOf = a => (BD.ext && a) ? (BD.ext.index.get(`${a.elem}:${a.side}`) ?? -1) : -1;
const normalLineEnd = (h, len = NL.length) => [h.x - len * h.nx, h.y - len * h.ny];

/** Apply a boundary point as the origin of a wall-normal line, keeping its length. */
function setNormalOrigin(L, h) {
  const E = BD.ext;
  const len = L.anchor ? L.anchor.length : NL.length;
  L.anchor = { elem: E.ids[2 * h.edge], side: E.ids[2 * h.edge + 1], node: h.node, t: h.t, nx: h.nx, ny: h.ny, length: len };
  L.x0 = h.x; L.y0 = h.y;
  [L.x1, L.y1] = normalLineEnd(h, len);
}
/** Change only the length of a wall-normal line so that its end is the projection of (x, y) onto the normal. */
function setNormalLength(L, x, y) {
  const a = L.anchor;
  const len = -((x - L.x0) * a.nx + (y - L.y0) * a.ny);
  L.anchor = { ...a, length: len };
  L.x1 = L.x0 - len * a.nx; L.y1 = L.y0 - len * a.ny;
}
/** Edges a wall-normal origin may slide along: the connected boundary run of its edge (same group at the current angle). */
function allowedEdgesFor(L) {
  const i = edgeIndexOf(L.anchor);
  if (i < 0) return null;
  if (BD.groups && BD.groups.angle === BD.angle) {
    const gi = BD.groups.edgeGroup[i];
    if (gi >= 0) return new Set(BD.groups.groups[gi].edges);
  }
  return new Set([i]);
}

function createNormalLine(h) {
  if (!h || !Number.isFinite(NL.length) || NL.length === 0) { toast('Set a non-zero length first', true); return; }
  const before = snapshot();
  const L = newLine(h.x, h.y);
  L.kind = 'normal';
  setNormalOrigin(L, h);
  pushHistory(`add wall-normal line ${lineName(L)}`, before);
  LP.xview = null; LP.yview = null;
  setLineWindow(true);
  renderLegend(); runLine(L); requestRender();
}

function drawNormalPreview() {
  const E = BD.ext; if (!E || !NL.mode) return;
  const light = isLight();
  const [vx0, vy0] = screenToData(0, H()), [vx1, vy1] = screenToData(W(), 0);
  // boundary GLL nodes of the visible edges (only when snapping, and when they are resolvable on screen)
  octx.fillStyle = light ? 'rgba(11,61,145,.6)' : 'rgba(92,200,255,.7)';
  for (let i = 0; NL.snap && i < E.nb; i++) {
    if (E.bbox[4 * i + 1] < vx0 || E.bbox[4 * i] > vx1 || E.bbox[4 * i + 3] < vy0 || E.bbox[4 * i + 2] > vy1) continue;
    const size = Math.max(E.bbox[4 * i + 1] - E.bbox[4 * i], E.bbox[4 * i + 3] - E.bbox[4 * i + 2]) * S.view.scale;
    if (size < 24) continue;
    for (let k = 0; k < E.n; k++) { const o = (i * E.n + k) * 2; const [sx, sy] = dataToScreen(E.nodes[o], E.nodes[o + 1]); octx.beginPath(); octx.arc(sx, sy, 2, 0, 2 * Math.PI); octx.fill(); }
  }
  const h = NL.hover;
  if (!h) return;
  const [ax, ay] = dataToScreen(h.x, h.y), [bx, by] = dataToScreen(...normalLineEnd(h));
  const col = light ? '#c2410c' : '#ffd166';
  octx.strokeStyle = col; octx.lineWidth = 1.5; octx.setLineDash([5, 4]);
  octx.beginPath(); octx.moveTo(ax, ay); octx.lineTo(bx, by); octx.stroke();
  octx.setLineDash([]);
  octx.fillStyle = col; octx.strokeStyle = light ? '#fff' : '#0d0f14'; octx.lineWidth = 1.2;
  octx.beginPath(); octx.arc(ax, ay, 5, 0, 2 * Math.PI); octx.fill(); octx.stroke();
  octx.beginPath(); octx.arc(bx, by, 3, 0, 2 * Math.PI); octx.fill();
}

$('btn-normal').onclick = () => setNormalMode(!NL.mode);
$('nl-length').onchange = ev => { const v = parseFloat(ev.target.value); if (Number.isFinite(v)) { NL.length = v; requestRender(); } else ev.target.value = NL.length ?? ''; };
$('nl-snap').onchange = ev => { NL.snap = ev.target.checked; NL.hover = null; requestRender(); };

// ----------------------------------------------------------------- collapsible sidebar sections
(function makeCollapsible() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('semscope.collapsed') || '{}'); } catch { /* ignore */ }
  for (const panel of document.querySelectorAll('#sidebar .panel')) {
    const h2 = panel.querySelector('h2');
    if (!h2) continue;
    const body = document.createElement('div'); body.className = 'panel-body';
    while (h2.nextSibling) body.appendChild(h2.nextSibling);
    panel.appendChild(body);
    const chev = document.createElement('span'); chev.className = 'chev'; chev.textContent = '▾';
    h2.appendChild(chev);
    const key = panel.dataset.panel || h2.textContent.trim();
    panel.classList.toggle('collapsed', saved[key] === true);
    h2.title = 'click to collapse / expand';
    h2.addEventListener('click', () => {
      panel.classList.toggle('collapsed');
      saved[key] = panel.classList.contains('collapsed');
      try { localStorage.setItem('semscope.collapsed', JSON.stringify(saved)); } catch { /* ignore */ }
    });
  }
})();

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
      rows.push(`${lineName(L)},${d.distance[i]},${d.x[i]},${d.y[i]},${gid},${vals[i]}`);
    }
  }
  return rows.join('\n') + '\n';
}

/** A Python snippet that reproduces the current view with the scripting API. */
function pythonSnippet() {
  const [x0, y0] = screenToData(0, H()), [x1, y1] = screenToData(W(), 0);
  let cmap = S.cmap;
  if (S.invert) cmap = cmap.endsWith('_r') ? cmap.slice(0, -2) : cmap + '_r';
  const field = S.field;
  const lines = [
    'import semscope',
    '',
    `data = semscope.load(${JSON.stringify(S.meta.path)}, step=${S.step})`,
  ];
  for (const d of calcDefs()) lines.push(`data.define(${JSON.stringify(d.name)}, ${JSON.stringify(d.expr)})`);
  lines.push(
    'pl = semscope.Plotter(figsize=(9, 6))',
    `pl.add_field(data, ${JSON.stringify(field)}, cmap=${JSON.stringify(cmap)}, clim=(${fmt(S.range.lo, 6)}, ${fmt(S.range.hi, 6)})${S.mode === 1 ? ', method="nodal"' : ''})`,
  );
  if (S.contours) lines.push(`pl.add_contours(data, ${JSON.stringify(field)}, levels=${S.nContours}, colors="w", linewidths=0.5)`);
  if (S.edges) lines.push('pl.add_mesh(data, color="k", linewidth=0.3)');
  if (S.nodes) lines.push('pl.add_nodes(data)');
  const bvar = b => 'b_' + b.name.replace(/\W+/g, '_');
  const wanted = BD.ext ? S.boundaries.filter(b => b.edges.length && (b.visible || FC.sel.has(b.id))) : [];
  if (wanted.some(b => b.source === 'auto')) lines.push(`boundaries = data.detect_boundaries(angle=${BD.angle})`);
  for (const b of wanted) {
    if (b.source === 'auto') lines.push(b.group !== undefined && b.group !== null ? `${bvar(b)} = boundaries[${b.group}]   # ${b.name}` : `${bvar(b)} = next(b for b in boundaries if b.name == ${JSON.stringify(b.name)})`);
    else lines.push(`${bvar(b)} = data.boundary([${b.edges.map(i => `(${BD.ext.ids[2 * i]}, ${BD.ext.ids[2 * i + 1]})`).join(', ')}], name=${JSON.stringify(b.name)})`);
  }
  const shown = wanted.filter(b => b.visible);
  if (shown.length) lines.push(`pl.add_boundaries(data, [${shown.map(bvar).join(', ')}])`);
  const forced = wanted.filter(b => FC.sel.has(b.id));
  if (forced.length) {
    const c = forceConfig(), lit = v => (typeof v === 'string' ? JSON.stringify(v) : String(v));
    const kw = [`u=${lit(c.u)}`, `v=${lit(c.v)}`, c.w ? `w=${lit(c.w)}` : null, `p=${lit(c.p)}`, `rho=${lit(c.rho)}`, `mu=${lit(c.mu)}`,
      `axes={${c.axes.map(a => `${JSON.stringify(a.name)}: (${a.dir[0]}, ${a.dir[1]})`).join(', ')}}`,
      `U_ref=${c.U_ref}`, `L_ref=${c.L_ref}`, `p_ref=${c.p_ref}`, c.rho_ref !== null ? `rho_ref=${c.rho_ref}` : null].filter(Boolean).join(', ');
    for (const b of forced) lines.push(`F_${bvar(b).slice(2)} = data.forces(${bvar(b)}, ${kw})`, `print(F_${bvar(b).slice(2)})   # pressure / viscous parts, coefficients; .distribution has Cp, Cf along the wall`);
  }
  for (const L of S.lines) {
    if (L.kind === 'normal' && L.anchor) {
      if (L.anchor.node !== null && L.anchor.node !== undefined) lines.push(`p0_${lineName(L)}, p1_${lineName(L)} = data.normal_line(${L.anchor.elem}, ${L.anchor.side}, ${L.anchor.node}, ${fmt(L.anchor.length, 7)})   # wall-normal from a boundary GLL node`);
      else lines.push(`p0_${lineName(L)}, p1_${lineName(L)} = data.normal_line_at(${L.anchor.elem}, ${L.anchor.side}, ${fmt(L.anchor.t, 7)}, ${fmt(L.anchor.length, 7)})   # wall-normal from edge parameter t`);
      lines.push(`dist_${lineName(L)}, vals_${lineName(L)} = data.sample_line(${JSON.stringify(field)}, p0_${lineName(L)}, p1_${lineName(L)}, 1000)`);
    } else {
      lines.push(`dist_${lineName(L)}, vals_${lineName(L)} = data.sample_line(${JSON.stringify(field)}, (${fmt(L.x0, 7)}, ${fmt(L.y0, 7)}), (${fmt(L.x1, 7)}, ${fmt(L.y1, 7)}), 1000)`);
    }
  }
  if (S.probePinned) lines.push(`probe = data.sample(${JSON.stringify(field)}, ${fmt(S.probePinned.x, 7)}, ${fmt(S.probePinned.y, 7)})`);
  lines.push(`pl.set_view((${fmt(x0, 7)}, ${fmt(x1, 7)}), (${fmt(y0, 7)}, ${fmt(y1, 7)}))`);
  lines.push(`pl.set_title(${JSON.stringify(`${S.meta.name}: ${S.field}`)})`);
  lines.push('pl.save("semscope_view.png")');
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
function closeCtx() { ctxEl.hidden = true; ctxEl.innerHTML = ''; ctxEl.dataset.menu = ''; for (const b of document.querySelectorAll('.menu-btn')) b.classList.remove('open'); }
function openCtx(items, clientX, clientY) {
  ctxEl.innerHTML = '';
  for (const it of items) {
    if (it === '-') { const d = document.createElement('div'); d.className = 'sep'; ctxEl.appendChild(d); continue; }
    if (it.header) { const d = document.createElement('div'); d.className = 'hd'; d.textContent = it.header; ctxEl.appendChild(d); continue; }
    const d = document.createElement('div');
    d.className = 'it' + (it.disabled ? ' disabled' : '');
    const label = document.createElement('span');
    if (it.checked !== undefined) { const chk = document.createElement('span'); chk.className = 'chk'; chk.textContent = it.checked ? '✓' : ''; label.appendChild(chk); }
    label.appendChild(document.createTextNode(it.label));
    d.appendChild(label);
    if (it.key) { const k = document.createElement('span'); k.className = 'key'; k.textContent = it.key; d.appendChild(k); }
    d.onclick = () => { closeCtx(); it.action(); };
    ctxEl.appendChild(d);
  }
  ctxEl.hidden = false;
  const v = $('app').getBoundingClientRect();
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
    items.push({ header: `line ${lineName(L)}: (${fmt(L.x0, 5)}, ${fmt(L.y0, 5)}) → (${fmt(L.x1, 5)}, ${fmt(L.y1, 5)})` });
    items.push({ label: L.visible ? `Hide ${lineName(L)} in the chart` : `Show ${lineName(L)} in the chart`, action: () => { pushHistory(`${L.visible ? 'hide' : 'show'} ${lineName(L)}`); L.visible = !L.visible; renderLegend(); showLinePanel(); drawLineChart(); requestRender(); } });
    items.push({ label: `Copy ${lineName(L)} samples (CSV)`, disabled: !L.data, action: () => copyText(lineCSV([L]), `${lineName(L)} samples copied`) });
    items.push({ label: `Download ${lineName(L)} samples (CSV)`, disabled: !L.data, action: () => download(`semscope_${S.meta.name}_${S.field}_${lineName(L)}.csv`, lineCSV([L]), 'text/csv') });
    items.push({ label: `Copy end points of ${lineName(L)}`, action: () => copyText(`${L.x0}\t${L.y0}\n${L.x1}\t${L.y1}`, 'End points copied') });
    items.push({ label: `Delete ${lineName(L)}`, key: 'Del', action: () => removeLine(L) });
  }
  const bhit = boundaryAt(px, py);
  if (bhit) {
    items.push('-');
    items.push({ header: `boundary ${bhit.name}: ${bhit.edges.length} edge${bhit.edges.length === 1 ? '' : 's'}${bhit.source === 'auto' ? ' (detected)' : ' (manual)'}` });
    items.push({ label: `Edit ${bhit.name} (pick edges)`, action: () => setPickMode(true, bhit) });
    items.push({ label: bhit.visible ? `Hide ${bhit.name}` : `Show ${bhit.name}`, action: () => { pushHistory(`${bhit.visible ? 'hide' : 'show'} boundary ${bhit.name}`); bhit.visible = !bhit.visible; renderBoundaryList(); requestRender(); } });
    items.push({ label: `Copy edges of ${bhit.name} (element, side)`, action: () => copyText('element\tside\n' + bhit.edges.map(i => `${S.mesh.elmap[BD.ext.ids[2 * i]]}\t${['bottom', 'right', 'top', 'left'][BD.ext.ids[2 * i + 1]]}`).join('\n'), 'Boundary edges copied') });
    items.push({ label: `Delete ${bhit.name}`, action: () => removeBoundary(bhit) });
  }
  if (BD.pick) { items.push('-'); items.push({ label: 'Finish picking edges', key: 'Esc', action: () => setPickMode(false) }); }
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
    { label: 'Download samples of visible lines (CSV)', disabled: !vis.length, action: () => download(`semscope_${S.meta.name}_${S.field}_lines.csv`, lineCSV(vis), 'text/csv') },
    '-',
    { label: 'Reset zoom', key: 'double-click', action: resetChartZoom },
    { label: 'Hide this window (lines are kept)', key: 'g', action: () => setLineWindow(false) },
    { label: 'Delete all lines', disabled: !S.lines.length, action: clearLines },
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
  if (FC.windowOpen && FC.sel.size) scheduleForces();
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
$('bg').onchange = ev => { S.bg = ev.target.value; document.body.classList.toggle('light', isLight()); if (S.lines.length) { renderLegend(); drawLineChart(); } if (FC.windowOpen) drawForceChart(); requestRender(); };
$('step-slider').oninput = ev => setStep(+ev.target.value);
$('btn-play').onclick = () => setPlaying(!S.playing);
$('btn-first').onclick = () => setStep(0);
$('btn-last').onclick = () => setStep(S.meta.nsteps - 1);
$('btn-prev').onclick = () => setStep(S.step - 1);
$('btn-next').onclick = () => setStep(S.step + 1);
$('fps').onchange = () => { if (S.playing) setPlaying(true); };
function setLineMode(on) {
  if (on) { setPickMode(false); if (NL.mode) setNormalMode(false); }
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
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT' || ev.target.tagName === 'TEXTAREA') return;
  if (!$('dialog').hidden) { if (ev.key === 'Escape') closeDialog(); return; }
  if (!$('save-dialog').hidden) { if (ev.key === 'Escape') closeSaveDialog(); return; }
  if (!$('help-dialog').hidden) { if (ev.key === 'Escape') $('help-dialog').hidden = true; return; }
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
    case 'S': if (ev.ctrlKey || ev.metaKey) { ev.preventDefault(); openSaveDialog(); } else return; break;
    case 'y': case 'Y': if (ev.ctrlKey || ev.metaKey) { ev.preventDefault(); redo(); } else return; break;
    case 's': if (ev.ctrlKey || ev.metaKey) { ev.preventDefault(); openSaveDialog(); } else screenshot(); break;
    case 'o': openDialog(); break;
    case 'l': setLineMode(!S.lineMode); break;
    case 'w': setNormalMode(!NL.mode); break;
    case 'g': setLineWindow(!LP.windowOpen); break;
    case 'k': setCalcWindow(!CP.windowOpen); break;
    case 'f': setForceWindow(!FC.windowOpen); break;
    case 'b': toggleSidebar(); break;
    case 'Escape':
      if (!ctxEl.hidden) { closeCtx(); break; }
      if (S.drag && S.drag.kind === 'line') { removeLine(S.drag.line, false); S.drag = null; glCanvas.style.cursor = ''; break; }
      if (BD.pick) { setPickMode(false); break; }
      if (NL.mode) { setNormalMode(false); break; }
      if (S.lineMode) { setLineMode(false); break; }
      if (S.probePinned) { pushHistory('unpin probe'); S.probePinned = null; updateProbe(); requestRender(); }
      break;
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
    a.download = `semscope_${S.meta ? S.meta.name : 'view'}_${S.field}_step${S.step}.png`;
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
      const tag = document.createElement('span'); tag.className = 'tag'; tag.textContent = e.type === 'dir' ? 'dir' : e.type === 'meta' ? 'series' : e.type === 'session' ? 'session' : 'field';
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
$('dlg-go').onclick = () => { const p = $('dlg-path').value.trim(); if (/\.nek5000$|\d\.f\d{5}$|\.sem(scope|view)\.json$/.test(p)) openPath(p); else browse(p); };
$('dlg-path').addEventListener('keydown', ev => { if (ev.key === 'Enter') $('dlg-go').click(); });

async function openPath(path, opts = {}) {
  closeDialog();
  if (isSessionFile(path)) return loadSessionPath(path);
  toast(`Opening ${path.split('/').pop()} …`);
  try {
    setPlaying(false);
    S.probePinned = null; clearLines(false); clearHistory();
    setPickMode(false); setNormalMode(false); NL.length = null; $('nl-length').value = ''; S.boundaries = []; BD.ext = null; BD.groups = null; resetForceWindow(); renderBoundaryList();
    S.meta = await api('open', { path });
    S.fieldCache.clear();
    await loadMesh();
    populateFields();
    setupTime();
    fitView();
    await loadField();
    if (!opts.quiet) toast(`Loaded ${S.meta.name}: ${S.meta.nelv.toLocaleString()} elements, N = ${S.meta.order}`);
  } catch (err) { toast(err.message, true); }
}

// ----------------------------------------------------------------- sessions
const SESSION_SUFFIX = '.semscope.json';
const isSessionFile = p => p.endsWith(SESSION_SUFFIX) || p.endsWith('.semview.json');   // older files keep working

function sessionState() {
  const [x0, y0] = screenToData(0, H()), [x1, y1] = screenToData(W(), 0);
  const p = $('line-panel');
  return {
    semscope_session: 1,
    saved: new Date().toISOString(),
    dataset: { path: S.meta.path, step: S.step },
    field: { name: S.field, cmap: S.cmap, invert: S.invert, range: { ...S.range } },
    render: { mode: S.mode, edges: S.edges, edgeWidth: S.edgeWidth, contours: S.contours, nContours: S.nContours, nodes: S.nodes, nodeSize: S.nodeSize, axes: S.axes, colorbar: S.colorbar, pxPerCell: S.pxPerCell, bg: S.bg },
    view: { cx: S.view.cx, cy: S.view.cy, scale: S.view.scale, xlim: [x0, x1], ylim: [y0, y1] },
    probe: { pinned: S.probePinned ? { ...S.probePinned } : null, snapAngle: S.snapAngle },
    lines: S.lines.map(L => ({ id: L.id, colorIdx: L.colorIdx, kind: L.kind || 'line', anchor: L.anchor || null, x0: L.x0, y0: L.y0, x1: L.x1, y1: L.y1, visible: L.visible })),
    normalLength: NL.length,
    normalSnap: NL.snap,
    activeLine: LP.active,
    boundaries: BD.ext ? S.boundaries.map(b => ({ name: b.name, colorIdx: b.colorIdx, source: b.source, group: b.group, visible: b.visible, closed: !!b.closed, edges: b.edges.map(i => [BD.ext.ids[2 * i], BD.ext.ids[2 * i + 1]]) })) : [],
    boundaryAngle: BD.angle,
    showBoundaries: S.showBoundaries,
    chart: { open: !p.hidden, ...panelGeometry(p), grid: LP.showGrid, elem: LP.showElem, xview: LP.xview, yview: LP.yview },
    calc: { defs: calcDefs().map(d => ({ name: d.name, expr: d.expr })), open: CP.windowOpen, ...panelGeometry($('calc-panel')) },
    forces: { open: FC.windowOpen, ...panelGeometry($('force-panel')), selection: [...FC.sel].map(id => (boundaryById(id) || {}).name).filter(Boolean), config: forceConfig(), plot: FC.plot, xview: FC.xview, yview: FC.yview },
  };
}

const setCheck = (id, v) => { const el = $(id); if (el && v !== undefined) el.checked = !!v; };
const setVal = (id, v) => { const el = $(id); if (el && v !== undefined && v !== null) el.value = v; };

/** Apply a session; opens its dataset first when it differs from the current one. */
async function applySession(sess) {
  if (!sess || !sess.dataset) { toast('Not a semscope session', true); return; }
  if (!S.meta || !S.meta.open || S.meta.path !== sess.dataset.path) {
    await openPath(sess.dataset.path, { quiet: true });
    if (!S.meta || S.meta.path !== sess.dataset.path) return;   // open failed (toast shown)
  }
  setPlaying(false);
  const f = sess.field || {}, r = sess.render || {}, v = sess.view || {}, pr = sess.probe || {}, ch = sess.chart || {};
  // rendering options
  if (r.bg && BG[r.bg]) { S.bg = r.bg; setVal('bg', r.bg); document.body.classList.toggle('light', isLight()); }
  if (r.mode !== undefined) { S.mode = +r.mode; for (const o of $('mode').querySelectorAll('button')) o.classList.toggle('on', +o.dataset.mode === S.mode); }
  for (const [k, id] of [['edges', 'show-edges'], ['contours', 'show-contours'], ['nodes', 'show-nodes'], ['axes', 'show-axes'], ['colorbar', 'show-colorbar']]) if (r[k] !== undefined) { S[k] = !!r[k]; setCheck(id, r[k]); }
  for (const [k, id] of [['edgeWidth', 'edge-width'], ['nContours', 'n-contours'], ['nodeSize', 'node-size'], ['pxPerCell', 'quality']]) if (r[k] !== undefined) { S[k] = +r[k]; setVal(id, r[k]); }
  // field, colormap, range
  if (f.cmap && S.colormaps[f.cmap]) { S.cmap = f.cmap; setVal('cmap', f.cmap); R.setColormap(S.colormaps[S.cmap]); }
  if (f.invert !== undefined) { S.invert = !!f.invert; setCheck('cmap-invert', f.invert); }
  if (f.range) { S.range = { ...S.range, ...f.range }; setCheck('range-auto', S.range.auto); setCheck('range-sym', S.range.sym); }
  S.step = clamp(+(sess.dataset.step || 0), 0, S.meta.nsteps - 1);
  $('step-slider').value = S.step;
  // field-calculator definitions (before the field is chosen: it may be one of them)
  const ca = sess.calc || {};
  await syncCalcDefs(ca.defs || []);
  applyPanelGeometry($('calc-panel'), ca);
  setCalcWindow(ca.open === true);
  if (f.name && S.meta.available.includes(f.name)) { S.field = f.name; setVal('field', f.name); }
  await loadField();
  if (f.range && !S.range.auto) setRange(f.range.lo, f.range.hi, false);
  // view: refit the stored extent to the current canvas
  if (v.xlim && v.ylim && W() > 1 && H() > 1) {
    const sx = W() / Math.max(v.xlim[1] - v.xlim[0], 1e-300), sy = H() / Math.max(v.ylim[1] - v.ylim[0], 1e-300);
    S.view = { cx: 0.5 * (v.xlim[0] + v.xlim[1]), cy: 0.5 * (v.ylim[0] + v.ylim[1]), scale: Math.min(sx, sy) };
  } else if (v.cx !== undefined) S.view = { cx: v.cx, cy: v.cy, scale: v.scale };
  // probes and lines
  S.probePinned = pr.pinned ? { x: pr.pinned.x, y: pr.pinned.y } : null;
  if (pr.snapAngle !== undefined) { S.snapAngle = !!pr.snapAngle; setCheck('snap-angle', pr.snapAngle); }
  S.lines = (sess.lines || []).map((L, i) => ({ id: L.id || i + 1, colorIdx: L.colorIdx ?? i, kind: L.kind || 'line', anchor: L.anchor || null, x0: L.x0, y0: L.y0, x1: L.x1, y1: L.y1, visible: L.visible !== false, data: null, field: null, step: null, hoverIdx: null, token: 0 }));
  if (sess.normalLength !== undefined && sess.normalLength !== null) { NL.length = +sess.normalLength; setVal('nl-length', NL.length); }
  if (sess.normalSnap !== undefined) { NL.snap = !!sess.normalSnap; setCheck('nl-snap', NL.snap); }
  LP.nextId = Math.max(1, ...S.lines.map(L => L.id + 1));
  LP.active = S.lines.some(L => L.id === sess.activeLine) ? sess.activeLine : (S.lines.length ? S.lines[S.lines.length - 1].id : null);
  LP.showGrid = ch.grid !== false; setCheck('lp-grid', LP.showGrid);
  LP.showElem = ch.elem !== false; setCheck('lp-elem', LP.showElem);
  LP.xview = ch.xview || null; LP.yview = ch.yview || null;
  applyPanelGeometry($('line-panel'), ch);
  renderLegend();
  setLineWindow(ch.open === true || (ch.open === undefined && S.lines.length > 0));
  for (const L of S.lines) runLine(L);
  // boundaries
  setPickMode(false);
  if (sess.boundaryAngle !== undefined) { BD.angle = +sess.boundaryAngle; setVal('bd-angle', BD.angle); }
  if (sess.showBoundaries !== undefined) { S.showBoundaries = !!sess.showBoundaries; setCheck('bd-show', S.showBoundaries); }
  S.boundaries = [];
  if (sess.boundaries && sess.boundaries.length) {
    try {
      const E = await loadBoundaryEdges();
      for (const sb of sess.boundaries) {
        const b = newBoundary(sb.source || 'manual', sb.name);
        if (sb.colorIdx !== undefined) b.colorIdx = sb.colorIdx;
        b.visible = sb.visible !== false; b.closed = !!sb.closed;
        if (sb.group !== undefined && sb.group !== null) b.group = sb.group;
        b.edges = (sb.edges || []).map(([e, sd]) => E.index.get(`${e}:${sd}`)).filter(i => i !== undefined);
      }
    } catch (err) { toast('Boundaries not restored: ' + err.message, true); }
  }
  renderBoundaryList();
  // forces window: selection by boundary name, settings, geometry
  const fo = sess.forces || {};
  FC.result = null; FC.hover = null;
  FC.sel = new Set(S.boundaries.filter(b => (fo.selection || []).includes(b.name)).map(b => b.id));
  if (fo.config) applyForceConfig(fo.config);
  if (fo.plot && FP_LABELS[fo.plot]) { FC.plot = fo.plot; setVal('fp-plot', fo.plot); }
  FC.xview = fo.xview || null; FC.yview = fo.yview || null;
  applyPanelGeometry($('force-panel'), fo);
  renderForceBoundaries();
  setForceWindow(fo.open === true);
  clearHistory();
  updateProbe();
  requestRender();
  toast(`Session restored: ${S.meta.name}, ${S.field}` + (S.lines.length ? `, ${S.lines.length} line${S.lines.length === 1 ? '' : 's'}` : ''));
}

function defaultSessionPath() {
  if (!S.meta || !S.meta.open) return '';
  const dir = S.meta.path.split('/').slice(0, -1).join('/');
  return `${dir}/${S.meta.name}${SESSION_SUFFIX}`;
}
function openSaveDialog() {
  if (!S.meta || !S.meta.open) { toast('Open a dataset first', true); return; }
  $('save-dialog').hidden = false;
  if (!$('sdlg-path').value) $('sdlg-path').value = defaultSessionPath();
  $('sdlg-path').focus();
}
const closeSaveDialog = () => { $('save-dialog').hidden = true; };
async function saveSessionTo(path, overwrite = false) {
  const res = await fetch(`/api/session/save?path=${encodeURIComponent(path)}&overwrite=${overwrite ? 1 : 0}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(sessionState()) });
  const out = await res.json();
  if (res.status === 409 && out.exists) {
    if (confirm(`${path} exists. Overwrite?`)) return saveSessionTo(path, true);
    return;
  }
  if (!res.ok) throw new Error(out.error || res.statusText);
  closeSaveDialog();
  toast(`Session saved: ${out.path}`);
}
$('btn-save-session').onclick = openSaveDialog;
$('sdlg-close').onclick = closeSaveDialog;
$('save-dialog').addEventListener('click', ev => { if (ev.target === $('save-dialog')) closeSaveDialog(); });
$('sdlg-save').onclick = () => { const p = $('sdlg-path').value.trim(); if (!p) return; saveSessionTo(p).catch(err => toast(err.message, true)); };
$('sdlg-path').addEventListener('keydown', ev => { if (ev.key === 'Enter') $('sdlg-save').click(); if (ev.key === 'Escape') closeSaveDialog(); });
$('sdlg-download').onclick = () => { download(`${S.meta.name}${SESSION_SUFFIX}`, JSON.stringify(sessionState(), null, 2), 'application/json'); closeSaveDialog(); };
$('dlg-import').addEventListener('change', async ev => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = '';
  if (!file) return;
  try {
    const sess = JSON.parse(await file.text());
    if (!sess.semscope_session && !sess.semview_session) throw new Error(`${file.name} is not a semscope session`);
    closeDialog();
    await applySession(sess);
  } catch (err) { toast(err.message, true); }
});
async function loadSessionPath(path) {
  closeDialog();
  toast(`Loading session ${path.split('/').pop()} …`);
  try {
    setPlaying(false);
    S.probePinned = null; clearLines(false); clearHistory();
    setPickMode(false); setNormalMode(false); S.boundaries = []; BD.groups = null; resetForceWindow(); renderBoundaryList();
    const meta = await api('session/load', { path });
    if (!(S.meta && S.meta.open && S.meta.path === meta.path)) BD.ext = null;
    const sameDataset = S.meta && S.meta.open && S.meta.path === meta.path;
    S.meta = meta;
    if (!sameDataset) { S.fieldCache.clear(); await loadMesh(); populateFields(); setupTime(); fitView(); }
    await applySession(meta.session);
  } catch (err) { toast(err.message, true); }
}

// ----------------------------------------------------------------- top menu
function toggleSidebar() { $('app').classList.toggle('no-sidebar'); requestRender(); }
const SHORTCUTS = [
  ['wheel / drag', 'zoom / pan'], ['double-click', 'reset view'], ['right-click', 'context menu'],
  ['hover / click', 'probe / pin probe'], ['Shift-drag or l', 'line probe'], ['w', 'wall-normal line'],
  ['Ctrl while dragging', 'snap line angle to 10°'], ['Del', 'delete selected line'], ['g', 'line chart window'], ['k', 'field calculator window'], ['f', 'forces window'],
  ['b', 'sidebar'], ['Space, ← →, Home, End', 'time steps'], ['1 / 2', 'spectral / nodal rendering'],
  ['m, c, n', 'element edges, iso-lines, GLL nodes'], ['r', 'reset view'], ['s', 'screenshot'],
  ['o', 'open dataset or session'], ['Ctrl+S', 'save session'], ['Ctrl+Z / Ctrl+Shift+Z', 'undo / redo'], ['Esc', 'leave a mode / unpin probe'],
];
function showHelp() {
  const w = Math.max(...SHORTCUTS.map(([k]) => k.length));
  $('help-body').textContent = SHORTCUTS.map(([k, v]) => `${k.padEnd(w + 2)}${v}`).join('\n');
  $('help-dialog').hidden = false;
}
$('help-close').onclick = () => { $('help-dialog').hidden = true; };
$('help-dialog').addEventListener('click', ev => { if (ev.target === $('help-dialog')) $('help-dialog').hidden = true; });

function menuItems(name) {
  const open = !!(S.meta && S.meta.open);
  if (name === 'file') return [
    { label: 'Open dataset or session…', key: 'o', action: openDialog },
    { label: 'Reload dataset', disabled: !open, action: () => openPath(S.meta.path) },
    '-',
    { label: 'Save session…', key: 'Ctrl+S', disabled: !open, action: openSaveDialog },
    { label: 'Download session', disabled: !open, action: () => download(`${S.meta.name}${SESSION_SUFFIX}`, JSON.stringify(sessionState(), null, 2), 'application/json') },
    '-',
    { label: 'Save screenshot (PNG)', key: 's', disabled: !open, action: screenshot },
    { label: 'Copy Python snippet of this view', disabled: !open, action: () => copyText(pythonSnippet(), 'Python snippet copied') },
  ];
  if (name === 'view') return [
    { label: 'Reset view', key: 'r', disabled: !open, action: resetViewUser },
    '-',
    { label: 'Line chart window', key: 'g', checked: LP.windowOpen, action: () => setLineWindow(!LP.windowOpen) },
    { label: 'Field calculator window', key: 'k', checked: CP.windowOpen, action: () => setCalcWindow(!CP.windowOpen) },
    { label: 'Forces window', key: 'f', checked: FC.windowOpen, action: () => setForceWindow(!FC.windowOpen) },
    { label: 'Sidebar', key: 'b', checked: !$('app').classList.contains('no-sidebar'), action: toggleSidebar },
    '-',
    { label: 'Axes', checked: S.axes, action: () => $('show-axes').click() },
    { label: 'Colorbar', checked: S.colorbar, action: () => $('show-colorbar').click() },
    { label: 'Element edges', key: 'm', checked: S.edges, action: () => $('show-edges').click() },
    { label: 'Iso-lines', key: 'c', checked: S.contours, action: () => $('show-contours').click() },
    { label: 'GLL nodes', key: 'n', checked: S.nodes, action: () => $('show-nodes').click() },
    { label: 'Boundaries', checked: S.showBoundaries, action: () => $('bd-show').click() },
    '-',
    { label: 'Spectral rendering (exact per pixel)', key: '1', checked: S.mode === 0, action: () => $('mode').querySelector('[data-mode="0"]').click() },
    { label: 'Nodal rendering (linear, fast)', key: '2', checked: S.mode === 1, action: () => $('mode').querySelector('[data-mode="1"]').click() },
    '-',
    { label: 'Light background', checked: isLight(), action: () => { $('bg').value = isLight() ? 'dark' : 'light'; $('bg').dispatchEvent(new Event('change')); } },
  ];
  return [
    { label: 'Keyboard and mouse…', action: showHelp },
    { label: 'About semscope', action: () => toast('semscope — spectral-element-aware viewer for 2D CG SEM data (Nek5000 / Neko), built on pySEMTools') },
  ];
}
function openMenu(btn) {
  const name = btn.dataset.menu;
  const rect = btn.getBoundingClientRect();
  openCtx(menuItems(name), rect.left, rect.bottom + 2);
  ctxEl.dataset.menu = name;
  btn.classList.add('open');
}
for (const btn of document.querySelectorAll('.menu-btn')) {
  btn.addEventListener('mousedown', ev => ev.stopPropagation());
  btn.addEventListener('click', () => { if (!ctxEl.hidden && ctxEl.dataset.menu === btn.dataset.menu) closeCtx(); else { closeCtx(); openMenu(btn); } });
  btn.addEventListener('mouseenter', () => { if (!ctxEl.hidden && ctxEl.dataset.menu && ctxEl.dataset.menu !== btn.dataset.menu) { closeCtx(); openMenu(btn); } });
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
window.semscope = { S, R, BD, NL, CP, FC, setForceWindow, computeForces, forceConfig, applyForceConfig, setCalcWindow, defineField, removeCalcField, syncCalcDefs, boundaryNodeAt, boundaryPointAt, edgePoint, createNormalLine, setNormalMode, api, requestRender, pythonSnippet, lineCSV, sessionState, applySession, detectBoundaries, setPickMode, toggleEdge, edgeAt };   // handy for debugging / scripting the GUI
(async () => {
  try {
    resize();
    await loadColormaps();
    await loadState();
  } catch (err) { toast(err.message, true); console.error(err); }
})();
