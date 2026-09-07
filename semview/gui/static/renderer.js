// WebGL2 renderer for 2D spectral-element fields.
//
// Data layout: nodal arrays (nelv, n, n) are uploaded *as they are* into R32F
// textures whose rows hold `epr` consecutive elements (n*n texels each), so no
// reshuffling is needed.  Elements are drawn instanced: one tessellated
// reference square per level of detail, the element index as the per-instance
// attribute; the vertex shader evaluates the polynomial geometry, the fragment
// shader evaluates the field at each pixel.

import { SURFACE_VS, SURFACE_FS, PICK_FS, NODES_VS, NODES_FS, MAXN } from './shaders.js';

const LEVELS = [2, 4, 8, 16, 32, 64];   // sub-cells per element side

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    throw new Error('shader compile error:\n' + log + '\n' + src.split('\n').map((l, i) => `${i + 1}: ${l}`).join('\n'));
  }
  return sh;
}

function link(gl, vs, fs, attribs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  for (const [name, loc] of Object.entries(attribs)) gl.bindAttribLocation(p, loc, name);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link error: ' + gl.getProgramInfoLog(p));
  const uniforms = {};
  const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(p, i);
    const name = info.name.replace(/\[0\]$/, '');
    uniforms[name] = gl.getUniformLocation(p, info.name);
  }
  return { prog: p, u: uniforms };
}

function hexToRGBA(hex, alpha = 1) {
  const h = hex.replace('#', '');
  const v = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255, alpha];
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL2 is required');
    this.gl = gl;
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const attribs = { aRS: 0, aElem: 1, aNode: 2 };
    this.surface = link(gl, SURFACE_VS, SURFACE_FS, attribs);
    this.pickProg = link(gl, SURFACE_VS, PICK_FS, attribs);
    this.nodesProg = link(gl, NODES_VS, NODES_FS, attribs);
    this.levels = LEVELS.map(k => this._makeLevel(k));
    this.mesh = null;
    this.field = null;
    this.cmapTex = null;
    this.pick = { fb: null, tex: null, w: 0, h: 0 };
    this.view = { cx: 0, cy: 0, scale: 1 };
    this.width = 1; this.height = 1; this.dpr = 1;
    this.lodKey = '';
    this.stats = { visible: 0, triangles: 0 };
  }

  // ------------------------------------------------------------------ setup
  _makeLevel(k) {
    const gl = this.gl;
    const m = k + 1;
    const rs = new Float32Array(m * m * 2);
    for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) {
      rs[2 * (j * m + i)] = -1 + 2 * i / k;
      rs[2 * (j * m + i) + 1] = -1 + 2 * j / k;
    }
    const idx = new Uint16Array(k * k * 6);
    let q = 0;
    for (let j = 0; j < k; j++) for (let i = 0; i < k; i++) {
      const a = j * m + i, b = a + 1, c = a + m, d = c + 1;
      idx[q++] = a; idx[q++] = b; idx[q++] = d; idx[q++] = a; idx[q++] = d; idx[q++] = c;
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, rs, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    const inst = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, inst);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.bindVertexArray(null);
    return { k, vao, inst, nIndex: idx.length, ids: null, count: 0 };
  }

  _texR32F(data, width, height) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
    return tex;
  }

  _padded(values) {
    const { epr, rows, n } = this.mesh;
    const need = rows * epr * n * n;
    if (values.length === need) return values;
    const out = new Float32Array(need);
    out.set(values);
    return out;
  }

  /** Upload a mesh: x, y Float32Array(nelv*n*n), n nodes per side, GLL nodes + barycentric weights. */
  setMesh({ x, y, n, nelv, nodes, bary }) {
    const gl = this.gl;
    if (n > MAXN) throw new Error(`polynomial order too high for the GPU renderer (n=${n} > ${MAXN})`);
    const nn = n * n;
    const epr = Math.min(Math.floor(this.maxTex / nn), nelv);
    const rows = Math.ceil(nelv / epr);
    if (rows > this.maxTex) throw new Error(`too many elements for GPU textures (${nelv})`);
    // bounding boxes + domain centre (coordinates are stored relative to it for precision)
    const bbox = new Float32Array(nelv * 4);
    let X0 = Infinity, X1 = -Infinity, Y0 = Infinity, Y1 = -Infinity;
    for (let e = 0; e < nelv; e++) {
      let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
      const off = e * nn;
      for (let k = 0; k < nn; k++) {
        const xv = x[off + k], yv = y[off + k];
        if (xv < a) a = xv; if (xv > b) b = xv; if (yv < c) c = yv; if (yv > d) d = yv;
      }
      bbox[4 * e] = a; bbox[4 * e + 1] = b; bbox[4 * e + 2] = c; bbox[4 * e + 3] = d;
      if (a < X0) X0 = a; if (b > X1) X1 = b; if (c < Y0) Y0 = c; if (d > Y1) Y1 = d;
    }
    const ox = 0.5 * (X0 + X1), oy = 0.5 * (Y0 + Y1);
    const xr = new Float32Array(rows * epr * nn), yr = new Float32Array(rows * epr * nn);
    for (let k = 0; k < x.length; k++) { xr[k] = x[k] - ox; yr[k] = y[k] - oy; }
    if (this.mesh) { gl.deleteTexture(this.mesh.texX); gl.deleteTexture(this.mesh.texY); }
    this.mesh = { n, nelv, epr, rows, width: epr * nn, bbox, bounds: [X0, X1, Y0, Y1], origin: [ox, oy], nodes, bary, x, y };
    this.mesh.texX = this._texR32F(xr, epr * nn, rows);
    this.mesh.texY = this._texR32F(yr, epr * nn, rows);
    for (const L of this.levels) { L.ids = new Uint32Array(nelv); L.count = 0; }
    this.visible = new Uint32Array(nelv);
    this.nVisible = 0;
    this.lodKey = '';
    // per-vertex node index buffer for the GLL node overlay
    if (this.nodesVao) gl.deleteVertexArray(this.nodesVao);
    this.nodesVao = gl.createVertexArray();
    gl.bindVertexArray(this.nodesVao);
    const nodeIdx = new Uint32Array(nn);
    for (let k = 0; k < nn; k++) nodeIdx[k] = k;
    const nb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nb);
    gl.bufferData(gl.ARRAY_BUFFER, nodeIdx, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribIPointer(2, 1, gl.UNSIGNED_INT, 0, 0);
    this.nodesInst = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodesInst);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.bindVertexArray(null);
    if (this.field) { gl.deleteTexture(this.field.tex); this.field = null; }
  }

  /** Upload field values Float32Array(nelv*n*n). */
  setField(values) {
    const gl = this.gl;
    if (!this.mesh) return;
    const data = this._padded(values);
    if (this.field) {
      gl.bindTexture(gl.TEXTURE_2D, this.field.tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.mesh.width, this.mesh.rows, gl.RED, gl.FLOAT, data);
    } else {
      this.field = { tex: this._texR32F(data, this.mesh.width, this.mesh.rows) };
    }
    this.field.values = values;
  }

  /** rgb: array of [r,g,b] (0..255) */
  setColormap(rgb) {
    const gl = this.gl;
    const N = rgb.length;
    const buf = new Uint8Array(N * 3);
    for (let i = 0; i < N; i++) { buf[3 * i] = rgb[i][0]; buf[3 * i + 1] = rgb[i][1]; buf[3 * i + 2] = rgb[i][2]; }
    if (!this.cmapTex) this.cmapTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.cmapTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, N, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, buf);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  resize(cssW, cssH, dpr) {
    this.width = cssW; this.height = cssH; this.dpr = dpr;
    const W = Math.max(1, Math.round(cssW * dpr)), H = Math.max(1, Math.round(cssH * dpr));
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; }
    this.lodKey = '';
  }

  setView(v) { this.view = { ...v }; }

  // ------------------------------------------------------------------ LOD
  /** Cull and bin visible elements by their on-screen size. */
  updateLOD(pxPerCell) {
    const m = this.mesh; if (!m) return;
    const { cx, cy, scale } = this.view;
    const key = `${cx}|${cy}|${scale}|${this.width}|${this.height}|${pxPerCell}`;
    if (key === this.lodKey) return;
    this.lodKey = key;
    const hw = 0.5 * this.width / scale, hh = 0.5 * this.height / scale;
    const vx0 = cx - hw, vx1 = cx + hw, vy0 = cy - hh, vy1 = cy + hh;
    for (const L of this.levels) L.count = 0;
    let nv = 0, tris = 0;
    const bb = m.bbox;
    for (let e = 0; e < m.nelv; e++) {
      const a = bb[4 * e], b = bb[4 * e + 1], c = bb[4 * e + 2], d = bb[4 * e + 3];
      if (b < vx0 || a > vx1 || d < vy0 || c > vy1) continue;
      const px = Math.max(b - a, d - c) * scale;
      let li = Math.ceil(Math.log2(Math.max(px / pxPerCell, 1)));
      li = Math.min(Math.max(li - 1, 0), LEVELS.length - 1);
      const L = this.levels[li];
      L.ids[L.count++] = e;
      this.visible[nv++] = e;
      tris += 2 * LEVELS[li] * LEVELS[li];
    }
    this.nVisible = nv;
    this.stats = { visible: nv, triangles: tris };
    const gl = this.gl;
    for (const L of this.levels) {
      if (!L.count) continue;
      gl.bindBuffer(gl.ARRAY_BUFFER, L.inst);
      gl.bufferData(gl.ARRAY_BUFFER, L.ids.subarray(0, L.count), gl.DYNAMIC_DRAW);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodesInst);
    gl.bufferData(gl.ARRAY_BUFFER, this.visible.subarray(0, nv), gl.DYNAMIC_DRAW);
  }

  _commonUniforms(P) {
    const gl = this.gl, m = this.mesh;
    gl.uniform1i(P.u.uN, m.n);
    gl.uniform1i(P.u.uEPR, m.epr);
    const nodes = new Float32Array(MAXN), bary = new Float32Array(MAXN);
    nodes.set(m.nodes); bary.set(m.bary);
    gl.uniform1fv(P.u.uNodes, nodes);
    gl.uniform1fv(P.u.uBary, bary);
    gl.uniform2f(P.u.uCenter, this.view.cx - m.origin[0], this.view.cy - m.origin[1]);
    gl.uniform2f(P.u.uScale, 2 * this.view.scale / this.width, 2 * this.view.scale / this.height);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, m.texX); gl.uniform1i(P.u.uX, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, m.texY); gl.uniform1i(P.u.uY, 1);
  }

  _drawSurface(P) {
    const gl = this.gl;
    for (const L of this.levels) {
      if (!L.count) continue;
      gl.bindVertexArray(L.vao);
      gl.drawElementsInstanced(gl.TRIANGLES, L.nIndex, gl.UNSIGNED_SHORT, 0, L.count);
    }
    gl.bindVertexArray(null);
  }

  // ------------------------------------------------------------------ frame
  render(o) {
    const gl = this.gl;
    const W = this.canvas.width, H = this.canvas.height;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    const bg = hexToRGBA(o.background);
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    if (!this.mesh || !this.field) return;
    this.updateLOD(o.pxPerCell);
    const P = this.surface;
    gl.useProgram(P.prog);
    this._commonUniforms(P);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.field.tex); gl.uniform1i(P.u.uField, 2);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.cmapTex); gl.uniform1i(P.u.uCmap, 3);
    gl.uniform2f(P.u.uRange, o.range[0], o.range[1] === o.range[0] ? o.range[0] + 1e-30 : o.range[1]);
    gl.uniform1i(P.u.uMode, o.mode);
    gl.uniform1i(P.u.uInvert, o.invert ? 1 : 0);
    gl.uniform1i(P.u.uContours, o.contours ? o.nContours : 0);
    gl.uniform1f(P.u.uContourWidth, o.contourWidth * this.dpr);
    gl.uniform4fv(P.u.uContourColor, hexToRGBA(o.contourColor, o.contourAlpha));
    gl.uniform1i(P.u.uEdges, o.edges ? 1 : 0);
    gl.uniform1f(P.u.uEdgeWidth, o.edgeWidth * this.dpr);
    gl.uniform4fv(P.u.uEdgeColor, hexToRGBA(o.edgeColor, o.edgeAlpha));
    gl.uniform4fv(P.u.uNanColor, hexToRGBA(o.nanColor));
    this._drawSurface(P);
    if (o.nodes && this.nVisible) {
      const NP = this.nodesProg;
      gl.useProgram(NP.prog);
      this._commonUniforms(NP);
      gl.uniform1f(NP.u.uSize, o.nodeSize * this.dpr);
      gl.uniform4fv(NP.u.uColor, hexToRGBA(o.nodeColor, 0.95));
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(this.nodesVao);
      gl.drawArraysInstanced(gl.POINTS, 0, this.mesh.n * this.mesh.n, this.nVisible);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    }
  }

  // ------------------------------------------------------------------ picking
  _ensurePick() {
    const gl = this.gl, W = this.canvas.width, H = this.canvas.height;
    if (this.pick.w === W && this.pick.h === H) return;
    if (this.pick.fb) { gl.deleteFramebuffer(this.pick.fb); gl.deleteTexture(this.pick.tex); }
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.pick = { fb, tex, w: W, h: H, key: '' };
  }

  /** Element index under CSS pixel (px, py), or -1.
   *  The element-id buffer is rendered once per view (cached by the LOD key), so
   *  hovering costs one 1x1 readPixels. */
  pickElement(px, py, pxPerCell) {
    const gl = this.gl;
    if (!this.mesh) return -1;
    this._ensurePick();
    this.updateLOD(pxPerCell);
    const x = Math.round(px * this.dpr), y = Math.round((this.height - py) * this.dpr);
    if (x < 0 || y < 0 || x >= this.pick.w || y >= this.pick.h) return -1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pick.fb);
    if (this.pick.key !== this.lodKey) {
      gl.viewport(0, 0, this.pick.w, this.pick.h);
      gl.disable(gl.SCISSOR_TEST);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.BLEND);
      const P = this.pickProg;
      gl.useProgram(P.prog);
      this._commonUniforms(P);
      this._drawSurface(P);
      this.pick.key = this.lodKey;
    }
    const out = new Uint8Array(4);
    gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const id = out[0] + 256 * out[1] + 65536 * out[2];
    return id === 0 ? -1 : id - 1;
  }
}
