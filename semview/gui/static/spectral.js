// Small CPU-side spectral toolkit (mirrors semview/spectral.py) used for the
// hover probe: invert the geometry map by Newton iteration and evaluate the
// field's Lagrange expansion at the exact cursor position.

export function barycentricWeights(nodes) {
  const n = nodes.length, w = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let p = 1;
    for (let k = 0; k < n; k++) if (k !== j) p *= nodes[j] - nodes[k];
    w[j] = 1 / p;
  }
  return w;
}

export function derivativeMatrix(nodes, w) {
  const n = nodes.length, D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) if (i !== j) { D[i * n + j] = (w[j] / w[i]) / (nodes[i] - nodes[j]); s += D[i * n + j]; }
    D[i * n + i] = -s;
  }
  return D;
}

/** Basis values L and derivatives dL at t. */
export function basis(nodes, w, D, t, L, dL) {
  const n = nodes.length;
  let hit = -1;
  for (let i = 0; i < n; i++) if (Math.abs(t - nodes[i]) < 1e-10) hit = i;
  if (hit >= 0) {
    for (let i = 0; i < n; i++) { L[i] = i === hit ? 1 : 0; dL[i] = D[hit * n + i]; }
    return;
  }
  let S1 = 0, S2 = 0;
  for (let i = 0; i < n; i++) { const d = t - nodes[i]; const q = w[i] / d; L[i] = q; S1 += q; S2 += q / d; }
  for (let i = 0; i < n; i++) { L[i] /= S1; dL[i] = L[i] * (S2 / S1 - 1 / (t - nodes[i])); }
}

export class Spectral {
  constructor(nodes) {
    this.nodes = Float64Array.from(nodes);
    this.n = nodes.length;
    this.w = barycentricWeights(this.nodes);
    this.D = derivativeMatrix(this.nodes, this.w);
    this.Lr = new Float64Array(this.n); this.dLr = new Float64Array(this.n);
    this.Ls = new Float64Array(this.n); this.dLs = new Float64Array(this.n);
  }

  /** Evaluate nodal data (flat, element offset `off`) at (r, s) using current basis arrays. */
  _eval(data, off, Lr, Ls) {
    const n = this.n; let v = 0;
    for (let j = 0; j < n; j++) {
      let acc = 0; const row = off + j * n;
      for (let i = 0; i < n; i++) acc += data[row + i] * Lr[i];
      v += acc * Ls[j];
    }
    return v;
  }

  evaluate(data, e, r, s) {
    basis(this.nodes, this.w, this.D, r, this.Lr, this.dLr);
    basis(this.nodes, this.w, this.D, s, this.Ls, this.dLs);
    return this._eval(data, e * this.n * this.n, this.Lr, this.Ls);
  }

  /** Newton inversion of the geometry map of element e for point (px, py). Returns {r, s, ok}. */
  invert(x, y, e, px, py, size = 1) {
    this.size = size;
    const off = e * this.n * this.n;
    let r = 0, s = 0, ok = false;
    for (let it = 0; it < 40; it++) {
      basis(this.nodes, this.w, this.D, r, this.Lr, this.dLr);
      basis(this.nodes, this.w, this.D, s, this.Ls, this.dLs);
      const X = this._eval(x, off, this.Lr, this.Ls), Y = this._eval(y, off, this.Lr, this.Ls);
      const Xr = this._eval(x, off, this.dLr, this.Ls), Yr = this._eval(y, off, this.dLr, this.Ls);
      const Xs = this._eval(x, off, this.Lr, this.dLs), Ys = this._eval(y, off, this.Lr, this.dLs);
      const fx = X - px, fy = Y - py;
      if (Math.hypot(fx, fy) < 1e-11 * this.size) { ok = true; break; }
      const det = Xr * Ys - Xs * Yr;
      if (Math.abs(det) < 1e-300) break;
      let dr = -(Ys * fx - Xs * fy) / det, ds = -(-Yr * fx + Xr * fy) / det;
      const step = Math.max(Math.abs(dr), Math.abs(ds));
      if (step > 1) { dr /= step; ds /= step; }
      r += dr; s += ds;
      if (Math.abs(r) > 3 || Math.abs(s) > 3) break;
      if (step < 1e-12) { ok = true; break; }
    }
    return { r, s, ok: ok && Math.abs(r) <= 1 + 1e-6 && Math.abs(s) <= 1 + 1e-6 };
  }
}
