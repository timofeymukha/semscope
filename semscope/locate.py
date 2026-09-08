"""Point location in a 2D spectral-element mesh.

Given physical points ``(x, y)`` find the element that contains each one and
its reference coordinates ``(r, s)``, by inverting the (polynomial) geometry
map of the element with Newton's method.  This is what allows semscope to
evaluate a field at an arbitrary point exactly, instead of interpolating
linearly between GLL nodes.

The search is two-stage and fully vectorised:

1. a uniform background grid over the domain bounding box maps every query
   point to a short list of candidate elements (those whose padded bounding box
   overlaps the grid cell);
2. Newton iteration on all (point, candidate) pairs at once; a pair is accepted
   when it converges to ``|r|, |s| <= 1 + tol``.
"""

from __future__ import annotations

import numpy as np

from . import spectral as sp

__all__ = ["ElementLocator", "Location"]


class Location:
    """Result of a point search: ``elem`` (``-1`` where not found), ``r``, ``s``."""

    __slots__ = ("elem", "r", "s")

    def __init__(self, elem, r, s):
        self.elem = elem
        self.r = r
        self.s = s

    @property
    def found(self) -> np.ndarray:
        return self.elem >= 0

    def __len__(self):
        return self.elem.size


class ElementLocator:
    """Locate points in a mesh given by nodal coordinates ``x, y`` of shape ``(nelv, n, n)``.

    Parameters
    ----------
    margin : relative bounding-box padding (fraction of the element size) that
        accounts for curved edges bulging beyond the GLL nodes.
    cells_per_element : resolution of the background grid.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, margin: float = 0.1, cells_per_element: float = 1.0):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.nelv, self.ns, self.nr = self.x.shape
        self.nodes_r = sp.gll_nodes(self.nr)
        self.nodes_s = sp.gll_nodes(self.ns)
        self._wr = sp.barycentric_weights(self.nodes_r)
        self._ws = sp.barycentric_weights(self.nodes_s)

        xmin = self.x.min(axis=(1, 2))
        xmax = self.x.max(axis=(1, 2))
        ymin = self.y.min(axis=(1, 2))
        ymax = self.y.max(axis=(1, 2))
        dx = xmax - xmin
        dy = ymax - ymin
        self.size = np.sqrt(np.maximum(dx * dy, 1e-300))
        pad_x = margin * dx
        pad_y = margin * dy
        self.bbox = np.stack([xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y], axis=1)
        self.domain = np.array([xmin.min(), xmax.max(), ymin.min(), ymax.max()])
        self._build_grid(cells_per_element)

    # ------------------------------------------------------------------ grid
    def _build_grid(self, cells_per_element: float):
        x0, x1, y0, y1 = self.domain
        Lx = max(x1 - x0, 1e-300)
        Ly = max(y1 - y0, 1e-300)
        ncells = max(1.0, cells_per_element * self.nelv)
        aspect = Lx / Ly
        self.ncx = max(1, int(round(np.sqrt(ncells * aspect))))
        self.ncy = max(1, int(round(np.sqrt(ncells / aspect))))
        self.hx = Lx / self.ncx
        self.hy = Ly / self.ncy
        b = self.bbox
        ix0 = np.clip(np.floor((b[:, 0] - x0) / self.hx).astype(int), 0, self.ncx - 1)
        ix1 = np.clip(np.floor((b[:, 1] - x0) / self.hx).astype(int), 0, self.ncx - 1)
        iy0 = np.clip(np.floor((b[:, 2] - y0) / self.hy).astype(int), 0, self.ncy - 1)
        iy1 = np.clip(np.floor((b[:, 3] - y0) / self.hy).astype(int), 0, self.ncy - 1)
        counts = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        elem = np.repeat(np.arange(self.nelv), counts)
        # enumerate the cells of each element's bbox
        offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        wx = np.repeat(ix1 - ix0 + 1, counts)
        cx = np.repeat(ix0, counts) + offs % wx
        cy = np.repeat(iy0, counts) + offs // wx
        cell = cy * self.ncx + cx
        order = np.argsort(cell, kind="stable")
        self._cell_elems = elem[order]
        self._cell_start = np.searchsorted(cell[order], np.arange(self.ncx * self.ncy + 1))

    def candidates(self, px: np.ndarray, py: np.ndarray):
        """Return arrays ``(point_index, element_index)`` of candidate pairs."""
        x0, x1, y0, y1 = self.domain
        eps_x = 1e-9 * max(x1 - x0, 1e-300)
        eps_y = 1e-9 * max(y1 - y0, 1e-300)
        inside = (px >= x0 - eps_x) & (px <= x1 + eps_x) & (py >= y0 - eps_y) & (py <= y1 + eps_y)
        pts = np.nonzero(inside)[0]
        cx = np.clip(np.floor((px[pts] - x0) / self.hx).astype(int), 0, self.ncx - 1)
        cy = np.clip(np.floor((py[pts] - y0) / self.hy).astype(int), 0, self.ncy - 1)
        cell = cy * self.ncx + cx
        start = self._cell_start[cell]
        stop = self._cell_start[cell + 1]
        cnt = stop - start
        pair_pt = np.repeat(pts, cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        pair_el = self._cell_elems[np.repeat(start, cnt) + offs]
        # exact bbox filter
        b = self.bbox[pair_el]
        keep = (px[pair_pt] >= b[:, 0]) & (px[pair_pt] <= b[:, 1]) & (py[pair_pt] >= b[:, 2]) & (py[pair_pt] <= b[:, 3])
        return pair_pt[keep], pair_el[keep]

    # ---------------------------------------------------------------- newton
    def _basis(self, r, s):
        Lr = sp.lagrange_basis(self.nodes_r, r, self._wr)
        Ls = sp.lagrange_basis(self.nodes_s, s, self._ws)
        return Lr, Ls

    @staticmethod
    def _eval_geometry(X, Y, Lr, Ls, dLr, dLs):
        # contract along r first
        Xi = np.einsum("pji,pi->pj", X, Lr)
        Yi = np.einsum("pji,pi->pj", Y, Lr)
        dXi = np.einsum("pji,pi->pj", X, dLr)
        dYi = np.einsum("pji,pi->pj", Y, dLr)
        x = np.einsum("pj,pj->p", Xi, Ls)
        y = np.einsum("pj,pj->p", Yi, Ls)
        xr = np.einsum("pj,pj->p", dXi, Ls)
        yr = np.einsum("pj,pj->p", dYi, Ls)
        xs = np.einsum("pj,pj->p", Xi, dLs)
        ys = np.einsum("pj,pj->p", Yi, dLs)
        return x, y, xr, xs, yr, ys

    @staticmethod
    def _initial_guess(X, Y, px, py):
        """Invert the bilinear (corner-vertex) approximation of each element: an
        exact start for affine elements, close for curved ones."""
        x00, x10, x01, x11 = X[:, 0, 0], X[:, 0, -1], X[:, -1, 0], X[:, -1, -1]
        y00, y10, y01, y11 = Y[:, 0, 0], Y[:, 0, -1], Y[:, -1, 0], Y[:, -1, -1]
        cx = 0.25 * (x00 + x10 + x01 + x11)
        cy = 0.25 * (y00 + y10 + y01 + y11)
        ar = 0.25 * (x10 - x00 + x11 - x01)
        as_ = 0.25 * (x01 - x00 + x11 - x10)
        br = 0.25 * (y10 - y00 + y11 - y01)
        bs = 0.25 * (y01 - y00 + y11 - y10)
        det = ar * bs - as_ * br
        ok = np.abs(det) > 1e-300
        det = np.where(ok, det, 1.0)
        dx = px - cx
        dy = py - cy
        r = np.where(ok, (bs * dx - as_ * dy) / det, 0.0)
        s = np.where(ok, (-br * dx + ar * dy) / det, 0.0)
        return np.clip(r, -1.5, 1.5), np.clip(s, -1.5, 1.5)

    def newton(self, el, px, py, r0=None, s0=None, tol=1e-10, max_iter=25):
        """Invert the geometry map for pairs ``(el, (px, py))``.

        Returns ``(r, s, converged)``.
        """
        P = el.size
        size = self.size[el]
        Xall = self.x[el]
        Yall = self.y[el]
        if r0 is None or s0 is None:
            r, s = self._initial_guess(Xall, Yall, px, py)
        else:
            r = np.array(r0, dtype=float)
            s = np.array(s0, dtype=float)
        active = np.ones(P, dtype=bool)
        converged = np.zeros(P, dtype=bool)
        Dr = sp.derivative_matrix(self.nr)
        Ds = sp.derivative_matrix(self.ns)
        for _ in range(max_iter):
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                break
            ra, sa = r[idx], s[idx]
            Lr, Ls = self._basis(ra, sa)
            dLr = _basis_derivative(self.nodes_r, ra, self._wr, Lr, Dr)
            dLs = _basis_derivative(self.nodes_s, sa, self._ws, Ls, Ds)
            if idx.size == P:
                X, Y = Xall, Yall
            else:
                X, Y = Xall[idx], Yall[idx]
            x, y, xr, xs, yr, ys = self._eval_geometry(X, Y, Lr, Ls, dLr, dLs)
            fx = x - px[idx]
            fy = y - py[idx]
            # converged pairs keep their current (r, s): do not apply another step
            res = np.hypot(fx, fy) / size[idx]
            hit = res < tol
            converged[idx[hit]] = True
            active[idx[hit]] = False
            det = xr * ys - xs * yr
            det = np.where(np.abs(det) < 1e-300, 1e-300, det)
            dr = -(ys * fx - xs * fy) / det
            ds = -(-yr * fx + xr * fy) / det
            # damp large steps: stay within a slightly enlarged reference square
            step = np.maximum(np.abs(dr), np.abs(ds))
            scale = np.where(step > 1.0, 1.0 / np.maximum(step, 1e-300), 1.0)
            upd = ~hit
            r[idx[upd]] = np.clip(ra[upd] + (dr * scale)[upd], -3.0, 3.0)
            s[idx[upd]] = np.clip(sa[upd] + (ds * scale)[upd], -3.0, 3.0)
            small = upd & (step * scale < tol)
            converged[idx[small]] = True
            active[idx[small]] = False
            # give up on pairs that wander far outside
            far = (np.abs(r[idx]) > 2.5) | (np.abs(s[idx]) > 2.5)
            active[idx[far]] = False
        return r, s, converged

    # ---------------------------------------------------------------- public
    def locate(self, px, py, tol_inside: float = 1e-8, chunk: int = 200_000) -> Location:
        """Find the containing element and ``(r, s)`` of each query point."""
        px = np.asarray(px, dtype=float).ravel()
        py = np.asarray(py, dtype=float).ravel()
        npts = px.size
        elem = np.full(npts, -1, dtype=np.int64)
        rout = np.full(npts, np.nan)
        sout = np.full(npts, np.nan)
        best = np.full(npts, np.inf)
        pair_pt, pair_el = self.candidates(px, py)
        for a in range(0, pair_pt.size, chunk):
            pp = pair_pt[a : a + chunk]
            pe = pair_el[a : a + chunk]
            r, s, conv = self.newton(pe, px[pp], py[pp])
            out = np.maximum(np.abs(r), np.abs(s))
            ok = conv & (out <= 1.0 + tol_inside)
            # retry pairs whose iteration did not converge from other starting points
            # (pairs that converged to a point outside the element are simply rejected)
            retry = np.nonzero(~conv)[0]
            if retry.size:
                for r0, s0 in ((-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)):
                    if retry.size == 0:
                        break
                    r2, s2, c2 = self.newton(
                        pe[retry], px[pp[retry]], py[pp[retry]], r0=np.full(retry.size, r0), s0=np.full(retry.size, s0)
                    )
                    o2 = np.maximum(np.abs(r2), np.abs(s2))
                    ok2 = c2 & (o2 <= 1.0 + tol_inside)
                    r[retry[ok2]] = r2[ok2]
                    s[retry[ok2]] = s2[ok2]
                    out[retry[ok2]] = o2[ok2]
                    ok[retry[ok2]] = True
                    retry = retry[~ok2]
            # keep, per point, the candidate that is most comfortably inside
            good = np.nonzero(ok)[0]
            if good.size:
                order = np.lexsort((out[good], pp[good]))
                g = good[order]
                first = np.concatenate([[True], pp[g][1:] != pp[g][:-1]])
                g = g[first]
                p = pp[g]
                better = out[g] < best[p]
                g, p = g[better], p[better]
                best[p] = out[g]
                elem[p] = pe[g]
                rout[p] = np.clip(r[g], -1.0, 1.0)
                sout[p] = np.clip(s[g], -1.0, 1.0)
        return Location(elem, rout, sout)


def _basis_derivative(nodes, x, w, L, D):
    """Derivatives ``l_j'(x)`` of the Lagrange basis at points ``x`` (``(p, n)``).

    Uses the barycentric formula away from the nodes and the differentiation
    matrix rows when ``x`` coincides with a node.
    """
    d = x[:, None] - nodes[None, :]
    # within ~1e-10 of a node the barycentric derivative formula cancels
    # catastrophically; fall back to the differentiation matrix there
    exact = np.abs(d) < 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        t = w / d  # (p, n)
        S1 = np.sum(t, axis=1, keepdims=True)
        S2 = np.sum(t / d, axis=1, keepdims=True)
        out = L * (S2 / S1 - 1.0 / d)
    hit = np.nonzero(np.any(exact, axis=1))[0]
    if hit.size:
        j = np.argmax(exact[hit], axis=1)
        out[hit] = D[j]
    return out
