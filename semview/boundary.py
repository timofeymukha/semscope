"""Boundary representation of a 2D spectral-element mesh.

An *external edge* is an element edge that no other element shares.  External
edges are chained into closed (or open) loops and split into *boundaries*
wherever the loop turns by more than a feature angle — with the turn measured
between the exact polynomial tangents of the two edges meeting at the vertex,
so smoothly curved boundaries (a cylinder wall) stay in one piece while the
corners of a box split it into four sides.

Edge sides are numbered counter-clockwise around the element::

    side 0: s = -1 (j = 0),      i increasing   (bottom)
    side 1: r = +1 (i = n - 1),  j increasing   (right)
    side 2: s = +1 (j = n - 1),  i decreasing   (top)
    side 3: r = -1 (i = 0),      j decreasing   (left)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import spectral as sp

__all__ = ["Boundary", "external_edges", "edge_nodes", "edge_normals", "edge_point", "normal_line", "normal_line_at", "chain_edges", "detect_boundaries", "SIDE_NAMES"]

SIDE_NAMES = ("bottom", "right", "top", "left")


# --------------------------------------------------------------------------- #
# edge geometry
# --------------------------------------------------------------------------- #
def edge_nodes(x: np.ndarray, y: np.ndarray, elem, side) -> tuple[np.ndarray, np.ndarray]:
    """GLL node coordinates ``(k, n)`` along edges ``(elem, side)`` in CCW order."""
    elem = np.atleast_1d(np.asarray(elem))
    side = np.atleast_1d(np.asarray(side))
    n = x.shape[-1]
    xe = np.empty((elem.size, n), dtype=x.dtype)
    ye = np.empty((elem.size, n), dtype=y.dtype)
    for s_, sl in enumerate(_side_slices(n)):
        m = side == s_
        if m.any():
            xe[m] = x[elem[m]][:, sl[0], sl[1]]
            ye[m] = y[elem[m]][:, sl[0], sl[1]]
    return xe, ye


def _side_slices(n):
    return ((0, slice(None)), (slice(None), n - 1), (n - 1, slice(None, None, -1)), (slice(None, None, -1), 0))


def _all_edges(x: np.ndarray, y: np.ndarray):
    """All ``nelv * 4`` edges as ``(elem, side)`` with their GLL nodes ``(N, n)``."""
    nelv, n, _ = x.shape
    elem = np.repeat(np.arange(nelv), 4)
    side = np.tile(np.arange(4), nelv)
    xe, ye = edge_nodes(x, y, elem, side)
    return elem, side, xe, ye


def _tolerance(xe, ye):
    chord = np.hypot(xe[:, -1] - xe[:, 0], ye[:, -1] - ye[:, 0])
    return 1e-4 * max(float(np.median(chord)), 1e-300)


def external_edges(x: np.ndarray, y: np.ndarray, tol: float | None = None) -> np.ndarray:
    """Edges not shared with another element, as ``(nb, 2)`` array of ``(elem, side)``.

    Two edges are the same edge when both end points coincide within ``tol``
    (default: 1e-4 of the median edge chord).  The result is sorted by
    ``(elem, side)``, so indices into it are stable.
    """
    from scipy.spatial import cKDTree

    elem, side, xe, ye = _all_edges(x, y)
    if tol is None:
        tol = _tolerance(xe, ye)
    mid = np.stack([0.5 * (xe[:, 0] + xe[:, -1]), 0.5 * (ye[:, 0] + ye[:, -1])], axis=1)
    tree = cKDTree(mid)
    pairs = tree.query_pairs(r=tol, output_type="ndarray")
    matched = np.zeros(elem.size, dtype=bool)
    if pairs.size:
        a, b = pairs[:, 0], pairs[:, 1]
        pa0 = np.stack([xe[a, 0], ye[a, 0]], 1)
        pa1 = np.stack([xe[a, -1], ye[a, -1]], 1)
        pb0 = np.stack([xe[b, 0], ye[b, 0]], 1)
        pb1 = np.stack([xe[b, -1], ye[b, -1]], 1)
        same = (np.linalg.norm(pa0 - pb1, axis=1) < tol) & (np.linalg.norm(pa1 - pb0, axis=1) < tol)
        same |= (np.linalg.norm(pa0 - pb0, axis=1) < tol) & (np.linalg.norm(pa1 - pb1, axis=1) < tol)
        matched[a[same]] = True
        matched[b[same]] = True
    ext = ~matched
    return np.stack([elem[ext], side[ext]], axis=1)


def jacobian_sign(x: np.ndarray, y: np.ndarray, elems) -> np.ndarray:
    """``+1`` for counter-clockwise (positive Jacobian) elements, ``-1`` for mirrored ones."""
    elems = np.atleast_1d(np.asarray(elems))
    xr, xs = sp.element_derivatives(x[elems])
    yr, ys = sp.element_derivatives(y[elems])
    det = (xr * ys - xs * yr).mean(axis=(1, 2))
    return np.where(det < 0, -1.0, 1.0)


def edge_normals(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Outward unit normals ``(k, n, 2)`` at the GLL nodes of edges ``(elem, side)``.

    Uses the spectral tangent along the edge; for counter-clockwise elements the
    domain lies to the left of a CCW side traversal, so the outward normal is
    the right-hand normal.  Mirrored elements are corrected via the Jacobian sign.
    """
    edges = np.asarray(edges).reshape(-1, 2)
    xe, ye = edge_nodes(x, y, edges[:, 0], edges[:, 1])
    n = xe.shape[1]
    D = sp.derivative_matrix(n)
    tx, ty = xe @ D.T, ye @ D.T
    nrm = np.stack([ty, -tx], axis=-1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-300)
    return nrm * jacobian_sign(x, y, edges[:, 0])[:, None, None]


def edge_point(x: np.ndarray, y: np.ndarray, elem: int, side: int, t: float):
    """Position and outward unit normal at parameter ``t`` in ``[-1, 1]`` along edge ``(elem, side)``.

    The edge geometry is the polynomial interpolant of its GLL nodes, so this is
    exact for the mesh as the solver sees it.  Returns ``(px, py, nx, ny)``.
    """
    xe, ye = edge_nodes(x, y, [elem], [side])
    n = xe.shape[1]
    J = sp.interpolation_matrix(sp.gll_nodes(n), [float(t)])[0]
    D = sp.derivative_matrix(n)
    px, py = float(J @ xe[0]), float(J @ ye[0])
    tx, ty = float(J @ (D @ xe[0])), float(J @ (D @ ye[0]))
    sgn = float(jacobian_sign(x, y, [elem])[0])
    nx, ny = sgn * ty, -sgn * tx
    nrm = max(np.hypot(nx, ny), 1e-300)
    return px, py, nx / nrm, ny / nrm


def normal_line_at(x: np.ndarray, y: np.ndarray, elem: int, side: int, t: float, length: float):
    """Wall-normal segment ``(p0, p1)`` from the point at parameter ``t`` of edge ``(elem, side)``."""
    px, py, nx, ny = edge_point(x, y, elem, side, t)
    p0 = np.array([px, py])
    return p0, p0 - length * np.array([nx, ny])


def normal_line(x: np.ndarray, y: np.ndarray, elem: int, side: int, node: int, length: float):
    """Segment from boundary GLL node ``node`` of edge ``(elem, side)`` along the inward normal.

    Returns ``(p0, p1)``; a negative ``length`` goes outward.
    """
    xe, ye = edge_nodes(x, y, [elem], [side])
    nrm = edge_normals(x, y, [[elem, side]])[0, node]
    p0 = np.array([xe[0, node], ye[0, node]])
    return p0, p0 - length * nrm


def _tangents(xe, ye):
    """Unit tangents at the start and end of each edge from the spectral derivative."""
    n = xe.shape[1]
    D = sp.derivative_matrix(n)
    dx = xe @ D.T
    dy = ye @ D.T
    t0 = np.stack([dx[:, 0], dy[:, 0]], 1)
    t1 = np.stack([dx[:, -1], dy[:, -1]], 1)
    t0 /= np.maximum(np.linalg.norm(t0, axis=1, keepdims=True), 1e-300)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-300)
    return t0, t1


def _turn_deg(t_end, t_start):
    """Turning angle in degrees between an incoming and an outgoing unit tangent."""
    cross = t_end[..., 0] * t_start[..., 1] - t_end[..., 1] * t_start[..., 0]
    dot = t_end[..., 0] * t_start[..., 0] + t_end[..., 1] * t_start[..., 1]
    return np.degrees(np.abs(np.arctan2(cross, dot)))


def chain_edges(x: np.ndarray, y: np.ndarray, edges: np.ndarray, tol: float | None = None):
    """Order external edges into loops.

    Returns a list of ``(indices, closed)`` where ``indices`` index into
    ``edges`` in traversal order (the end of one edge is the start of the next).
    """
    from scipy.spatial import cKDTree

    xe, ye = edge_nodes(x, y, edges[:, 0], edges[:, 1])
    if tol is None:
        tol = _tolerance(xe, ye)
    starts = np.stack([xe[:, 0], ye[:, 0]], 1)
    ends = np.stack([xe[:, -1], ye[:, -1]], 1)
    t0, t1 = _tangents(xe, ye)
    tree = cKDTree(starts)
    cand = tree.query_ball_point(ends, r=tol)
    nxt = np.full(len(edges), -1)
    for i, c in enumerate(cand):
        c = [j for j in c if j != i]
        if len(c) == 1:
            nxt[i] = c[0]
        elif len(c) > 1:  # non-manifold vertex: continue as straight as possible
            nxt[i] = min(c, key=lambda j: _turn_deg(t1[i], t0[j]))
    has_prev = np.zeros(len(edges), dtype=bool)
    has_prev[nxt[nxt >= 0]] = True
    visited = np.zeros(len(edges), dtype=bool)
    loops = []
    # open chains first (start at edges without a predecessor), then closed loops
    order = list(np.nonzero(~has_prev)[0]) + list(range(len(edges)))
    for s in order:
        if visited[s]:
            continue
        seq = []
        i = s
        while i >= 0 and not visited[i]:
            visited[i] = True
            seq.append(i)
            i = nxt[i]
        closed = i == s and len(seq) > 1
        loops.append((np.array(seq), closed))
    return loops


# --------------------------------------------------------------------------- #
@dataclass
class Boundary:
    """A named set of external edges of a mesh (usually one contiguous run)."""

    name: str
    edges: np.ndarray  # (k, 2) of (elem, side), in traversal order when chained
    closed: bool = False
    x: np.ndarray = field(repr=False, default=None)
    y: np.ndarray = field(repr=False, default=None)

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)

    def __len__(self):
        return len(self.edges)

    @property
    def n(self) -> int:
        return self.x.shape[-1]

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """GLL node coordinates along the edges, ``(k, n)`` each."""
        return edge_nodes(self.x, self.y, self.edges[:, 0], self.edges[:, 1])

    def coords(self, m: int | None = None) -> np.ndarray:
        """Points along the boundary, spectrally resampled to ``m`` per edge: ``(k, m, 2)``."""
        xe, ye = self.nodes()
        if m is None or m == self.n:
            return np.stack([xe, ye], axis=-1)
        J = sp.interpolation_matrix(sp.gll_nodes(self.n), sp.uniform_nodes(m))
        return np.stack([xe @ J.T, ye @ J.T], axis=-1)

    def polyline(self, m: int | None = None) -> np.ndarray:
        """A single ``(P, 2)`` polyline (shared vertices merged; closed loops repeat the first point)."""
        c = self.coords(m)
        pts = [c[0]]
        for seg in c[1:]:
            pts.append(seg[1:] if np.allclose(seg[0], pts[-1][-1]) else seg)
        out = np.concatenate(pts)
        if self.closed and not np.allclose(out[0], out[-1]):
            out = np.concatenate([out, out[:1]])
        return out

    def length(self) -> float:
        """Arc length from GLL quadrature of the edge tangents (spectrally accurate)."""
        xe, ye = self.nodes()
        D = sp.derivative_matrix(self.n)
        w = sp.gll_weights(self.n)
        speed = np.hypot(xe @ D.T, ye @ D.T)
        return float(np.sum(speed * w))

    def values(self, data_or_field) -> np.ndarray:
        """Nodal values ``(k, n)`` of a field along the boundary edges."""
        f = np.asarray(data_or_field)
        return edge_nodes(f, f, self.edges[:, 0], self.edges[:, 1])[0]

    def normals(self) -> np.ndarray:
        """Outward unit normals ``(k, n, 2)`` at the boundary GLL nodes."""
        return edge_normals(self.x, self.y, self.edges)

    def normal_line(self, k: int, node: int, length: float):
        """Wall-normal segment ``(p0, p1)`` from node ``node`` of the ``k``-th edge, ``length`` into the domain."""
        return normal_line(self.x, self.y, int(self.edges[k, 0]), int(self.edges[k, 1]), node, length)

    def normal_line_at(self, k: int, t: float, length: float):
        """Wall-normal segment from the point at edge parameter ``t`` (``-1..1``) of the ``k``-th edge."""
        return normal_line_at(self.x, self.y, int(self.edges[k, 0]), int(self.edges[k, 1]), t, length)

    def to_dict(self) -> dict:
        return {"name": self.name, "closed": bool(self.closed), "edges": self.edges.tolist()}


def detect_boundaries(x: np.ndarray, y: np.ndarray, angle: float = 90.0, edges: np.ndarray | None = None, prefix: str = "B", angle_tol: float = 0.5) -> list[Boundary]:
    """Find the external edges, chain them and split at corners sharper than ``angle`` degrees.

    The turning angle is measured between the spectral tangents of the two
    edges meeting at a vertex; ``angle_tol`` (degrees) absorbs the rounding of
    those tangents so that right angles split at ``angle=90``.
    """
    if edges is None:
        edges = external_edges(x, y)
    if len(edges) == 0:
        return []
    xe, ye = edge_nodes(x, y, edges[:, 0], edges[:, 1])
    t0, t1 = _tangents(xe, ye)
    out = []
    for seq, closed in chain_edges(x, y, edges):
        k = len(seq)
        turns = np.array([_turn_deg(t1[seq[i]], t0[seq[(i + 1) % k]]) for i in range(k)]) if k > 1 else np.zeros(1)
        cut = turns >= angle - angle_tol  # cut after edge i
        if not closed:
            cut[-1] = True
        if closed and not cut.any():
            out.append(Boundary("", edges[seq], closed=True, x=x, y=y))
            continue
        # start each group right after a cut
        start = (int(np.nonzero(cut)[0][-1]) + 1) % k if closed else 0
        order = [(start + i) % k for i in range(k)]
        group = []
        for i in order:
            group.append(seq[i])
            if cut[i]:
                out.append(Boundary("", edges[np.array(group)], closed=False, x=x, y=y))
                group = []
        if group:
            out.append(Boundary("", edges[np.array(group)], closed=False, x=x, y=y))
    for i, b in enumerate(out):
        b.name = f"{prefix}{i + 1}"
    return out
