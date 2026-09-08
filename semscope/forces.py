"""Forces and force coefficients on boundaries.

The traction exerted by the fluid on a wall is integrated along the boundary
edges with the GLL quadrature of each edge::

    F = ∮ (p n − μ (∇u + ∇uᵀ)·n) dl

where ``n`` is the unit normal pointing out of the fluid (into the wall), so
``F`` is the force *on the body* per unit depth — the same convention as the
``torque_calc`` / ``force_torque`` routines of Nek5000 and Neko.  The pressure
and viscous parts are kept separately.  Velocity gradients are the exact
derivatives of the element polynomials, evaluated on the boundary elements.

Coefficients use the dynamic pressure ``q = ½ ρ_ref U_ref²``::

    C_a = (F · a) / (q L_ref)        for a projection axis a (unit vector)
    Cp  = (p − p_ref) / q            along the boundary
    Cf  = τ_w / q                    wall shear along the boundary

``τ_w`` is the tangential component of the viscous traction on the wall,
positive along the traversal direction of the boundary (the fluid to its left,
i.e. counter-clockwise around the domain, clockwise around a hole).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import spectral as sp
from .boundary import Boundary, chain_edges, edge_nodes, edge_normals, jacobian_sign, orient_rows

__all__ = ["Forces", "compute_forces", "sum_forces", "edge_values", "edge_weights", "edge_arclength", "DEFAULT_AXES"]

DEFAULT_AXES = (("x", (1.0, 0.0)), ("y", (0.0, 1.0)))


# --------------------------------------------------------------------------- edge helpers
def edge_values(f: np.ndarray, elem, side) -> np.ndarray:
    """Nodal values ``(k, n)`` of an element array along edges ``(elem, side)`` in CCW order."""
    return edge_nodes(f, f, elem, side)[0]


def edge_weights(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Line-quadrature weights ``(k, n)``: ``w_k |dx/dt|`` so that ``sum(W * f)`` is ``∮ f dl``."""
    edges = np.asarray(edges).reshape(-1, 2)
    xe, ye = edge_nodes(x, y, edges[:, 0], edges[:, 1])
    n = xe.shape[1]
    D = sp.derivative_matrix(n)
    speed = np.hypot(xe @ D.T, ye @ D.T)
    return speed * sp.gll_weights(n)[None, :]


def edge_arclength(xe: np.ndarray, ye: np.ndarray, fine: int = 8) -> np.ndarray:
    """Cumulative arc length ``(k, n)`` at the GLL nodes of edges given by their nodes ``(k, n)``.

    The edge curve is resampled spectrally onto ``fine * n`` points and the
    chord lengths accumulated; good to a small fraction of a percent, which
    is plenty for plotting distributions against arc length.
    """
    n = xe.shape[1]
    m = max(fine * n, 4)
    t = sp.gll_nodes(n)
    tf = np.linspace(-1.0, 1.0, m)
    J = sp.interpolation_matrix(t, tf)
    xf, yf = xe @ J.T, ye @ J.T
    seg = np.hypot(np.diff(xf, axis=1), np.diff(yf, axis=1))
    cum = np.concatenate([np.zeros((len(xe), 1)), np.cumsum(seg, axis=1)], axis=1)
    k = np.clip(np.searchsorted(tf, t) - 1, 0, m - 2)
    frac = (t - tf[k]) / (tf[k + 1] - tf[k])
    return cum[:, k] + frac * (cum[:, k + 1] - cum[:, k])


# --------------------------------------------------------------------------- result
@dataclass
class Forces:
    """Forces on one boundary (per unit depth) and the distributions along it."""

    name: str
    edges: np.ndarray
    length: float
    pressure: np.ndarray          # force vector, components ``x, y[, z]``
    viscous: np.ndarray
    components: tuple[str, ...]
    axes: list[tuple[str, np.ndarray]]
    q: float
    rho_ref: float
    U_ref: float
    L_ref: float
    p_ref: float
    distribution: dict = field(default_factory=dict, repr=False)

    @property
    def total(self) -> np.ndarray:
        return self.pressure + self.viscous

    def project(self, direction) -> tuple[float, float, float]:
        """``(pressure, viscous, total)`` force components along a unit direction."""
        a = _unit(direction, len(self.components))
        return float(self.pressure @ a), float(self.viscous @ a), float(self.total @ a)

    def coefficient(self, direction) -> float:
        """Force coefficient ``(F · a) / (q L_ref)`` along a direction."""
        return self.project(direction)[2] / (self.q * self.L_ref)

    def table(self) -> list[dict]:
        """One row per projection axis: name, direction, pressure, viscous, total, coefficient."""
        rows = []
        for name, a in self.axes:
            fp, fv, ft = self.project(a)
            rows.append({"name": name, "dir": [float(c) for c in a], "pressure": fp, "viscous": fv, "total": ft, "coefficient": ft / (self.q * self.L_ref)})
        return rows

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "nedges": int(len(self.edges)),
            "length": self.length,
            "components": list(self.components),
            "pressure": self.pressure.tolist(),
            "viscous": self.viscous.tolist(),
            "total": self.total.tolist(),
            "axes": self.table(),
            "q": self.q,
            "rho_ref": self.rho_ref,
            "U_ref": self.U_ref,
            "L_ref": self.L_ref,
            "p_ref": self.p_ref,
            "dist": {k: _jsonable(v) for k, v in self.distribution.items()},
        }
        return d

    def __repr__(self):
        rows = ", ".join(f"C{r['name']}={r['coefficient']:.5g}" for r in self.table())
        return f"Forces({self.name!r}, F={np.array2string(self.total, precision=5)}, pressure={np.array2string(self.pressure, precision=5)}, viscous={np.array2string(self.viscous, precision=5)}, {rows})"


def _jsonable(v):
    a = np.asarray(v)
    if a.dtype.kind in "iu":
        return a.tolist()
    return [None if not np.isfinite(x) else float(x) for x in a.ravel()]


def _unit(direction, ncomp: int) -> np.ndarray:
    a = np.zeros(ncomp)
    d = np.asarray(direction, dtype=float).ravel()
    a[: min(len(d), ncomp)] = d[:ncomp]
    nrm = np.linalg.norm(a)
    if not np.isfinite(nrm) or nrm == 0:
        raise ValueError("a projection axis must be a non-zero vector")
    return a / nrm


def _axes(axes, ncomp: int) -> list[tuple[str, np.ndarray]]:
    if axes is None:
        axes = DEFAULT_AXES
    items = list(axes.items()) if isinstance(axes, dict) else list(axes)
    out = []
    for it in items:
        if isinstance(it, dict):
            name, d = it["name"], it["dir"]
        else:
            name, d = it
        out.append((str(name), _unit(d, ncomp)))
    return out


def _resolve(data, spec, edges):
    """A constant, a field name/expression or an element array -> scalar or edge values ``(k, n)``."""
    if spec is None:
        return None
    if isinstance(spec, (int, float, np.floating, np.integer)) and not isinstance(spec, bool):
        return float(spec)
    if isinstance(spec, str):
        s = spec.strip()
        try:
            return float(s)
        except ValueError:
            arr = data[s]
    else:
        arr = np.asarray(spec, dtype=float)
    return edge_values(arr, edges[:, 0], edges[:, 1])


# --------------------------------------------------------------------------- main entry point
def compute_forces(
    data,
    boundary,
    u: str = "u",
    v: str = "v",
    w: str | None = None,
    p: str = "p",
    rho=1.0,
    mu=1.0,
    axes=None,
    U_ref: float = 1.0,
    L_ref: float = 1.0,
    p_ref: float = 0.0,
    rho_ref: float | None = None,
    name: str | None = None,
    chain: bool = True,
) -> Forces:
    """Forces on the wall formed by external edges (see the module docstring).

    Parameters
    ----------
    data : SEMData2D
    boundary : :class:`~semscope.boundary.Boundary` or ``(k, 2)`` array of ``(elem, side)``
    u, v, w, p : field names (or calculator expressions); ``w`` adds a spanwise force
    rho, mu : constants or field names — ``mu`` is the dynamic viscosity in the
        stress, ``rho`` only enters the reference dynamic pressure
    axes : ``{name: (dx, dy)}`` (or a sequence of pairs) of projection axes, default x and y
    U_ref, L_ref, p_ref : reference velocity, length and pressure of the coefficients
    rho_ref : reference density; default the material ``rho`` (its boundary
        average when it is a field)
    chain : order the distributions by walking along the boundary
    """
    if isinstance(boundary, Boundary):
        edges = boundary.edges
        name = boundary.name if name is None else name
    else:
        edges = np.asarray(boundary, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        raise ValueError("the boundary has no edges")
    if name is None:
        name = ""
    x, y = data.x, data.y
    el, sd = edges[:, 0], edges[:, 1]
    W = edge_weights(x, y, edges)
    N = edge_normals(x, y, edges)
    nx, ny = N[..., 0], N[..., 1]
    ev = lambda a: edge_values(a, el, sd)  # noqa: E731

    pe = _resolve(data, p, edges)
    mue = _resolve(data, mu, edges)
    rhoe = _resolve(data, rho, edges)
    if not isinstance(pe, np.ndarray):
        pe = np.full(W.shape, float(pe))
    ux, uy = data.gradient(u)
    vx, vy = data.gradient(v)
    Sxx, Sxy, Syy = 2 * ev(ux), ev(uy) + ev(vx), 2 * ev(vy)
    # traction on the wall: n points out of the fluid, so t = p n - mu S n
    tpx, tpy = pe * nx, pe * ny
    tvx = -mue * (Sxx * nx + Sxy * ny)
    tvy = -mue * (Sxy * nx + Syy * ny)
    comps = ("x", "y")
    Fp = [float(np.sum(W * tpx)), float(np.sum(W * tpy))]
    Fv = [float(np.sum(W * tvx)), float(np.sum(W * tvy))]
    if w:
        wx, wy = data.gradient(w)
        tvz = -mue * (ev(wx) * nx + ev(wy) * ny)
        comps = ("x", "y", "z")
        Fp.append(0.0)
        Fv.append(float(np.sum(W * tvz)))
    length = float(np.sum(W))
    if rho_ref is None:
        rho_ref = float(rhoe) if not isinstance(rhoe, np.ndarray) else float(np.sum(W * rhoe) / length)
    q = 0.5 * float(rho_ref) * float(U_ref) ** 2
    if not (q > 0):
        raise ValueError("the reference dynamic pressure ½ ρ_ref U_ref² must be positive")
    if not (float(L_ref) > 0):
        raise ValueError("L_ref must be positive")

    # wall shear along the traversal tangent (-ny, nx); distributions ordered along the boundary
    # (mirrored elements run their nodes backwards: walk them in reverse)
    tau = -tvx * ny + tvy * nx
    xe, ye = edge_nodes(x, y, el, sd)
    sgn = jacobian_sign(x, y, el)
    s_local = edge_arclength(orient_rows(xe, sgn), orient_rows(ye, sgn))   # along the traversal direction
    order = chain_edges(x, y, edges) if chain and len(edges) > 1 else [(np.arange(len(edges)), False)]
    n = xe.shape[1]
    idx_e, idx_k, s_all, chain_id = [], [], [], []
    offset = 0.0
    for ci, (seq, closed) in enumerate(order):
        for j, e in enumerate(seq):
            ks = np.arange(n) if j == 0 else np.arange(1, n)      # traversal-order positions
            idx_e.append(np.full(ks.size, e))
            idx_k.append(ks if sgn[e] > 0 else n - 1 - ks)         # side index of each position
            s_all.append(offset + s_local[e, ks])
            chain_id.append(np.full(ks.size, ci))
            offset += s_local[e, -1]
    ie, ik = np.concatenate(idx_e), np.concatenate(idx_k)
    pick = lambda a: a[ie, ik]  # noqa: E731
    dist = {
        "s": np.concatenate(s_all),
        "x": pick(xe), "y": pick(ye), "nx": pick(nx), "ny": pick(ny),
        "p": pick(pe), "cp": (pick(pe) - float(p_ref)) / q,
        "tau": pick(tau), "cf": pick(tau) / q,
        "chain": np.concatenate(chain_id).astype(int),
        "edge": ie.astype(int),          # index into ``edges`` of the edge each point lies on
        "elem": el[ie].astype(int),      # local element index of that edge
    }
    return Forces(
        name=name, edges=edges, length=length,
        pressure=np.array(Fp), viscous=np.array(Fv), components=comps,
        axes=_axes(axes, len(comps)), q=q, rho_ref=float(rho_ref), U_ref=float(U_ref), L_ref=float(L_ref), p_ref=float(p_ref),
        distribution=dist,
    )


def sum_forces(results: list[Forces], name: str = "total") -> Forces:
    """Sum the forces of several boundaries (same reference values); distributions are concatenated."""
    if not results:
        raise ValueError("no forces to sum")
    r0 = results[0]
    dist = {}
    if all(r.distribution for r in results):
        offset, cid = 0.0, 0
        parts = {k: [] for k in r0.distribution}
        for r in results:
            for k, v in r.distribution.items():
                if k == "s":
                    parts[k].append(np.asarray(v) + offset)
                elif k == "chain":
                    parts[k].append(np.asarray(v) + cid)
                else:
                    parts[k].append(np.asarray(v))
            offset += float(np.asarray(r.distribution["s"])[-1]) if len(r.distribution["s"]) else 0.0
            cid += int(np.asarray(r.distribution["chain"]).max()) + 1 if len(r.distribution["chain"]) else 0
        dist = {k: np.concatenate(v) for k, v in parts.items()}
    return Forces(
        name=name,
        edges=np.concatenate([r.edges for r in results]),
        length=float(sum(r.length for r in results)),
        pressure=np.sum([r.pressure for r in results], axis=0),
        viscous=np.sum([r.viscous for r in results], axis=0),
        components=r0.components, axes=r0.axes, q=r0.q, rho_ref=r0.rho_ref, U_ref=r0.U_ref, L_ref=r0.L_ref, p_ref=r0.p_ref,
        distribution=dist,
    )
