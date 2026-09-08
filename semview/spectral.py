"""Spectral (GLL / Lagrange) machinery for tensor-product quadrilateral elements.

Everything here is pure numpy and independent of the file format.  The
conventions follow Nek5000 / Neko / pysemtools:

* An element of polynomial order ``N = n - 1`` carries ``n x n`` Gauss-Lobatto-
  Legendre (GLL) nodes.
* Nodal arrays are indexed ``[element, j, i]`` where ``i`` runs along the
  reference coordinate ``r`` (the fastest, "x" index) and ``j`` along ``s``.
* The reference element is ``[-1, 1]^2``.

The key operation that makes SEM data special is that the nodal values define
a polynomial on every element, so the field can be evaluated *anywhere* — not
only at the GLL nodes.  :func:`interpolation_matrix` builds the 1D Lagrange
operator for that, :func:`tensor_apply` applies it in both directions and
:func:`evaluate_at` evaluates at scattered ``(r, s)`` points.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "gll_nodes",
    "gll_weights",
    "barycentric_weights",
    "lagrange_basis",
    "interpolation_matrix",
    "derivative_matrix",
    "tensor_apply",
    "resample_elements",
    "evaluate_at",
    "element_derivatives",
    "uniform_nodes",
    "legendre_vandermonde",
]


# --------------------------------------------------------------------------- #
# 1D nodes, weights, basis
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _gll(n: int) -> tuple[np.ndarray, np.ndarray]:
    """GLL nodes (ascending) and weights for ``n`` points, computed to double precision."""
    if n < 2:
        raise ValueError("A GLL rule needs at least 2 points")
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    # Interior nodes are the roots of P'_{n-1}, i.e. the eigenvalues of the
    # Jacobi matrix of the (1,1) Jacobi polynomials; refine with Newton.
    N = n - 1
    k = np.arange(1, N)
    # Chebyshev-Gauss-Lobatto initial guess
    x = -np.cos(np.pi * np.arange(n) / N)
    # Newton iterations on the Legendre Vandermonde (Pozrikidis / Canuto style)
    P = np.zeros((n, n))
    xold = np.full(n, 2.0)
    for _ in range(100):
        if np.max(np.abs(x - xold)) < 1e-15:
            break
        xold = x.copy()
        P[:, 0] = 1.0
        P[:, 1] = x
        for kk in range(2, n):
            P[:, kk] = ((2 * kk - 1) * x * P[:, kk - 1] - (kk - 1) * P[:, kk - 2]) / kk
        x = xold - (x * P[:, N] - P[:, N - 1]) / (n * P[:, N])
    x[0], x[-1] = -1.0, 1.0
    x[1:-1] = np.sort(x[1:-1])
    # weights: w_i = 2 / (N (N+1) P_N(x_i)^2)
    P[:, 0] = 1.0
    P[:, 1] = x
    for kk in range(2, n):
        P[:, kk] = ((2 * kk - 1) * x * P[:, kk - 1] - (kk - 1) * P[:, kk - 2]) / kk
    w = 2.0 / (N * n * P[:, N] ** 2)
    x.setflags(write=False)
    w.setflags(write=False)
    del k
    return x, w


def gll_nodes(n: int) -> np.ndarray:
    """Return the ``n`` GLL nodes on ``[-1, 1]`` in ascending order."""
    return _gll(n)[0]


def gll_weights(n: int) -> np.ndarray:
    """Return the ``n`` GLL quadrature weights."""
    return _gll(n)[1]


def uniform_nodes(m: int, closed: bool = True) -> np.ndarray:
    """``m`` equispaced points on ``[-1, 1]`` (end points included when ``closed``)."""
    if closed:
        return np.linspace(-1.0, 1.0, m)
    return -1.0 + (2.0 * np.arange(m) + 1.0) / m


def barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    """Barycentric weights ``w_j = 1 / prod_{k != j} (x_j - x_k)`` of a node set."""
    nodes = np.asarray(nodes, dtype=float)
    diff = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(diff, 1.0)
    return 1.0 / np.prod(diff, axis=1)


def lagrange_basis(nodes: np.ndarray, x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Evaluate all Lagrange basis polynomials of ``nodes`` at points ``x``.

    Returns an array of shape ``x.shape + (n,)`` with ``out[..., j] = l_j(x)``.
    Uses the numerically stable second barycentric formula and is exact at the
    nodes themselves.
    """
    nodes = np.asarray(nodes, dtype=float)
    x = np.asarray(x, dtype=float)
    if weights is None:
        weights = barycentric_weights(nodes)
    d = x[..., None] - nodes  # (..., n)
    exact = d == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = weights / d
        out = t / np.sum(t, axis=-1, keepdims=True)
    hit = np.any(exact, axis=-1)
    if np.any(hit):
        out[hit] = exact[hit].astype(float)
    return out


def interpolation_matrix(nodes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Matrix ``J`` (``m x n``) with ``J[a, j] = l_j(targets[a])``.

    ``J @ f`` interpolates nodal values ``f`` to the target points.
    """
    return lagrange_basis(nodes, np.asarray(targets, dtype=float).ravel())


@lru_cache(maxsize=None)
def _interp_matrix_cached(n: int, m: int, kind: str) -> np.ndarray:
    nodes = gll_nodes(n)
    if kind == "uniform":
        tgt = uniform_nodes(m)
    elif kind == "gll":
        tgt = gll_nodes(m)
    elif kind == "cell_centers":
        tgt = uniform_nodes(m, closed=False)
    else:
        raise ValueError(kind)
    J = interpolation_matrix(nodes, tgt)
    J.setflags(write=False)
    return J


def gll_to_uniform(n: int, m: int) -> np.ndarray:
    """Cached interpolation matrix from ``n`` GLL nodes to ``m`` equispaced points."""
    return _interp_matrix_cached(n, m, "uniform")


@lru_cache(maxsize=None)
def derivative_matrix(n: int) -> np.ndarray:
    """Differentiation matrix ``D`` on the GLL nodes: ``(D f)_i = f'(x_i)``.

    Built from the barycentric weights (Berrut & Trefethen 2004, eq. 9.4),
    which is stable for high orders.
    """
    x = gll_nodes(n)
    w = barycentric_weights(x)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
        D[i, i] = -np.sum(D[i, :])
    D.setflags(write=False)
    return D


@lru_cache(maxsize=None)
def legendre_vandermonde(n: int) -> np.ndarray:
    """Vandermonde matrix ``V[i, k] = P_k(x_i)`` of the Legendre polynomials on the GLL nodes.

    ``solve(V, f)`` gives the modal (Legendre) coefficients of nodal data.
    """
    x = gll_nodes(n)
    V = np.zeros((n, n))
    V[:, 0] = 1.0
    if n > 1:
        V[:, 1] = x
    for k in range(2, n):
        V[:, k] = ((2 * k - 1) * x * V[:, k - 1] - (k - 1) * V[:, k - 2]) / k
    V.setflags(write=False)
    return V


# --------------------------------------------------------------------------- #
# Tensor-product operations on element arrays [e, j, i]
# --------------------------------------------------------------------------- #
def tensor_apply(data: np.ndarray, Ar: np.ndarray, As: np.ndarray | None = None) -> np.ndarray:
    """Apply ``As`` along ``s`` (axis -2) and ``Ar`` along ``r`` (axis -1).

    ``data`` has shape ``(..., ns, nr)``; the result has shape ``(..., ms, mr)``
    where ``As`` is ``(ms, ns)`` and ``Ar`` is ``(mr, nr)``.  If ``As`` is None,
    ``Ar`` is used for both directions.
    """
    if As is None:
        As = Ar
    data = np.asarray(data)
    # (..., ns, nr) @ (nr, mr) -> (..., ns, mr); then As @ -> (..., ms, mr)
    tmp = data @ Ar.T
    return As @ tmp


def resample_elements(data: np.ndarray, m: int, kind: str = "uniform") -> np.ndarray:
    """Spectrally resample element data ``(nelv, n, n)`` onto an ``m x m`` grid per element.

    ``kind`` is ``"uniform"`` (closed equispaced grid, ideal for plotting),
    ``"gll"`` (GLL grid of another order, i.e. p-refinement/coarsening) or
    ``"cell_centers"``.
    """
    data = np.asarray(data)
    n = data.shape[-1]
    J = _interp_matrix_cached(n, m, kind)
    return tensor_apply(data, J)


def evaluate_at(
    data: np.ndarray,
    elem: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    nodes_r: np.ndarray | None = None,
    nodes_s: np.ndarray | None = None,
    chunk: int = 65536,
) -> np.ndarray:
    """Evaluate nodal data at scattered reference points.

    Parameters
    ----------
    data : (nelv, ns, nr) array (or (nfields, nelv, ns, nr) for several fields)
    elem : (npts,) int array — element that contains each point
    r, s : (npts,) reference coordinates of each point
    """
    data = np.asarray(data)
    multi = data.ndim == 4
    if not multi:
        data = data[None]
    nf, _, ns, nr = data.shape
    if nodes_r is None:
        nodes_r = gll_nodes(nr)
    if nodes_s is None:
        nodes_s = gll_nodes(ns)
    wr = barycentric_weights(nodes_r)
    ws = barycentric_weights(nodes_s)
    elem = np.asarray(elem)
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    npts = elem.size
    out = np.empty((nf, npts), dtype=np.result_type(data.dtype, np.float64))
    for a in range(0, npts, chunk):
        b = min(npts, a + chunk)
        Lr = lagrange_basis(nodes_r, r[a:b], wr)  # (p, nr)
        Ls = lagrange_basis(nodes_s, s[a:b], ws)  # (p, ns)
        block = data[:, elem[a:b]]  # (nf, p, ns, nr)
        tmp = np.einsum("fpji,pi->fpj", block, Lr)
        out[:, a:b] = np.einsum("fpj,pj->fp", tmp, Ls)
    return out if multi else out[0]


def element_derivatives(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference-space derivatives ``(d/dr, d/ds)`` of nodal data ``(..., n, n)``."""
    data = np.asarray(data)
    ns, nr = data.shape[-2:]
    Dr = derivative_matrix(nr)
    Ds = derivative_matrix(ns)
    d_dr = data @ Dr.T  # differentiate along i (r)
    d_ds = Ds @ data  # differentiate along j (s)
    return d_dr, d_ds
