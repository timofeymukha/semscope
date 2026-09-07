"""Analytic test meshes and fields.

Handy for unit tests, benchmarks and for trying the tool without a solver
run.  All generators return :class:`~semview.dataset.SEMData2D` objects whose
geometry is the exact polynomial interpolant of a smooth mapping, so curved
elements are genuinely curved.
"""

from __future__ import annotations

import numpy as np

from . import spectral as sp
from .dataset import SEMData2D

__all__ = ["box", "annulus", "wavy_box", "add_analytic_field"]


def _structured(nx: int, ny: int, n: int, mapping):
    """Build ``nx * ny`` elements of order ``n-1`` from a mapping ``(xi, eta) in [0,1]^2 -> (x, y)``."""
    g = sp.gll_nodes(n)
    R, S = np.meshgrid(g, g)  # R[j, i] = g[i]
    xs, ys = [], []
    for ey in range(ny):
        for ex in range(nx):
            xi = (ex + (R + 1) / 2) / nx
            eta = (ey + (S + 1) / 2) / ny
            x, y = mapping(xi, eta)
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)


def box(nx: int = 8, ny: int = 6, n: int = 8, xlim=(0.0, 2.0), ylim=(0.0, 1.0), name="box") -> SEMData2D:
    """A Cartesian box of ``nx x ny`` straight-sided elements."""

    def mapping(xi, eta):
        return xlim[0] + xi * (xlim[1] - xlim[0]), ylim[0] + eta * (ylim[1] - ylim[0])

    x, y = _structured(nx, ny, n, mapping)
    return SEMData2D(x, y, {}, name=name, elmap=np.arange(1, nx * ny + 1))


def annulus(nr: int = 4, ntheta: int = 24, n: int = 8, r_in: float = 0.5, r_out: float = 1.5, theta=(0.0, 2 * np.pi), name="annulus") -> SEMData2D:
    """A (sector of an) annulus with curved elements."""

    def mapping(xi, eta):
        r = r_in + xi * (r_out - r_in)
        th = theta[0] + eta * (theta[1] - theta[0])
        return r * np.cos(th), r * np.sin(th)

    x, y = _structured(nr, ntheta, n, mapping)
    return SEMData2D(x, y, {}, name=name, elmap=np.arange(1, nr * ntheta + 1))


def wavy_box(nx: int = 10, ny: int = 5, n: int = 7, amplitude: float = 0.08, name="wavy") -> SEMData2D:
    """A box whose element edges follow a sinusoidal perturbation (all elements curved)."""

    def mapping(xi, eta):
        x = 2.0 * xi + amplitude * np.sin(2 * np.pi * eta) * np.sin(np.pi * xi)
        y = eta + amplitude * np.sin(2 * np.pi * xi) * np.sin(np.pi * eta)
        return x, y

    x, y = _structured(nx, ny, n, mapping)
    return SEMData2D(x, y, {}, name=name, elmap=np.arange(1, nx * ny + 1))


def add_analytic_field(data: SEMData2D, name: str, func) -> np.ndarray:
    """Sample ``func(x, y)`` at the GLL nodes and store it as field ``name``."""
    vals = func(data.x, data.y)
    data.add_field(name, np.asarray(vals, dtype=float))
    return vals


def taylor_green(data: SEMData2D, k: float = np.pi) -> SEMData2D:
    """Add a Taylor-Green-like velocity ``u = sin(kx) cos(ky), v = -cos(kx) sin(ky)`` and its pressure."""
    add_analytic_field(data, "u", lambda x, y: np.sin(k * x) * np.cos(k * y))
    add_analytic_field(data, "v", lambda x, y: -np.cos(k * x) * np.sin(k * y))
    add_analytic_field(data, "p", lambda x, y: 0.25 * (np.cos(2 * k * x) + np.cos(2 * k * y)))
    return data
