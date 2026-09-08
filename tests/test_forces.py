"""Forces on boundaries: analytic checks of the pressure and viscous parts, coefficients and distributions."""

import numpy as np
import pytest

from semscope import synthetic
from semscope.boundary import detect_boundaries
from semscope.forces import compute_forces, edge_arclength, edge_weights, sum_forces


def _annulus(n=8, r_in=0.5, r_out=1.5):
    d = synthetic.annulus(3, 32, n, r_in=r_in, r_out=r_out)
    inner, outer = None, None
    for b in detect_boundaries(d.x, d.y, angle=90):
        xe, ye = b.nodes()
        r = np.hypot(xe, ye).mean()
        if abs(r - r_in) < 1e-6:
            inner = b
        elif abs(r - r_out) < 1e-6:
            outer = b
    assert inner is not None and outer is not None and inner.closed and outer.closed
    return d, inner, outer


def test_edge_quadrature_and_arclength():
    d, inner, outer = _annulus()
    assert abs(edge_weights(d.x, d.y, inner.edges).sum() - 2 * np.pi * 0.5) < 1e-9
    assert abs(edge_weights(d.x, d.y, outer.edges).sum() - 2 * np.pi * 1.5) < 1e-9
    xe, ye = inner.nodes()
    s = edge_arclength(xe, ye)
    assert np.all(np.diff(s, axis=1) > 0)
    assert np.allclose(s[:, -1], 2 * np.pi * 0.5 / len(inner), rtol=1e-4)


def test_pressure_force_on_cylinder():
    # p = x on a hole of radius R: the force on the body is -pi R^2 e_x (pushed towards -x)
    d, inner, outer = _annulus()
    synthetic.add_analytic_field(d, "u", lambda x, y: 0 * x)
    synthetic.add_analytic_field(d, "v", lambda x, y: 0 * x)
    synthetic.add_analytic_field(d, "p", lambda x, y: x)
    F = compute_forces(d, inner, mu=1.0)
    assert np.allclose(F.pressure, [-np.pi * 0.5**2, 0.0], atol=1e-8)
    assert np.allclose(F.viscous, 0.0, atol=1e-10)
    assert abs(F.length - np.pi) < 1e-9
    # the outer circle is the domain's outer wall: opposite sign, radius 1.5
    G = compute_forces(d, outer)
    assert np.allclose(G.pressure, [np.pi * 1.5**2, 0.0], atol=1e-8)
    # coefficients: q = 1/2 rho U^2 with the defaults rho = U = L = 1
    rows = {r["name"]: r for r in F.table()}
    assert abs(rows["x"]["coefficient"] - (-np.pi * 0.25) / 0.5) < 1e-8 and abs(rows["y"]["coefficient"]) < 1e-8
    # reference values and a rotated axis system
    H = compute_forces(d, inner, U_ref=2.0, L_ref=0.5, rho=1.2, axes={"d": (1, 1), "l": (-1, 1)})
    assert abs(H.q - 0.5 * 1.2 * 4.0) < 1e-12
    hd = {r["name"]: r for r in H.table()}
    assert abs(hd["d"]["total"] - (-np.pi * 0.25) / np.sqrt(2)) < 1e-8
    assert abs(hd["d"]["coefficient"] - hd["d"]["total"] / (H.q * 0.5)) < 1e-12
    # Cp distribution follows p - p_ref along the wall, ordered by arc length
    dist = F.distribution
    assert np.all(np.diff(dist["s"]) >= -1e-12) and abs(dist["s"][-1] - np.pi) < 1e-3
    assert np.allclose(dist["cp"], dist["x"] / 0.5, atol=1e-10)
    assert len(dist["edge"]) == len(dist["elem"]) == len(dist["s"]) and set(dist["elem"]) == set(inner.edges[:, 0])
    assert np.count_nonzero(np.diff(dist["edge"])) == len(inner) - 1   # one crossing per element boundary
    assert np.allclose(np.hypot(dist["x"], dist["y"]), 0.5, atol=1e-9)


def test_dalembert_potential_flow():
    # potential flow around the cylinder: Bernoulli pressure gives zero force (d'Alembert)
    d, inner, outer = _annulus(n=10, r_in=1.0, r_out=6.0)
    U, R = 1.0, 1.0
    r2 = lambda x, y: x**2 + y**2  # noqa: E731
    synthetic.add_analytic_field(d, "u", lambda x, y: U * (1 - R**2 * (x**2 - y**2) / r2(x, y) ** 2))
    synthetic.add_analytic_field(d, "v", lambda x, y: -U * R**2 * 2 * x * y / r2(x, y) ** 2)
    synthetic.add_analytic_field(d, "p", lambda x, y: 0.5 * U**2 - 0.5 * (d["u"] ** 2 + d["v"] ** 2))
    F = compute_forces(d, inner, mu=0.01)
    assert np.all(np.abs(F.pressure) < 1e-4)
    # surface pressure coefficient of potential flow: Cp = 1 - 4 sin^2(theta)
    th = np.arctan2(F.distribution["y"], F.distribution["x"])
    assert np.allclose(F.distribution["cp"], 1 - 4 * np.sin(th) ** 2, atol=2e-3)


def test_viscous_force_shear_flow():
    # u = (y, 0) between the walls y = 0 and y = 1 of a box: shear stress mu on both walls
    d = synthetic.box(6, 3, 7, xlim=(0, 2), ylim=(0, 1))
    synthetic.add_analytic_field(d, "u", lambda x, y: y)
    synthetic.add_analytic_field(d, "v", lambda x, y: 0 * x)
    synthetic.add_analytic_field(d, "p", lambda x, y: 0 * x + 3.0)
    synthetic.add_analytic_field(d, "visc", lambda x, y: 0 * x + 0.1)
    sides = {}
    for b in detect_boundaries(d.x, d.y, angle=90):
        xe, ye = b.nodes()
        key = "bottom" if np.allclose(ye, 0) else "top" if np.allclose(ye, 1) else "left" if np.allclose(xe, 0) else "right"
        sides[key] = b
    bottom = compute_forces(d, sides["bottom"], mu=0.1)
    top = compute_forces(d, sides["top"], mu="visc")
    # the fluid drags the bottom wall forward (+x) and the top wall backward (-x); pressure pushes into the walls
    assert np.allclose(bottom.viscous, [0.1 * 2.0, 0.0], atol=1e-10)
    assert np.allclose(top.viscous, [-0.1 * 2.0, 0.0], atol=1e-10)
    assert np.allclose(bottom.pressure, [0.0, -3.0 * 2.0], atol=1e-10)
    assert np.allclose(top.pressure, [0.0, 3.0 * 2.0], atol=1e-10)
    # Cf along the traversal direction: +x on the bottom (domain to the left when walking +x)
    assert np.allclose(bottom.distribution["cf"], 0.1 / 0.5, atol=1e-10)
    assert np.allclose(bottom.distribution["s"][-1], 2.0, rtol=1e-4)
    # a spanwise velocity adds a z force through its wall-normal gradient
    synthetic.add_analytic_field(d, "w", lambda x, y: 2 * y)
    b3 = compute_forces(d, sides["bottom"], w="w", mu=0.1)
    assert b3.components == ("x", "y", "z") and abs(b3.viscous[2] - 0.1 * 2 * 2.0) < 1e-10
    # density as a field: the reference density is its boundary average
    synthetic.add_analytic_field(d, "dens", lambda x, y: 2.0 + 0 * x)
    b4 = compute_forces(d, sides["bottom"], rho="dens", U_ref=3.0)
    assert abs(b4.rho_ref - 2.0) < 1e-12 and abs(b4.q - 0.5 * 2.0 * 9.0) < 1e-12
    # sums
    tot = sum_forces([bottom, top])
    assert np.allclose(tot.total, [0.0, 0.0], atol=1e-10) and abs(tot.length - 4.0) < 1e-9
    assert len(tot.distribution["s"]) == len(bottom.distribution["s"]) + len(top.distribution["s"])
    with pytest.raises(ValueError):
        compute_forces(d, sides["top"], axes={"a": (0, 0)})
    with pytest.raises(ValueError):
        compute_forces(d, sides["top"], U_ref=0.0)
    with pytest.raises(KeyError):
        compute_forces(d, sides["top"], p="nope")
