import numpy as np
import pytest

from semview import spectral as sp


@pytest.mark.parametrize("n", [2, 3, 5, 8, 12, 16])
def test_gll_nodes_and_weights(n):
    x = sp.gll_nodes(n)
    w = sp.gll_weights(n)
    assert x[0] == -1.0 and x[-1] == 1.0
    assert np.all(np.diff(x) > 0)
    assert abs(w.sum() - 2.0) < 1e-13
    # GLL integrates polynomials up to degree 2n-3 exactly
    for k in range(0, 2 * n - 2):
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        assert abs(np.sum(w * x**k) - exact) < 1e-12


def test_gll_matches_pysemtools():
    pysem = pytest.importorskip("pysemtools.datatypes.coef")
    for n in (4, 8, 10):
        x, w = pysem.GLL_pwts(n)
        assert np.allclose(sp.gll_nodes(n), np.flip(x), atol=1e-14)
        assert np.allclose(sp.gll_weights(n), np.flip(w), atol=1e-14)
        _, _, _, dn = pysem.get_derivative_matrix(n, 2, apply_1d_operators=True)
        assert np.allclose(sp.derivative_matrix(n), dn, atol=1e-12)


@pytest.mark.parametrize("n", [3, 6, 9])
def test_lagrange_interpolation_is_exact_for_polynomials(n, rng):
    nodes = sp.gll_nodes(n)
    c = rng.standard_normal(n)  # degree n-1
    xt = np.linspace(-1, 1, 77)
    J = sp.interpolation_matrix(nodes, xt)
    assert J.shape == (77, n)
    assert np.allclose(J @ np.polyval(c, nodes), np.polyval(c, xt), atol=1e-12)
    # partition of unity and exactness at nodes
    assert np.allclose(J.sum(axis=1), 1.0)
    assert np.allclose(sp.lagrange_basis(nodes, nodes), np.eye(n))


def test_derivative_matrix(rng):
    n = 8
    nodes = sp.gll_nodes(n)
    c = rng.standard_normal(n)
    D = sp.derivative_matrix(n)
    assert np.allclose(D @ np.polyval(c, nodes), np.polyval(np.polyder(c), nodes), atol=1e-11)
    assert np.allclose(D @ np.ones(n), 0.0, atol=1e-13)


def test_tensor_operations(rng):
    n = 7
    g = sp.gll_nodes(n)
    R, S = np.meshgrid(g, g)

    def f(r, s):
        return 1 + 2 * r - 3 * s + r * s**2 - 0.5 * r**3 * s**4 + r**6

    data = np.stack([f(R, S), -f(R, S)])
    m = 11
    out = sp.resample_elements(data, m)
    u = sp.uniform_nodes(m)
    U, V = np.meshgrid(u, u)
    assert out.shape == (2, m, m)
    assert np.allclose(out[0], f(U, V), atol=1e-12)
    assert np.allclose(out[1], -f(U, V), atol=1e-12)
    rr = rng.uniform(-1, 1, 500)
    ss = rng.uniform(-1, 1, 500)
    el = rng.integers(0, 2, 500)
    vals = sp.evaluate_at(data, el, rr, ss)
    assert np.allclose(vals, np.where(el == 0, 1, -1) * f(rr, ss), atol=1e-12)
    # multi-field variant
    vals2 = sp.evaluate_at(np.stack([data, 2 * data]), el, rr, ss)
    assert np.allclose(vals2[1], 2 * vals, atol=1e-12)
    dr, ds = sp.element_derivatives(data)
    assert np.allclose(dr[0], 2 + S**2 - 1.5 * R**2 * S**4 + 6 * R**5, atol=1e-10)
    assert np.allclose(ds[0], -3 + 2 * R * S - 2.0 * R**3 * S**3, atol=1e-10)


def test_legendre_vandermonde_recovers_modes():
    n = 6
    V = sp.legendre_vandermonde(n)
    x = sp.gll_nodes(n)
    # P_3(x) = (5x^3 - 3x)/2 should have a single modal coefficient
    coeffs = np.linalg.solve(V, 0.5 * (5 * x**3 - 3 * x))
    assert np.allclose(coeffs, [0, 0, 0, 1, 0, 0], atol=1e-12)
