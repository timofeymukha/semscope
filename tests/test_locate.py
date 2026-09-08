import numpy as np

from semscope import spectral as sp
from semscope import synthetic
from semscope.locate import ElementLocator


def _roundtrip(data, rng, npts=3000):
    loc = ElementLocator(data.x, data.y)
    el = rng.integers(0, data.nelv, npts)
    r = rng.uniform(-1, 1, npts)
    s = rng.uniform(-1, 1, npts)
    px = sp.evaluate_at(data.x, el, r, s)
    py = sp.evaluate_at(data.y, el, r, s)
    res = loc.locate(px, py)
    assert res.found.all()
    assert np.array_equal(res.elem, el)
    assert np.allclose(res.r, r, atol=1e-8)
    assert np.allclose(res.s, s, atol=1e-8)
    return loc


def test_locate_box(rng):
    _roundtrip(synthetic.box(6, 4, 6), rng)


def test_locate_curved_annulus(rng):
    data = synthetic.annulus(3, 16, 8)
    loc = _roundtrip(data, rng)
    # points in the hole and outside are not found
    res = loc.locate([0.0, 0.2, 3.0], [0.0, -0.1, 0.0])
    assert (res.elem == -1).all()
    # every GLL node is found (also those on the outer boundary)
    res = loc.locate(data.x.ravel(), data.y.ravel())
    assert res.found.all()


def test_locate_wavy(rng):
    _roundtrip(synthetic.wavy_box(8, 4, 7, amplitude=0.1), rng)


def test_sample_is_spectrally_exact(rng):
    data = synthetic.wavy_box(6, 3, 9)
    # a polynomial of total degree <= 8 in (x, y) is exactly representable only
    # on affine elements; here we test a smooth field and expect ~1e-8 accuracy
    synthetic.add_analytic_field(data, "f", lambda x, y: np.sin(1.5 * x) * np.cos(2.0 * y))
    px = rng.uniform(0.1, 1.9, 400)
    py = rng.uniform(0.1, 0.9, 400)
    vals = data.sample("f", px, py)
    assert np.all(np.isfinite(vals))
    assert np.max(np.abs(vals - np.sin(1.5 * px) * np.cos(2.0 * py))) < 1e-6


def test_to_grid_and_line(rng):
    data = synthetic.box(4, 2, 6, xlim=(0, 1), ylim=(0, 0.5))
    synthetic.add_analytic_field(data, "g", lambda x, y: x**2 + 3 * y)
    xg, yg, vals = data.to_grid("g", nx=50)
    assert vals.shape == (len(yg), len(xg))
    XG, YG = np.meshgrid(xg, yg)
    assert np.allclose(vals, XG**2 + 3 * YG, atol=1e-10)
    dist, line = data.sample_line("g", (0, 0.1), (1, 0.1), 33)
    assert np.allclose(line, np.linspace(0, 1, 33) ** 2 + 0.3, atol=1e-10)
    outside = data.sample("g", [-1.0, 2.0], [0.1, 0.1])
    assert np.isnan(outside).all()
