import numpy as np
import pytest

import semview
from semview import synthetic


def test_derived_quantities_taylor_green():
    k = np.pi
    data = synthetic.taylor_green(synthetic.wavy_box(8, 4, 10, amplitude=0.05), k=k)
    x, y = data.x, data.y
    vort = data["vorticity"]
    exact = 2 * k * np.sin(k * x) * np.sin(k * y)
    assert np.max(np.abs(vort - exact)) < 1e-4  # spectral accuracy at order 9
    assert np.max(np.abs(data["divergence"])) < 1e-4
    assert np.allclose(data["speed"], np.hypot(data["u"], data["v"]))
    assert "vorticity" in data.available and "vorticity" in data
    assert data["du/dx"].shape == x.shape
    with pytest.raises(KeyError):
        data["nope"]


def test_jacobian_of_box_is_constant():
    data = synthetic.box(3, 2, 5, xlim=(0, 3), ylim=(0, 1))
    # each element is 1 x 0.5 -> dx/dr = 0.5, dy/ds = 0.25
    assert np.allclose(data["jacobian"], 0.125)


def test_resample_and_vertices():
    data = synthetic.annulus(2, 8, 6)
    X, Y, F = data.resample([], 5)
    assert X.shape == (16, 5, 5)
    assert data.vertices.shape == (16, 4, 2)
    assert np.allclose(np.hypot(X, Y).min(), 0.5, atol=1e-12)


def test_spectral_decay_indicator():
    data = synthetic.box(2, 2, 8, xlim=(0, 1), ylim=(0, 1))
    synthetic.add_analytic_field(data, "smooth", lambda x, y: x + y)
    synthetic.add_analytic_field(data, "rough", lambda x, y: np.sin(40 * x * y))
    assert data.spectral_decay("smooth").max() < 1e-20
    assert data.spectral_decay("rough").max() > 1e-3


def test_read_mixlay(mixlay):
    data = semview.load(mixlay)
    assert data.nelv == 1600 and data.n == 8
    assert data.field_names == ["u", "v", "p", "t", "s0", "s1"]
    assert abs(data.time - 148.752677327) < 1e-6
    assert data.bounds == (0.0, 20.0, 0.0, 14.0)
    # the velocity is (nearly) divergence free and evaluation at nodes reproduces nodal values
    v = data.sample("u", data.x[0, 3, 4], data.y[0, 3, 4])
    assert abs(v - data["u"][0, 3, 4]) < 1e-9


def test_series_shares_mesh(flat_plate_meta):
    ds = semview.open(flat_plate_meta)
    assert len(ds) == 4
    assert ds.mesh.nelv == 187
    d3 = ds[3]
    assert d3.x.shape == ds[0].x.shape
    assert d3.time > ds[0].time
    assert ds.step_at_time(1e-5) == 1
    assert ds[-1] is d3


def test_resolve_single_step_collects_siblings(flat_plate_meta):
    import os

    from semview import io

    step1 = os.path.join(os.path.dirname(flat_plate_meta), "flat_plate_2d0.f00001")
    series = io.resolve_files(step1)
    assert len(series) == 4 and series.name == "flat_plate_2d"
    series = io.resolve_files(os.path.join(os.path.dirname(flat_plate_meta), "flat_plate_2d0.f0000[12]"))
    assert len(series) == 2


def test_write_roundtrip(tmp_path):
    data = synthetic.taylor_green(synthetic.annulus(2, 6, 6))
    path = str(tmp_path / "tg0.f00000")
    data.write(path)
    back = semview.load(path)
    assert back.nelv == data.nelv and back.n == data.n
    assert np.allclose(back.x, data.x, atol=1e-6)
    assert np.allclose(back["u"], data["u"], atol=1e-6)
    assert set(back.field_names) == {"u", "v", "p"}


def test_rejects_3d(tmp_path):
    from semview import io

    path = "/tmp/sa-flat-average-smoke.wMUkFV/field0.f00000"
    import os

    if not os.path.exists(path):
        pytest.skip("no 3D sample")
    with pytest.raises(ValueError, match="3D"):
        io.read_step(path)
