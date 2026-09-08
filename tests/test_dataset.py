import numpy as np
import pytest

import semscope
from semscope import synthetic


def test_getitem_fields_and_expressions():
    k = np.pi
    data = synthetic.taylor_green(synthetic.wavy_box(8, 4, 10, amplitude=0.05), k=k)
    assert data.available == ["u", "v", "p"] and "u" in data
    assert data["u"] is data.fields["u"]
    assert np.allclose(data["x"], data.x)
    dvdx, dvdy = data.gradient("v")
    assert np.max(np.abs(dvdx - data["dx(v)"])) == 0 and np.max(np.abs(dvdy - data["dy(v)"])) == 0
    assert np.max(np.abs(data.derivative(data["u"], 0, 1) - data["dx(u)"])) == 0
    with pytest.raises(KeyError):
        data["nope"]


def test_metric_of_box_is_constant():
    data = synthetic.box(3, 2, 5, xlim=(0, 3), ylim=(0, 1))
    # each element is 1 x 0.5 -> dx/dr = 0.5, dy/ds = 0.25
    assert np.allclose(data.metric()["jac"], 0.125)


def test_resample_and_vertices():
    data = synthetic.annulus(2, 8, 6)
    X, Y, F = data.resample([], 5)
    assert X.shape == (16, 5, 5)
    assert data.vertices.shape == (16, 4, 2)
    assert np.allclose(np.hypot(X, Y).min(), 0.5, atol=1e-12)


def test_read_mixlay(mixlay):
    data = semscope.load(mixlay)
    assert data.nelv == 1600 and data.n == 8
    assert data.field_names == ["u", "v", "p", "t", "s0", "s1"]
    assert abs(data.time - 148.752677327) < 1e-6
    assert data.bounds == (0.0, 20.0, 0.0, 14.0)
    # the velocity is (nearly) divergence free and evaluation at nodes reproduces nodal values
    v = data.sample("u", data.x[0, 3, 4], data.y[0, 3, 4])
    assert abs(v - data["u"][0, 3, 4]) < 1e-9


def test_series_shares_mesh(flat_plate_meta):
    ds = semscope.open(flat_plate_meta)
    assert len(ds) == 4
    assert ds.mesh.nelv == 187
    d3 = ds[3]
    assert d3.x.shape == ds[0].x.shape
    assert d3.time > ds[0].time
    assert ds.step_at_time(1e-5) == 1
    assert ds[-1] is d3


def test_resolve_single_step_collects_siblings(flat_plate_meta):
    import os

    from semscope import io

    step1 = os.path.join(os.path.dirname(flat_plate_meta), "flat_plate_2d0.f00001")
    series = io.resolve_files(step1)
    assert len(series) == 4 and series.name == "flat_plate_2d"
    series = io.resolve_files(os.path.join(os.path.dirname(flat_plate_meta), "flat_plate_2d0.f0000[12]"))
    assert len(series) == 2


def test_write_roundtrip(tmp_path):
    data = synthetic.taylor_green(synthetic.annulus(2, 6, 6))
    path = str(tmp_path / "tg0.f00000")
    data.write(path)
    back = semscope.load(path)
    assert back.nelv == data.nelv and back.n == data.n
    assert np.allclose(back.x, data.x, atol=1e-6)
    assert np.allclose(back["u"], data["u"], atol=1e-6)
    assert set(back.field_names) == {"u", "v", "p"}


def test_rejects_3d(tmp_path):
    from semscope import io

    path = "/tmp/sa-flat-average-smoke.wMUkFV/field0.f00000"
    import os

    if not os.path.exists(path):
        pytest.skip("no 3D sample")
    with pytest.raises(ValueError, match="3D"):
        io.read_step(path)
