import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import semview  # noqa: E402
from semview import synthetic  # noqa: E402
from semview.plotting import Plotter, quick_plot  # noqa: E402


@pytest.fixture
def data():
    return synthetic.taylor_green(synthetic.annulus(2, 12, 6), k=2.0)


def test_all_layers_render(tmp_path, data):
    pl = Plotter(figsize=(5, 5), dpi=60)
    pl.add_field(data, "dx(v) - dy(u)", cmap="RdBu_r", clim=(-4, 4))
    pl.add_contours(data, "p", levels=6, colors="k")
    pl.add_mesh(data)
    pl.add_nodes(data, size=1)
    pl.add_vectors(data, spacing=8)
    pl.add_streamlines(data, density=0.5, resolution=60)
    pl.add_element_ids(data, fontsize=4)
    pl.add_points([1.0], [0.0])
    pl.set_labels()
    out = pl.save(str(tmp_path / "all.png"))
    assert os.path.getsize(out) > 1000
    pl.close()


@pytest.mark.parametrize("method", ["spectral", "nodal", "pixel"])
def test_field_methods(tmp_path, data, method):
    pl = Plotter(figsize=(4, 4), dpi=50)
    pl.add_field(data, "u", method=method, resolution=4 if method != "pixel" else "auto")
    pl.set_view((0.5, 1.5), (-0.5, 0.5))
    pl.render()
    assert pl._artists
    pl.save(str(tmp_path / f"{method}.png"))
    pl.close()


def test_auto_resolution_grows_when_zooming(data):
    pl = Plotter(figsize=(6, 6), dpi=100)
    pl.add_field(data, "u")
    pl.render()
    m_full = pl._resolution_per_element(data, *pl._limits(), "auto")
    pl.set_view((0.9, 1.1), (-0.1, 0.1))
    pl.render()
    m_zoom = pl._resolution_per_element(data, *pl._limits(), "auto")
    assert m_zoom.max() > m_full.max()
    pl.close()


def test_quick_plot_and_log(tmp_path, data):
    data.add_field("pos", np.abs(data["u"]) + 1e-3)
    pl = quick_plot(data, "pos", mesh=True, log=True, cmap="magma")
    pl.save(str(tmp_path / "q.png"))
    pl.close()


def test_cli_info_and_png(tmp_path, data, capsys):
    from semview.cli import main

    path = str(tmp_path / "tg0.f00000")
    data.write(path)
    assert main(["--info", path]) == 0
    out = capsys.readouterr().out
    assert "order N = 5" in out and "['u', 'v', 'p']" in out
    png = str(tmp_path / "tg.png")
    assert main(["--png", png, "--field", "dx(v) - dy(u)", "--mesh", path]) == 0
    assert os.path.getsize(png) > 1000
