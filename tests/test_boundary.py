import numpy as np
import pytest

from semscope import synthetic
from semscope.boundary import Boundary, chain_edges, detect_boundaries, external_edges


def test_box_has_four_sides():
    d = synthetic.box(6, 4, 6, xlim=(0, 3), ylim=(0, 2))
    ext = d.external_edges()
    assert len(ext) == 2 * (6 + 4)
    b = d.detect_boundaries(90)
    assert len(b) == 4
    assert sorted(len(x) for x in b) == [4, 4, 6, 6]
    assert all(not x.closed for x in b)
    lengths = sorted(round(x.length(), 12) for x in b)
    assert lengths == [2.0, 2.0, 3.0, 3.0]
    # a very large feature angle keeps the whole loop together
    whole = d.detect_boundaries(179)
    assert len(whole) == 1 and whole[0].closed and len(whole[0]) == 20
    assert abs(whole[0].length() - 10.0) < 1e-12


def test_annulus_inner_and_outer():
    d = synthetic.annulus(3, 24, 7, r_in=0.5, r_out=1.5)
    b = d.detect_boundaries(90)
    assert len(b) == 2 and all(x.closed for x in b)
    lengths = sorted(x.length() for x in b)
    assert abs(lengths[0] - 2 * np.pi * 0.5) < 1e-8
    assert abs(lengths[1] - 2 * np.pi * 1.5) < 1e-8
    # outward normals point away from the origin on the outer circle, towards it on the inner
    outer = max(b, key=lambda x: x.length())
    xe, ye = outer.nodes()
    nrm = outer.normals()
    radial = np.stack([xe, ye], -1) / 1.5
    assert np.allclose(np.sum(nrm * radial, axis=-1), 1.0, atol=1e-8)
    poly = outer.polyline(10)
    assert np.allclose(poly[0], poly[-1])
    assert np.allclose(np.hypot(poly[:, 0], poly[:, 1]), 1.5, atol=1e-10)


def test_annulus_sector_and_curved_box():
    d = synthetic.annulus(2, 8, 6, theta=(0.0, np.pi / 2))
    assert len(d.detect_boundaries(90)) == 4
    # 45 degree splitting also separates the arcs from the radial sides; a tiny angle splits every edge
    assert len(d.detect_boundaries(45)) == 4
    assert len(d.detect_boundaries(1e-9)) == len(d.external_edges())
    w = synthetic.wavy_box(8, 4, 7, amplitude=0.08)
    assert len(w.detect_boundaries(90)) == 4


def test_chain_and_manual_boundary():
    d = synthetic.box(3, 2, 5)
    ext = d.external_edges()
    loops = chain_edges(d.x, d.y, ext)
    assert len(loops) == 1 and loops[0][1] and len(loops[0][0]) == 10
    # manual selection of the bottom edges (elements 0..2, side 0) chains into one open boundary
    b = d.boundary([(0, 0), (2, 0), (1, 0)], name="wall")
    assert isinstance(b, Boundary) and not b.closed and len(b) == 3
    assert list(b.edges[:, 0]) == [0, 1, 2]
    assert abs(b.length() - 2.0) < 1e-12
    vals = b.values(d.x)
    assert vals.shape == (3, 5) and np.all(np.diff(vals.ravel()) >= -1e-12)
    assert b.to_dict()["edges"] == [[0, 0], [1, 0], [2, 0]]


def test_plotter_boundaries(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import semscope

    d = synthetic.taylor_green(synthetic.annulus(2, 12, 6))
    pl = semscope.Plotter(figsize=(4, 4), dpi=50)
    pl.add_field(d, "u")
    pl.add_boundaries(d)
    pl.save(str(tmp_path / "b.png"))
    assert any(L.kind == "boundaries" for L in pl._layers)
    pl.close()


def test_normal_line_and_mirrored_elements():
    d = synthetic.annulus(2, 8, 6, r_in=0.5, r_out=1.0)
    outer = max(d.detect_boundaries(90), key=lambda b: b.length())
    p0, p1 = outer.normal_line(0, 2, 0.25)
    assert abs(np.hypot(*p0) - 1.0) < 1e-12 and abs(np.hypot(*p1) - 0.75) < 1e-9   # inward
    p0, p1 = d.normal_line(int(outer.edges[0, 0]), int(outer.edges[0, 1]), 2, -0.25)
    assert abs(np.hypot(*p1) - 1.25) < 1e-9   # negative length goes outward
    # a mirrored mesh (negative Jacobian) must still give outward normals
    m = synthetic.box(3, 2, 5, xlim=(0, 1), ylim=(0, 1))
    from semscope.dataset import SEMData2D
    from semscope.boundary import edge_normals, jacobian_sign

    mir = SEMData2D(-m.x, m.y, {}, name="mirrored")
    assert np.all(jacobian_sign(mir.x, mir.y, np.arange(mir.nelv)) == -1)
    ext = mir.external_edges()
    nrm = edge_normals(mir.x, mir.y, ext)
    xe, ye = __import__("semscope.boundary", fromlist=["edge_nodes"]).edge_nodes(mir.x, mir.y, ext[:, 0], ext[:, 1])
    # outward: moving a little along the normal leaves the unit square [-1,0]x[0,1]
    px, py = xe + 1e-3 * nrm[..., 0], ye + 1e-3 * nrm[..., 1]
    outside = (px < -1) | (px > 0) | (py < 0) | (py > 1)
    assert outside.all()


def test_normal_line_at_parametric():
    from semscope.boundary import edge_point
    from semscope import spectral as sp

    d = synthetic.annulus(2, 8, 6, r_in=0.5, r_out=1.0)
    outer = max(d.detect_boundaries(90), key=lambda b: b.length())
    e, sd = int(outer.edges[3, 0]), int(outer.edges[3, 1])
    # at a GLL node the parametric version agrees with the node version
    g = sp.gll_nodes(d.n)
    p0a, p1a = d.normal_line(e, sd, 2, 0.3)
    p0b, p1b = d.normal_line_at(e, sd, g[2], 0.3)
    assert np.allclose(p0a, p0b) and np.allclose(p1a, p1b)
    # anywhere along the edge: on the circle, normal radial, inward step
    for t in (-0.73, 0.11, 0.9):
        px, py, nx, ny = edge_point(d.x, d.y, e, sd, t)
        # the interpolated edge deviates from the exact circle by the geometry's own interpolation error (~1e-7 at order 5)
        assert abs(np.hypot(px, py) - 1.0) < 1e-5
        assert abs(nx * px + ny * py - 1.0) < 1e-5
        p0, p1 = outer.normal_line_at(3, t, 0.25)
        assert abs(np.hypot(*p1) - 0.75) < 1e-5


def test_mixed_orientation_mesh_chains_and_orders():
    """Meshes often mix positive and mirrored elements; boundaries must still chain into one loop."""
    from semscope.boundary import chain_edges, jacobian_sign
    from semscope.dataset import SEMData2D
    from semscope.forces import compute_forces

    box = synthetic.box(9, 4, 6, xlim=(0, 3), ylim=(0, 1))
    x, y = box.x.copy(), box.y.copy()
    flip = np.arange(box.nelv) % 3 != 1          # mirror two thirds of the elements (swap the r direction)
    x[flip] = x[flip][:, :, ::-1]
    y[flip] = y[flip][:, :, ::-1]
    d = SEMData2D(x, y, {}, name="mixed")
    assert set(jacobian_sign(d.x, d.y, np.arange(d.nelv))) == {-1.0, 1.0}
    ext = d.external_edges()
    assert len(ext) == 2 * (9 + 4)
    loops = chain_edges(d.x, d.y, ext)
    assert len(loops) == 1 and loops[0][1]       # one closed loop
    sides = d.detect_boundaries(90)
    assert len(sides) == 4 and sorted(len(b) for b in sides) == [4, 4, 9, 9]
    for b in sides:
        pl = b.polyline()                        # continuous: consecutive points are close
        assert len(pl) == len(b) * (b.n - 1) + 1
        assert np.all(np.hypot(*np.diff(pl, axis=0).T) < 0.4)
        xe, ye = b.nodes()                       # traversal order: end of one edge = start of the next
        assert np.allclose(xe[1:, 0], xe[:-1, -1]) and np.allclose(ye[1:, 0], ye[:-1, -1])
        nrm = b.normals()
        px, py = xe + 1e-3 * nrm[..., 0], ye + 1e-3 * nrm[..., 1]
        assert np.all((px < 0) | (px > 3) | (py < 0) | (py > 1))   # outward
    bottom = next(b for b in sides if np.allclose(b.nodes()[1], 0))
    p0, p1 = bottom.normal_line(0, 0, 0.1)
    assert abs(p0[0] - bottom.nodes()[0][0, 0]) < 1e-12 and abs(p1[1] - 0.1) < 1e-12
    synthetic.add_analytic_field(d, "u", lambda x, y: y)
    synthetic.add_analytic_field(d, "v", lambda x, y: 0 * x)
    synthetic.add_analytic_field(d, "p", lambda x, y: x)
    F = compute_forces(d, bottom)
    s_ = F.distribution["s"]
    assert np.all(np.diff(s_) > 0) and abs(s_[-1] - 3.0) < 1e-3
    assert np.allclose(F.distribution["p"], F.distribution["x"], atol=1e-12)   # values follow the walk
    assert np.allclose(F.viscous, [3.0, 0.0], atol=1e-10)
