"""API-level tests of the GUI server (no browser): run the stdlib server in a
thread and talk to it over HTTP."""

import json
import struct
import urllib.request

import numpy as np
import pytest

from semview import synthetic
from semview.gui.server import DataService, browse, make_handler
from http.server import ThreadingHTTPServer


@pytest.fixture(scope="module")
def dataset_file(tmp_path_factory):
    d = synthetic.taylor_green(synthetic.annulus(2, 12, 6), k=2.0)
    p = tmp_path_factory.mktemp("gui") / "tg0.f00000"
    d.write(str(p))
    return str(p)


@pytest.fixture(scope="module")
def server(dataset_file):
    service = DataService()
    service.open(dataset_file)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    httpd.daemon_threads = True
    import threading

    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", service
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read(), dict(r.headers)


def test_state_and_static(server):
    base, service = server
    body, hdr = get(base + "/api/state")
    st = json.loads(body)
    assert st["open"] and st["nelv"] == 24 and st["n"] == 6
    assert st["fields"] == ["u", "v", "p"]
    assert "vorticity" in st["available"]
    assert len(st["gll"]) == 6 and abs(st["gll"][0] + 1) < 1e-15
    html, hdr = get(base + "/")
    assert b"<title>semview</title>" in html
    for f in ("app.js", "renderer.js", "shaders.js", "spectral.js", "style.css"):
        body, hdr = get(base + "/" + f)
        assert len(body) > 100


def test_mesh_and_field_binary(server):
    base, service = server
    body, _ = get(base + "/api/mesh")
    n = 24 * 36
    assert len(body) == 2 * n * 4
    x = np.frombuffer(body[: n * 4], dtype="<f4")
    d = service.get_step(0)
    assert np.allclose(x, d.x.ravel(), atol=1e-6)
    body, hdr = get(base + "/api/field?name=u&step=0")
    u = np.frombuffer(body, dtype="<f4")
    assert np.allclose(u, d["u"].ravel(), atol=1e-6)
    assert abs(float(hdr["X-Min"]) - d["u"].min()) < 1e-12
    body, hdr = get(base + "/api/field?name=vorticity&step=0")
    assert len(body) == n * 4
    body, hdr = get(base + "/api/field?name=decay:u&step=0")
    dec = np.frombuffer(body, dtype="<f4").reshape(24, 6, 6)
    assert np.allclose(dec, dec[:, :1, :1])  # per-element constant
    body, _ = get(base + "/api/elmap")
    assert np.frombuffer(body, dtype="<i4")[0] == 1


def test_probe_and_line(server):
    base, _ = server
    st = json.loads(get(base + "/api/probe?x=1.0&y=0.0&step=0&fields=u,v")[0])
    assert st["found"]
    assert abs(st["values"]["u"] - np.sin(2.0) * np.cos(0.0)) < 1e-6
    st = json.loads(get(base + "/api/probe?x=0.0&y=0.0&step=0&fields=u")[0])
    assert not st["found"]  # in the hole
    ln = json.loads(get(base + "/api/line?x0=0.6&y0=0&x1=1.4&y1=0&step=0&fields=u&n=9")[0])
    assert len(ln["distance"]) == 9 and len(ln["elem"]) == 9
    xs = np.array(ln["x"])
    assert np.allclose(ln["values"]["u"], np.sin(2.0 * xs), atol=1e-6)


def test_browse_and_errors(server, dataset_file):
    import os

    base, _ = server
    b = json.loads(get(base + f"/api/browse?dir={os.path.dirname(dataset_file)}")[0])
    assert any(e["name"] == "tg0.f00000" and e["type"] == "step" for e in b["entries"])
    assert browse(os.path.dirname(dataset_file))["dir"] == os.path.dirname(dataset_file)
    cm = json.loads(get(base + "/api/colormaps")[0])
    assert len(cm["viridis"]) == 256
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base + "/api/field?name=nope&step=0")
    assert exc.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base + "/nonexistent.js")
    assert exc.value.code == 404


def test_boundary_endpoints(server):
    base, service = server
    body, _ = get(base + "/api/boundary/edges?m=5")
    nb, m, n = np.frombuffer(body[:12], dtype="<i4")
    assert m == 5 and n == 6 and nb == 2 * 12  # inner + outer circle of the 2 x 12 annulus
    off = 12
    ids = np.frombuffer(body[off : off + nb * 8], dtype="<i4").reshape(nb, 2)
    off += nb * 8
    coords = np.frombuffer(body[off : off + nb * m * 8], dtype="<f4").reshape(nb, m, 2)
    off += nb * m * 8
    nodes = np.frombuffer(body[off : off + nb * n * 8], dtype="<f4").reshape(nb, n, 2)
    off += nb * n * 8
    normals = np.frombuffer(body[off:], dtype="<f4").reshape(nb, n, 2)
    r = np.hypot(coords[..., 0], coords[..., 1])
    assert np.allclose(np.sort(np.unique(np.round(r, 3))), [0.5, 1.5])
    assert set(ids[:, 1].tolist()) == {1, 3}
    # normals point away from the fluid: radially outward on the outer circle, towards the centre on the inner
    rn = np.hypot(nodes[..., 0], nodes[..., 1])
    radial = nodes / rn[..., None]
    dots = np.sum(normals * radial, axis=-1)
    assert np.allclose(dots[rn > 1], 1.0, atol=1e-5) and np.allclose(dots[rn < 1], -1.0, atol=1e-5)
    det = json.loads(get(base + "/api/boundary/detect?angle=90")[0])
    assert len(det["groups"]) == 2 and all(g["closed"] for g in det["groups"])
    assert sorted(len(g["edges"]) for g in det["groups"]) == [12, 12]
    assert all(0 <= i < nb for g in det["groups"] for i in g["edges"])
