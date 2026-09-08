import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from semscope import synthetic
from semscope.gui.server import DataService, browse, make_handler
from semscope.session import load_session, plotter_from_session, save_session


def _session(path, **over):
    sess = {
        "semscope_session": 1,
        "dataset": {"path": path, "step": 0},
        "field": {"name": "vort", "cmap": "RdBu", "invert": True, "range": {"auto": False, "sym": True, "lo": -4.0, "hi": 4.0}},
        "calc": {"defs": [{"name": "vort", "expr": "dx(v) - dy(u)"}], "open": True},
        "render": {"mode": 0, "edges": True, "contours": True, "nContours": 8, "nodes": False},
        "view": {"xlim": [-1.2, 1.2], "ylim": [-0.9, 0.9]},
        "probe": {"pinned": {"x": 1.0, "y": 0.0}, "snapAngle": False},
        "lines": [{"id": 1, "colorIdx": 0, "x0": 0.6, "y0": 0.0, "x1": 1.4, "y1": 0.0, "visible": True}, {"id": 2, "colorIdx": 1, "x0": 0, "y0": 0.6, "x1": 0, "y1": 1.4, "visible": False}],
        "activeLine": 1,
        "chart": {"open": True, "grid": True, "elem": False},
    }
    sess.update(over)
    return sess


@pytest.fixture(scope="module")
def dataset_file(tmp_path_factory):
    d = synthetic.taylor_green(synthetic.annulus(2, 12, 6), k=2.0)
    p = tmp_path_factory.mktemp("sess") / "tg0.f00000"
    d.write(str(p))
    return str(p)


def test_save_load_roundtrip(tmp_path, dataset_file):
    out = save_session(str(tmp_path / "my"), _session(dataset_file))
    assert out.endswith(".semscope.json") and os.path.exists(out)
    back = load_session(out)
    assert back["dataset"]["path"] == dataset_file
    assert back["field"]["name"] == "vort" and "saved" in back
    # relative dataset paths are resolved against the session file location
    rel = _session(os.path.basename(dataset_file))
    p2 = save_session(str(tmp_path / "rel.semscope.json"), rel)
    back2 = load_session(p2)
    assert back2["dataset"]["path"] == str(tmp_path / os.path.basename(dataset_file))
    with pytest.raises(FileExistsError):
        save_session(out, _session(dataset_file), overwrite=False)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"foo": 1}))
    with pytest.raises(ValueError):
        load_session(str(bad))


def test_plotter_from_session(tmp_path, dataset_file):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    path = save_session(str(tmp_path / "fig.semscope.json"), _session(dataset_file))
    pl, data = plotter_from_session(path, figsize=(4, 4), dpi=60)
    kinds = [L.kind for L in pl._layers]
    assert kinds == ["field", "contours", "mesh", "segments", "points"]
    assert data.defined == ["vort"] and pl._layers[0].opts["name"] == "vort"
    assert pl._layers[0].opts["cmap"] == "RdBu_r" and pl._layers[0].opts["clim"] == (-4.0, 4.0)
    assert len(pl._layers[3].opts["segments"]) == 1  # hidden line is not drawn
    xl, yl = pl._limits()
    assert xl == (-1.2, 1.2) and yl == (-0.9, 0.9)
    out = pl.save(str(tmp_path / "fig.png"))
    assert os.path.getsize(out) > 1000
    pl.close()


def test_cli_png_from_session(tmp_path, dataset_file):
    pytest.importorskip("matplotlib")
    from semscope.cli import main

    path = save_session(str(tmp_path / "cli.semscope.json"), _session(dataset_file))
    out = str(tmp_path / "cli.png")
    assert main(["--png", out, path]) == 0
    assert os.path.getsize(out) > 1000


@pytest.fixture(scope="module")
def server(dataset_file):
    service = DataService()
    service.open(dataset_file)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", service
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_server_session_endpoints(server, tmp_path, dataset_file):
    base, service = server
    target = str(tmp_path / "srv.semscope.json")
    status, out = _post(f"{base}/api/session/save?path={target}", _session(dataset_file))
    assert status == 200 and out["ok"] and os.path.exists(target)
    status, out = _post(f"{base}/api/session/save?path={target}", _session(dataset_file))
    assert status == 409 and out["exists"]
    status, out = _post(f"{base}/api/session/save?path={target}&overwrite=1", _session(dataset_file, activeLine=2))
    assert status == 200
    st = _get(f"{base}/api/session/load?path={target}")
    assert st["open"] and st["path"] == dataset_file
    assert st["session"]["activeLine"] == 2 and st["session"]["dataset"]["path"] == dataset_file
    assert any(e["type"] == "session" and e["name"] == "srv.semscope.json" for e in browse(str(tmp_path))["entries"])
    # state keeps handing out the session until another dataset is opened
    assert _get(f"{base}/api/state")["session"]["activeLine"] == 2
    st = _get(f"{base}/api/open?path={dataset_file}")
    assert st["session"] is None


def test_legacy_semview_session_files_still_load(tmp_path, dataset_file):
    """Sessions written before the rename (.semview.json, key semview_session) keep working."""
    sess = _session(dataset_file)
    sess["semview_session"] = sess.pop("semscope_session")
    old = tmp_path / "old.semview.json"
    old.write_text(json.dumps(sess))
    back = load_session(str(old))
    assert back["semscope_session"] == 1 and "semview_session" not in back
    assert back["dataset"]["path"] == dataset_file
    assert save_session(str(tmp_path / "again"), back).endswith(".semscope.json")
    assert any(e["type"] == "session" and e["name"] == "old.semview.json" for e in browse(str(tmp_path))["entries"])
    service = DataService()
    st = service.open(str(old))
    assert st["open"] and st["session"]["semscope_session"] == 1
