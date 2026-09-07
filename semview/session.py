"""GUI sessions: a JSON description of what the viewer shows.

A session file (``*.semview.json``) records the dataset and time step, the
field, colormap and range, the rendering options, the view extent, the
pinned probe and the line probes.  The GUI saves and reloads them, and
:func:`plotter_from_session` turns one into a matplotlib
:class:`~semview.plotting.Plotter`, so a view composed interactively can be
reproduced as a publication figure.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

__all__ = ["SESSION_VERSION", "SESSION_SUFFIX", "load_session", "save_session", "plotter_from_session"]

SESSION_VERSION = 1
SESSION_SUFFIX = ".semview.json"


def load_session(path: str) -> dict:
    """Read and validate a session file; the dataset path is made absolute."""
    with open(path) as fh:
        sess = json.load(fh)
    if not isinstance(sess, dict) or "semview_session" not in sess:
        raise ValueError(f"{path} is not a semview session file")
    if int(sess["semview_session"]) > SESSION_VERSION:
        raise ValueError(f"{path}: session version {sess['semview_session']} is newer than this semview")
    ds = sess.setdefault("dataset", {})
    p = ds.get("path")
    if not p:
        raise ValueError(f"{path}: session has no dataset path")
    if not os.path.isabs(p):
        ds["path"] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), p))
    ds.setdefault("step", 0)
    sess["_file"] = os.path.abspath(path)
    return sess


def save_session(path: str, session: dict, overwrite: bool = True) -> str:
    """Write a session file (adds the ``.semview.json`` suffix if missing)."""
    if not path.endswith(SESSION_SUFFIX):
        path = path + SESSION_SUFFIX if not path.endswith(".json") else path[: -len(".json")] + SESSION_SUFFIX
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(path)
    session = dict(session)
    session.pop("_file", None)
    session["semview_session"] = SESSION_VERSION
    session.setdefault("saved", _dt.datetime.now().isoformat(timespec="seconds"))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(session, fh, indent=2)
    return os.path.abspath(path)


def plotter_from_session(session, figsize=(9, 6), dpi=150, comm=None):
    """Build a :class:`~semview.plotting.Plotter` reproducing a GUI session.

    ``session`` is a path or an already loaded dict.  Returns the plotter and
    the loaded :class:`~semview.dataset.SEMData2D` (``None`` on non-root MPI ranks).
    """
    from .dataset import Dataset
    from .plotting import Plotter

    sess = load_session(session) if isinstance(session, str) else session
    ds = Dataset(sess["dataset"]["path"], comm=comm)
    data = ds[int(sess["dataset"].get("step", 0))].gather(ds.comm)
    if data is None:
        return None, None
    fld = sess.get("field", {})
    name = fld.get("name", data.field_names[0])
    if name.startswith("decay:"):
        import numpy as np

        base = name[6:]
        dec = np.log10(np.maximum(data.spectral_decay(base), 1e-16))[:, None, None] * np.ones((1, data.n, data.n))
        data.add_field(name, dec)
    cmap = fld.get("cmap", "viridis")
    if fld.get("invert"):
        cmap = cmap[:-2] if cmap.endswith("_r") else cmap + "_r"
    rng = fld.get("range", {})
    clim = None if rng.get("auto", True) else (rng.get("lo"), rng.get("hi"))
    render = sess.get("render", {})
    pl = Plotter(figsize=figsize, dpi=dpi)
    pl.add_field(data, name, cmap=cmap, clim=clim, method="nodal" if render.get("mode") == 1 else "spectral")
    if render.get("contours"):
        pl.add_contours(data, name, levels=int(render.get("nContours", 12)), colors="w", linewidths=0.5)
    if render.get("edges", True):
        pl.add_mesh(data, color="k", linewidth=0.3)
    if render.get("nodes"):
        pl.add_nodes(data)
    lines = sess.get("lines", [])
    if lines:
        pl.add_segments([((L["x0"], L["y0"]), (L["x1"], L["y1"])) for L in lines if L.get("visible", True)], color="w", linewidth=1.2)
    pinned = (sess.get("probe") or {}).get("pinned")
    if pinned:
        pl.add_points([pinned["x"]], [pinned["y"]], color="magenta", size=30, marker="+")
    view = sess.get("view", {})
    if "xlim" in view and "ylim" in view:
        pl.set_view(view["xlim"], view["ylim"])
    pl.set_title(f"{ds.name}: {name}  (t = {data.time:g})")
    return pl, data
