"""HTTP server for the semscope GUI.

Pure standard library: a ``ThreadingHTTPServer`` serves the static front-end
and a small JSON/binary API.  Field data goes to the browser as raw little-
endian ``float32`` arrays, which the front-end uploads straight into GPU
textures.

Under ``mpirun`` only rank 0 serves; the other ranks sit in
:func:`worker_loop`, take part in the collective (parallel) reads that
pysemtools performs and gather their elements to rank 0.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import mpi
from .. import spectral as sp
from ..boundary import detect_boundaries, edge_nodes, edge_normals
from ..calc import ExpressionError, syntax_help
from ..dataset import Dataset, SEMData2D
from ..forces import compute_forces, sum_forces
from ..session import SESSION_SUFFIXES, is_session_file, load_session, save_session

__all__ = ["serve", "DataService", "worker_loop"]

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_STEP_FILE = re.compile(r".*\d\.f\d{5}$")

COLORMAPS = [
    "viridis", "magma", "inferno", "plasma", "cividis", "turbo",
    "coolwarm", "RdBu_r", "seismic", "Spectral_r", "PiYG", "BrBG",
    "gray", "bone", "cubehelix", "jet", "twilight", "hsv",
]


def colormap_table(name: str, size: int = 256) -> list[list[int]]:
    """RGB lookup table (0..255) for a matplotlib colormap; a fallback gradient if matplotlib is missing."""
    try:
        import matplotlib

        cm = matplotlib.colormaps[name]
        rgba = cm(np.linspace(0, 1, size))
        return (rgba[:, :3] * 255).round().astype(int).tolist()
    except Exception:  # noqa: BLE001
        t = np.linspace(0, 1, size)
        return np.stack([255 * t, 255 * t**2, 255 * (1 - t)], axis=1).round().astype(int).tolist()


class DataService:
    """Owns the open dataset and answers the API.  MPI-collective operations are
    serialised through a lock and mirrored to the worker ranks."""

    def __init__(self, comm=None):
        self.comm = comm or mpi.world()
        self.lock = threading.Lock()
        self.ds: Dataset | None = None
        self.path: str | None = None
        self._gathered: dict[int, SEMData2D] = {}
        self._gather_order: list[int] = []
        self.session: dict | None = None   # session the front-end should apply
        self._boundary = None              # cached external edges of the open mesh
        self.max_cached = 16
        self.max_cached_bytes = 4 << 30

    # ------------------------------------------------------------ MPI mirror
    def _bcast(self, cmd):
        if self.comm.Get_size() > 1:
            self.comm.bcast(cmd, root=0)

    def open(self, path: str, session: dict | None = None):
        if is_session_file(path):
            return self.open_session(path)
        with self.lock:
            if os.path.abspath(path) != (os.path.abspath(self.path) if self.path else None):
                self._bcast(("open", path))
                self.ds = Dataset(path, comm=self.comm)
                self.path = path
                self._gathered.clear()
                self._gather_order.clear()
                self._boundary = None
                self.step(0)
            self.session = session
        return self.state()

    def open_session(self, path: str):
        """Open the dataset referenced by a session file and hand the session to the front-end."""
        sess = load_session(path)
        return self.open(sess["dataset"]["path"], session=sess)

    def save_session(self, path: str, session: dict, overwrite: bool) -> str:
        return save_session(path, session, overwrite=overwrite)

    def step(self, i: int) -> SEMData2D:
        """Return step ``i`` gathered on rank 0 (call with the lock held or from ``open``)."""
        if self.ds is None:
            raise RuntimeError("no dataset open")
        if i < 0:
            i += len(self.ds)
        if i in self._gathered:
            return self._gathered[i]
        self._bcast(("step", i))
        data = self.ds[i].gather(self.comm)
        self._gathered[i] = data
        self._gather_order.append(i)
        while len(self._gather_order) > 1 and (
            len(self._gather_order) > self.max_cached or sum(d.nbytes for d in self._gathered.values()) > self.max_cached_bytes
        ):
            old = self._gather_order.pop(0)
            self._gathered.pop(old, None)
        return data

    def get_step(self, i: int) -> SEMData2D:
        with self.lock:
            return self.step(i)

    def close(self):
        self._bcast(("quit",))

    # ------------------------------------------------------------ API payloads
    def state(self) -> dict:
        if self.ds is None:
            return {"open": False, "cwd": os.getcwd()}
        d0 = self._gathered.get(0) or self.get_step(0)
        return {
            "open": True,
            "name": self.ds.name,
            "path": os.path.abspath(self.path),
            "nsteps": len(self.ds),
            "times": [float(t) for t in self.ds.times],
            "nelv": int(d0.nelv),
            "n": int(d0.n),
            "order": int(d0.order),
            "fields": d0.field_names,
            "available": d0.available,
            "calc": self.ds.expressions.to_list(),
            "calc_syntax": syntax_help(),
            "bounds": list(d0.bounds),
            "gll": sp.gll_nodes(d0.n).tolist(),
            "bary": sp.barycentric_weights(sp.gll_nodes(d0.n)).tolist(),
            "mpi_ranks": self.comm.Get_size(),
            "cwd": os.getcwd(),
            "session": self.session,
        }

    def mesh_bytes(self) -> bytes:
        d = self.get_step(0)
        return np.concatenate([d.x.ravel(), d.y.ravel()]).astype("<f4").tobytes()

    def elmap_bytes(self) -> bytes:
        d = self.get_step(0)
        ids = d.elmap if d.elmap is not None else np.arange(1, d.nelv + 1)
        return np.asarray(ids).astype("<i4").tobytes()

    def field(self, name: str, step: int):
        d = self.get_step(step)
        arr = d[name]
        lo, hi = _finite_range(arr)
        return np.ascontiguousarray(arr, dtype="<f4").tobytes(), {"time": d.time, "min": lo, "max": hi}

    # ------------------------------------------------------------ field calculator
    def calc_define(self, name: str, expr: str, step: int = 0) -> dict:
        """Add or replace a calculator field; evaluates it once on ``step`` to report its range."""
        if self.ds is None:
            raise RuntimeError("no dataset open")
        with self.lock:
            previous = self.ds.expressions.get(name)
            name = self.ds.define(name, expr)
            try:
                arr = self.step(step)[name]
            except Exception:
                if previous is None:
                    self.ds.expressions.remove(name)
                else:
                    self.ds.expressions.define(name, previous, stored=self.ds.field_names)
                raise
            lo, hi = _finite_range(arr)
            return {"ok": True, "name": name, "expr": self.ds.expressions[name], "min": lo, "max": hi, "calc": self.ds.expressions.to_list()}

    def calc_remove(self, name: str) -> dict:
        if self.ds is None:
            raise RuntimeError("no dataset open")
        with self.lock:
            self.ds.undefine(name)
            return {"ok": True, "calc": self.ds.expressions.to_list()}

    def calc_check(self, expr: str, name: str | None = None) -> dict:
        """Validate an expression (syntax, names, cycles) without evaluating it."""
        if self.ds is None:
            raise RuntimeError("no dataset open")
        trial = self.ds.expressions.__class__(self.ds.expressions)
        try:
            trial.define(name or "_check_", expr, stored=self.ds.field_names)
        except ExpressionError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    # ------------------------------------------------------------ boundaries
    def _external(self):
        if self._boundary is None:
            d = self.get_step(0)
            ext = d.external_edges()
            self._boundary = {"edges": ext, "index": {(int(e), int(s)): i for i, (e, s) in enumerate(ext)}}
        return self._boundary

    def boundary_edges_bytes(self, m: int = 16) -> bytes:
        """Binary description of the external edges.

        Layout: int32 ``(nb, m, n)``; int32 ``(elem, side) * nb``; float32 polyline
        ``(x, y) * m * nb`` (spectrally resampled); float32 GLL nodes ``(x, y) * n * nb``;
        float32 outward unit normals ``(nx, ny) * n * nb`` at those nodes.
        """
        d = self.get_step(0)
        ext = self._external()["edges"]
        m = max(2, int(m))
        xe, ye = edge_nodes(d.x, d.y, ext[:, 0], ext[:, 1])
        J = sp.interpolation_matrix(sp.gll_nodes(d.n), sp.uniform_nodes(m))
        coords = np.stack([xe @ J.T, ye @ J.T], axis=-1).astype("<f4")
        nodes = np.stack([xe, ye], axis=-1).astype("<f4")
        normals = edge_normals(d.x, d.y, ext).astype("<f4") if len(ext) else np.zeros((0, d.n, 2), "<f4")
        head = np.array([len(ext), m, d.n], dtype="<i4").tobytes()
        return head + np.ascontiguousarray(ext, dtype="<i4").tobytes() + coords.tobytes() + nodes.tobytes() + normals.tobytes()

    def boundary_detect(self, angle: float) -> dict:
        d = self.get_step(0)
        b = self._external()
        groups = detect_boundaries(d.x, d.y, angle=angle, edges=b["edges"])
        return {
            "angle": angle,
            "groups": [{"name": g.name, "closed": g.closed, "edges": [b["index"][(int(e), int(s_))] for e, s_ in g.edges]} for g in groups],
        }

    def forces(self, body: dict) -> dict:
        """Forces on the selected boundaries (edge indices into the external-edge list) for one step."""
        step = int(body.get("step", 0))
        d = self.get_step(step)
        ext = self._external()["edges"]
        kw = {k: body[k] for k in ("u", "v", "p") if body.get(k)}
        kw["w"] = body.get("w") or None
        for k in ("rho", "mu"):
            if body.get(k) is not None and body.get(k) != "":
                kw[k] = body[k]
        for k in ("U_ref", "L_ref", "p_ref"):
            if body.get(k) is not None and body.get(k) != "":
                kw[k] = float(body[k])
        if body.get("rho_ref") not in (None, ""):
            kw["rho_ref"] = float(body["rho_ref"])
        if body.get("axes"):
            kw["axes"] = [(a["name"], a["dir"]) for a in body["axes"]]
        results = []
        for b in body.get("boundaries", []):
            idx = np.asarray(b.get("edges", []), dtype=np.int64)
            if idx.size == 0:
                continue
            if idx.min() < 0 or idx.max() >= len(ext):
                raise ValueError(f"boundary {b.get('name', '')!r}: edge index out of range")
            results.append(compute_forces(d, ext[idx], name=str(b.get("name", "")), **kw))
        if not results:
            raise ValueError("select at least one boundary with edges")
        out = {"step": step, "time": d.time, "q": results[0].q, "rho_ref": results[0].rho_ref, "boundaries": [r.to_dict() for r in results]}
        tot = sum_forces(results).to_dict()
        tot.pop("dist", None)
        out["total"] = tot
        return out

    def probe(self, x: float, y: float, step: int, names: list[str]) -> dict:
        d = self.get_step(step)
        loc = d.locate([x], [y])
        if not loc.found[0]:
            return {"found": False}
        from .. import spectral as sp

        vals = {nm: float(sp.evaluate_at(d[nm], loc.elem, loc.r, loc.s)[0]) for nm in names}
        e = int(loc.elem[0])
        gid = int(d.elmap[e]) if d.elmap is not None else e + 1
        return {"found": True, "elem": e, "elem_global": gid, "r": float(loc.r[0]), "s": float(loc.s[0]), "values": vals}

    def line(self, p0, p1, step: int, names: list[str], npts: int) -> dict:
        d = self.get_step(step)
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        t = np.linspace(0.0, 1.0, npts)
        pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        loc = d.locate(pts[:, 0], pts[:, 1])
        out = {"distance": (t * np.linalg.norm(p1 - p0)).tolist(), "x": pts[:, 0].tolist(), "y": pts[:, 1].tolist(), "elem": loc.elem.tolist(), "values": {}}
        found = loc.found
        for nm in names:
            v = np.full(npts, np.nan)
            if found.any():
                v[found] = sp.evaluate_at(d[nm], loc.elem[found], loc.r[found], loc.s[found])
            out["values"][nm] = [None if not np.isfinite(a) else float(a) for a in v]
        return out


def _finite_range(arr) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)


def browse(directory: str) -> dict:
    """Directory listing for the open-file dialog (directories, metafiles, step files)."""
    directory = os.path.abspath(os.path.expanduser(directory or os.getcwd()))
    entries = []
    try:
        names = sorted(os.listdir(directory), key=str.lower)
    except OSError as exc:
        return {"dir": directory, "error": str(exc), "entries": []}
    for nm in names:
        if nm.startswith("."):
            continue
        full = os.path.join(directory, nm)
        if os.path.isdir(full):
            entries.append({"name": nm, "type": "dir"})
        elif nm.endswith(SESSION_SUFFIXES):
            entries.append({"name": nm, "type": "session", "size": os.path.getsize(full)})
        elif nm.endswith(".nek5000"):
            entries.append({"name": nm, "type": "meta", "size": os.path.getsize(full)})
        elif _STEP_FILE.match(nm):
            entries.append({"name": nm, "type": "step", "size": os.path.getsize(full)})
    entries.sort(key=lambda e: ({"dir": 0, "session": 1, "meta": 2, "step": 3}[e["type"]], e["name"].lower()))
    return {"dir": directory, "parent": os.path.dirname(directory), "entries": entries}


# --------------------------------------------------------------------------- #
def make_handler(service: DataService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "semscope/0.1"

        def log_message(self, fmt, *args):  # quieter log: API only
            if os.environ.get("SEMSCOPE_DEBUG"):
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        # -------------------------------------------------------- helpers
        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _error(self, status: int, msg: str):
            self._json({"error": msg}, status)

        # -------------------------------------------------------- routing
        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path.startswith("/api/"):
                    self.api(url.path[5:], q)
                else:
                    self.static(url.path)
            except FileNotFoundError as exc:
                self._error(404, str(exc))
            except ExpressionError as exc:
                self._error(400, str(exc))
            except (KeyError, ValueError, IndexError, RuntimeError) as exc:
                self._error(400, f"{type(exc).__name__}: {exc}")
            except BrokenPipeError:
                pass
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                self._error(500, f"{type(exc).__name__}: {exc}")

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if url.path == "/api/session/save":
                    try:
                        out = service.save_session(q["path"], body, overwrite=q.get("overwrite", "0") == "1")
                    except FileExistsError as exc:
                        self._json({"error": f"{exc} exists", "exists": True}, 409)
                        return
                    self._json({"ok": True, "path": out})
                elif url.path == "/api/calc/define":
                    self._json(service.calc_define(str(body.get("name", "")), str(body.get("expr", "")), int(body.get("step", 0))))
                elif url.path == "/api/calc/remove":
                    self._json(service.calc_remove(str(body.get("name", ""))))
                elif url.path == "/api/forces":
                    self._json(service.forces(body))
                else:
                    raise FileNotFoundError(url.path)
            except FileNotFoundError as exc:
                self._error(404, str(exc))
            except ExpressionError as exc:
                self._error(400, str(exc))
            except (KeyError, ValueError, OSError, RuntimeError) as exc:
                self._error(400, f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                self._error(500, f"{type(exc).__name__}: {exc}")

        def static(self, path: str):
            if path in ("/", ""):
                path = "/index.html"
            full = os.path.normpath(os.path.join(STATIC_DIR, path.lstrip("/")))
            if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
                raise FileNotFoundError(path)
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            if full.endswith(".js"):
                ctype = "application/javascript"
            with open(full, "rb") as fh:
                self._send(200, fh.read(), ctype)

        def api(self, route: str, q: dict):
            if route == "state":
                self._json(service.state())
            elif route == "open":
                self._json(service.open(q["path"]))
            elif route == "session/load":
                self._json(service.open_session(q["path"]))
            elif route == "browse":
                self._json(browse(q.get("dir", "")))
            elif route == "colormaps":
                self._json({nm: colormap_table(nm) for nm in COLORMAPS})
            elif route == "mesh":
                self._send(200, service.mesh_bytes(), "application/octet-stream")
            elif route == "elmap":
                self._send(200, service.elmap_bytes(), "application/octet-stream")
            elif route == "field":
                data, meta = service.field(q["name"], int(q.get("step", 0)))
                self._send(200, data, "application/octet-stream", {"X-Time": repr(meta["time"]), "X-Min": repr(meta["min"]), "X-Max": repr(meta["max"])})
            elif route == "calc":
                self._json({"calc": service.ds.expressions.to_list() if service.ds else []})
            elif route == "calc/check":
                self._json(service.calc_check(q.get("expr", ""), q.get("name") or None))
            elif route == "boundary/edges":
                self._send(200, service.boundary_edges_bytes(int(q.get("m", 16))), "application/octet-stream")
            elif route == "boundary/detect":
                self._json(service.boundary_detect(float(q.get("angle", 90))))
            elif route == "probe":
                names = [s for s in q.get("fields", "").split(",") if s]
                self._json(service.probe(float(q["x"]), float(q["y"]), int(q.get("step", 0)), names))
            elif route == "line":
                names = [s for s in q.get("fields", "").split(",") if s]
                self._json(service.line((float(q["x0"]), float(q["y0"])), (float(q["x1"]), float(q["y1"])), int(q.get("step", 0)), names, int(q.get("n", 300))))
            elif route == "shutdown":
                self._json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                raise FileNotFoundError(route)

    return Handler


def worker_loop(comm):
    """Non-root ranks: follow rank 0's collective read/gather commands."""
    ds = None
    while True:
        cmd = comm.bcast(None, root=0)
        if cmd[0] == "quit":
            break
        if cmd[0] == "open":
            ds = Dataset(cmd[1], comm=comm)
        elif cmd[0] == "step":
            ds[cmd[1]].gather(comm)


def serve(path: str | None = None, port: int = 8765, host: str = "127.0.0.1", open_browser: bool = True, comm=None, block: bool = True):
    """Start the GUI server (rank 0) and, under MPI, the worker loops.

    Returns the ``ThreadingHTTPServer`` on rank 0 when ``block=False``.
    """
    comm = comm or mpi.world()
    if comm.Get_rank() != 0:
        worker_loop(comm)
        return None
    service = DataService(comm)
    if path:
        service.open(path)   # a dataset or a *.semscope.json session file
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    httpd.daemon_threads = True
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"semscope GUI at {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        httpd.server_close()
    return None
