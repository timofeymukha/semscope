"""In-memory representation of a 2D spectral-element field and time series.

:class:`SEMData2D` holds the nodal coordinates and fields of one time step as
plain arrays ``(nelv, n, n)`` indexed ``[element, j, i]`` (``i`` along ``r``).
It knows how to evaluate the spectral expansion anywhere (``sample``,
``sample_line``, ``to_grid``), how to oversample elements for rendering
(``resample``) and how to evaluate calculator expressions on the nodal
data (``evaluate``/``define``) with spectrally exact derivatives.

:class:`Dataset` wraps a :class:`~semview.io.Nek5000Series` and loads steps on
demand, reusing the mesh of the first file for mesh-less steps (the usual
Nek5000/Neko output layout).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from . import calc
from . import spectral as sp
from .locate import ElementLocator, Location

__all__ = ["SEMData2D", "Dataset", "open", "load"]


@dataclass
class SEMData2D:
    """One time step of 2D CG spectral-element data."""

    x: np.ndarray
    y: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    time: float = 0.0
    elmap: np.ndarray | None = None
    name: str = ""
    source: str | None = None
    offset_el: int = 0
    glb_nelv: int | None = None
    expressions: calc.Expressions | None = None   # calculator definitions, shared between steps

    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=float)
        self.y = np.ascontiguousarray(self.y, dtype=float)
        if self.x.ndim != 3 or self.x.shape != self.y.shape:
            raise ValueError("x and y must be (nelv, n, n) arrays of equal shape")
        for k, v in list(self.fields.items()):
            v = np.asarray(v)
            if v.shape != self.x.shape:
                raise ValueError(f"field {k!r} has shape {v.shape}, expected {self.x.shape}")
            self.fields[k] = v
        if self.glb_nelv is None:
            self.glb_nelv = self.x.shape[0]
        if self.expressions is None:
            self.expressions = calc.Expressions()
        elif not isinstance(self.expressions, calc.Expressions):
            self.expressions = calc.Expressions(self.expressions)
        self._cache: dict = {}
        self._calc_version = self.expressions.version

    # ------------------------------------------------------------ basic info
    @property
    def nelv(self) -> int:
        return self.x.shape[0]

    @property
    def n(self) -> int:
        """Number of GLL nodes per direction (polynomial order + 1)."""
        return self.x.shape[-1]

    @property
    def order(self) -> int:
        return self.n - 1

    @property
    def field_names(self) -> list[str]:
        return list(self.fields.keys())

    @property
    def defined(self) -> list[str]:
        """Names defined with the field calculator (:meth:`define`)."""
        return list(self.expressions)

    @property
    def available(self) -> list[str]:
        """Stored fields followed by the calculator definitions."""
        return list(self.fields) + [k for k in self.expressions if k not in self.fields]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` of the GLL nodes."""
        return float(self.x.min()), float(self.x.max()), float(self.y.min()), float(self.y.max())

    @property
    def is_distributed(self) -> bool:
        return self.glb_nelv != self.nelv

    @property
    def nbytes(self) -> int:
        """Memory held by the fields (the mesh is usually shared between steps)."""
        return int(sum(v.nbytes for v in self.fields.values()))

    def __repr__(self):
        return (
            f"SEMData2D(nelv={self.nelv}, order={self.order}, fields={self.field_names}, "
            f"t={self.time:g}, bounds={tuple(round(b, 4) for b in self.bounds)})"
        )

    # ------------------------------------------------------------ fields
    def __getitem__(self, name: str) -> np.ndarray:
        """Nodal values of a stored field, a calculator definition, or an expression.

        ``data["u"]`` returns the stored field, ``data["vort"]`` a field defined
        with :meth:`define`, and ``data["dx(v) - dy(u)"]`` evaluates the
        expression on the fly (see :mod:`semview.calc` for the syntax).
        """
        if name in self.fields:
            return self.fields[name]
        if name in self.expressions:
            return self._calc(name, self.expressions[name])
        if calc.is_identifier(name):
            if name in calc.COORDINATES:
                return self.x if name == "x" else self.y
            raise KeyError(f"unknown field {name!r}; available: {self.available}")
        return self._calc(name, name)

    def __contains__(self, name: str) -> bool:
        return name in self.fields or name in self.expressions

    def add_field(self, name: str, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.shape != self.x.shape:
            raise ValueError(f"expected shape {self.x.shape}, got {values.shape}")
        self.fields[name] = values
        self._purge_calc()

    def metric(self) -> dict[str, np.ndarray]:
        """Geometric factors at the GLL nodes: ``xr, xs, yr, ys, jac, rx, ry, sx, sy``."""
        if "metric" not in self._cache:
            xr, xs = sp.element_derivatives(self.x)
            yr, ys = sp.element_derivatives(self.y)
            jac = xr * ys - xs * yr
            with np.errstate(divide="ignore", invalid="ignore"):
                self._cache["metric"] = dict(
                    xr=xr, xs=xs, yr=yr, ys=ys, jac=jac, rx=ys / jac, ry=-xs / jac, sx=-yr / jac, sy=xr / jac
                )
        return self._cache["metric"]

    def gradient(self, field) -> tuple[np.ndarray, np.ndarray]:
        """Spectrally exact physical gradient ``(df/dx, df/dy)`` of a field (name, expression or array), element-wise."""
        f = self[field] if isinstance(field, str) else np.asarray(field, dtype=float)
        fr, fs = sp.element_derivatives(f)
        m = self.metric()
        return fr * m["rx"] + fs * m["sx"], fr * m["ry"] + fs * m["sy"]

    def derivative(self, values, direction: int, order: int = 1) -> np.ndarray:
        """``order``-th physical derivative along x (``direction=0``) or y (``direction=1``).

        Applied repeatedly for higher orders; each application is the exact
        derivative of the element polynomial, collocated back onto the GLL nodes.
        """
        arr = np.asarray(values, dtype=float)
        if arr.shape != self.x.shape:   # a constant: its derivative vanishes
            return np.zeros_like(self.x)
        for _ in range(int(order)):
            arr = self.gradient(arr)[direction]
        return arr

    # ------------------------------------------------------------ field calculator
    def evaluate(self, expr: str) -> np.ndarray:
        """Evaluate a calculator expression (see :mod:`semview.calc`) at the GLL nodes."""
        return self._calc(expr, expr)

    def define(self, name: str, expr: str) -> str:
        """Define ``name`` as ``expr`` for this dataset (and every step sharing its definitions).

        The expression is checked for syntax, unknown names and circular
        references; evaluation happens lazily on first access and is cached.
        """
        name = self.expressions.define(name, expr, stored=self.fields)
        self._purge_calc()
        return name

    def undefine(self, name: str) -> None:
        """Remove a calculator definition (refused while other definitions use it)."""
        self.expressions.remove(name)
        self._purge_calc()

    def _purge_calc(self):
        for k in [k for k in self._cache if isinstance(k, tuple) and k[0] == "calc"]:
            del self._cache[k]
        self._calc_version = self.expressions.version

    def _calc(self, key: str, expr: str, _stack: tuple = ()) -> np.ndarray:
        if self._calc_version != self.expressions.version:
            self._purge_calc()
        ck = ("calc", key)
        if ck in self._cache:
            return self._cache[ck]

        def resolve(nm):
            if nm in self.fields:
                return self.fields[nm]
            if nm in calc.COORDINATES:
                return self.x if nm == "x" else self.y
            if nm in self.expressions:
                if nm in _stack:
                    raise calc.ExpressionError("circular definition: " + " -> ".join(_stack + (nm,)))
                return self._calc(nm, self.expressions[nm], _stack + (nm,))
            raise calc.ExpressionError(f"unknown field {nm!r}; available: {self.available + list(calc.COORDINATES)}")

        out = calc.evaluate(expr, resolve, self.derivative)
        out = np.asarray(out, dtype=float)
        if out.shape != self.x.shape:
            out = np.broadcast_to(out, self.x.shape).copy()
        self._cache[ck] = out
        return out

    # ------------------------------------------------------------ geometry
    @property
    def vertices(self) -> np.ndarray:
        """Element corner coordinates ``(nelv, 4, 2)`` ordered (r,s) = (-,-), (+,-), (-,+), (+,+)."""
        idx = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        return np.stack([np.stack([self.x[:, j, i], self.y[:, j, i]], axis=-1) for j, i in idx], axis=1)

    def element_bounds(self, margin: float = 0.0) -> np.ndarray:
        """Per-element bounding boxes ``(nelv, 4)`` as ``xmin, xmax, ymin, ymax``."""
        xmin = self.x.min(axis=(1, 2))
        xmax = self.x.max(axis=(1, 2))
        ymin = self.y.min(axis=(1, 2))
        ymax = self.y.max(axis=(1, 2))
        if margin:
            px = margin * (xmax - xmin)
            py = margin * (ymax - ymin)
            return np.stack([xmin - px, xmax + px, ymin - py, ymax + py], axis=1)
        return np.stack([xmin, xmax, ymin, ymax], axis=1)

    @property
    def locator(self) -> ElementLocator:
        if "locator" not in self._cache:
            self._cache["locator"] = ElementLocator(self.x, self.y)
        return self._cache["locator"]

    def locate(self, x, y) -> Location:
        """Find the element and reference coordinates of physical points."""
        return self.locator.locate(x, y)

    # ------------------------------------------------------------ evaluation
    def resample(self, names, m: int, kind: str = "uniform"):
        """Oversample geometry and fields on an ``m x m`` grid per element.

        Returns ``(X, Y, F)`` with ``X, Y`` of shape ``(nelv, m, m)`` and ``F`` a
        dict ``name -> (nelv, m, m)`` (or a single array if ``names`` is a str).
        """
        single = isinstance(names, str)
        if single:
            names = [names]
        X = sp.resample_elements(self.x, m, kind)
        Y = sp.resample_elements(self.y, m, kind)
        F = {nm: sp.resample_elements(self[nm], m, kind) for nm in names}
        return X, Y, (F[names[0]] if single else F)

    def sample(self, names, x, y, fill_value=np.nan):
        """Evaluate fields at arbitrary physical points (spectrally exact).

        ``names`` may be a field name or a list of names.  Points outside the
        domain get ``fill_value``.
        """
        single = isinstance(names, str)
        if single:
            names = [names]
        x = np.asarray(x, dtype=float)
        shape = x.shape
        loc = self.locate(x.ravel(), np.asarray(y, dtype=float).ravel())
        found = loc.found
        out = {}
        data = np.stack([self[nm] for nm in names])
        vals = np.full((len(names), loc.elem.size), fill_value, dtype=float)
        if found.any():
            vals[:, found] = sp.evaluate_at(data, loc.elem[found], loc.r[found], loc.s[found])
        for k, nm in enumerate(names):
            out[nm] = vals[k].reshape(shape)
        return out[names[0]] if single else out

    def sample_line(self, names, p0, p1, npts: int = 200):
        """Sample along the segment ``p0 -> p1``. Returns ``(distance, values)``."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        t = np.linspace(0.0, 1.0, npts)
        pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        vals = self.sample(names, pts[:, 0], pts[:, 1])
        return t * np.linalg.norm(p1 - p0), vals

    def to_grid(self, names, xlim=None, ylim=None, nx: int = 400, ny: int | None = None, fill_value=np.nan):
        """Resample onto a uniform Cartesian grid (pixel-exact spectral evaluation).

        Returns ``(xg, yg, values)`` where ``xg, yg`` are 1D and ``values`` is
        ``(ny, nx)`` (or a dict of such arrays).
        """
        b = self.bounds
        xlim = xlim or (b[0], b[1])
        ylim = ylim or (b[2], b[3])
        if ny is None:
            ny = max(2, int(round(nx * (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1e-300))))
        xg = np.linspace(xlim[0], xlim[1], nx)
        yg = np.linspace(ylim[0], ylim[1], ny)
        XG, YG = np.meshgrid(xg, yg)
        return xg, yg, self.sample(names, XG, YG, fill_value=fill_value)

    # ------------------------------------------------------------ boundaries
    def external_edges(self) -> np.ndarray:
        """Element edges not shared with another element, ``(nb, 2)`` of ``(elem, side)``."""
        if "external_edges" not in self._cache:
            from .boundary import external_edges

            self._cache["external_edges"] = external_edges(self.x, self.y)
        return self._cache["external_edges"]

    def detect_boundaries(self, angle: float = 90.0):
        """Boundaries: external edges chained and split at corners sharper than ``angle`` degrees."""
        from .boundary import detect_boundaries

        return detect_boundaries(self.x, self.y, angle=angle, edges=self.external_edges())

    def boundary(self, edges, name: str = "boundary"):
        """A :class:`~semview.boundary.Boundary` from explicit ``(elem, side)`` pairs (chained if possible)."""
        from .boundary import Boundary, chain_edges

        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        closed = False
        if len(edges) > 1:
            loops = chain_edges(self.x, self.y, edges)
            if len(loops) == 1:
                edges = edges[loops[0][0]]
                closed = loops[0][1]
        return Boundary(name, edges, closed=closed, x=self.x, y=self.y)

    def normal_line(self, elem: int, side: int, node: int, length: float):
        """Wall-normal probe segment ``(p0, p1)`` starting at a boundary GLL node (see :func:`semview.boundary.normal_line`)."""
        from .boundary import normal_line

        return normal_line(self.x, self.y, int(elem), int(side), int(node), float(length))

    def normal_line_at(self, elem: int, side: int, t: float, length: float):
        """Wall-normal probe segment from the point at parameter ``t`` in ``[-1, 1]`` along a boundary edge."""
        from .boundary import normal_line_at

        return normal_line_at(self.x, self.y, int(elem), int(side), float(t), float(length))

    # ------------------------------------------------------------ MPI
    def gather(self, comm=None, root: int = 0) -> "SEMData2D | None":
        """Gather an element-distributed dataset to ``root`` (returns ``None`` elsewhere).

        A no-op copy when the data is not distributed.
        """
        from . import mpi

        comm = comm or mpi.world()
        if comm.Get_size() == 1 or not self.is_distributed:
            return self
        x = mpi.gatherv_elements(self.x, comm, root)
        y = mpi.gatherv_elements(self.y, comm, root)
        fields = {k: mpi.gatherv_elements(v, comm, root) for k, v in self.fields.items()}
        elmap = mpi.gatherv_elements(self.elmap, comm, root) if self.elmap is not None else None
        if comm.Get_rank() != root:
            return None
        return SEMData2D(x, y, fields, time=self.time, elmap=elmap, name=self.name, source=self.source, expressions=self.expressions)

    # ------------------------------------------------------------ IO
    @classmethod
    def from_pysemtools(cls, msh, fld=None, name: str = "") -> "SEMData2D":
        """Build from pysemtools ``Mesh`` and ``FieldRegistry`` objects (2D, lz == 1)."""
        if msh.lz != 1:
            raise ValueError("semview handles 2D data only (msh.lz must be 1)")
        fields = {}
        t = 0.0
        if fld is not None:
            fields = {k: np.ascontiguousarray(v[:, 0]) for k, v in fld.registry.items()}
            t = float(getattr(fld, "t", 0.0))
        return cls(
            np.ascontiguousarray(msh.x[:, 0]),
            np.ascontiguousarray(msh.y[:, 0]),
            fields,
            time=t,
            elmap=np.asarray(msh.elmap).copy() if getattr(msh, "elmap", None) is not None else None,
            name=name,
            offset_el=int(msh.offset_el),
            glb_nelv=int(msh.glb_nelv),
        )

    def write(self, path: str, comm=None) -> None:
        """Write to a Nek5000 field file via pysemtools (``pynekwrite``)."""
        from . import mpi
        from pysemtools.datatypes.field import FieldRegistry
        from pysemtools.datatypes.msh import Mesh
        from pysemtools.io.ppymech.neksuite import pynekwrite

        comm = comm or mpi.world()
        z = np.zeros_like(self.x)
        # Nek5000 stores a 1-based global element map
        elmap = self.elmap if self.elmap is not None else np.arange(1, self.nelv + 1) + self.offset_el
        msh = Mesh(comm, x=self.x[:, None], y=self.y[:, None], z=z[:, None], elmap=np.asarray(elmap, dtype=np.int32), create_connectivity=False)
        fld = FieldRegistry(comm)
        # velocity components must come first and in order (Nek convention)
        order = [k for k in ("u", "v") if k in self.fields] + [k for k in self.fields if k not in ("u", "v")]
        for k in order:
            v = self.fields[k]
            fld.add_field(comm, field_name=k, field=np.ascontiguousarray(v[:, None]), dtype=v.dtype)
        fld.t = self.time
        pynekwrite(path, comm, msh=msh, fld=fld, wdsz=4, istep=0, write_mesh=True)


# --------------------------------------------------------------------------- #
class Dataset:
    """A time series of :class:`SEMData2D` steps backed by Nek5000 files.

    Steps are read lazily with pysemtools (MPI-parallel when run under
    ``mpirun``) and cached.  The mesh is read once and shared by all steps.
    """

    def __init__(
        self,
        path: str,
        comm=None,
        dtype=np.float64,
        cache_steps: int = 8,
        cache_bytes: int = 2 << 30,
        mesh_file: str | None = None,
    ):
        from . import io, mpi

        self.series = io.resolve_files(path)
        self.path = path
        self.comm = comm or mpi.world()
        self.dtype = dtype
        self._cache: OrderedDict[int, SEMData2D] = OrderedDict()
        self._cache_size = cache_steps
        self._cache_bytes = cache_bytes
        self._mesh = None  # (x, y, elmap, offset_el, glb_nelv)
        self._mesh_file = mesh_file
        self.expressions = calc.Expressions()   # calculator definitions, shared by all steps
        self._load_mesh()

    # ------------------------------------------------------------ properties
    @property
    def name(self) -> str:
        return self.series.name

    @property
    def files(self) -> list[str]:
        return self.series.files

    def __len__(self) -> int:
        return len(self.series)

    @property
    def times(self) -> np.ndarray:
        return self.series.times()

    def __repr__(self):
        return f"Dataset({self.name!r}, steps={len(self)}, nelv={self.mesh.glb_nelv}, order={self.mesh.order})"

    # ------------------------------------------------------------ mesh
    def _load_mesh(self):
        from . import io

        candidates = [self._mesh_file] if self._mesh_file else []
        candidates += [f for f in self.series.files]
        for f in candidates:
            if f is None:
                continue
            if io.read_header(f).has_mesh:
                coords, fields, t, elmap = io.read_step(f, self.comm, self.dtype)
                nelv = coords[0].shape[0]
                offset = self.comm.scan(nelv) - nelv if self.comm.Get_size() > 1 else 0
                glb = self.comm.allreduce(nelv) if self.comm.Get_size() > 1 else nelv
                self._mesh = (coords[0], coords[1], elmap, offset, glb)
                self._mesh_from = f
                # if this is also a step, cache its fields
                try:
                    i = self.series.files.index(f)
                    self._store(i, SEMData2D(coords[0], coords[1], fields, time=t, elmap=elmap, name=self.name, source=f, offset_el=offset, glb_nelv=glb, expressions=self.expressions))
                except ValueError:
                    pass
                return
        raise ValueError(
            f"No mesh coordinates found in {self.path}. Pass mesh_file= pointing to a file that contains the mesh."
        )

    @property
    def mesh(self) -> SEMData2D:
        """The mesh (as an empty-field :class:`SEMData2D`)."""
        x, y, elmap, offset, glb = self._mesh
        return SEMData2D(x, y, {}, elmap=elmap, name=self.name, offset_el=offset, glb_nelv=glb, expressions=self.expressions)

    # ------------------------------------------------------------ steps
    def _store(self, i: int, data: SEMData2D):
        """Cache a step; evict the least recently used ones beyond the count/byte budget."""
        self._cache[i] = data
        self._cache.move_to_end(i)
        while len(self._cache) > 1 and (len(self._cache) > self._cache_size or sum(d.nbytes for d in self._cache.values()) > self._cache_bytes):
            self._cache.popitem(last=False)

    def __getitem__(self, i: int) -> SEMData2D:
        """Load (or fetch from cache) step ``i``; negative indices count from the end."""
        from . import io

        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        x, y, elmap, offset, glb = self._mesh
        coords, fields, t, elmap_i = io.read_step(self.series.files[i], self.comm, self.dtype)
        if coords is not None:
            x, y = coords
            elmap = elmap_i
        data = SEMData2D(x, y, fields, time=t, elmap=elmap, name=self.name, source=self.series.files[i], offset_el=offset, glb_nelv=glb, expressions=self.expressions)
        self._store(i, data)
        return data

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def step_at_time(self, t: float) -> int:
        """Index of the step closest to physical time ``t``."""
        return int(np.argmin(np.abs(self.times - t)))

    @property
    def field_names(self) -> list[str]:
        return self[0].field_names

    # ------------------------------------------------------------ field calculator
    def define(self, name: str, expr: str) -> str:
        """Define a calculator field for every step of the series (see :meth:`SEMData2D.define`)."""
        return self.expressions.define(name, expr, stored=self.field_names)

    def undefine(self, name: str) -> None:
        self.expressions.remove(name)

    def gather(self, i: int, root: int = 0) -> SEMData2D | None:
        """Load step ``i`` and gather it to ``root``."""
        return self[i].gather(self.comm, root)


def open(path: str, **kwargs) -> Dataset:  # noqa: A001 - mirrors the file-open idiom
    """Open a Nek5000 ``.nek5000`` metafile, field file, or glob as a :class:`Dataset`."""
    return Dataset(path, **kwargs)


def load(path: str, step: int = -1, gather: bool = True, **kwargs) -> SEMData2D:
    """Convenience: open ``path`` and return one step (the last by default).

    Under MPI the step is gathered to rank 0 unless ``gather=False``; other
    ranks then receive ``None``.
    """
    ds = Dataset(path, **kwargs)
    data = ds[step]
    if gather:
        return data.gather(ds.comm)
    return data
