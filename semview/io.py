"""File-format layer: Nek5000/Neko field files through pysemtools' MPI readers.

Only this module (and :mod:`semview.mpi`) talk to pysemtools and mpi4py.  The
rest of the package works on plain numpy arrays held by
:class:`semview.dataset.SEMData2D`.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("PYSEMTOOLS_HIDE_LOG", "true")  # also set in semview/__init__.py

__all__ = ["Nek5000Series", "read_header", "resolve_files", "read_step"]


# --------------------------------------------------------------------------- #
# Metafile handling
# --------------------------------------------------------------------------- #
_STEP_RE = re.compile(r"^(?P<base>.*?)(?P<fid>\d)\.f(?P<step>\d{5})$")


@dataclass
class Nek5000Series:
    """A (possibly single-file) sequence of Nek5000 field files.

    Attributes
    ----------
    files : list of file paths, in time-step order
    name : short case name used for labels
    """

    files: list[str]
    name: str = ""
    metafile: str | None = None
    _times: dict[int, float] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.files)

    def time(self, i: int) -> float:
        """Physical time of step ``i`` (read from the file header, cached)."""
        if i not in self._times:
            self._times[i] = read_header(self.files[i]).time
        return self._times[i]

    def times(self) -> np.ndarray:
        return np.array([self.time(i) for i in range(len(self))])


def parse_metafile(path: str) -> Nek5000Series:
    """Parse a ``.nek5000`` metafile.

    The format is three ``key: value`` lines::

        filetemplate:         case%01d.f%05d
        firsttimestep:     0
        numtimesteps:     4

    Some writers use ``%s`` for the case name; that is substituted with the
    metafile's base name.  Missing files are dropped (with a warning).
    """
    keys: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if ":" in line:
                k, v = line.split(":", 1)
                keys[k.strip().lower()] = v.strip()
    if "filetemplate" not in keys:
        raise ValueError(f"{path}: no 'filetemplate' entry, not a Nek5000 metafile")
    template = keys["filetemplate"]
    first = int(keys.get("firsttimestep", 0))
    num = int(keys.get("numtimesteps", 1))
    base = os.path.splitext(os.path.basename(path))[0]
    dirname = os.path.dirname(os.path.abspath(path))
    if "%s" in template:
        template = template.replace("%s", base, 1)
    nfmt = template.count("%")
    files = []
    for step in range(first, first + num):
        if nfmt >= 2:
            fname = template % (0, step)
        elif nfmt == 1:
            fname = template % step
        else:
            fname = template
        fpath = fname if os.path.isabs(fname) else os.path.join(dirname, fname)
        if os.path.exists(fpath):
            files.append(fpath)
        else:
            import warnings

            warnings.warn(f"metafile {path}: missing step file {fpath}", stacklevel=2)
    if not files:
        raise FileNotFoundError(f"{path}: none of the {num} step files exist")
    return Nek5000Series(files=files, name=base, metafile=os.path.abspath(path))


def resolve_files(path: str) -> Nek5000Series:
    """Turn ``path`` into a :class:`Nek5000Series`.

    ``path`` may be a ``.nek5000`` metafile, a single ``case0.f00012`` file (in
    which case all sibling steps of that case are collected), a glob pattern,
    or a ``.fld`` style prefix such as ``case0.f*``.
    """
    if os.path.isdir(path):
        metas = sorted(glob.glob(os.path.join(path, "*.nek5000")))
        if len(metas) == 1:
            return parse_metafile(metas[0])
        raise ValueError(f"{path} is a directory; give a .nek5000 metafile or field file")
    if path.endswith(".nek5000"):
        return parse_metafile(path)
    if any(ch in path for ch in "*?["):
        files = sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(path)
        name = re.sub(r"\d\.f.*$", "", os.path.basename(files[0]))
        return Nek5000Series(files=[os.path.abspath(f) for f in files], name=name)
    m = _STEP_RE.match(os.path.basename(path))
    if m and os.path.exists(path):
        # A single step file: pick up its siblings (same base and file id)
        dirname = os.path.dirname(os.path.abspath(path))
        pattern = os.path.join(dirname, f"{m['base']}{m['fid']}.f[0-9][0-9][0-9][0-9][0-9]")
        files = sorted(glob.glob(pattern))
        if os.path.abspath(path) not in files:
            files = [os.path.abspath(path)]
        return Nek5000Series(files=files, name=m["base"])
    if os.path.exists(path):
        return Nek5000Series(files=[os.path.abspath(path)], name=os.path.basename(path))
    raise FileNotFoundError(path)


# --------------------------------------------------------------------------- #
# Reading through pysemtools
# --------------------------------------------------------------------------- #
@dataclass
class Header:
    wdsz: int
    orders: tuple[int, int, int]
    nelv: int
    variables: str
    time: float
    istep: int
    has_mesh: bool
    nb_vars: tuple[int, ...]


def read_header(path: str) -> Header:
    """Read the 132-byte Nek5000 header (serial, cheap)."""
    from pysemtools.io.ppymech.neksuite import read_nekheader

    h = read_nekheader(path)
    return Header(
        wdsz=int(h.wdsz),
        orders=tuple(int(o) for o in h.orders),
        nelv=int(h.nb_elems),
        variables=str(h.variables),
        time=float(h.time),
        istep=int(h.istep),
        has_mesh="X" in str(h.variables).upper(),
        nb_vars=tuple(int(v) for v in h.nb_vars),
    )


def read_step(path: str, comm=None, dtype=np.float64, mesh=None):
    """Read one field file with pysemtools (MPI-parallel over elements).

    Returns ``(coords, fields, time, elmap)`` on every rank for the rank's own
    share of elements: ``coords`` is ``(x, y)`` with shape ``(nelv, n, n)`` or
    ``None`` if the file carries no mesh, ``fields`` is a ``dict`` name ->
    ``(nelv, n, n)`` array.
    """
    from mpi4py import MPI
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.io.ppymech.neksuite import pynekread

    if comm is None:
        comm = MPI.COMM_WORLD
    hdr = read_header(path)
    if hdr.orders[2] > 1:
        raise ValueError(
            f"{path}: this is a 3D file (lz={hdr.orders[2]}); semview handles 2D "
            "spectral-element data (lz == 1)."
        )
    msh = Mesh(comm, create_connectivity=False) if hdr.has_mesh else None
    fld = FieldRegistry(comm)
    pynekread(path, comm, data_dtype=dtype, msh=msh, fld=fld)

    coords = None
    elmap = None
    if msh is not None and getattr(msh, "x", None) is not None:
        coords = (np.ascontiguousarray(msh.x[:, 0, :, :]), np.ascontiguousarray(msh.y[:, 0, :, :]))
        elmap = np.asarray(msh.elmap).copy()
    fields = {k: np.ascontiguousarray(v[:, 0, :, :]) for k, v in fld.registry.items()}
    return coords, fields, float(fld.t), elmap


def read_mesh_only(path: str, comm=None, dtype=np.float64):
    coords, _fields, _t, elmap = read_step(path, comm=comm, dtype=dtype)
    if coords is None:
        raise ValueError(f"{path} contains no mesh coordinates")
    return coords, elmap
