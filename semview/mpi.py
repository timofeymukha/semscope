"""Small MPI helpers.  Everything degrades gracefully to serial when the
communicator has a single rank."""

from __future__ import annotations

import numpy as np

__all__ = ["world", "rank", "size", "is_root", "gatherv_elements", "bcast"]


def world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def rank(comm=None) -> int:
    comm = comm or world()
    return comm.Get_rank()


def size(comm=None) -> int:
    comm = comm or world()
    return comm.Get_size()


def is_root(comm=None, root: int = 0) -> bool:
    return rank(comm) == root


def bcast(obj, comm=None, root: int = 0):
    comm = comm or world()
    if comm.Get_size() == 1:
        return obj
    return comm.bcast(obj, root=root)


def gatherv_elements(arr: np.ndarray, comm=None, root: int = 0):
    """Gather an element-distributed array ``(nelv_local, ...)`` to ``root``.

    Returns the concatenated ``(nelv_global, ...)`` array on ``root`` and
    ``None`` elsewhere.  Works for any trailing shape and dtype.
    """
    comm = comm or world()
    if comm.Get_size() == 1:
        return arr
    from mpi4py import MPI

    arr = np.ascontiguousarray(arr)
    trailing = arr.shape[1:]
    per_el = int(np.prod(trailing, dtype=np.int64)) if trailing else 1
    counts = comm.gather(arr.shape[0] * per_el, root=root)
    recv = None
    if comm.Get_rank() == root:
        total = int(sum(counts))
        recv = np.empty((total // per_el,) + tuple(trailing), dtype=arr.dtype)
        displs = np.concatenate([[0], np.cumsum(counts)[:-1]])
        recvbuf = [recv, (counts, displs), MPI._typedict[arr.dtype.char]]
    else:
        recvbuf = None
    comm.Gatherv(arr, recvbuf, root=root)
    return recv
