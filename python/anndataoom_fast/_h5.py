"""Describe an on-disk h5ad CSR matrix for the C++ reader.

h5py is used once, up front, to find where every chunk of ``data`` and
``indices`` lives in the file; after that the C++ side reads raw bytes with
``pread`` and decodes them itself, on all threads, without libhdf5.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import h5py
import numpy as np

from . import _core

_DTYPE_CODES = {
    np.dtype("<f4"): 0, np.dtype("<f8"): 1,
    np.dtype("i1"): 2, np.dtype("<i2"): 3, np.dtype("<i4"): 4, np.dtype("<i8"): 5,
    np.dtype("u1"): 6, np.dtype("<u2"): 7, np.dtype("<u4"): 8, np.dtype("<u8"): 9,
}
_SUPPORTED_FILTERS = {1, 2, 3}  # deflate, shuffle, fletcher32


class Unsupported(Exception):
    """The on-disk layout cannot be read by the C++ reader."""


def _dtype_code(dt: np.dtype) -> int:
    dt = np.dtype(dt)
    if dt.byteorder == ">":
        raise Unsupported(f"big-endian dtype {dt}")
    code = _DTYPE_CODES.get(dt.newbyteorder("<") if dt.byteorder == "=" else dt)
    if code is None:
        raise Unsupported(f"dtype {dt}")
    return code


def dataset(path: str, ds: h5py.Dataset) -> _core.Dataset:
    """Build a :class:`_core.Dataset` describing a 1-D HDF5 dataset."""
    if ds.ndim != 1:
        raise Unsupported(f"{ds.name} is not 1-D")
    code = _dtype_code(ds.dtype)
    n = int(ds.shape[0])
    dcpl = ds.id.get_create_plist()
    filters = [int(dcpl.get_filter(i)[0]) for i in range(dcpl.get_nfilters())]
    if ds.chunks is None:
        if filters:
            raise Unsupported(f"{ds.name}: filters on a contiguous dataset")
        off = ds.id.get_offset()
        if off is None:
            if n:
                raise Unsupported(f"{ds.name}: storage not allocated")
            off = 0
        return _core.Dataset(path, code, n, int(off))
    if set(filters) - _SUPPORTED_FILTERS:
        raise Unsupported(f"{ds.name}: HDF5 filters {filters}")
    ce = int(ds.chunks[0])
    n_chunks = max(1, math.ceil(n / ce)) if n else 0
    addr = np.full(n_chunks, -1, dtype=np.int64)
    nbytes = np.zeros(n_chunks, dtype=np.int64)
    mask = np.zeros(n_chunks, dtype=np.uint32)

    def visit(info):
        c = int(info.chunk_offset[0]) // ce
        addr[c] = info.byte_offset
        nbytes[c] = info.size
        mask[c] = info.filter_mask

    try:
        ds.id.chunk_iter(visit)
    except (AttributeError, NotImplementedError):  # HDF5 < 1.12.3
        for i in range(ds.id.get_num_chunks()):
            visit(ds.id.get_chunk_info(i))
    return _core.Dataset(path, code, n, ce, addr, nbytes, mask, filters)


@dataclass
class CSRHandle:
    path: str
    key: str
    shape: tuple[int, int]
    src: "_core.CSRSource"
    indptr: np.ndarray


_HANDLES: dict[tuple, CSRHandle] = {}


def open_csr(path: str, key: str = "X", target_block_elems: int = 1 << 21) -> CSRHandle:
    """Open the CSR matrix stored at ``key`` of an h5ad file (cached)."""
    path = os.path.realpath(os.fspath(path))
    st = os.stat(path)
    cache_key = (path, key, st.st_mtime_ns, st.st_size, target_block_elems)
    h = _HANDLES.get(cache_key)
    if h is not None:
        return h
    with h5py.File(path, "r") as f:
        if key not in f:
            raise Unsupported(f"{key!r} not in {path}")
        g = f[key]
        if not isinstance(g, h5py.Group):
            raise Unsupported(f"{key!r} is a dense array")
        enc = g.attrs.get("encoding-type", b"")
        enc = enc.decode() if isinstance(enc, bytes) else str(enc)
        if enc != "csr_matrix":
            raise Unsupported(f"{key!r} has encoding {enc!r}, need csr_matrix")
        shape = tuple(int(x) for x in g.attrs["shape"])
        indptr = np.ascontiguousarray(g["indptr"][:], dtype=np.int64)
        data = dataset(path, g["data"])
        indices = dataset(path, g["indices"])
    src = _core.CSRSource(data, indices, indptr, shape[1], target_block_elems)
    h = CSRHandle(path, key, shape, src, indptr)
    _HANDLES.clear()  # keep at most one open matrix per process
    _HANDLES[cache_key] = h
    return h


def read_rows(path: str, key: str, r0: int, r1: int):
    """Rows [r0, r1) of an on-disk CSR matrix via h5py (for verification)."""
    import scipy.sparse as sp
    with h5py.File(path, "r") as f:
        g = f[key]
        ip = g["indptr"][r0:r1 + 1]
        a, b = int(ip[0]), int(ip[-1])
        shape = tuple(int(x) for x in g.attrs["shape"])
        return sp.csr_matrix((g["data"][a:b], g["indices"][a:b], ip - a),
                             shape=(r1 - r0, shape[1]))
