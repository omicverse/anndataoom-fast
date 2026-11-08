"""Map an anndataoom lazy array onto the original on-disk CSR matrix.

anndataoom represents every preprocessing step as a lazy wrapper:

    ScaledBackedArray / TransformedBackedArray   (size factors, log1p, z-score)
      -> _SubsetBackedArray                      (QC / gene filters: row & column index)
        -> BackedArray                           (the on-disk matrix)

:func:`resolve` walks that chain and returns a :class:`View` describing the
same matrix as (file, rows, columns, transforms), which the C++ passes can
stream directly.  Anything it does not recognise returns ``None`` and the
caller falls back to anndataoom's own implementation.
"""
from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass, field

import numpy as np

from ._h5 import CSRHandle, Unsupported, open_csr, read_rows

# Root BackedArray -> (path, h5 key).  Weak, so closed datasets drop out.
_ROOTS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _anndataoom():
    import anndataoom
    from anndataoom import _chunked_ops, _core
    from anndataoom._backed_array import BackedArray
    return BackedArray, _core._SubsetBackedArray, _chunked_ops.TransformedBackedArray, \
        _chunked_ops.ScaledBackedArray


def digest(*arrays) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        if a is None:
            h.update(b"<none>")
            continue
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _same_rows(root, path: str, key: str, n: int = 8) -> bool:
    """Do the first rows of `root` equal rows of `key` on disk?"""
    import scipy.sparse as sp
    m = min(n, root.shape[0])
    if m == 0:
        return True
    a = root._read_rows(0, m)
    b = read_rows(path, key, 0, m)
    a = sp.csr_matrix(a)
    return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
            and np.array_equal(a.indices, b.indices) and np.array_equal(a.data, b.data))


def register_root(root, path: str, key: str) -> None:
    _ROOTS[root] = (path, key)


def identify_roots(adata) -> None:
    """Register the on-disk source of adata.X / layers when it can be proven."""
    BackedArray, Sub, Tr, _ = _anndataoom()
    path = getattr(adata, "filename", None)
    if not path:
        return
    import h5py
    try:
        with h5py.File(path, "r") as f:
            keys = [k for k in ["X"] + [f"layers/{n}" for n in f.get("layers", {})]
                    if k in f and isinstance(f[k], h5py.Group)
                    and str(f[k].attrs.get("encoding-type", b"").decode()
                            if isinstance(f[k].attrs.get("encoding-type", b""), bytes)
                            else f[k].attrs.get("encoding-type", "")) == "csr_matrix"]
            shapes = {k: tuple(int(x) for x in f[k].attrs["shape"]) for k in keys}
    except OSError:
        return
    arrays = [adata.X] + list(getattr(adata.layers, "_in_memory", {}).values())
    for arr in arrays:
        root = _root_of(arr)
        if root is None or root in _ROOTS:
            continue
        for k in keys:
            if shapes[k] == tuple(root.shape) and _same_rows(root, path, k):
                register_root(root, path, k)
                break


def _root_of(arr):
    BackedArray, Sub, Tr, _ = _anndataoom()
    while isinstance(arr, (Tr, Sub)):
        arr = arr._parent
    if type(arr) is BackedArray and getattr(arr, "_is_rs", False):
        return arr
    return None


@dataclass
class View:
    handle: CSRHandle
    rows: np.ndarray | None          # sorted original rows (None = all)
    cols: np.ndarray | None          # original columns in view order (None = all)
    norm_factors: np.ndarray | None  # per view row
    log1p: bool
    scale_mean: np.ndarray | None = None
    scale_std: np.ndarray | None = None
    max_value: float | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def raw(self) -> bool:
        return self.norm_factors is None and not self.log1p and self.scale_mean is None

    @property
    def n_rows(self) -> int:
        return self.handle.shape[0] if self.rows is None else len(self.rows)

    @property
    def n_cols(self) -> int:
        return self.handle.shape[1] if self.cols is None else len(self.cols)

    @property
    def row_keep(self) -> np.ndarray:
        if "row_keep" not in self._cache:
            m = np.zeros(self.handle.shape[0], dtype=np.uint8)
            m[slice(None) if self.rows is None else self.rows] = 1
            self._cache["row_keep"] = m
        return self._cache["row_keep"]

    def col_map(self, sel: np.ndarray | None = None) -> np.ndarray:
        """int32 map original column -> position in the view (or in `sel`)."""
        cols = np.arange(self.handle.shape[1]) if self.cols is None else self.cols
        if sel is not None:
            cols = cols[sel]
        m = np.full(self.handle.shape[1], -1, dtype=np.int32)
        m[cols] = np.arange(len(cols), dtype=np.int32)
        return m

    def col_keep(self) -> np.ndarray:
        return (self.col_map() >= 0).astype(np.uint8)

    def to_orig_rows(self, per_view_row: np.ndarray, fill, dtype) -> np.ndarray:
        out = np.full(self.handle.shape[0], fill, dtype=dtype)
        out[slice(None) if self.rows is None else self.rows] = per_view_row
        return out

    def from_orig_rows(self, per_orig_row: np.ndarray) -> np.ndarray:
        return per_orig_row if self.rows is None else per_orig_row[self.rows]

    def from_orig_cols(self, per_orig_col: np.ndarray) -> np.ndarray:
        return per_orig_col if self.cols is None else per_orig_col[self.cols]

    def recip(self) -> np.ndarray:
        """float32 per-original-row reciprocal size factor, as anndataoom applies it."""
        if self.norm_factors is None:
            return np.ones(self.handle.shape[0], dtype=np.float32)
        f = np.asarray(self.norm_factors).astype(np.float64)
        f[f == 0] = 1.0
        return self.to_orig_rows(1.0 / f.astype(np.float32), 1.0, np.float32)

    def key(self) -> tuple:
        if "key" not in self._cache:
            self._cache["key"] = (self.handle.path, self.handle.key,
                                  digest(self.rows), digest(self.cols))
        return self._cache["key"]


def resolve(arr) -> View | None:
    """Describe `arr` as a view of an on-disk matrix, or None."""
    BackedArray, Sub, Tr, Scaled = _anndataoom()
    mean = std = max_value = None
    nf, log1p = None, False
    if type(arr) is Scaled:
        mean, std, max_value = arr._scale_mean, arr._scale_std, arr._max_value
    if type(arr) in (Tr, Scaled):
        nf, log1p = arr._norm_factors, bool(arr._apply_log1p)
        arr = arr._parent
    elif isinstance(arr, Tr):  # Pearson / regressed / other wrappers
        return None
    rows = cols = None
    if type(arr) is Sub:
        rows, cols = arr._obs_idx, arr._var_idx
        arr = arr._parent
    if type(arr) is not BackedArray or not getattr(arr, "_is_rs", False):
        return None
    src = _ROOTS.get(arr)
    if src is None:
        return None
    try:
        handle = open_csr(*src)
    except (Unsupported, OSError):
        return None
    if tuple(handle.shape) != tuple(arr.shape):
        return None
    if rows is not None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and np.any(np.diff(rows) <= 0):
            return None  # reordered / duplicated rows: keep anndataoom's path
    if cols is not None:
        cols = np.asarray(cols, dtype=np.int64)
        if np.unique(cols).size != cols.size:
            return None
    v = View(handle, rows, cols, None if nf is None else np.asarray(nf), log1p,
             mean, std, max_value)
    if nf is not None and len(nf) != v.n_rows:
        return None
    return v
