"""Exact, fast port of omicverse's CPU Leiden path (leidenalg).

``leiden(adjacency, r, random_state=s)`` returns the same membership, element
for element, as omicverse ``ov.pp.leiden(flavor='leidenalg')``, i.e.::

    g = get_igraph_from_adjacency(adjacency, directed=True)
    leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                             weights=np.array(g.es['weight']).astype(np.float64),
                             resolution_parameter=r, n_iterations=-1, seed=s
                             ).membership

The C++ core (``src/cpp/leiden``) is a port of leidenalg / libleidenalg
(V.A. Traag, GPL-3.0-or-later) and the parts of the igraph C core it uses
(GPL-2.0-or-later).  Distributed under GPL-3.0-or-later.

``leiden_brisc`` / ``leiden_brisc_multi`` are a port of brisc's Leiden
(``brisc.SingleCell.cluster``; ``src/cpp/leiden_brisc``, ported from
brisc/leiden.pyx, Copyright (c) 2026 Michael Wainberg and Keon Arbabi, MIT):
they return exactly brisc's labels.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _leiden_cpp as _cpp_leiden  # type: ignore[attr-defined]
from . import _leiden_brisc_cpp as _cpp_brisc  # type: ignore[attr-defined]

__all__ = ["leiden", "leiden_multi", "leiden_brisc", "leiden_brisc_multi", "edges_from_adjacency"]


def edges_from_adjacency(adjacency, use_weights: bool = True):
    """Edge list exactly as ``get_igraph_from_adjacency`` builds it.

    ``sources, targets = adjacency.nonzero()`` (stored, row-major order for
    CSR; explicit zeros dropped) and ``weights = adjacency[sources, targets]``
    converted to float64.  Returns ``(n, src, dst, weights_or_None)``.
    """
    import scipy.sparse as sp

    n = int(adjacency.shape[0])
    if sp.issparse(adjacency) and adjacency.format == "csr" and adjacency.has_canonical_format:
        # Fast path: no duplicate entries, so nonzero() order is the storage
        # order and adjacency[i, j] is the stored value.
        indptr = np.asarray(adjacency.indptr)
        data = np.asarray(adjacency.data)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.int64))
        cols = np.asarray(adjacency.indices, dtype=np.int64)
        mask = data != 0
        if not mask.all():
            rows, cols, data = rows[mask], cols[mask], data[mask]
        w = np.ascontiguousarray(data, dtype=np.float64) if use_weights else None
        return n, np.ascontiguousarray(rows), np.ascontiguousarray(cols), w
    # Generic path: literally what omicverse does.
    sources, targets = adjacency.nonzero()
    src = np.ascontiguousarray(sources, dtype=np.int64)
    dst = np.ascontiguousarray(targets, dtype=np.int64)
    w = None
    if use_weights:
        if len(src):
            w = np.asarray(adjacency[sources, targets]).ravel()
            w = np.ascontiguousarray(w, dtype=np.float64)
        else:
            w = np.zeros(0, dtype=np.float64)
    return n, src, dst, w


def _seed(random_state) -> int:
    if random_state is None:
        import time
        return int(time.time()) & 0xFFFFFFFFFFFFFFFF
    if isinstance(random_state, np.random.RandomState):
        raise TypeError("random_state must be an int (leidenalg only accepts integer seeds)")
    return int(random_state) & 0xFFFFFFFFFFFFFFFF  # Py_ssize_t -> size_t


def leiden_multi(
    adjacency,
    resolutions: Sequence[float],
    *,
    random_state: int = 0,
    directed: bool = True,
    n_iterations: int = -1,
    use_weights: bool = True,
    n_threads: int = 0,
) -> list[np.ndarray]:
    """Leiden memberships for several resolutions, one OpenMP thread each.

    Every resolution is an independent run with its own RNG seeded with
    ``random_state``, so ``leiden_multi(A, rs)[i]`` equals ``leiden(A, rs[i])``.
    ``n_threads`` (0 = all OpenMP threads) are split into one thread per
    resolution, the remainder being used for graph aggregation inside each
    run.  The result never depends on the number of threads.
    """
    n, src, dst, w = edges_from_adjacency(adjacency, use_weights)
    res = [float(r) for r in resolutions]
    return list(_cpp_leiden.find_partition(
        n, src, dst, w, bool(directed), res, _seed(random_state), int(n_iterations), int(n_threads)))


def leiden(
    adjacency,
    resolution: float = 1.0,
    *,
    random_state: int = 0,
    directed: bool = True,
    n_iterations: int = -1,
    use_weights: bool = True,
    n_threads: int = 0,
) -> np.ndarray:
    """leidenalg RBConfigurationVertexPartition membership (int64), exact.

    The node-moving phases are inherently sequential (they consume one RNG
    stream); ``n_threads`` (0 = all OpenMP threads) only parallelises graph
    aggregation, so the result never depends on it.
    """
    return leiden_multi(adjacency, [resolution], random_state=random_state, directed=directed,
                        n_iterations=n_iterations, use_weights=use_weights,
                        n_threads=n_threads)[0]


# ---------------------------------------------------------------------------
# brisc's Leiden
# ---------------------------------------------------------------------------

def _brisc_graph(adjacency):
    """CSR arrays as brisc expects them: float32 weights, no self loops, no
    explicit zeros.  The order of the remaining entries is kept (it affects
    the result exactly as it does in brisc)."""
    import scipy.sparse as sp

    if not sp.issparse(adjacency):
        raise TypeError("adjacency must be a scipy sparse matrix")
    A = adjacency if adjacency.format == "csr" else adjacency.tocsr()
    n = int(A.shape[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {A.shape}")
    if n == 0 or A.nnz == 0:
        raise ValueError("adjacency is an empty graph")
    indptr = np.asarray(A.indptr)
    indices = np.asarray(A.indices)
    data = np.asarray(A.data)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    if not np.isfinite(data).all():
        raise ValueError("adjacency contains NaN or infinite weights")
    if (data < 0).any():
        raise ValueError("adjacency contains negative weights")
    if (data == 0).any() or np.asarray(A.diagonal()).any():
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        keep = (data != 0) & (indices != rows)
        if not keep.all():
            counts = np.bincount(rows[keep], minlength=n)
            indptr = np.concatenate(([0], np.cumsum(counts))).astype(indptr.dtype)
            indices = indices[keep]
            data = data[keep]
            if len(data) == 0:
                raise ValueError("adjacency is an empty graph")
        del rows, keep
    idx_t = np.int32 if (indptr.dtype == np.int32 and indices.dtype == np.int32) else np.int64
    return (np.ascontiguousarray(data), np.ascontiguousarray(indices, dtype=idx_t),
            np.ascontiguousarray(indptr, dtype=idx_t))


def _brisc_neighbors(n, neighbors, min_cluster_size, graph):
    """(nbr, nbr_indptr, k) for the small-cluster merge."""
    if min_cluster_size <= 1:
        return np.zeros(0, dtype=np.uint32), None, 0
    if neighbors is not None:
        nbr = np.asarray(neighbors)
        if nbr.ndim != 2 or nbr.shape[0] != n:
            raise ValueError(f"neighbors must have shape (n, k) with n = {n}")
        if not np.issubdtype(nbr.dtype, np.integer):
            raise TypeError("neighbors must be an integer array")
        if nbr.size and (nbr.min() < 0 or nbr.max() >= n):
            raise ValueError(f"some nearest-neighbor indices are < 0 or >= the total number "
                             f"of cells, {n:,}")
        return np.ascontiguousarray(nbr, dtype=np.uint32).ravel(), None, int(nbr.shape[1])
    # No kNN given: each node's graph neighbours by decreasing edge weight
    # (ties in storage order).
    data, indices, indptr = graph
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    order = np.lexsort((np.arange(len(data)), -data.astype(np.float64), rows))
    return (np.ascontiguousarray(indices[order], dtype=np.uint32),
            np.ascontiguousarray(indptr, dtype=np.int64), 0)


def leiden_brisc_multi(
    adjacency,
    resolutions: Sequence[float],
    *,
    random_state: int = 0,
    min_cluster_size: int = 1,
    neighbors=None,
    n_threads: int = 0,
) -> list[np.ndarray]:
    """brisc Leiden labels (int64) for several resolutions, one thread each.

    Port of ``brisc.SingleCell.cluster(resolution=resolutions, seed=random_state,
    min_cluster_size=...)``: ``adjacency`` plays the role of
    ``obsp['shared_neighbors']`` and must be symmetric (not checked, as in
    brisc).  Self loops and explicit zeros are dropped and weights are
    converted to float32, as brisc requires.  Labels are brisc's (cluster
    ``c`` is brisc's ``'c'``), element for element.

    ``neighbors`` (``obsm['neighbors']``, an ``(n, k)`` integer array of each
    cell's nearest neighbours, nearest first, excluding the cell itself) is
    only used when ``min_cluster_size > 1``, to merge cells of too-small
    clusters into the cluster of their nearest neighbour in a large enough
    cluster.  If it is None, each cell's graph neighbours sorted by
    decreasing edge weight are used instead.

    Each resolution is an independent run seeded with ``random_state``, so
    ``leiden_brisc_multi(A, rs)[i]`` equals ``leiden_brisc(A, rs[i])`` and
    the result does not depend on ``n_threads`` (0 = all OpenMP threads,
    at most one per resolution).
    """
    res = [float(np.float32(r)) for r in resolutions]
    if len(res) == 0:
        raise ValueError("resolutions is empty")
    for r in res:
        if not r > 0:
            raise ValueError(f"a resolution is {r:,}, but all resolutions must be > 0")
    min_cluster_size = int(min_cluster_size)
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be a positive integer")
    graph = _brisc_graph(adjacency)
    n = len(graph[2]) - 1
    nbr, nbr_indptr, k = _brisc_neighbors(n, neighbors, min_cluster_size, graph)
    out = np.empty((len(res), n), dtype=np.uint32)
    seed = int(random_state) & 0xFFFFFFFFFFFFFFFF
    _cpp_brisc.leiden(graph[0], graph[1], graph[2], nbr, nbr_indptr, k, out, res,
                      min_cluster_size, seed, int(n_threads))
    return [row.astype(np.int64) for row in out]


def leiden_brisc(
    adjacency,
    resolution: float = 1.0,
    *,
    random_state: int = 0,
    min_cluster_size: int = 1,
    neighbors=None,
    n_threads: int = 0,
) -> np.ndarray:
    """brisc Leiden labels (int64), identical to ``brisc.SingleCell.cluster``.

    The algorithm is sequential (deterministic); see
    :func:`leiden_brisc_multi` for the parameters.
    """
    return leiden_brisc_multi(adjacency, [resolution], random_state=random_state,
                              min_cluster_size=min_cluster_size, neighbors=neighbors,
                              n_threads=n_threads)[0]
