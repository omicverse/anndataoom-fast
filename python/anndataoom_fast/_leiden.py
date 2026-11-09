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
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _leiden_cpp as _cpp_leiden  # type: ignore[attr-defined]

__all__ = ["leiden", "leiden_multi", "edges_from_adjacency"]


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
