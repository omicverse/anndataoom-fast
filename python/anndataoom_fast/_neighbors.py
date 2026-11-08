"""k-nearest neighbours on an embedding (e.g. PCs), scanpy conventions."""
from __future__ import annotations

import math

import numpy as np

from . import _core


def default_n_clusters(n: int) -> int:
    """Inverted-file cluster count used by brisc: ceil(min(4 sqrt(n), n / 100))."""
    return max(1, math.ceil(min(4 * math.sqrt(n), n / 100)))


def knn(X, n_neighbors: int = 15, *, n_clusters: int | None = None, n_probe: int = 64,
        kmeans_iters: int = 4, exact: bool = False, random_state: int = 0):
    """Nearest neighbours of every row of ``X`` (Euclidean).

    Returns ``(indices, distances)`` of shape ``(n, n_neighbors)`` in scanpy's
    convention: column 0 is the cell itself at distance 0, then the
    ``n_neighbors - 1`` nearest other cells, nearest first.

    The search is an inverted-file index (k-means into ``n_clusters`` cells,
    each query scans its ``n_probe`` nearest clusters); ``exact=True`` scans
    every cluster and returns exact brute-force neighbours.  The output does
    not depend on the number of threads.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")
    if n_neighbors > n:
        raise ValueError(f"n_neighbors={n_neighbors} exceeds the number of cells ({n})")
    k = n_neighbors - 1
    if k == 0:
        return np.arange(n, dtype=np.int64)[:, None], np.zeros((n, 1), np.float32)
    nc = min(n, default_n_clusters(n) if n_clusters is None else int(n_clusters))
    init = np.sort(np.random.RandomState(random_state).choice(n, nc, replace=False)).astype(np.int64)
    probe = nc if exact else int(n_probe)
    idx, dist, _, _ = _core.knn_ivf(X, k, init, probe, int(kmeans_iters))
    indices = np.empty((n, n_neighbors), dtype=np.int64)
    distances = np.empty((n, n_neighbors), dtype=np.float32)
    indices[:, 0] = np.arange(n)
    distances[:, 0] = 0.0
    indices[:, 1:] = idx
    distances[:, 1:] = dist
    return indices, distances


def fuzzy_simplicial_set(knn_indices, knn_dists, n_neighbors: int | None = None, *,
                         set_op_mix_ratio: float = 1.0, local_connectivity: float = 1.0,
                         return_sigmas_rhos: bool = False):
    """UMAP connectivities from a kNN graph, as ``umap.umap_.fuzzy_simplicial_set``.

    Reproduces umap-learn 0.5's float32 arithmetic (smooth-kNN binary search,
    membership strengths, fuzzy union), so the result matches what scanpy /
    omicverse compute with umap-learn on the same neighbours.  Returns a
    float32 ``scipy.sparse.csr_matrix`` of shape ``(n, n)``.
    """
    import scipy.sparse as sp
    idx = np.ascontiguousarray(knn_indices, dtype=np.int64)
    dist = np.ascontiguousarray(knn_dists, dtype=np.float32)
    n, k = idx.shape
    nn = float(k if n_neighbors is None else n_neighbors)
    indptr, indices, data, sigmas, rhos = _core.fuzzy_simplicial_set(
        idx, dist, nn, float(local_connectivity), float(set_op_mix_ratio), 1.0)
    G = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    G.has_sorted_indices = True
    if return_sigmas_rhos:
        return G, sigmas, rhos
    return G
