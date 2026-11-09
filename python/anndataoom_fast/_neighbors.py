"""k-nearest neighbours on an embedding (e.g. PCs), scanpy conventions."""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

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


class IVFKNNTransformer:
    """sklearn-style transformer returning the kNN distance graph.

    Output format matches ``pynndescent.PyNNDescentTransformer``: a CSR matrix
    with ``n_neighbors + 1`` stored entries per row (the cell itself at
    distance 0 first, then its neighbours by increasing distance).
    """

    def __init__(self, n_neighbors=15, metric="euclidean", random_state=0, n_jobs=None, **kwargs):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, "metric": self.metric,
                "random_state": self.random_state, "n_jobs": self.n_jobs}

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)
        rs = self.random_state if isinstance(self.random_state, (int, np.integer)) else 0
        k = min(self.n_neighbors + 1, X.shape[0])
        idx, dist = knn(X, k, random_state=int(rs))
        n = X.shape[0]
        D = sp.csr_matrix((dist.ravel(), idx.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
        D.has_sorted_indices = False
        return D
