"""k-nearest neighbours on an embedding (e.g. PCs), scanpy conventions."""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from . import _core


def default_n_clusters(n: int) -> int:
    """Inverted-file cluster count used by brisc: ceil(min(4 sqrt(n), n / 100))."""
    return max(1, math.ceil(min(4 * math.sqrt(n), n / 100)))


# One cached kNN result per process: ov.pp.neighbors searches a few extra
# neighbours, and a later embedding (LocalMAP / PaCMAP) on the same
# representation reuses them instead of searching again.  Searching k
# neighbours and keeping the first j gives exactly the j-neighbour result.
_KNN_CACHE: dict = {}


def _digest(X: np.ndarray) -> bytes:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(X.shape, dtype=np.int64).tobytes())
    h.update(memoryview(X).cast("B"))
    return h.digest()


def cached_knn(X, n_neighbors: int, **search):
    """The cached result of :func:`knn` on ``X`` if it has ``>= n_neighbors``
    columns and the same search settings, else ``None``."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    c = _KNN_CACHE.get("last")
    if c is None or c["settings"] != _settings(**search) or c["idx"].shape[1] < n_neighbors:
        return None
    if c["shape"] != X.shape or c["digest"] != _digest(X):
        return None
    return c["idx"][:, :n_neighbors].copy(), c["dist"][:, :n_neighbors].copy()


def _settings(n_clusters=None, n_probe=64, kmeans_iters=4, exact=False, random_state=0):
    return (n_clusters, int(n_probe), int(kmeans_iters), bool(exact), int(random_state))


def knn(X, n_neighbors: int = 15, *, n_clusters: int | None = None, n_probe: int = 64,
        kmeans_iters: int = 4, exact: bool = False, random_state: int = 0,
        cache_neighbors: int | None = None):
    """Nearest neighbours of every row of ``X`` (Euclidean).

    Returns ``(indices, distances)`` of shape ``(n, n_neighbors)`` in scanpy's
    convention: column 0 is the cell itself at distance 0, then the
    ``n_neighbors - 1`` nearest other cells, nearest first.

    The search is an inverted-file index (k-means into ``n_clusters`` cells,
    each query scans its ``n_probe`` nearest clusters); ``exact=True`` scans
    every cluster and returns exact brute-force neighbours.  The output does
    not depend on the number of threads.

    ``cache_neighbors``: search ``max(n_neighbors, cache_neighbors)``
    neighbours, return the first ``n_neighbors`` (identical to searching
    ``n_neighbors``) and keep the full result for :func:`cached_knn`.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    settings = dict(n_clusters=n_clusters, n_probe=n_probe, kmeans_iters=kmeans_iters,
                    exact=exact, random_state=random_state)
    hit = cached_knn(X, n_neighbors, **settings)
    if hit is not None:
        return hit
    if cache_neighbors is not None and min(int(cache_neighbors), n) > n_neighbors:
        full_i, full_d = knn(X, min(int(cache_neighbors), n), **settings)
        _KNN_CACHE["last"] = dict(settings=_settings(**settings), shape=X.shape,
                                  digest=_digest(X), idx=full_i, dist=full_d)
        return full_i[:, :n_neighbors].copy(), full_d[:, :n_neighbors].copy()
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
        # A few extra neighbours are cached for LocalMAP / PaCMAP (20 needed).
        idx, dist = knn(X, k, random_state=int(rs), cache_neighbors=21)
        n = X.shape[0]
        D = sp.csr_matrix((dist.ravel(), idx.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
        D.has_sorted_indices = False
        return D
