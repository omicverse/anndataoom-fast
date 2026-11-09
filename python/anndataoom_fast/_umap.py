"""UMAP embedding of a fuzzy graph, following umap-learn's simplicial_set_embedding.

``sgd='parallel'`` (default) runs the layout optimisation on all threads and
is deterministic: each epoch updates every vertex from a snapshot of the
previous epoch, one vertex per thread, with negative samples drawn from a
hash of (seed, edge, epoch), so the embedding is the same for any number of
threads.  ``sgd='sequential'`` is a C++ port of umap-learn's single-threaded
optimize_layout_euclidean (same edge schedule, same Tausworthe RNG streams,
same float32/float64 arithmetic; umap compiles it with numba fastmath, so the
two agree to rounding); ``sgd='hogwild'`` is its lock-free parallel version
(like umap-learn's parallel=True, not reproducible).  The spectral
initialisation solves for the same eigenvectors (the 2nd and 3rd smallest of
the normalised Laplacian) as the largest eigenvectors of D^-1/2 W D^-1/2,
which Lanczos resolves far faster, with a thread-count independent matvec and
single-threaded BLAS.  ``spectral='exact'`` calls umap-learn's own
spectral_layout instead.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import _core

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def find_ab_params(spread: float = 1.0, min_dist: float = 0.5):
    """umap-learn's find_ab_params (same curve fit), without importing umap."""
    from scipy.optimize import curve_fit

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


def _spectral_fast(graph: sp.coo_matrix, dim: int, tol: float = 1e-4):
    import scipy.sparse.linalg as sla
    from threadpoolctl import threadpool_limits
    n = graph.shape[0]
    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    D = sp.spdiags(1.0 / sqrt_deg, 0, n, n)
    M = (D * graph * D).tocsr()
    M.sort_indices()
    ip = M.indptr.astype(np.int64)
    ix = M.indices.astype(np.int32)
    dat = M.data.astype(np.float64)

    def mv(x):
        return _core.csr_matvec(ip, ix, dat, np.ascontiguousarray(x, dtype=np.float64).ravel())

    op = sla.LinearOperator(M.shape, matvec=mv, dtype=np.float64)
    k = dim + 1
    # Single-threaded BLAS for ARPACK's dense steps (thread-count independent);
    # the matvec itself runs on our OpenMP threads, so limit only BLAS.
    with threadpool_limits(1, user_api="blas"):
        w, v = sla.eigsh(op, k, which="LA", ncv=max(2 * k + 1, 40), tol=tol, v0=sqrt_deg,
                         maxiter=n * 5)
    order = np.argsort(w)[::-1][1:k]  # skip the trivial eigenvalue 1
    return v[:, order]


def _multi_component_layout(data, graph, n_comp, labels, dim, random_state):
    """umap.spectral.multi_component_layout with the fast per-component solver.

    Same meta-embedding and placement rules and the same random_state draws;
    without ``data`` the component positions come from ``random_state``
    (umap-learn falls back to the unseeded global numpy RNG there).
    """
    from sklearn.manifold import SpectralEmbedding
    from sklearn.metrics import pairwise_distances
    n = graph.shape[0]
    result = np.empty((n, dim), dtype=np.float32)
    if n_comp > 2 * dim:
        if data is None:
            meta = random_state.random_sample(size=(n_comp, dim)) * 10.0
        else:
            data = np.asarray(data)
            cent = np.empty((n_comp, data.shape[1]), dtype=np.float64)
            for lab in range(n_comp):
                cent[lab] = data[labels == lab].mean(axis=0)
            dm = pairwise_distances(cent)
            aff = np.exp(-(dm ** 2))
            meta = SpectralEmbedding(n_components=dim, affinity="precomputed",
                                     random_state=random_state).fit_transform(aff)
            meta /= meta.max()
    else:
        k = int(np.ceil(n_comp / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta = np.vstack([base, -base])[:n_comp]
    G = graph.tocsr()
    for lab in range(n_comp):
        sel = labels == lab
        cg = G[sel, :].tocsc()[:, sel].tocoo()
        d = pairwise_distances([meta[lab]], meta)
        data_range = d[d > 0.0].min() / 2.0
        if cg.shape[0] < 2 * dim or cg.shape[0] <= dim + 1:
            result[sel] = random_state.uniform(low=-data_range, high=data_range,
                                               size=(cg.shape[0], dim)) + meta[lab]
        else:
            random_state.normal(size=(cg.shape[0], dim + 1))  # umap's lobpcg start draw
            ce = _spectral_fast(cg, dim)
            ce *= data_range / np.max(np.abs(ce))
            result[sel] = ce + meta[lab]
    return result


def spectral_init(graph, dim: int = 2, random_state=None, *, data=None, method: str = "fast"):
    """Spectral layout of a (pruned) fuzzy graph, as umap-learn's spectral_layout."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)
    graph = graph.tocoo()
    if method == "exact":
        from threadpoolctl import threadpool_limits
        from umap.spectral import spectral_layout
        with threadpool_limits(1):
            return spectral_layout(data, graph, dim, random_state, metric="euclidean", metric_kwds={})
    n_comp, labels = connected_components(graph)
    if n_comp > 1:
        return _multi_component_layout(data, graph, n_comp, labels, dim, random_state)
    # umap draws an (n, dim+1) normal block for its lobpcg start even on the
    # eigsh path; consume it so the RNG stream stays aligned with umap-learn.
    random_state.normal(size=(graph.shape[0], dim + 1))
    return _spectral_fast(graph, dim)


def simplicial_set_embedding(graph, n_components: int = 2, *, a: float, b: float,
                             n_epochs: int | None = None, init="spectral", random_state=0,
                             gamma: float = 1.0, initial_alpha: float = 1.0,
                             negative_sample_rate: int = 5, spectral: str = "fast",
                             parallel: bool = False, data=None, sgd: str | None = None):
    """Drop-in for umap.umap_.simplicial_set_embedding (euclidean output, no densMAP).

    ``sgd``: 'parallel' (deterministic, multithreaded; default), 'sequential'
    (umap-learn's algorithm) or 'hogwild'; ``parallel=True`` means 'hogwild'.

    Returns the float32 embedding of shape (n, n_components).
    """
    from sklearn.utils import check_random_state
    rs = check_random_state(random_state)
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]
    default_epochs = 500 if graph.shape[0] <= 10000 else 200
    if n_epochs is None:
        n_epochs = default_epochs
    if n_epochs > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0
    graph.eliminate_zeros()

    if isinstance(init, str) and init == "spectral":
        emb = spectral_init(graph, n_components, rs, data=data, method=spectral)
        expansion = 10.0 / np.abs(emb).max()
        emb = (emb * expansion).astype(np.float32)
        emb = emb + rs.normal(scale=0.0001, size=emb.shape).astype(np.float32)
    elif isinstance(init, str) and init == "random":
        emb = rs.uniform(low=-10.0, high=10.0, size=(graph.shape[0], n_components)).astype(np.float32)
    else:
        emb = np.asarray(init, dtype=np.float32)

    weights = graph.data
    n_samples = n_epochs * (weights / weights.max())
    eps = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    eps[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    rng_state = rs.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    emb = (10.0 * (emb - np.min(emb, 0)) / (np.max(emb, 0) - np.min(emb, 0))).astype(np.float32, order="C")
    if sgd is None:
        sgd = "hogwild" if parallel else "parallel"
    if sgd == "parallel":
        g = graph.tocsr()  # rows of a summed COO are sorted; keep the same edge order
        assert np.array_equal(g.indices, graph.col.astype(g.indices.dtype))
        M = (1 << 64) - 1
        r0, r1, r2 = (int(x) & M for x in rng_state)
        seed = ((r0 * 0x9E3779B97F4A7C15) & M) ^ r1 ^ ((r2 << 32) & M)
        _core.umap_optimize_euclidean_det(emb, g.indptr.astype(np.int64), g.indices.astype(np.int32),
                                          eps, seed, float(a), float(b), float(gamma),
                                          float(initial_alpha), float(negative_sample_rate),
                                          int(n_epochs), True)
        return emb
    if sgd not in ("sequential", "hogwild"):
        raise ValueError(f"sgd must be 'parallel', 'sequential' or 'hogwild', not {sgd!r}")
    parallel = sgd == "hogwild"
    _core.umap_optimize_euclidean(emb, graph.row.astype(np.int32), graph.col.astype(np.int32),
                                  eps, rng_state, float(a), float(b), float(gamma),
                                  float(initial_alpha), float(negative_sample_rate), int(n_epochs),
                                  True, bool(parallel))
    return emb


def umap(connectivities, n_components: int = 2, *, min_dist: float = 0.5, spread: float = 1.0,
         n_epochs: int | None = None, random_state=0, a=None, b=None, gamma: float = 1.0,
         alpha: float = 1.0, negative_sample_rate: int = 5, init="spectral", spectral: str = "fast",
         parallel: bool = False, sgd: str | None = None):
    """UMAP embedding from a connectivities matrix (scanpy / omicverse parameters)."""
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    return simplicial_set_embedding(connectivities, n_components, a=a, b=b, n_epochs=n_epochs,
                                    init=init, random_state=random_state, gamma=gamma,
                                    initial_alpha=alpha, negative_sample_rate=negative_sample_rate,
                                    spectral=spectral, parallel=parallel, sgd=sgd)
