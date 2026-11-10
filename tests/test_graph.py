import numpy as np
import pytest
import scipy.sparse as sp

import anndataoom_fast as aof
from anndataoom_fast import _core
from anndataoom_fast._markers import cosg, group_stats
from anndataoom_fast._neighbors import fuzzy_simplicial_set, knn


def _blobs(n=4000, d=20, k=12, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(k, d))
    lab = rng.integers(0, k, n)
    return (centers[lab] + rng.normal(size=(n, d))).astype(np.float32), lab


def test_knn_exact_matches_bruteforce():
    X, _ = _blobs()
    idx, dist = knn(X, 15, exact=True)
    D = ((X[:, None, :].astype(np.float64) - X[None, :, :]) ** 2).sum(-1)
    ref = np.argsort(D, axis=1, kind="stable")[:, :15]
    assert np.array_equal(idx[:, 0], np.arange(len(X)))
    same = np.mean([set(a) == set(b) for a, b in zip(idx, ref)])
    assert same == 1.0
    np.testing.assert_allclose(dist[:, 1:], np.sqrt(np.take_along_axis(D, idx[:, 1:], 1)), rtol=1e-5)


def test_knn_approx_recall_and_threads():
    X, _ = _blobs(6000)
    ie, _ = knn(X, 15, exact=True)
    outs = []
    for nt in (1, 5):
        aof.set_num_threads(nt)
        outs.append(knn(X, 15))
    assert all(np.array_equal(a, b) for a, b in zip(outs[0], outs[1]))
    recall = np.mean([len(set(a[1:]) & set(b[1:])) / 14 for a, b in zip(outs[0][0], ie)])
    assert recall > 0.99


def test_fuzzy_simplicial_set_matches_umap_learn():
    umap_ = pytest.importorskip("umap.umap_")
    X, _ = _blobs(3000)
    idx, dist = knn(X, 15)
    U, us, ur = umap_.fuzzy_simplicial_set(sp.coo_matrix((len(X), 1)), 15, None, None,
                                           knn_indices=idx, knn_dists=dist)
    U = U.tocsr()
    U.sort_indices()
    G, s, r = fuzzy_simplicial_set(idx, dist, 15, return_sigmas_rhos=True)
    assert np.array_equal(s, us) and np.array_equal(r, ur)
    assert np.array_equal(G.indptr, U.indptr) and np.array_equal(G.indices, U.indices)
    assert np.array_equal(G.data, U.data)


def test_umap_first_epoch_matches_umap_learn_and_is_deterministic():
    umap_ = pytest.importorskip("umap.umap_")
    from umap.layouts import optimize_layout_euclidean
    X, _ = _blobs(2000)
    idx, dist = knn(X, 15)
    G = fuzzy_simplicial_set(idx, dist, 15).tocoo()
    a, b = umap_.find_ab_params(1.0, 0.5)
    rs = np.random.RandomState(0)
    init = rs.uniform(0, 10, size=(G.shape[0], 2)).astype(np.float32)
    eps = umap_.make_epochs_per_sample(G.data, 20)
    rng = rs.randint(-2 ** 31 + 1, 2 ** 31 - 2, 3).astype(np.int64)
    e1 = optimize_layout_euclidean(init.copy(), init.copy(), G.row, G.col, 1, G.shape[1], eps, a, b,
                                   rng.copy(), 1.0, 1.0, 5, parallel=False, move_other=True)
    e2 = init.copy()
    _core.umap_optimize_euclidean(e2, G.row.astype(np.int32), G.col.astype(np.int32), eps,
                                  rng.copy(), a, b, 1.0, 1.0, 5.0, 1, True, False)
    assert np.array_equal(e1, e2)  # no negative samples in epoch 0: no fastmath effects yet
    aof.set_num_threads(1)
    y1 = aof.umap(G.tocsr(), n_epochs=30)
    aof.set_num_threads(4)
    y4 = aof.umap(G.tocsr(), n_epochs=30)
    assert np.array_equal(y1, y4)


def test_group_stats_and_cosg_against_dense():
    rng = np.random.default_rng(1)
    X = sp.random(3000, 400, density=0.05, random_state=2, format="csr", dtype=np.float32)
    X.data = np.log1p(X.data * 10).astype(np.float32)
    lab = rng.integers(0, 7, 3000).astype(str)
    groups, sizes, S, Q, Z = group_stats(X, lab)
    D = X.toarray().astype(np.float64)
    for ci, g in enumerate(groups):
        m = lab == g
        np.testing.assert_allclose(S[ci], D[m].sum(0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(Q[ci], (D[m] ** 2).sum(0), rtol=1e-12, atol=1e-12)
        assert np.array_equal(Z[ci], (X[m] != 0).sum(0).A1)
    from sklearn.metrics.pairwise import cosine_similarity
    onehot = np.stack([(lab == g).astype(float) for g in groups])
    cos = cosine_similarity(D.T, onehot)
    e2 = cos ** 2
    lam = e2 / e2.sum(1, keepdims=True) * cos
    out = cosg(X, lab, np.array([f"g{i}" for i in range(400)]), n_genes=20)
    for ci, g in enumerate(groups):
        np.testing.assert_allclose(out["scores"][g], np.sort(lam[:, ci])[::-1][:20].astype(np.float32),
                                   rtol=1e-6)


def test_umap_parallel_sgd_deterministic_and_close_to_sequential():
    from sklearn.neighbors import NearestNeighbors
    X, lab = _blobs(4000)
    idx, dist = knn(X, 15)
    G = fuzzy_simplicial_set(idx, dist, 15).tocsr()
    runs = []
    for nt in (1, 3, 8):
        aof.set_num_threads(nt)
        runs.append(aof.umap(G, n_epochs=100, init="random", sgd="parallel"))
    assert all(np.array_equal(runs[0], r) for r in runs[1:])
    assert not np.array_equal(runs[0], aof.umap(G, n_epochs=100, init="random", sgd="parallel",
                                                random_state=1))
    seq = aof.umap(G, n_epochs=100, init="random", sgd="sequential")

    def preservation(E):
        _, I = NearestNeighbors(n_neighbors=16).fit(E).kneighbors(E)
        return np.mean([len(set(a[1:]) & set(b[1:])) / 15 for a, b in zip(I, idx)])

    assert preservation(runs[0]) > 0.95 * preservation(seq)


def test_knn_cache_reused_and_exact():
    from anndataoom_fast import _neighbors
    X, _ = _blobs(3000, seed=3)
    _neighbors._KNN_CACHE.clear()
    ref16 = knn(X, 16)
    ref21 = knn(X, 21)
    _neighbors._KNN_CACHE.clear()
    got16 = knn(X, 16, cache_neighbors=21)          # searches 21, returns 16
    assert all(np.array_equal(a, b) for a, b in zip(got16, ref16))
    hit = _neighbors.cached_knn(X, 21)
    assert hit is not None and all(np.array_equal(a, b) for a, b in zip(hit, ref21))
    assert _neighbors.cached_knn(X + 1, 21) is None   # different data: no hit
    assert _neighbors.cached_knn(X, 22) is None       # not enough neighbours
    # LocalMAP takes its 20 neighbours from the cache
    a = aof.localmap(X, num_iterations=20)
    _neighbors._KNN_CACHE.clear()
    sq = np.square(ref21[1][:, 1:])
    b = aof.localmap(X, knn_indices=ref21[0][:, 1:], knn_dists=sq, num_iterations=20)
    assert np.array_equal(a, b)


def test_group_stats_any_accepts_device_arrays():
    """omicverse's GPU mode keeps .X on the GPU; COSG copies it to the host."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    import cupyx.scipy.sparse as csp
    from anndataoom_fast import _omicverse
    rng = np.random.default_rng(3)
    X = sp.random(500, 80, density=0.1, random_state=4, format="csr", dtype=np.float32)
    lab = rng.integers(0, 4, 500).astype(str)
    ref = _omicverse._group_stats_any(X, lab)
    _omicverse._STATS_CACHE.clear()
    got = _omicverse._group_stats_any(csp.csr_matrix(X), lab)
    for a, b in zip(ref[1:], got[1:]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _dask_blocks(X, rows, to_device=lambda b: b):
    import dask
    import dask.array as da
    parts = []
    for r0 in range(0, X.shape[0], rows):
        b = X[r0:r0 + rows]
        parts.append(da.from_delayed(dask.delayed(to_device)(b), shape=b.shape, dtype=X.dtype,
                                     meta=to_device(X[:0])))
    return da.concatenate(parts, axis=0)


@pytest.mark.parametrize("dense", [False, True])
def test_group_stats_any_streams_dask_row_blocks(dense):
    """omicverse's chunked mode (ov.read(backend='dask')): one pass over blocks."""
    pytest.importorskip("dask")
    from anndataoom_fast import _omicverse
    rng = np.random.default_rng(5)
    X = sp.random(700, 90, density=0.1, random_state=6, format="csr", dtype=np.float32)
    lab = rng.integers(0, 5, 700).astype(str)
    ref = _omicverse._group_stats_any(X, lab)
    _omicverse._STATS_CACHE.clear()
    D = _dask_blocks(X.toarray() if dense else X, 160)
    got = _omicverse._group_stats_any(D, lab)
    assert list(ref[0]) == list(got[0])
    for a, b in zip(ref[1:], got[1:]):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12)


def test_group_stats_any_streams_dask_device_blocks():
    pytest.importorskip("dask")
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    import cupyx.scipy.sparse as csp
    from anndataoom_fast import _omicverse
    rng = np.random.default_rng(7)
    X = sp.random(700, 90, density=0.1, random_state=8, format="csr", dtype=np.float32)
    lab = rng.integers(0, 5, 700).astype(str)
    ref = _omicverse._group_stats_any(X, lab)
    _omicverse._STATS_CACHE.clear()
    got = _omicverse._group_stats_any(_dask_blocks(X, 160, csp.csr_matrix), lab)
    for a, b in zip(ref[1:], got[1:]):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12)
