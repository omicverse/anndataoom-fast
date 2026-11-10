import h5py
import numpy as np
import pytest
import scipy.sparse as sp

import anndataoom_fast as aof
from anndataoom_fast import _core
from anndataoom_fast._h5 import open_csr


def _counts(n=3000, g=700, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.3, 2.0, size=g) * rng.lognormal(0, 0.6, size=(n, 1))
    X = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
    X.sort_indices()
    return X


LAYOUTS = {
    "gzip": dict(compression="gzip", chunks=(997,)),
    "gzip+shuffle": dict(compression="gzip", shuffle=True, chunks=(1500,)),
    "gzip+fletcher32": dict(compression="gzip", fletcher32=True, chunks=(777,)),
    "contiguous": dict(),
    "chunked-raw": dict(chunks=(4096,)),
}


def _write(path, X, data_dtype=np.float32, index_dtype=np.int64, **ds_kw):
    with h5py.File(path, "w") as f:
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.array(X.shape)
        g.create_dataset("data", data=X.data.astype(data_dtype), **ds_kw)
        g.create_dataset("indices", data=X.indices.astype(index_dtype), **ds_kw)
        g.create_dataset("indptr", data=X.indptr.astype(np.int64))


@pytest.fixture(params=sorted(LAYOUTS))
def h5(tmp_path, request):
    X = _counts()
    p = str(tmp_path / f"{request.param}.h5")
    _write(p, X, **LAYOUTS[request.param])
    return p, X


@pytest.mark.parametrize("dt", [np.float32, np.float64, np.int32, np.uint16])
def test_reader_dtypes(tmp_path, dt):
    X = _counts(500, 300)
    p = str(tmp_path / "d.h5")
    _write(p, X, data_dtype=dt, index_dtype=np.int32, compression="gzip", chunks=(333,))
    with h5py.File(p) as f:
        d = aof._h5.dataset(p, f["X/data"])
        assert np.array_equal(d.read(0, d.size), f["X/data"][:].astype(np.float64))
        d = aof._h5.dataset(p, f["X/indices"])
        assert np.array_equal(d.read(17, 4000), f["X/indices"][17:4000].astype(np.float64))


def test_reader_layouts(h5):
    p, X = h5
    with h5py.File(p) as f:
        d = aof._h5.dataset(p, f["X/data"])
        rng = np.random.default_rng(1)
        for _ in range(20):
            a = int(rng.integers(0, X.nnz))
            b = int(rng.integers(a, X.nnz + 1))
            assert np.array_equal(d.read(a, b), X.data[a:b].astype(np.float64))


@pytest.mark.parametrize("out_dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("n_threads", [1, 3, 8])
def test_read_into_matches_h5py(h5, out_dtype, n_threads):
    """read_into decodes straight into a caller buffer, converted to its
    dtype, on any number of threads (omicverse's Dask reader)."""
    p, X = h5
    rng = np.random.default_rng(2)
    with h5py.File(p) as f:
        for key, ref in (("X/data", X.data), ("X/indices", X.indices)):
            d = aof._h5.dataset(p, f[key])
            for a, b in [(0, X.nnz), (5, 6), (1234, 1234)] + [
                    tuple(sorted(rng.integers(0, X.nnz + 1, 2))) for _ in range(10)]:
                out = np.full(b - a, -1, dtype=out_dtype)
                d.read_into(int(a), int(b), out, n_threads)
                assert np.array_equal(out, ref[a:b].astype(out_dtype))


def test_read_into_rejects_bad_buffers(h5):
    p, X = h5
    with h5py.File(p) as f:
        d = aof._h5.dataset(p, f["X/data"])
    with pytest.raises(ValueError):
        d.read_into(0, 10, np.empty(9, np.float32))
    with pytest.raises((ValueError, TypeError)):
        d.read_into(0, 10, np.empty(10, np.uint8))
    with pytest.raises((IndexError, ValueError)):
        d.read_into(0, X.nnz + 1, np.empty(X.nnz + 1, np.float32))


def test_qc_pass_matches_numpy(h5):
    p, X = h5
    h = open_csr(p, target_block_elems=5000)
    assert h.src.n_blocks > 10
    gg = np.zeros(X.shape[1], np.uint8)
    gg[:20] |= 1
    gg[50:80] |= 2
    r = _core.qc_pass(h.src, gg, [])
    tot = np.asarray(X.sum(axis=1)).ravel()  # float32, like anndataoom
    assert np.array_equal(r["total"], tot.astype(np.float64))
    assert np.array_equal(r["nnz"], np.diff(X.indptr))
    den = tot.astype(np.float64).copy()
    den[den == 0] = 1
    mt = np.asarray(X[:, :20].sum(axis=1)).ravel().astype(np.float64)
    assert np.array_equal(r["mt"], mt / den)
    X64 = X.astype(np.float64)
    assert np.array_equal(r["g_stored"], np.bincount(X.indices, minlength=X.shape[1]))
    assert np.array_equal(r["g_sum"], np.asarray(X64.sum(0)).ravel())
    assert np.array_equal(r["g_sumsq"], np.asarray(X64.multiply(X64).sum(0)).ravel())


def test_qc_rules_and_row_mask(h5):
    p, X = h5
    h = open_csr(p, target_block_elems=5000)
    gg = np.zeros(X.shape[1], np.uint8)
    r = _core.qc_pass(h.src, gg, [(0, 2, 150.0), (1, 3, 60.0)])  # total > 150, nnz >= 60
    tot = np.asarray(X.sum(axis=1)).ravel()
    keep = (tot > 150) & (np.diff(X.indptr) >= 60)
    assert np.array_equal(r["keep"].astype(bool), keep)
    assert np.array_equal(r["g_stored"], np.bincount(X[keep].indices, minlength=X.shape[1]))
    rows = np.zeros(X.shape[0], np.uint8)
    rows[::3] = 1
    r2 = _core.qc_pass(h.src, gg, [], rows)
    assert np.array_equal(r2["g_stored"], np.bincount(X[::3].indices, minlength=X.shape[1]))


def test_thread_count_does_not_change_results(h5):
    p, X = h5
    h = open_csr(p, target_block_elems=3000)
    keep = np.ones(X.shape[0], np.uint8)
    cm = np.arange(X.shape[1], dtype=np.int32)
    ex = np.zeros(X.shape[1], np.uint8)
    sg = np.asarray(X.astype(np.float64).sum(0)).ravel()
    outs = []
    for nt in (1, 3, 8):
        aof.set_num_threads(nt)
        outs.append(_core.stats_pass(h.src, keep, cm, ex, sg, 1e4, True, True, True, 100.0,
                                     float(np.sqrt(X.shape[0])), float(sg.sum())))
    for o in outs[1:]:
        for k in o:
            assert np.array_equal(o[k], outs[0][k]), k


def test_pearson_residual_variance_matches_dense(h5):
    p, X = h5
    h = open_csr(p, target_block_elems=4000)
    n, g = X.shape
    D = X.toarray().astype(np.float64)
    sc = D.sum(1)
    sg = D.sum(0)
    tot = sg.sum()
    theta, clip = 100.0, np.sqrt(n)
    mu = np.outer(sc, sg) / tot
    den = np.sqrt(mu + mu ** 2 / theta)
    den[den == 0] = 1
    res = np.clip((D - mu) / den, -clip, clip)
    ref = (res ** 2).sum(0) / n - (res.sum(0) / n) ** 2

    st = _core.stats_pass(h.src, np.ones(n, np.uint8), np.arange(g, dtype=np.int32),
                          np.zeros(g, np.uint8), sg, 1.0, False, False, True, theta, clip, tot,
                          cells_override=sc)
    su, m = np.unique(sc, return_counts=True)
    z1, z2 = _core.pearson_zero_terms(su, m.astype(np.float64), sg, tot, theta, clip)
    s1, s2 = z1 + st["d1"], z2 + st["d2"]
    got = s2 / n - (s1 / n) ** 2
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_scaled_operator_matches_dense(h5):
    p, X = h5
    h = open_csr(p, target_block_elems=4000)
    n, g = X.shape
    sel = np.arange(0, g, 3)
    cps = np.asarray(X.sum(1)).ravel().astype(np.float64)
    nf = cps / 1e4
    nf[nf == 0] = 1
    recip = (1.0 / nf.astype(np.float32)).astype(np.float32)
    L = X[:, sel].multiply(recip[:, None]).tocsr().astype(np.float32)
    L.data = np.log1p(L.data.astype(np.float64)).astype(np.float32)  # see log1p_f32
    D = L.toarray().astype(np.float64)
    mean, std = D.mean(0), D.std(0, ddof=1)
    std[std == 0] = 1
    A = np.clip((D - mean) / std, -10, 10).astype(np.float32)
    cmap = np.full(g, -1, np.int32)
    cmap[sel] = np.arange(len(sel), dtype=np.int32)
    op = _core.extract_scaled(h.src, np.ones(n, np.uint8), cmap, mean, std, recip, True, 10.0)
    rng = np.random.default_rng(3)
    Q = rng.standard_normal((len(sel), 7))
    P = rng.standard_normal((n, 7))
    np.testing.assert_allclose(op.matmul(Q), A.astype(np.float64) @ Q, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(op.rmatmul(P), A.astype(np.float64).T @ P, rtol=1e-10, atol=1e-8)
    ip, ix, val, z = op.csr()
    dense = np.repeat(z[None, :], n, 0)
    rows = np.repeat(np.arange(n), np.diff(ip))
    dense[rows, ix] = val
    assert np.array_equal(dense, A)


def test_orthonormalize_and_svd_deterministic(h5):
    from anndataoom_fast._svd import randomized_svd
    p, X = h5
    h = open_csr(p, target_block_elems=4000)
    n, g = X.shape
    cmap = np.arange(g, dtype=np.int32)
    D = X.toarray().astype(np.float64)
    mean, std = D.mean(0), D.std(0, ddof=1)
    std[std == 0] = 1
    op = _core.extract_scaled(h.src, np.ones(n, np.uint8), cmap, mean, std,
                              np.ones(n, np.float32), False, 0.0)
    Y = np.random.default_rng(0).standard_normal((n, 12))
    Q = _core.orthonormalize(Y)
    np.testing.assert_allclose(Q.T @ Q, np.eye(12), atol=1e-12)
    np.testing.assert_allclose(Q @ (Q.T @ Y), Y, atol=1e-9)  # same column space
    ref = None
    for nt in (1, 4, 7):
        aof.set_num_threads(nt)
        U, S, Vt = randomized_svd(op, 10, n_oversamples=10, n_iter=6, random_state=0)
        if ref is None:
            ref = (U, S, Vt)
        else:
            assert all(np.array_equal(a, b) for a, b in zip((U, S, Vt), ref))
    # Same start and iterations as sklearn => same subspace, only rounding differs.
    from sklearn.utils.extmath import randomized_svd as sk_rsvd
    A = (D - mean) / std
    Us, Ss, Vts = sk_rsvd(A, 10, n_oversamples=10, n_iter=6, random_state=0, flip_sign=False)
    np.testing.assert_allclose(S, Ss, rtol=1e-7)
    cos = np.abs((U * Us).sum(0)) / (np.linalg.norm(U, axis=0) * np.linalg.norm(Us, axis=0))
    assert cos.min() > 1 - 1e-6
