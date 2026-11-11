"""In-memory CSR helpers used by omicverse's storage='memory' path."""
import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from anndataoom_fast import memory


def _csr(n=700, g=300, density=0.1, dtype=np.float32, seed=0, wide_index=False):
    rng = np.random.default_rng(seed)
    X = sp.random(n, g, density=density, format="csr", dtype=dtype, random_state=rng)
    X.data = np.ceil(X.data * 20).astype(dtype)
    if wide_index:
        X.indices = X.indices.astype(np.int64)
        X.indptr = X.indptr.astype(np.int64)
    return X


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("wide_index", [False, True])
def test_subset_csr_matches_scipy(dtype, wide_index):
    X = _csr(dtype=dtype, wide_index=wide_index)
    rng = np.random.default_rng(1)
    rows = rng.random(X.shape[0]) < 0.6
    cols = rng.random(X.shape[1]) < 0.5
    cases = [(rows, None), (None, cols), (rows, cols), (np.flatnonzero(rows)[::-1].copy(), cols),
             (None, None), (np.zeros(X.shape[0], bool), cols)]
    for r, c in cases:
        ref = X if r is None else X[r]
        ref = ref if c is None else ref[:, c]
        got = memory.subset_csr(X, r, c)
        assert got.shape == ref.shape
        assert got.indices.dtype == got.indptr.dtype
        np.testing.assert_array_equal(got.toarray(), ref.toarray())
    got = memory.subset_csr(X, None, cols)
    assert got.has_sorted_indices  # stored order is kept


def test_scale_rows_bit_identical_to_numpy():
    X = _csr(dtype=np.float32)
    f = np.random.default_rng(2).uniform(0.5, 3.0, X.shape[0])
    ref = X.data / np.repeat(f, np.diff(X.indptr))
    np.testing.assert_array_equal(memory.scale_rows(X, f), ref)
    ref = X.data * np.repeat(f, np.diff(X.indptr))
    np.testing.assert_array_equal(memory.scale_rows(X, f, divide=False), ref)


def test_nonnegative_integers():
    d = np.arange(1000, dtype=np.float32)
    assert memory.nonnegative_integers(d)
    d2 = d.copy(); d2[777] = 1.5
    assert not memory.nonnegative_integers(d2)
    d3 = d.copy(); d3[3] = -1
    assert not memory.nonnegative_integers(d3)


def test_scale_columns_equal_full_scale():
    X = _csr(dtype=np.float32)
    full, mean, std = memory.scale(X, max_value=10)
    cols = np.random.default_rng(3).random(X.shape[1]) < 0.3
    part, _, _ = memory.scale(X, max_value=10, mean=mean, std=std, cols=cols)
    np.testing.assert_array_equal(part, full[:, cols])


@pytest.mark.parametrize("compression", [None, "gzip"])
def test_read_csr_matches_h5py(tmp_path, compression):
    X = _csr(dtype=np.float32)
    p = str(tmp_path / "m.h5")
    with h5py.File(p, "w") as f:
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.array(X.shape)
        kw = dict(compression=compression, chunks=(997,)) if compression else {}
        g.create_dataset("data", data=X.data, **kw)
        g.create_dataset("indices", data=X.indices.astype(np.int64), **kw)
        g.create_dataset("indptr", data=X.indptr.astype(np.int64))
    got = memory.read_csr(p)
    assert got.shape == X.shape and got.indices.dtype == np.int32
    np.testing.assert_array_equal(got.toarray(), X.toarray())


def test_empty_is_a_plain_array():
    a = memory.empty(10, np.float64)
    assert a.shape == (10,) and a.dtype == np.float64
