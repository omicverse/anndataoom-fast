"""Exactness tests: anndataoom_fast.leiden_brisc vs brisc's Leiden.

brisc is not a test dependency, so its labels for a set of deterministic
graphs (tests/_brisc_graphs.py) are stored in tests/data/brisc_leiden_fixtures.npz
(regenerate with tests/data/make_brisc_fixtures.py in an environment that has
brisc).  Labels must be identical element for element.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from anndataoom_fast import leiden_brisc, leiden_brisc_multi

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_brisc_graphs", HERE / "_brisc_graphs.py")
bg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bg)

FIXTURES = np.load(HERE / "data" / "brisc_leiden_fixtures.npz")
_GRAPHS = {}


def graph(gi):
    if gi not in _GRAPHS:
        G, nbr = bg.make_graph(*bg.GRAPHS[gi])
        assert bg.graph_digest(G, nbr) == str(FIXTURES[f"digest_{gi}"]), \
            "test graph generator no longer reproduces the fixture graphs"
        _GRAPHS[gi] = (G, nbr)
    return _GRAPHS[gi]


def modularity(G, labels, resolution=1.0):
    two_m = G.sum()
    k = np.asarray(G.sum(axis=1)).ravel()
    coo = G.tocoo()
    same = labels[coo.row] == labels[coo.col]
    e_in = coo.data[same].astype(np.float64).sum()
    deg = np.bincount(labels, weights=k)
    return e_in / two_m - resolution * (deg ** 2).sum() / two_m ** 2


@pytest.mark.parametrize("gi", range(len(bg.GRAPHS)))
@pytest.mark.parametrize("m", bg.MIN_CLUSTER_SIZES)
def test_matches_brisc(gi, m):
    G, nbr = graph(gi)
    kw = {"neighbors": nbr} if m > 1 else {}
    multi = leiden_brisc_multi(G, bg.RESOLUTIONS, min_cluster_size=m, **kw)
    for ri, r in enumerate(bg.RESOLUTIONS):
        ref = FIXTURES[f"labels_{gi}_{ri}_{m}"].astype(np.int64)
        single = leiden_brisc(G, r, min_cluster_size=m, **kw)
        assert single.dtype == np.int64
        np.testing.assert_array_equal(single, ref)
        np.testing.assert_array_equal(multi[ri], ref)


def test_thread_count_independent():
    G, nbr = graph(1)
    res = [0.25, 0.5, 1.0, 1.5, 2.0]
    base = leiden_brisc_multi(G, res, n_threads=1, min_cluster_size=5, neighbors=nbr)
    for nt in (2, 3, 5, 0):
        out = leiden_brisc_multi(G, res, n_threads=nt, min_cluster_size=5, neighbors=nbr)
        for a, b in zip(base, out):
            np.testing.assert_array_equal(a, b)
    for r, a in zip(res, base):
        np.testing.assert_array_equal(
            leiden_brisc(G, r, min_cluster_size=5, neighbors=nbr), a)


def test_modularity_sanity():
    G, _ = graph(2)
    lab = leiden_brisc(G, 1.0)
    # labels are 0..K-1, all used
    assert lab.min() == 0 and np.bincount(lab).min() > 0
    q = modularity(G, lab)
    assert q > 0.5
    # better than the planted partition's random relabelling, and a local
    # optimum: no worse than merging everything or splitting everything
    assert q > modularity(G, np.zeros_like(lab))
    assert q > modularity(G, np.arange(len(lab)))
    rng = np.random.default_rng(0)
    assert q > modularity(G, rng.permutation(lab))
    # higher resolution never gives fewer clusters here
    n1 = len(np.unique(leiden_brisc(G, 0.25)))
    n2 = len(np.unique(lab))
    assert n1 <= n2


def test_input_handling():
    G, _ = graph(0)
    ref = leiden_brisc(G, 1.0)
    # float64, int64 indices, CSC, explicit zeros and self loops are normalised
    H = G.astype(np.float64)
    H.indices = H.indices.astype(np.int64)
    H.indptr = H.indptr.astype(np.int64)
    np.testing.assert_array_equal(leiden_brisc(H, 1.0), ref)
    np.testing.assert_array_equal(leiden_brisc(G.tocsc(), 1.0), ref)
    D = (G + sp.identity(G.shape[0], dtype=np.float32, format="csr")).tocsr()
    D.sort_indices()
    np.testing.assert_array_equal(leiden_brisc(D, 1.0), ref)
    Z = G.copy()
    Z.data[::7] = 0  # explicit zeros: same as removing those edges
    Zc = Z.copy()
    Zc.eliminate_zeros()
    np.testing.assert_array_equal(leiden_brisc(Z, 1.0), leiden_brisc(Zc, 1.0))
    # seeds change the result deterministically
    a = leiden_brisc(G, 1.0, random_state=7)
    np.testing.assert_array_equal(a, leiden_brisc(G, 1.0, random_state=7))
    # min_cluster_size without an explicit kNN uses graph neighbours, which
    # cannot rescue the disconnected 2- and 3-node components from size 5
    lab = leiden_brisc(G, 1.0, min_cluster_size=2)
    assert np.bincount(lab).min() >= 2
    with pytest.raises(ValueError, match="cannot be assigned a cluster label"):
        leiden_brisc(G, 1.0, min_cluster_size=5)


def test_errors():
    G, nbr = graph(0)
    with pytest.raises(ValueError, match="empty graph"):
        leiden_brisc(sp.csr_matrix((10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="must be > 0"):
        leiden_brisc(G, 0.0)
    with pytest.raises(ValueError, match="negative"):
        leiden_brisc(-G)
    bad = nbr.copy()
    bad[:, 0] = np.arange(G.shape[0])
    with pytest.raises(ValueError, match="own neighbor"):
        leiden_brisc(G, 1.0, min_cluster_size=5, neighbors=bad)
    # disjoint edges: every node ends up alone
    n = 6
    P = sp.csr_matrix((np.ones(n, np.float32), ([0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4])),
                      shape=(n, n))
    with pytest.raises(ValueError, match="cluster by itself"):
        leiden_brisc(P, 100.0)
