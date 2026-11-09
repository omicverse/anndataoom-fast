"""Exactness tests: anndataoom_fast._leiden vs leidenalg (omicverse's CPU path).

The reference is built exactly like omicverse ``ov.pp.leiden(flavor='leidenalg')``:
``get_igraph_from_adjacency(adjacency, directed=True)`` and
``leidenalg.find_partition(g, RBConfigurationVertexPartition,
weights=np.array(g.es['weight']).astype(np.float64), resolution_parameter=r,
n_iterations=-1, seed=s)``.  Memberships must be identical element for element.
"""
import numpy as np
import pytest
import scipy.sparse as sp

ig = pytest.importorskip("igraph")
leidenalg = pytest.importorskip("leidenalg")

from anndataoom_fast._leiden import leiden, leiden_multi  # noqa: E402

try:  # omicverse's own graph construction, if importable
    from omicverse.pp._leiden import get_igraph_from_adjacency
except Exception:  # pragma: no cover - verbatim copy of omicverse's helper
    def get_igraph_from_adjacency(adjacency, *, directed=False):
        sources, targets = adjacency.nonzero()
        weights = np.asarray(adjacency[sources, targets]).ravel() if len(sources) else []
        g = ig.Graph(directed=directed)
        g.add_vertices(adjacency.shape[0])
        g.add_edges(list(zip(sources, targets)))
        g.es["weight"] = weights
        return g


def reference(A, resolution, seed, *, directed=True, use_weights=True, n_iterations=-1):
    g = get_igraph_from_adjacency(A, directed=directed)
    kw = {}
    if use_weights:
        kw["weights"] = np.array(g.es["weight"]).astype(np.float64)
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution,
        n_iterations=n_iterations, seed=seed, **kw)
    return np.array(part.membership)


def _symmetrize(A):
    A = A.maximum(A.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    return A.astype(np.float32)


def random_graph(n, k, seed):
    """Random sparse symmetric weighted graph (float32 CSR, no diagonal)."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), k)
    cols = rng.integers(0, n, n * k)
    vals = rng.random(n * k).astype(np.float32)
    return _symmetrize(sp.coo_matrix((vals, (rows, cols)), shape=(n, n)))


def clustered_graph(n, n_clusters, k, seed):
    """kNN-like graph with planted clusters and UMAP-style fuzzy-union weights."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_clusters, n)
    rows, cols = [], []
    for i in range(n):
        same = np.flatnonzero(labels == labels[i])
        rows.append(np.full(k, i))
        pick = rng.random(k) < 0.85
        cols.append(np.where(pick, rng.choice(same, k), rng.integers(0, n, k)))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    vals = rng.random(len(rows)).astype(np.float32)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.data = np.minimum(A.data, 1).astype(np.float32)
    A = (A + A.T - A.multiply(A.T)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    return A.astype(np.float32)


def disconnected_graph(seed):
    """Several components of different sizes plus isolated vertices."""
    blocks = [random_graph(150, 5, seed), clustered_graph(300, 4, 8, seed + 1),
              random_graph(40, 3, seed + 2), sp.csr_matrix((7, 7), dtype=np.float32)]
    A = sp.block_diag(blocks, format="csr").astype(np.float32)
    perm = np.random.default_rng(seed).permutation(A.shape[0])
    A = A[perm][:, perm].tocsr()
    A.sort_indices()
    return A


GRAPHS = {
    "random": lambda: random_graph(400, 6, 0),
    "clustered": lambda: clustered_graph(600, 6, 10, 1),
    "disconnected": lambda: disconnected_graph(2),
}
_cache = {}


def graph(name):
    if name not in _cache:
        _cache[name] = GRAPHS[name]()
    return _cache[name]


@pytest.mark.parametrize("name", list(GRAPHS))
@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("resolution", [0.25, 0.5, 1.0, 1.5, 2.0])
def test_exact_vs_leidenalg(name, seed, resolution):
    A = graph(name)
    expected = reference(A, resolution, seed)
    got = leiden(A, resolution, random_state=seed)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("name", list(GRAPHS))
def test_multi_matches_single_and_reference(name):
    A = graph(name)
    rs = [0.25, 0.5, 1.0, 1.5, 2.0]
    multi = leiden_multi(A, rs, random_state=42, n_threads=5)
    assert len(multi) == len(rs)
    for r, m in zip(rs, multi):
        np.testing.assert_array_equal(m, reference(A, r, 42))
        np.testing.assert_array_equal(m, leiden(A, r, random_state=42, n_threads=1))


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("resolution", [0.5, 1.0, 2.0])
def test_undirected(seed, resolution):
    # directed=False: every stored entry becomes an undirected edge (pairs of
    # multi-edges), as omicverse's get_igraph_from_adjacency(directed=False).
    A = graph("clustered")
    np.testing.assert_array_equal(
        leiden(A, resolution, random_state=seed, directed=False),
        reference(A, resolution, seed, directed=False))


@pytest.mark.parametrize("n_iterations", [1, 2, 5])
def test_fixed_number_of_iterations(n_iterations):
    A = graph("clustered")
    np.testing.assert_array_equal(
        leiden(A, 1.0, random_state=0, n_iterations=n_iterations),
        reference(A, 1.0, 0, n_iterations=n_iterations))


def test_unweighted():
    A = graph("random")
    np.testing.assert_array_equal(
        leiden(A, 1.0, random_state=1, use_weights=False),
        reference(A, 1.0, 1, use_weights=False))


def test_generic_input_path():
    # unsorted indices, explicit zeros, float64 values, a self loop and an
    # asymmetric entry: handled through the literal nonzero()/fancy-index path
    A = graph("random").astype(np.float64).tolil()
    A[3, 3] = 0.7
    A[5, 17] = 0.3
    A = A.tocsr()
    A.data[::50] = 0.0  # explicit zeros are dropped by nonzero()
    rng = np.random.default_rng(0)
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i + 1]
        p = rng.permutation(e - s)
        A.indices[s:e] = A.indices[s:e][p]
        A.data[s:e] = A.data[s:e][p]
    A.has_sorted_indices = False
    for r in (0.5, 1.5):
        np.testing.assert_array_equal(leiden(A, r, random_state=0), reference(A, r, 0))
    # dense input
    D = graph("random").toarray()
    np.testing.assert_array_equal(leiden(D, 1.0, random_state=0), reference(D, 1.0, 0))


def test_thread_count_does_not_change_result():
    # large enough for the multithreaded aggregation paths (> 1e5 edges)
    A = clustered_graph(12000, 30, 15, 3)
    assert A.nnz > 100_000
    expected = reference(A, 1.0, 0)
    for nt in (1, 2, 7):
        np.testing.assert_array_equal(leiden(A, 1.0, random_state=0, n_threads=nt), expected)
    expected = reference(A, 1.0, 0, directed=False)
    for nt in (1, 4):
        np.testing.assert_array_equal(
            leiden(A, 1.0, random_state=0, directed=False, n_threads=nt), expected)


def test_negative_weights_rejected():
    A = graph("random").copy()
    A.data[0] = -1.0
    with pytest.raises(ValueError):
        leiden(A, 1.0)


def test_empty_and_edgeless():
    A = sp.csr_matrix((5, 5), dtype=np.float32)
    np.testing.assert_array_equal(leiden(A, 1.0), reference(A, 1.0, 0))
    assert leiden(sp.csr_matrix((0, 0), dtype=np.float32), 1.0).shape == (0,)
