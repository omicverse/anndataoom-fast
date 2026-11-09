"""Deterministic test graphs for the brisc Leiden port.

UMAP-like symmetric weighted kNN graphs with planted clusters plus a few tiny
disconnected components (so that ``min_cluster_size`` merging is exercised).
Only integer draws, float32 uniform draws, sorting and IEEE +,-,* are used, so
the graphs are bit-identical on every platform.  Used by
tests/test_leiden_brisc.py and by tests/data/make_brisc_fixtures.py (which
runs brisc itself to produce the reference labels).
"""
import hashlib

import numpy as np
import scipy.sparse as sp

# (n, n_clusters, k, seed, p_within)
GRAPHS = [(2000, 8, 10, 0, 0.8), (8000, 20, 15, 1, 0.7), (20000, 40, 15, 2, 0.6)]
RESOLUTIONS = [0.25, 1.0, 2.0]
MIN_CLUSTER_SIZES = [1, 5]
N_NEIGHBORS = 10  # columns of the kNN array used for the min_cluster_size merge


def make_graph(n, n_clusters, k, seed, p_within):
    """Return (adjacency float32 CSR, neighbors uint32 (n, N_NEIGHBORS))."""
    rng = np.random.default_rng(seed)
    n_tiny = max(4, n // 200)            # nodes in tiny components (pairs / triples)
    n_reg = n - n_tiny
    cluster = rng.integers(0, n_clusters, n_reg)
    order = np.argsort(cluster, kind="stable")
    sizes = np.bincount(cluster, minlength=n_clusters)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))

    rank = np.empty(n_reg, dtype=np.int64)       # position of each node in its cluster
    rank[order] = np.arange(n_reg) - np.repeat(starts, sizes)

    # Directed kNN-style edges of the regular nodes: with probability
    # p_within a "local" neighbour (nearby along a 1-D manifold inside the
    # cluster, so higher resolutions split clusters), otherwise any node.
    rows = np.repeat(np.arange(n_reg), k)
    c = np.repeat(cluster, k)
    same = rng.random(n_reg * k) < p_within
    step = rng.integers(-4 * k, 4 * k + 1, n_reg * k)
    within = order[starts[c] + (np.repeat(rank, k) + step) % sizes[c]]
    anywhere = rng.integers(0, n_reg, n_reg * k)
    cols = np.where(same, within, anywhere)
    # UMAP-like membership strengths: nearest neighbour 1, then decreasing.
    w = -np.sort(-rng.random((n_reg, k), dtype=np.float32), axis=1)
    w[:, 0] = 1
    w = w.ravel()

    # Tiny components: consecutive pairs / triples, fully connected.
    t_rows, t_cols = [], []
    pos = n_reg
    size = 2
    while pos < n:
        members = np.arange(pos, min(pos + size, n))
        for a in members:
            for b in members:
                if a != b:
                    t_rows.append(a)
                    t_cols.append(b)
        pos += size
        size = 5 - size  # alternate 2, 3
    rows = np.concatenate((rows, np.array(t_rows, dtype=rows.dtype)))
    cols = np.concatenate((cols, np.array(t_cols, dtype=cols.dtype)))
    w = np.concatenate((w, rng.random(len(t_rows), dtype=np.float32) * 0.5 + 0.5))

    # Drop self edges and duplicate (row, col) pairs (keep the first).
    keep = rows != cols
    rows, cols, w = rows[keep], cols[keep], w[keep]
    _, first = np.unique(rows.astype(np.int64) * n + cols, return_index=True)
    rows, cols, w = rows[first], cols[first], w[first]
    A = sp.csr_matrix((w, (rows, cols)), shape=(n, n), dtype=np.float32)
    # Fuzzy union, as in UMAP: A + A.T - A * A.T
    At = A.T.tocsr()
    G = (A + At - A.multiply(At)).tocsr().astype(np.float32)
    G.eliminate_zeros()
    G.sort_indices()

    # kNN array (nearest first, never the node itself): the node's graph
    # neighbours by decreasing weight, then deterministic filler nodes.
    nbr = np.empty((n, N_NEIGHBORS), dtype=np.uint32)
    for i in range(n):
        s, e = G.indptr[i], G.indptr[i + 1]
        idx = G.indices[s:e][np.argsort(-G.data[s:e], kind="stable")][:N_NEIGHBORS]
        fill = (i * 7 + 13 * np.arange(N_NEIGHBORS - len(idx)) + 1) % n_reg
        fill[fill == i] = (fill[fill == i] + 1) % n_reg
        nbr[i] = np.concatenate((idx, fill))
    return G, nbr


def graph_digest(G, nbr):
    h = hashlib.sha256()
    for a in (G.indptr.astype(np.int64), G.indices.astype(np.int64),
              G.data.astype(np.float32), nbr.astype(np.uint32)):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
