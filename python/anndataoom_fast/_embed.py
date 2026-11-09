"""PaCMAP and LocalMAP embeddings, ported from brisc.

The C++ code (``anndataoom_fast._embed_cpp``) is a port of brisc's
``brisc/embed.pyx`` (Copyright (c) 2026 Michael Wainberg and Keon Arbabi, MIT
License, https://github.com/briscverse/brisc).  Given the same principal
components and nearest neighbours it reproduces ``SingleCell.pacmap()`` /
``SingleCell.localmap()`` bit for bit -- the result brisc gives with
``num_threads > 1`` (or ``num_threads=1, match_parallel=True``) -- and, like
that code path, the result does not depend on the number of threads.
"""
from __future__ import annotations

import numpy as np

__all__ = ["pacmap", "localmap", "brisc_knn"]


def _iterations(num_iterations):
    if isinstance(num_iterations, (tuple, list)):
        if len(num_iterations) != 3:
            raise ValueError("num_iterations must be an integer or a length-3 tuple")
        its = tuple(int(x) for x in num_iterations)
    else:
        its = (100, 100, int(num_iterations))
    if min(its) < 1:
        raise ValueError("num_iterations must be positive")
    return its


def brisc_knn(X, k: int = 20, *, random_state: int = 0):
    """``k`` nearest neighbours (excluding self) and squared Euclidean distances.

    Uses anndataoom-fast's inverted-file search (:func:`anndataoom_fast.knn`),
    which is close to, but not identical to, brisc's ``SingleCell.neighbors()``.
    Returns ``(uint32 (n, k), float32 (n, k))`` in brisc's layout.
    """
    from ._neighbors import knn
    idx, dist = knn(X, k + 1, random_state=random_state)
    return (np.ascontiguousarray(idx[:, 1:], dtype=np.uint32),
            np.ascontiguousarray(np.square(dist[:, 1:]), dtype=np.float32))


def _prepare_knn(X, knn_indices, knn_dists, k_needed, random_state):
    n = X.shape[0]
    if knn_indices is None:
        if knn_dists is not None:
            raise ValueError("knn_dists given without knn_indices")
        # Reuse the neighbours of a previous kNN search on the same
        # coordinates (e.g. ov.pp.neighbors with backend='cpp') if it has
        # enough of them; otherwise search with brisc's method.
        from ._neighbors import cached_knn
        hit = cached_knn(X, k_needed + 1)
        if hit is not None:
            idx, dist = hit
            return (np.ascontiguousarray(idx[:, 1:], dtype=np.uint32),
                    np.ascontiguousarray(np.square(dist[:, 1:]), dtype=np.float32))
        return brisc_knn(X, k_needed, random_state=random_state)
    if knn_dists is None:
        raise ValueError("knn_indices given without knn_dists")
    idx = np.asarray(knn_indices)
    dist = np.asarray(knn_dists)
    if idx.ndim != 2 or idx.shape != dist.shape or idx.shape[0] != n:
        raise ValueError("knn_indices and knn_dists must both have shape (n_cells, k)")
    # scanpy convention (cell itself in column 0): drop that column
    if idx.shape[1] > 0 and np.array_equal(idx[:, 0], np.arange(n)):
        idx, dist = idx[:, 1:], dist[:, 1:]
    if idx.shape[1] < k_needed:
        raise ValueError(f"need at least {k_needed} neighbours per cell "
                         f"(n_neighbors + n_extra_neighbors), got {idx.shape[1]}")
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError("knn_indices contains out-of-range indices")
    return (np.ascontiguousarray(idx, dtype=np.uint32),
            np.ascontiguousarray(dist, dtype=np.float32))


def _run(localmap_, X_pcs, knn_indices, knn_dists, n_neighbors, n_extra_neighbors,
         n_mid_near_pairs, n_further_pairs, num_iterations, learning_rate, max_distance,
         random_state, n_threads, return_pairs):
    from . import _embed_cpp
    X = np.ascontiguousarray(X_pcs, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X_pcs must be a 2-D array with at least 2 columns")
    n = X.shape[0]
    n_neighbors, n_extra_neighbors = int(n_neighbors), int(n_extra_neighbors)
    n_mid_near_pairs, n_further_pairs = int(n_mid_near_pairs), int(n_further_pairs)
    if not 1 <= n_neighbors < n:
        raise ValueError(f"n_neighbors must be >= 1 and < the number of cells ({n})")
    if n_extra_neighbors < 0:
        raise ValueError("n_extra_neighbors must be >= 0")
    k_needed = n_neighbors + n_extra_neighbors
    if k_needed < 6:
        raise ValueError("n_neighbors + n_extra_neighbors must be at least 6")
    if k_needed >= n:
        raise ValueError("n_neighbors + n_extra_neighbors must be < the number of cells")
    for v, name in ((n_mid_near_pairs, "n_mid_near_pairs"), (n_further_pairs, "n_further_pairs")):
        if not 1 <= v <= n:
            raise ValueError(f"{name} must be between 1 and the number of cells")
    if n < n_mid_near_pairs + 6 or n < n_neighbors + n_further_pairs + 1:
        raise ValueError("too few cells for the requested numbers of pairs")
    if not learning_rate > 0:
        raise ValueError("learning_rate must be positive")
    if not max_distance > 0:
        raise ValueError("max_distance must be positive")
    it1, it2, it3 = _iterations(num_iterations)
    idx, dist = _prepare_knn(X, knn_indices, knn_dists, k_needed, random_state)

    out = np.empty((n, 2), dtype=np.float32)
    pairs = None
    if return_pairs:
        pairs = {"neighbor_pairs": np.empty((n, n_neighbors), np.uint32),
                 "mid_near_pairs": np.empty((n, n_mid_near_pairs), np.uint32),
                 "further_pairs": np.empty((n, n_further_pairs), np.uint32)}
    _embed_cpp.embed(X, idx, dist, out, bool(localmap_), n_neighbors, n_mid_near_pairs,
                     n_further_pairs, it1, it2, it3, float(learning_rate), float(max_distance),
                     int(random_state) & 0xFFFFFFFFFFFFFFFF,
                     0 if n_threads is None else int(n_threads),
                     **(pairs or {}))
    return (out, pairs) if return_pairs else out


_COMMON_DOC = """
    Parameters
    ----------
    X_pcs
        ``(n_cells, n_pcs)`` input coordinates, usually principal components
        (cast to float32, as brisc requires).  The embedding is initialised
        with ``0.01 * X_pcs[:, :2]``.
    knn_indices, knn_dists
        Optional nearest neighbours of every cell, nearest first, excluding the
        cell itself (a leading self column, scanpy-style, is dropped), and the
        **squared** Euclidean distances -- brisc's ``obsm['neighbors']`` and
        ``obsm['distances']``.  All columns are used, as in brisc.  When
        omitted they are computed with :func:`brisc_knn`
        (``n_neighbors + n_extra_neighbors`` neighbours).
    n_neighbors
        Nearest-neighbour pairs kept per cell, chosen among the kNN by
        PaCMAP's scaled distance.
    n_extra_neighbors
        Extra neighbours searched before pruning to ``n_neighbors``.
    n_mid_near_pairs, n_further_pairs
        Mid-near and further pairs sampled per cell.
    num_iterations
        ``(phase1, phase2, phase3)`` iterations, or an int for phase 3 (phases
        1 and 2 then get 100 each).
    learning_rate
        Adam learning rate.
    random_state
        brisc's ``seed``.
    n_threads
        OpenMP threads (``0``/``None`` = all).  Does not change the result.
    return_pairs
        Also return the sampled pair lists (for testing).

    Returns
    -------
    float32 array of shape ``(n_cells, 2)``.
"""


def pacmap(X_pcs, *, knn_indices=None, knn_dists=None, n_neighbors: int = 10,
           n_extra_neighbors: int = 10, n_mid_near_pairs: int = 5, n_further_pairs: int = 20,
           num_iterations=(100, 100, 250), learning_rate: float = 1.0, random_state: int = 0,
           n_threads: int | None = 0, return_pairs: bool = False):
    return _run(False, X_pcs, knn_indices, knn_dists, n_neighbors, n_extra_neighbors,
                n_mid_near_pairs, n_further_pairs, num_iterations, learning_rate, 10.0,
                random_state, n_threads, return_pairs)


def localmap(X_pcs, *, knn_indices=None, knn_dists=None, n_neighbors: int = 10,
             n_extra_neighbors: int = 10, n_mid_near_pairs: int = 5, n_further_pairs: int = 20,
             num_iterations=(100, 100, 250), learning_rate: float = 1.0,
             max_distance: float = 10.0, random_state: int = 0, n_threads: int | None = 0,
             return_pairs: bool = False):
    return _run(True, X_pcs, knn_indices, knn_dists, n_neighbors, n_extra_neighbors,
                n_mid_near_pairs, n_further_pairs, num_iterations, learning_rate, max_distance,
                random_state, n_threads, return_pairs)


pacmap.__doc__ = ("PaCMAP embedding (brisc ``SingleCell.pacmap``), bit-identical to brisc.\n"
                  + _COMMON_DOC)
localmap.__doc__ = ("LocalMAP embedding (brisc ``SingleCell.localmap``), bit-identical to "
                    "brisc.\n" + _COMMON_DOC.replace(
                        "    learning_rate\n        Adam learning rate.\n",
                        "    learning_rate\n        Adam learning rate.\n    max_distance\n"
                        "        LocalMAP's distance cut-off for re-sampled further pairs "
                        "and\n        nearest-neighbour force scaling.\n"))
