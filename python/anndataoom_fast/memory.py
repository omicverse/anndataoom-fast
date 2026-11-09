"""C++ kernels for in-memory scipy CSR matrices, matching omicverse's CPU path.

These are the building blocks omicverse's ``backend='cpp'`` uses for regular
(in-memory) AnnData objects.  Each function documents which arithmetic it
reproduces; all run multithreaded and give results independent of the number
of threads.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import _core
from ._h5 import memory_csr


def _flags(n_cols, **masks):
    g = np.zeros(n_cols, dtype=np.uint8)
    for bit, name in ((1, "mt"), (2, "ribo"), (4, "hb")):
        m = masks.get(name)
        if m is not None:
            g[np.asarray(m, dtype=bool)] |= np.uint8(bit)
    return g


def qc_metrics(X, mt=None, ribo=None, hb=None):
    """Per-cell QC metrics as omicverse's in-memory ``qc_cpu``.

    Returns float32 ``nUMIs``, float32 ``mito_perc``/``ribo_perc``/``hb_perc``
    (group sum / total in float32; 0/0 gives NaN), int32 ``detected_genes``
    (stored entries) and int64 ``n_positive`` (entries > 0).
    """
    m = memory_csr(X)
    r = _core.qc_pass(m.src, _flags(m.shape[1], mt=mt, ribo=ribo, hb=hb), [])
    tot = r["total"].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "nUMIs": tot,
            "mito_perc": r["mt_sum"] / tot,
            "ribo_perc": r["ribo_sum"] / tot,
            "hb_perc": r["hb_sum"] / tot,
        }
    out["detected_genes"] = r["nnz"].astype(np.int32)
    out["n_positive"] = r["nnz_pos"].astype(np.int64)
    return out


def positive_counts(X, rows=None):
    """Per-gene number of entries > 0 over ``rows`` (bool mask or None)."""
    m = memory_csr(X)
    keep = None if rows is None else np.asarray(rows, dtype=np.uint8)
    r = _core.qc_pass(m.src, np.zeros(m.shape[1], np.uint8), [], keep)
    return r["g_nonzero"].astype(np.int64)


def stored_counts(X):
    """Per-gene number of stored entries (``X.getnnz(axis=0)``)."""
    m = memory_csr(X)
    r = _core.qc_pass(m.src, np.zeros(m.shape[1], np.uint8), [])
    return r["g_stored"].astype(np.int64)


def size_factor_counts(X, *, exclude_highly_expressed=True, max_fraction=0.2):
    """Counts per cell used by ``normalize_total`` (float32) and the kept-gene mask.

    With ``exclude_highly_expressed`` a gene is excluded when it exceeds
    ``fl32(rowsum * fl32(max_fraction))`` in any cell (compared in float32),
    exactly as omicverse, but without densifying the matrix.
    """
    m = memory_csr(X)
    n, G = m.shape
    rows = np.ones(n, np.uint8)
    cols = np.ones(G, np.uint8)
    if not exclude_highly_expressed:
        r = _core.qc_pass(m.src, np.zeros(G, np.uint8), [])
        return r["total"].astype(np.float32), np.ones(G, dtype=bool)
    hi = _core.hi_pass(m.src, rows, cols, float(max_fraction))
    gene_subset = hi["hi_count"] == 0
    g = np.zeros(G, np.uint8)
    g[~gene_subset] = 0x80  # dropped from the size factor
    r = _core.qc_pass(m.src, g, [])
    return r["total"].astype(np.float32), gene_subset


def pearson_residual_variance(counts, *, theta=100.0, clip=None, sums_genes=None,
                              sums_cells=None, sum_total=None):
    """Variance of clipped Pearson residuals per gene (omicverse in-memory kernel).

    Same definition as ``_calculate_res_sparse``: mu = sum_gene * sum_cell /
    total, residuals clipped at ``clip`` (default sqrt(n)), population
    variance; computed from the stored entries plus a closed form for zeros.
    Without precomputed sums they are taken in float32 as omicverse does, and
    the product sum_gene * sum_cell is then rounded to float32 like its kernel.
    """
    m = memory_csr(counts)
    n, G = m.shape
    f32 = sums_genes is None or sums_cells is None
    if f32:
        q = _core.qc_pass(m.src, np.zeros(G, np.uint8), [])
        sums_cells = q["total"].astype(np.float32).astype(np.float64)
        sums_genes = q["g_sum"].astype(np.float32).astype(np.float64)
        sum_total = float(np.sum(sums_genes.astype(np.float32)))
    else:
        sums_cells = np.asarray(sums_cells, dtype=np.float64)
        sums_genes = np.asarray(sums_genes, dtype=np.float64)
        if sum_total is None:
            sum_total = float(np.sum(sums_genes))
    if clip is None:
        clip = float(np.sqrt(n))
    st = _core.stats_pass(m.src, np.ones(n, np.uint8), np.arange(G, dtype=np.int32),
                          np.zeros(G, np.uint8), sums_genes, 1.0, False, False, True,
                          float(theta), float(clip), float(sum_total), cells_override=sums_cells,
                          mu_f32_product=f32)
    su, mult = np.unique(sums_cells, return_counts=True)
    z1, z2 = _core.pearson_zero_terms(su, mult.astype(np.float64), sums_genes, float(sum_total),
                                      float(theta), float(clip), f32)
    s1 = z1 + st["d1"]
    s2 = z2 + st["d2"]
    mean = s1 / n
    return s2 / n - mean * mean


def mean_var(X):
    """Per-gene mean and sample variance (ddof=1), float64."""
    m = memory_csr(X)
    n, G = m.shape
    st = _core.stats_pass(m.src, np.ones(n, np.uint8), np.arange(G, dtype=np.int32),
                          np.zeros(G, np.uint8), np.zeros(G), 1.0, False, True, False, 100.0,
                          0.0, 0.0, recip_override=np.ones(n, np.float32))
    mean = st["s1"] / n
    var = np.clip(st["s2"] - n * mean * mean, 0.0, None) / max(n - 1, 1)
    return mean, var


def scale(X, *, max_value=10.0, mean=None, std=None):
    """Dense float64 z-score of a CSR matrix, clipped to +-max_value.

    Returns ``(scaled, mean, std)`` with ``std == 0`` replaced by 1, as
    omicverse's ``scale_array(zero_center=True)``.
    """
    X = sp.csr_matrix(X)
    if mean is None or std is None:
        mean, var = mean_var(X)
        std = np.sqrt(var)
        std[std == 0] = 1.0
    m = memory_csr(X)
    n, G = m.shape
    out = np.empty((n, G), dtype=np.float64)
    clip = 0.0 if max_value is None else float(max_value)
    _core.scale_dense(m.src, np.ones(n, np.uint8), np.arange(G, dtype=np.int32),
                      np.ascontiguousarray(mean, np.float64), np.ascontiguousarray(std, np.float64),
                      clip, out)
    return out, mean, std


def gene_sums(X):
    """Per-gene sum and sum of squares (float64) of a CSR matrix."""
    m = memory_csr(X)
    r = _core.qc_pass(m.src, np.zeros(m.shape[1], np.uint8), [])
    return r["g_sum"], r["g_sumsq"]


class _PCAResult:
    """The attributes omicverse reads from a fitted sklearn PCA."""


def pca_covariance_eigh(X, n_components):
    """sklearn 1.7 ``PCA(svd_solver='covariance_eigh').fit_transform`` on dense X.

    Same steps (mean, X^T X - n mean mean^T, eigh, descending order, negative
    eigenvalues to 0, svd_flip on Vt, projection X C^T - mean C^T); the
    2000 x 2000 eigendecomposition runs on single-threaded LAPACK, which on
    many-core nodes is ~50x faster than letting OpenBLAS spawn threads for it
    and does not depend on the thread count.
    """
    from scipy import linalg
    from sklearn.utils.extmath import svd_flip
    from threadpoolctl import threadpool_limits
    X = np.ascontiguousarray(np.asarray(X))
    n, p = X.shape
    mean = np.mean(X, axis=0).reshape(-1)
    C = X.T @ X
    C -= n * mean.reshape(-1, 1) * mean.reshape(1, -1)
    C /= n - 1
    with threadpool_limits(1, user_api="blas"):
        w, V = np.linalg.eigh(C)
    w = np.ascontiguousarray(np.flip(w, axis=0))
    V = np.ascontiguousarray(np.flip(V, axis=1))
    w[w < 0.0] = 0.0
    S = np.sqrt(w * (n - 1))
    _, Vt = svd_flip(None, V.T, u_based_decision=False)
    ratio = w / np.sum(w)
    k = int(n_components)
    m = _PCAResult()
    m.mean_ = mean
    m.components_ = np.ascontiguousarray(Vt[:k])
    m.n_components_ = k
    m.explained_variance_ = w[:k]
    m.explained_variance_ratio_ = ratio[:k]
    m.singular_values_ = S[:k]
    m.noise_variance_ = float(np.mean(w[k:])) if k < min(n, p) else 0.0
    m.n_samples_, m.n_features_in_ = n, p
    Ct = np.ascontiguousarray(m.components_.T)
    Xt = X @ Ct
    Xt -= mean.reshape(1, -1) @ Ct
    return Xt, m
