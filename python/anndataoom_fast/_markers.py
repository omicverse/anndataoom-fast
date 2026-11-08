"""Marker genes by COSG (Dai et al. 2022), as ``ov.single.find_markers(method='cosg')``.

One multithreaded pass over the expression matrix gives, for every
(group, gene), the sum, sum of squares and number of stored entries; the
cosine similarities, the COSG specificity score, log-fold changes and
detection fractions are all closed-form functions of these, so nothing is
transposed, normalised or densified.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import _core


def group_stats(X, labels):
    """Per-group sums, sums of squares and stored-entry counts of a CSR matrix.

    Returns ``(groups, sizes, sum, sumsq, nnz)``; the last three have shape
    ``(n_groups, n_genes)``.  ``groups`` are the sorted unique labels.
    """
    X = sp.csr_matrix(X)
    labels = np.asarray(labels)
    groups, codes = np.unique(labels, return_inverse=True)
    order = np.argsort(codes, kind="stable").astype(np.int64)
    sizes = np.bincount(codes, minlength=len(groups)).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    data = X.data if X.data.dtype in (np.float32, np.float64) else X.data.astype(np.float64)
    s, q, z = _core.group_stats_csr(X.indptr.astype(np.int64, copy=False), X.indices, data,
                                    int(X.shape[1]), order, offsets)
    return groups, sizes, s, q, z


def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    return reference_indices[partition][partial_indices]


def cosg(X, labels, var_names, *, mu: float = 1.0, n_genes: int = 50,
         remove_lowly_expressed: bool = False, expressed_pct: float = 0.1,
         calculate_logfoldchanges: bool = True, expm1=np.expm1):
    """COSG marker genes for every group.

    Returns a dict with per-group arrays ``names``, ``scores``,
    ``logfoldchanges`` (top ``n_genes`` per group, best first) and the full
    ``pts`` / ``pts_rest`` detection fractions (n_genes_total x n_groups).
    """
    groups, sizes, S, Q, Z = group_stats(X, labels)
    var_names = np.asarray(var_names)
    n = int(sizes.sum())
    # cosine(gene g, indicator of group c) = S[c, g] / (||x_g|| sqrt(n_c))
    norm_g = np.sqrt(Q.sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (S / np.sqrt(sizes)[:, None]) / norm_g[None, :]
    cos = np.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0).T  # genes x groups
    nz = cos != 0
    e2 = cos * cos
    e2_sum = e2.sum(axis=1, keepdims=True)
    lam = np.zeros_like(e2)
    if mu == 1:
        lam[nz] = (e2 / e2_sum)[nz]
    else:
        lam[nz] = (e2 / ((1 - mu) * e2 + mu * e2_sum))[nz]
    lam[nz] = lam[nz] * cos[nz]

    total = S.sum(axis=0)
    out = {"groups": groups, "names": {}, "scores": {}, "logfoldchanges": {}}
    n_top = min(int(n_genes), lam.shape[0])
    for ci, g in enumerate(groups):
        scores = lam[:, ci].copy()
        if remove_lowly_expressed:
            scores[Z[ci] < sizes[ci] * expressed_pct] = -1
        top = _select_top_n(scores, n_top)
        out["names"][g] = var_names[top]
        out["scores"][g] = scores[top].astype(np.float32)
        if calculate_logfoldchanges:
            mean_g = S[ci] / sizes[ci]
            mean_r = (total - S[ci]) / max(n - sizes[ci], 1)
            fc = (expm1(mean_g) + 1e-9) / (expm1(mean_r) + 1e-9)
            out["logfoldchanges"][g] = np.log2(fc[top]).astype(np.float32)
    z_total = Z.sum(axis=0)
    out["pts"] = (Z / sizes[:, None]).T
    rest = np.maximum(n - sizes, 1)
    out["pts_rest"] = ((z_total[None, :] - Z) / rest[:, None]).T
    return out
