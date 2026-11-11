"""Drop-in replacements for anndataoom's chunked preprocessing functions.

Each function has the signature and side effects of its anndataoom
counterpart.  When the input is a lazy view of an on-disk CSR matrix that the
C++ engine can read, the work runs as multithreaded C++ passes; otherwise the
original anndataoom implementation is called unchanged.

Passes are shared across calls through a small cache: the QC pass also
gathers the per-gene counts that robust-gene selection needs, and the
size-factor pass of ``normalize_total`` also accumulates the statistics that
``scale`` and the Pearson-residual HVG step need.  A later call reuses them
only when its inputs (rows, columns, size factors, parameters) match exactly.
"""
from __future__ import annotations

import logging
import math
from collections import OrderedDict

import numpy as np

from . import _core
from ._svd import randomized_svd
from ._view import digest, identify_roots, resolve

logger = logging.getLogger("anndataoom_fast")

_ORIG: dict = {}
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_CACHE_MAX = 32
STATS = {"fast": 0, "fallback": 0, "cache_hits": 0}


def _remember(key, value):
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def _recall(key):
    v = _CACHE.get(key)
    if v is not None:
        STATS["cache_hits"] += 1
    return v


def _fallback(name, *args, **kwargs):
    STATS["fallback"] += 1
    fn = _ORIG.get(name)
    if fn is None:  # not installed as a patch: call anndataoom's implementation
        from anndataoom import _chunked_ops
        fn = getattr(_chunked_ops, name)
        if getattr(fn, "__module__", "").startswith("anndataoom_fast"):
            raise RuntimeError(f"no original implementation recorded for {name}")
    return fn(*args, **kwargs)


def _view(adata, arr=None):
    from ._config import enabled
    if not enabled():
        return None
    if adata is not None:
        identify_roots(adata)
    try:
        return resolve(adata.X if arr is None else arr)
    except Exception as e:  # never let the fast path break a pipeline
        logger.debug("anndataoom_fast: cannot resolve view (%s)", e)
        return None


def _gene_group(v, masks: dict) -> np.ndarray:
    """uint8 per original column: bit0 mt, bit1 ribo, bit2 hb, bit7 dropped."""
    g = np.full(v.handle.shape[1], 0x80, dtype=np.uint8)
    cols = np.arange(v.handle.shape[1]) if v.cols is None else v.cols
    g[cols] = 0
    for bit, m in masks.items():
        m = np.asarray(m, dtype=bool)
        g[cols[m]] |= np.uint8(bit)
    return g


def _qc_cached(v, adata=None, extra_mask=None):
    """Run (or recall) the QC pass for view `v`; returns the result dict."""
    masks = {}
    if adata is not None:
        for bit, col in ((1, "mt"), (2, "ribo"), (4, "hb")):
            if col in adata.var.columns and adata.var[col].dtype == bool:
                masks[bit] = adata.var[col].values
    if extra_mask is not None:
        masks = {1: extra_mask}
    key = ("qc",) + v.key() + (digest(*[masks[b] for b in sorted(masks)]), tuple(sorted(masks)))
    hit = _recall(key)
    if hit is not None:
        return hit, masks
    res = _core.qc_pass(v.handle.src, _gene_group(v, masks), [],
                        None if v.rows is None else v.row_keep)
    out = {k: (v.from_orig_rows(res[k]) if k in ("total", "nnz", "mt", "ribo", "hb")
               else v.from_orig_cols(res[k]) if k.startswith("g_") else res[k])
           for k in res}
    out["masks"] = {b: np.asarray(m, dtype=bool).copy() for b, m in masks.items()}
    _remember(key, out)
    # Per-gene counts over these rows/columns do not depend on gene groups.
    _remember(("genes",) + v.key(), out)
    # Nor on the column selection: keep them in original column space, so a
    # view of the same rows with fewer columns (after gene filtering) reuses
    # this pass.
    _remember(_genes_rows_key(v), {k: res[k] for k in res if k.startswith("g_")})
    return out, masks


def _genes_rows_key(v):
    path, key, rows_digest, _ = v.key()
    return ("genes_rows", path, key, rows_digest)


def _gene_counts(v):
    """Per-gene QC counts for view `v`, from any cached pass over the same rows."""
    res = _recall(("genes",) + v.key())
    if res is not None:
        return res
    raw = _recall(_genes_rows_key(v))
    if raw is not None:
        return {k: v.from_orig_cols(a) for k, a in raw.items()}
    return _qc_cached(v)[0]


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------
def chunked_qc_metrics(adata, *, percent_top=None, chunk_size=None, **kw):
    v = _view(adata)
    if v is None or not v.raw or percent_top:
        return _fallback("chunked_qc_metrics", adata, percent_top=percent_top,
                         **({} if chunk_size is None else {"chunk_size": chunk_size}), **kw)
    STATS["fast"] += 1
    res, _ = _qc_cached(v, adata)
    adata.obs["nUMIs"] = np.asarray(res["total"], dtype=np.float64)
    adata.obs["detected_genes"] = np.asarray(res["nnz"], dtype=np.int64)
    adata.var["n_cells"] = res["g_stored"].astype(np.int64)


def chunked_gene_group_pct(adata, gene_mask, *, chunk_size=None, **kw):
    v = _view(adata)
    if v is None or not v.raw:
        return _fallback("chunked_gene_group_pct", adata, gene_mask,
                         **({} if chunk_size is None else {"chunk_size": chunk_size}), **kw)
    STATS["fast"] += 1
    gene_mask = np.asarray(gene_mask, dtype=bool)
    # Reuse a QC pass that already summed this exact gene group.
    for key in reversed(list(_CACHE)):
        if key[0] != "qc" or key[1:5] != v.key():
            continue
        r = _CACHE[key]
        for bit, name in ((1, "mt"), (2, "ribo"), (4, "hb")):
            m = r["masks"].get(bit)
            if m is not None and m.shape == gene_mask.shape and np.array_equal(m, gene_mask):
                STATS["cache_hits"] += 1
                return np.asarray(r[name], dtype=np.float64)
    res, _ = _qc_cached(v, None, extra_mask=gene_mask)
    return np.asarray(res["mt"], dtype=np.float64)


def chunked_identify_robust_genes(adata, percent_cells=0.05, chunk_size=None, **kw):
    v = _view(adata)
    if v is None or not v.raw:
        return _fallback("chunked_identify_robust_genes", adata, percent_cells=percent_cells,
                         **({} if chunk_size is None else {"chunk_size": chunk_size}), **kw)
    STATS["fast"] += 1
    res = _gene_counts(v)
    n_cells = res["g_nonzero"].astype(np.int64)
    adata.var["n_cells"] = n_cells
    nonzero_mask = n_cells > 0
    if not nonzero_mask.all():
        adata._inplace_subset_var(nonzero_mask)
        n_cells = n_cells[nonzero_mask]
        adata.var["n_cells"] = n_cells
    adata.var["percent_cells"] = (adata.var["n_cells"] / adata.n_obs) * 100
    adata.var["robust"] = adata.var["percent_cells"] >= percent_cells
    adata.var["highly_variable_features"] = adata.var["robust"]


# ---------------------------------------------------------------------------
# normalize_total (+ speculative scale / Pearson statistics)
# ---------------------------------------------------------------------------
def _pearson_key(v, sums_cells, sums_genes, sum_total, theta, clip):
    return ("pearson",) + v.key() + (digest(sums_cells), digest(sums_genes),
                                     float(sum_total), float(theta), float(clip))


def _scale_key(v, norm_factors, log1p):
    return ("scale",) + v.key() + (digest(np.asarray(norm_factors, dtype=np.float64)),
                                   bool(log1p))


def chunked_normalize_total(adata, *, target_sum=None, exclude_highly_expressed=False,
                            max_fraction=0.05, chunk_size=None, **kw):
    v = _view(adata)
    if v is None or not v.raw:
        return _fallback("chunked_normalize_total", adata, target_sum=target_sum,
                         exclude_highly_expressed=exclude_highly_expressed,
                         max_fraction=max_fraction,
                         **({} if chunk_size is None else {"chunk_size": chunk_size}), **kw)
    STATS["fast"] += 1
    from anndataoom._chunked_ops import TransformedBackedArray

    X = adata.X
    n_obs, n_vars = v.n_rows, v.n_cols
    adata.layers["counts"] = X
    col_map = v.col_map()
    excl = np.zeros(v.handle.shape[1], dtype=np.uint8)
    sums_genes = sq_sums_genes = None
    # Pearson residuals use each cell's total over *all* genes, also when the
    # size factors leave the highly expressed ones out.
    full_rows = None
    if exclude_highly_expressed:
        hi = _core.hi_pass(v.handle.src, v.row_keep, v.col_keep(), float(max_fraction))
        full_rows = np.ascontiguousarray(hi["rowsum"], dtype=np.float64)
        hi_count = v.from_orig_cols(hi["hi_count"])
        sums_genes = v.from_orig_cols(hi["g_sum"])
        sq_sums_genes = v.from_orig_cols(hi["g_sumsq"])
        gene_subset = hi_count == 0
        n_excluded = int((~gene_subset).sum())
        if n_excluded > 0:
            logger.info("Excluding %d highly-expressed genes from normalization", n_excluded)
        cols = np.arange(v.handle.shape[1]) if v.cols is None else v.cols
        excl[cols[~gene_subset]] = 1

    # One pass: size factors, plus the statistics later steps will ask for
    # (scale on log1p of these size factors; Pearson residuals with default
    # theta/clip) whenever their inputs are already known.
    speculate_scale = target_sum is not None
    speculate_pearson = sums_genes is not None
    theta, clip = 100.0, float(np.sqrt(n_obs))
    sum_total = float(sums_genes.sum()) if speculate_pearson else 0.0
    st = _core.stats_pass(
        v.handle.src, v.row_keep, col_map, excl,
        np.ascontiguousarray(sums_genes if speculate_pearson else np.zeros(n_vars)),
        float(target_sum) if target_sum is not None else 1.0,
        True, speculate_scale, speculate_pearson, theta, clip, sum_total,
        cells_override=full_rows)
    counts_per_cell = v.from_orig_rows(st["cps"]).astype(np.float64)
    pearson_cells = counts_per_cell if full_rows is None else v.from_orig_rows(full_rows)
    if sums_genes is None:
        sums_genes = v.from_orig_cols(st["g_sum"])
        sq_sums_genes = v.from_orig_cols(st["g_sumsq"])

    if target_sum is None:
        nonzero = counts_per_cell[counts_per_cell > 0]
        target_sum = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
    norm_factors = counts_per_cell / target_sum
    norm_factors = np.nan_to_num(norm_factors, nan=1.0, posinf=1.0, neginf=1.0)
    norm_factors[norm_factors == 0] = 1.0
    adata.obs["_norm_factor"] = norm_factors
    adata.uns["_pearson_precompute"] = {
        "sums_cells": np.array(pearson_cells, dtype=np.float64),
        "sums_genes": sums_genes,
        "sq_sums_genes": sq_sums_genes,
        "sum_total": float(sums_genes.sum()),
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
    }
    adata.X = TransformedBackedArray(X, norm_factors=norm_factors)

    if speculate_scale:
        _remember(_scale_key(v, norm_factors, True),
                  {"s1": st["s1"], "s2": st["s2"], "n": n_obs})
    if speculate_pearson:
        _remember(_pearson_key(v, pearson_cells, sums_genes, sum_total, theta, clip),
                  {"d1": st["d1"], "d2": st["d2"]})


# ---------------------------------------------------------------------------
# Pearson residual variance (HVG)
# ---------------------------------------------------------------------------
def chunked_pearson_residual_variance(X, *, theta=100.0, clip=None, chunk_size=None,
                                      precomputed=None, **kw):
    v = _view(None, X)
    if v is None or not v.raw:
        return _fallback("chunked_pearson_residual_variance", X, theta=theta, clip=clip,
                         precomputed=precomputed,
                         **({} if chunk_size is None else {"chunk_size": chunk_size}), **kw)
    STATS["fast"] += 1
    n_obs, n_vars = v.n_rows, v.n_cols
    if clip is None:
        clip = np.sqrt(n_obs)
    if precomputed is not None:
        sums_cells = np.asarray(precomputed["sums_cells"], dtype=np.float64)
        sums_genes = np.asarray(precomputed["sums_genes"], dtype=np.float64)
        sq_sums_genes = np.asarray(precomputed["sq_sums_genes"], dtype=np.float64)
        if sums_cells.shape[0] != n_obs or sums_genes.shape[0] != n_vars:
            raise ValueError(
                "Stale precomputed Pearson stats: shapes "
                f"({sums_cells.shape[0]}, {sums_genes.shape[0]}) do not "
                f"match X ({n_obs}, {n_vars}). Was the adata subset between "
                "normalize_total and HVG?")
        sum_total = float(precomputed.get("sum_total", sums_genes.sum()))
    else:
        res = _recall(("genes",) + v.key())
        if res is None:
            res, _ = _qc_cached(v)
        sums_cells = np.asarray(res["total"], dtype=np.float64)
        sums_genes = res["g_sum"]
        sq_sums_genes = res["g_sumsq"]
        sum_total = sums_genes.sum()
    gene_mean = sums_genes / n_obs
    gene_var = sq_sums_genes / n_obs - gene_mean ** 2

    key = _pearson_key(v, sums_cells, sums_genes, sum_total, theta, clip)
    hit = _recall(key)
    if hit is None:
        st = _core.stats_pass(
            v.handle.src, v.row_keep, v.col_map(), np.zeros(v.handle.shape[1], np.uint8),
            np.ascontiguousarray(sums_genes), 1.0, False, False, True, float(theta),
            float(clip), float(sum_total),
            cells_override=v.to_orig_rows(sums_cells, 0.0, np.float64))
        hit = {"d1": st["d1"], "d2": st["d2"]}
        _remember(key, hit)
    s_unique, mult = np.unique(sums_cells, return_counts=True)
    z1, z2 = _core.pearson_zero_terms(np.ascontiguousarray(s_unique, dtype=np.float64),
                                      mult.astype(np.float64), np.ascontiguousarray(sums_genes),
                                      float(sum_total), float(theta), float(clip))
    sum_res = z1 + hit["d1"]
    sum_res_sq = z2 + hit["d2"]
    mean_res = sum_res / n_obs
    residual_var = sum_res_sq / n_obs - mean_res ** 2
    return residual_var, gene_mean, gene_var


# ---------------------------------------------------------------------------
# scale (mean / var)
# ---------------------------------------------------------------------------
def chunked_mean_var(adata, *, layer=None, chunk_size=None, **kw):
    X = adata.layers[layer] if layer is not None else adata.X
    v = _view(adata, X)
    if v is None or v.scale_mean is not None:
        return _fallback("chunked_mean_var", adata, layer=layer,
                         **({} if chunk_size is None else {"chunk_size": chunk_size}), **kw)
    STATS["fast"] += 1
    n = v.n_rows
    nf = np.ones(n) if v.norm_factors is None else v.norm_factors
    key = _scale_key(v, nf, v.log1p) if v.norm_factors is not None else None
    hit = _recall(key) if key is not None else None
    if hit is None:
        st = _core.stats_pass(
            v.handle.src, v.row_keep, v.col_map(), np.zeros(v.handle.shape[1], np.uint8),
            np.zeros(v.n_cols), 1.0, v.log1p, True, False, 100.0, 0.0, 0.0,
            recip_override=v.recip())
        hit = {"s1": st["s1"], "s2": st["s2"], "n": n}
        if key is not None:
            _remember(key, hit)
    mean = hit["s1"] / n
    ssd = np.clip(hit["s2"] - n * mean * mean, 0.0, None)
    var = ssd / max(n - 1, 1)
    return mean, var


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def chunked_pca(adata, *, layer="scaled", n_comps=50, n_oversamples=10, n_power_iters=4,
                chunk_size=None, random_state=0, materialize_threshold_gb=16.0,
                use_implicit_centering=True, use_highly_variable=True, **kw):
    X = adata.layers[layer] if layer in adata.layers else adata.X
    v = _view(adata, X)
    if v is None or v.scale_mean is None:
        return _fallback("chunked_pca", adata, layer=layer, n_comps=n_comps,
                         n_oversamples=n_oversamples, n_power_iters=n_power_iters,
                         random_state=random_state,
                         materialize_threshold_gb=materialize_threshold_gb,
                         use_implicit_centering=use_implicit_centering,
                         use_highly_variable=use_highly_variable,
                         **({} if chunk_size is None else {"chunk_size": chunk_size}), **kw)
    STATS["fast"] += 1
    from anndataoom._chunked_ops import _resolve_hvg_mask
    hvg_idx = _resolve_hvg_mask(adata, use_highly_variable)
    sel = np.arange(v.n_cols) if hvg_idx is None else np.asarray(hvg_idx)
    n_obs, n_vars = v.n_rows, len(sel)
    mean = np.ascontiguousarray(v.scale_mean[sel], dtype=np.float64)
    std = np.ascontiguousarray(v.scale_std[sel], dtype=np.float64)
    clip = 0.0 if v.max_value is None else float(v.max_value)
    op = _core.extract_scaled(v.handle.src, v.row_keep, v.col_map(sel), mean, std,
                              v.recip(), v.log1p, clip)
    k = min(n_comps + n_oversamples, n_obs, n_vars)
    U, S, Vt = randomized_svd(op, k, n_oversamples=0, n_iter=n_power_iters,
                              random_state=random_state)
    n_comps_eff = min(n_comps, k)
    X_pca = (U[:, :n_comps_eff] * S[:n_comps_eff]).astype(np.float32)
    components = Vt[:n_comps_eff]
    variance_ratio = (S[:n_comps_eff] ** 2 / (S ** 2).sum()).astype(np.float64)
    return X_pca, components, variance_ratio


PATCHED = (
    "chunked_qc_metrics",
    "chunked_gene_group_pct",
    "chunked_identify_robust_genes",
    "chunked_normalize_total",
    "chunked_pearson_residual_variance",
    "chunked_mean_var",
    "chunked_pca",
)
