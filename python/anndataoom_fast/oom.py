"""C++-accelerated equivalents of ``anndataoom``'s chunked preprocessing functions.

Same names, signatures and side effects as ``anndataoom.chunked_*``; inputs
the C++ engine cannot read are passed to anndataoom's own implementation.
Use these directly (e.g. ``from anndataoom_fast import oom; oom.chunked_pca(...)``)
instead of patching anndataoom with :func:`anndataoom_fast.install`.
"""
from ._ops import (  # noqa: F401
    chunked_gene_group_pct,
    chunked_identify_robust_genes,
    chunked_mean_var,
    chunked_normalize_total,
    chunked_pca,
    chunked_pearson_residual_variance,
    chunked_qc_metrics,
)


def chunked_log1p(adata):
    from anndataoom import chunked_log1p as _f
    return _f(adata)


def chunked_scale(adata, **kwargs):
    """anndataoom.chunked_scale computing mean/var with the C++ pass."""
    from anndataoom import _chunked_ops
    orig = _chunked_ops.chunked_mean_var
    _chunked_ops.chunked_mean_var = chunked_mean_var
    try:
        return _chunked_ops.chunked_scale(adata, **kwargs)
    finally:
        _chunked_ops.chunked_mean_var = orig


def chunked_highly_variable_genes_pearson(adata, **kwargs):
    """anndataoom's Pearson-residual HVG with the C++ residual-variance pass."""
    from anndataoom import _chunked_ops
    orig = _chunked_ops.chunked_pearson_residual_variance
    _chunked_ops.chunked_pearson_residual_variance = chunked_pearson_residual_variance
    try:
        return _chunked_ops.chunked_highly_variable_genes_pearson(adata, **kwargs)
    finally:
        _chunked_ops.chunked_pearson_residual_variance = orig
