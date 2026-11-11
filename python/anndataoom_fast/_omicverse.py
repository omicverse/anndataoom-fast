"""Route omicverse's CPU workflow through anndataoom-fast kernels.

Patched (only for the inputs each kernel supports; everything else falls
through to the original code):

* kNN: the pynndescent transformer omicverse picks for >= 8192 cells is
  replaced by the IVF transformer (Euclidean metric only);
* UMAP connectivities: ``omicverse.pp._connectivity.umap`` (bit-identical);
* UMAP layout: ``umap.umap_.simplicial_set_embedding`` (euclidean output,
  no densMAP);
* COSG marker genes: ``omicverse.single._cosg.cosg`` and the pts helper of
  ``find_markers`` (in-memory CSR and anndataoom lazy matrices).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_ORIG: dict = {}


# ---------------------------------------------------------------------------
# kNN transformer
# ---------------------------------------------------------------------------
from ._neighbors import IVFKNNTransformer  # noqa: E402


def _patched_handle_transformer(self, method, transformer, *, knn, kwds):
    conn_method, tr, shortcut = _ORIG["handle_transformer"](self, method, transformer, knn=knn, kwds=kwds)
    name = type(tr).__name__
    if name == "PyNNDescentTransformer" and kwds.get("metric", "euclidean") == "euclidean":
        tr = IVFKNNTransformer(n_neighbors=kwds["n_neighbors"], metric="euclidean",
                               random_state=kwds.get("random_state", 0))
    return conn_method, tr, shortcut


def _patched_connectivity_umap(knn_indices, knn_dists, *, n_obs, n_neighbors,
                               set_op_mix_ratio=1.0, local_connectivity=1.0):
    from ._neighbors import fuzzy_simplicial_set
    return fuzzy_simplicial_set(knn_indices, knn_dists, n_neighbors,
                                set_op_mix_ratio=set_op_mix_ratio,
                                local_connectivity=local_connectivity).tocsr()


# ---------------------------------------------------------------------------
# UMAP layout
# ---------------------------------------------------------------------------
def _patched_simplicial_set_embedding(data, graph, n_components, initial_alpha, a, b, gamma,
                                      negative_sample_rate, n_epochs, init, random_state, metric,
                                      metric_kwds, densmap, densmap_kwds, output_dens,
                                      output_metric=None, output_metric_kwds=None,
                                      euclidean_output=True, parallel=False, verbose=False,
                                      tqdm_kwds=None, **kwargs):
    # umap-learn >= 0.6 adds layout options; anything but its legacy (0.5.x)
    # layout goes back to umap-learn.
    new_layout = (kwargs.get("compatibility_layout") is False
                  or kwargs.get("optimizer") not in (None, "compatibility"))
    if densmap or output_dens or not euclidean_output or new_layout:
        return _ORIG["simplicial_set_embedding"](
            data, graph, n_components, initial_alpha, a, b, gamma, negative_sample_rate,
            n_epochs, init, random_state, metric, metric_kwds, densmap, densmap_kwds,
            output_dens, **({} if output_metric is None else {"output_metric": output_metric}),
            euclidean_output=euclidean_output, parallel=parallel, verbose=verbose,
            **({} if tqdm_kwds is None else {"tqdm_kwds": tqdm_kwds}), **kwargs)
    from ._umap import simplicial_set_embedding
    emb = simplicial_set_embedding(graph, n_components, a=a, b=b, n_epochs=n_epochs, init=init,
                                   random_state=random_state, gamma=gamma,
                                   initial_alpha=initial_alpha,
                                   negative_sample_rate=negative_sample_rate, parallel=parallel,
                                   data=data)
    return emb, {}


# ---------------------------------------------------------------------------
# COSG
# ---------------------------------------------------------------------------
_STATS_CACHE: dict = {}


def _expression_matrix(adata, use_raw, layer):
    if layer is not None:
        return adata.layers[layer], adata.var_names
    if use_raw and getattr(adata, "raw", None) is not None:
        return adata.raw.X, adata.raw.var_names
    return adata.X, adata.var_names


def _to_host(X):
    """A CPU copy of a CuPy / cupyx.scipy.sparse matrix (omicverse's GPU mode
    keeps .X on the device); other inputs are returned unchanged."""
    mod = type(X).__module__ or ""
    if mod.startswith(("cupy", "cupyx")) and hasattr(X, "get"):
        return X.get()
    return X


def _is_dask(X):
    return type(X).__module__.split(".")[0] == "dask" and hasattr(X, "to_delayed")


_GROUP_KERNEL_SRC = r"""
template <typename T, typename I>
__global__ void group_rows(const I* indptr, const int* indices, const T* data, long long n_rows,
                           const long long* codes, long long n_genes,
                           double* gs, double* gq, double* gz) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    long long base = codes[row] * n_genes;
    for (I i = indptr[row]; i < indptr[row + 1]; ++i) {
        double v = (double)data[i];
        long long j = base + indices[i];
        atomicAdd(gs + j, v);
        atomicAdd(gq + j, v * v);
        if (v != 0) atomicAdd(gz + j, 1.0);
    }
}
"""
_GROUP_KERNELS: dict = {}


def _group_block_cuda(blk, codes, n_groups):
    """GPU per-group statistics of a cupyx CSR block: one thread per row."""
    import cupy as cp
    blk = blk.tocsr()
    if blk.data.dtype not in (cp.float32, cp.float64):
        blk = blk.astype(cp.float32)
    indices = blk.indices.astype(cp.int32, copy=False)
    T = "float" if blk.data.dtype == cp.float32 else "double"
    I = "int" if blk.indptr.dtype == cp.int32 else "long long"
    name = f"group_rows<{T}, {I}>"
    if not _GROUP_KERNELS:
        exprs = [f"group_rows<{t}, {i}>" for t in ("float", "double") for i in ("int", "long long")]
        _GROUP_KERNELS["module"] = cp.RawModule(code=_GROUP_KERNEL_SRC, options=("-std=c++14",),
                                                name_expressions=exprs)
    k = _GROUP_KERNELS.get(name) or _GROUP_KERNELS.setdefault(
        name, _GROUP_KERNELS["module"].get_function(name))
    n, G = blk.shape
    out = [cp.zeros((n_groups, G)) for _ in range(3)]
    threads = 128
    k(((n + threads - 1) // threads,), (threads,),
      (blk.indptr, indices, blk.data, cp.int64(n), cp.asarray(codes, dtype=cp.int64), cp.int64(G),
       *out))
    return [o.get() for o in out]


def _group_block(blk, codes, n_groups):
    """Per-group sums, sums of squares and stored-entry counts of one row
    block (cupyx / scipy CSR or dense), float64."""
    if type(blk).__module__.startswith("cupyx"):
        return _group_block_cuda(blk, codes, n_groups)
    if type(blk).__module__.startswith(("cupy", "cupyx")):
        import cupy as xp
        import cupyx.scipy.sparse as sps
    else:
        xp, sps = np, sp
    codes = xp.asarray(codes)
    n = codes.shape[0]
    onehot = sps.csr_matrix((xp.ones(n), (codes, xp.arange(n))), shape=(n_groups, n))
    if sps.issparse(blk):
        blk = blk.tocsr()
        d = blk.data.astype(xp.float64)
        mk = lambda v: sps.csr_matrix((v, blk.indices, blk.indptr), shape=blk.shape)  # noqa: E731
        out = [onehot @ mk(v) for v in (d, d * d, (d != 0).astype(xp.float64))]
        out = [o.toarray() for o in out]
    else:
        D = xp.asarray(blk, dtype=xp.float64)
        out = [onehot @ D, onehot @ (D * D), onehot @ (D != 0).astype(xp.float64)]
    return [o.get() if hasattr(o, "get") and xp is not np else np.asarray(o) for o in out]


def _group_stats_dask(X, labels):
    """group_stats of a Dask matrix of row blocks: one pass, any device."""
    import dask
    groups, codes = np.unique(np.asarray(labels), return_inverse=True)
    sizes = np.bincount(codes, minlength=len(groups)).astype(np.int64)
    if X.numblocks[1] != 1:
        X = X.rechunk({1: -1})
    tasks, r = [], 0
    for b, c in zip(X.to_delayed(optimize_graph=False).ravel(), X.chunks[0]):
        tasks.append(dask.delayed(_group_block)(b, codes[r:r + c].astype(np.int64), len(groups)))
        r += c
    try:  # a distributed client schedules itself; otherwise bound the threads
        from distributed import default_client
        default_client()
        kw = {}
    except (ImportError, ValueError):
        import os
        kw = {"scheduler": "threads", "num_workers": int(os.environ.get("OV_DASK_WORKERS", "8"))}
    parts = dask.compute(*tasks, **kw)
    s, q, z = (sum(p[i] for p in parts) for i in range(3))
    return groups, sizes, s, q, z


def _group_stats_any(X, labels):
    """group_stats for in-memory CSR, anndataoom lazy and Dask matrices (cached)."""
    from ._markers import group_stats
    from ._view import resolve, digest
    key = (id(X), digest(np.asarray(labels).astype(str)))
    hit = _STATS_CACHE.get(key)
    if hit is not None:
        return hit
    v = None
    try:
        v = None if _is_dask(X) else resolve(X)
    except Exception:
        v = None
    if _is_dask(X):
        out = _group_stats_dask(X, labels)
    elif v is not None and v.scale_mean is None:
        from . import _core
        groups, codes = np.unique(np.asarray(labels), return_inverse=True)
        sizes = np.bincount(codes, minlength=len(groups)).astype(np.int64)
        grp = v.to_orig_rows(codes.astype(np.int32), -1, np.int32)
        s, q, z = _core.group_pass(v.handle.src, grp, len(groups), v.col_map(), v.recip(), v.log1p)
        out = (groups, sizes, s, q, z)
    else:
        X = _to_host(X)
        if not sp.issparse(X):
            X = sp.csr_matrix(np.asarray(X))
        out = group_stats(X, labels)
    _STATS_CACHE.clear()
    _STATS_CACHE[key] = out
    return out


def ov_cosg(adata, groupby="CellTypes", groups="all", mu=1, remove_lowly_expressed=False,
                  expressed_pct=0.1, n_genes_user=50, key_added=None,
                  calculate_logfoldchanges=True, use_raw=True, layer=None, reference="rest",
                  copy=False):
    import pandas as pd
    if groups != "all" or reference != "rest":
        if "cosg" not in _ORIG:
            from omicverse.single._cosg import cosg as _c
            _ORIG["cosg"] = _c
        return _ORIG["cosg"](adata, groupby=groupby, groups=groups, mu=mu,
                             remove_lowly_expressed=remove_lowly_expressed,
                             expressed_pct=expressed_pct, n_genes_user=n_genes_user,
                             key_added=key_added, calculate_logfoldchanges=calculate_logfoldchanges,
                             use_raw=use_raw, layer=layer, reference=reference, copy=copy)
    adata = adata.copy() if copy else adata
    if layer is not None and use_raw:
        raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
    key_added = key_added or "rank_genes_groups"
    adata.uns[key_added] = {"params": dict(groupby=groupby, reference=reference, groups=groups,
                                           method="cosg", use_raw=use_raw, layer=layer)}
    labels = adata.obs[groupby]
    # omicverse scores on adata.X (or the layer) and takes log-fold changes from
    # raw when use_raw and raw exist; mirror that exactly.
    X_scores = adata.layers[layer] if layer is not None else adata.X
    groups_, sizes, S, Q, Z = _group_stats_any(X_scores, labels.values)
    n = int(sizes.sum())
    norm_g = np.sqrt(Q.sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (S / np.sqrt(sizes)[:, None]) / norm_g[None, :]
    cos = np.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0).T
    nz = cos != 0
    e2 = cos * cos
    e2_sum = e2.sum(axis=1, keepdims=True)
    lam = np.zeros_like(e2)
    if mu == 1:
        lam[nz] = (e2 / e2_sum)[nz]
    else:
        lam[nz] = (e2 / ((1 - mu) * e2 + mu * e2_sum))[nz]
    lam[nz] = lam[nz] * cos[nz]
    if calculate_logfoldchanges:
        Xl, _ = _expression_matrix(adata, use_raw, layer)
        g2, sz2, S2, _, _ = _group_stats_any(Xl, labels.values) if Xl is not X_scores else (groups_, sizes, S, Q, Z)
        if "log1p" in adata.uns_keys() and adata.uns["log1p"].get("base") is not None:
            base = adata.uns["log1p"]["base"]
            expm1 = lambda x: np.expm1(x * np.log(base))  # noqa: E731
        else:
            expm1 = np.expm1
        tot2 = S2.sum(axis=0)
    from ._markers import _select_top_n
    rank_stats = None
    n_top = min(int(n_genes_user), lam.shape[0])
    for ci, g in enumerate(groups_):
        scores = lam[:, ci].copy()
        if remove_lowly_expressed:
            scores[Z[ci] < sizes[ci] * expressed_pct] = -1
        top = _select_top_n(scores, n_top)
        if rank_stats is None:
            rank_stats = pd.DataFrame(columns=pd.MultiIndex.from_tuples([(g, "names")]))
        rank_stats[g, "names"] = adata.var_names[top]
        rank_stats[g, "scores"] = scores[top]
        if calculate_logfoldchanges:
            mean_g = S2[ci] / sz2[ci]
            mean_r = (tot2 - S2[ci]) / max(int(sz2.sum()) - sz2[ci], 1)
            fc = (expm1(mean_g) + 1e-9) / (expm1(mean_r) + 1e-9)
            rank_stats[g, "logfoldchanges"] = np.log2(fc[top])
    dtypes = {"names": "O", "scores": "float32", "logfoldchanges": "float32"}
    rank_stats.columns = rank_stats.columns.swaplevel()
    for col in rank_stats.columns.levels[0]:
        adata.uns[key_added][col] = pd.DataFrame(rank_stats[col].to_records(index=False,
                                                                         column_dtypes=dtypes[col]))
    p_shape = adata.uns[key_added]["scores"].shape
    adata.uns[key_added]["pvals"] = np.zeros(p_shape, dtype="float32")
    adata.uns[key_added]["pvals_adj"] = np.zeros(p_shape, dtype="float32")
    if key_added != "cosg":
        adata.uns["cosg"] = adata.uns.get(key_added, adata.uns.get("cosg", {}))
    adata.uns.setdefault("cosg_markers", adata.uns.get(key_added, {}))
    print("**finished identifying marker genes by COSG**")
    try:
        from omicverse.single._cosg import add_reference
        add_reference(adata, "COSG", "marker gene identification with COSG")
    except Exception:
        pass
    return adata if copy else None


def ov_cosg_add_pts(adata, key_added, groupby, *, use_raw=True, layer=None):
    import pandas as pd
    result = adata.uns[key_added]
    names = result.get("names")
    if names is None:
        return
    group_names = names.columns.tolist() if isinstance(names, pd.DataFrame) else list(names.dtype.names)
    X, var_names = _expression_matrix(adata, use_raw, layer)
    labels = adata.obs[groupby].astype(str).values
    groups_, sizes, _, _, Z = _group_stats_any(X, labels)
    gi = {str(g): i for i, g in enumerate(groups_)}
    n = int(sizes.sum())
    zt = Z.sum(axis=0)
    pts = {g: Z[gi[g]] / sizes[gi[g]] for g in group_names}
    rest = {g: (zt - Z[gi[g]]) / max(n - sizes[gi[g]], 1) for g in group_names}
    result["pts"] = pd.DataFrame(pts, index=var_names)
    result["pts_rest"] = pd.DataFrame(rest, index=var_names)


# ---------------------------------------------------------------------------
def install() -> None:
    import umap.umap_ as umap_mod
    import omicverse.pp._connectivity as conn_mod
    import omicverse.pp._neighbors as nb_mod
    import omicverse.single._cosg as cosg_mod
    import omicverse.single._markers as markers_mod
    cls = next(c for c in vars(nb_mod).values()
               if isinstance(c, type) and hasattr(c, "_handle_transformer"))
    if "handle_transformer" not in _ORIG:
        _ORIG["handle_transformer"] = cls._handle_transformer
        _ORIG["connectivity_umap"] = conn_mod.umap
        _ORIG["simplicial_set_embedding"] = umap_mod.simplicial_set_embedding
        _ORIG["cosg"] = cosg_mod.cosg
        _ORIG["cosg_add_pts"] = getattr(markers_mod, "_cosg_add_pts", None)
        _ORIG["cls"] = cls
    cls._handle_transformer = _patched_handle_transformer
    conn_mod.umap = _patched_connectivity_umap
    umap_mod.simplicial_set_embedding = _patched_simplicial_set_embedding
    cosg_mod.cosg = ov_cosg
    if _ORIG["cosg_add_pts"] is not None:
        markers_mod._cosg_add_pts = ov_cosg_add_pts
    try:
        from . import _leiden_omicverse
        _leiden_omicverse.install()
    except ImportError:
        pass


def uninstall() -> None:
    if not _ORIG:
        return
    import umap.umap_ as umap_mod
    import omicverse.pp._connectivity as conn_mod
    import omicverse.single._cosg as cosg_mod
    import omicverse.single._markers as markers_mod
    _ORIG["cls"]._handle_transformer = _ORIG["handle_transformer"]
    conn_mod.umap = _ORIG["connectivity_umap"]
    umap_mod.simplicial_set_embedding = _ORIG["simplicial_set_embedding"]
    cosg_mod.cosg = _ORIG["cosg"]
    if _ORIG["cosg_add_pts"] is not None:
        markers_mod._cosg_add_pts = _ORIG["cosg_add_pts"]
