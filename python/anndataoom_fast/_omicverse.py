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
class IVFKNNTransformer:
    """sklearn-style transformer returning the kNN distance graph.

    Output format matches ``pynndescent.PyNNDescentTransformer``: a CSR matrix
    with ``n_neighbors + 1`` stored entries per row (the cell itself at
    distance 0 first, then its neighbours by increasing distance).
    """

    def __init__(self, n_neighbors=15, metric="euclidean", random_state=0, n_jobs=None, **kwargs):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, "metric": self.metric,
                "random_state": self.random_state, "n_jobs": self.n_jobs}

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        from ._neighbors import knn
        X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)
        rs = self.random_state if isinstance(self.random_state, (int, np.integer)) else 0
        k = min(self.n_neighbors + 1, X.shape[0])
        idx, dist = knn(X, k, random_state=int(rs))
        n = X.shape[0]
        D = sp.csr_matrix((dist.ravel(), idx.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
        D.has_sorted_indices = False
        return D


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
                                      tqdm_kwds=None):
    if densmap or output_dens or not euclidean_output:
        return _ORIG["simplicial_set_embedding"](
            data, graph, n_components, initial_alpha, a, b, gamma, negative_sample_rate,
            n_epochs, init, random_state, metric, metric_kwds, densmap, densmap_kwds,
            output_dens, **({} if output_metric is None else {"output_metric": output_metric}),
            euclidean_output=euclidean_output, parallel=parallel, verbose=verbose)
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


def _group_stats_any(X, labels):
    """group_stats for in-memory CSR or anndataoom lazy matrices (cached)."""
    from ._markers import group_stats
    from ._view import resolve, digest
    key = (id(X), digest(np.asarray(labels).astype(str)))
    hit = _STATS_CACHE.get(key)
    if hit is not None:
        return hit
    v = None
    try:
        v = resolve(X)
    except Exception:
        v = None
    if v is not None and v.scale_mean is None:
        from . import _core
        groups, codes = np.unique(np.asarray(labels), return_inverse=True)
        sizes = np.bincount(codes, minlength=len(groups)).astype(np.int64)
        grp = v.to_orig_rows(codes.astype(np.int32), -1, np.int32)
        s, q, z = _core.group_pass(v.handle.src, grp, len(groups), v.col_map(), v.recip(), v.log1p)
        out = (groups, sizes, s, q, z)
    else:
        if not sp.issparse(X):
            X = sp.csr_matrix(np.asarray(X))
        out = group_stats(X, labels)
    _STATS_CACHE.clear()
    _STATS_CACHE[key] = out
    return out


def _patched_cosg(adata, groupby="CellTypes", groups="all", mu=1, remove_lowly_expressed=False,
                  expressed_pct=0.1, n_genes_user=50, key_added=None,
                  calculate_logfoldchanges=True, use_raw=True, layer=None, reference="rest",
                  copy=False):
    import pandas as pd
    if groups != "all" or reference != "rest":
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


def _patched_cosg_add_pts(adata, key_added, groupby, *, use_raw=True, layer=None):
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
    cosg_mod.cosg = _patched_cosg
    if _ORIG["cosg_add_pts"] is not None:
        markers_mod._cosg_add_pts = _patched_cosg_add_pts
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
