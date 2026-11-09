"""Run the omicverse standard workflow with backend='python' and backend='cpp'
and compare every output (keys, dtypes, values).

    python backend_parity.py COUNTS.h5ad OUTDIR [--oom] [--n-cells N] [--groupby cell_type]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

RES = [0.5, 1.0]


def workflow(path, backend, oom, n_cells, groupby):
    import omicverse as ov
    import omicverse.single  # noqa: F401
    ov.settings.cpu_init()
    ov.settings.backend = backend
    t = {}

    def step(name, fn):
        s = time.perf_counter()
        r = fn()
        t[name] = time.perf_counter() - s
        print(f"[{backend}] {name}: {t[name]:.2f}s", file=sys.stderr, flush=True)
        return r

    if oom:
        import anndataoom as oom_
        adata = step("load", lambda: oom_.read(path))
    else:
        import anndata as ad
        a = ad.read_h5ad(path)
        if n_cells and n_cells < a.n_obs:
            a = a[np.random.RandomState(0).choice(a.n_obs, n_cells, replace=False)].copy()
        adata = a
    out = step("qc", lambda: ov.pp.qc(adata, mode="seurat", doublets=False))
    if out is not None:
        adata = out
    step("preprocess", lambda: ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000))
    step("scale", lambda: ov.pp.scale(adata, max_value=10))
    step("pca", lambda: ov.pp.pca(adata, n_pcs=50, layer="scaled"))
    step("neighbors", lambda: ov.pp.neighbors(adata, n_neighbors=15, use_rep="scaled|original|X_pca"))
    step("leiden", lambda: [ov.pp.leiden(adata, resolution=r, key_added=f"leiden_{r}") for r in RES])
    step("umap", lambda: ov.pp.umap(adata))
    step("markers", lambda: ov.single.find_markers(adata, groupby=groupby, method="cosg", n_genes=50))
    return adata, t


def describe(x):
    if sp.issparse(x):
        return ("sparse", x.format, str(x.dtype), x.shape)
    if hasattr(x, "dtype") and hasattr(x, "shape"):
        return ("array", str(x.dtype), x.shape)
    return (type(x).__name__,)


def compare_arrays(a, b):
    if sp.issparse(a) or sp.issparse(b):
        a = sp.csr_matrix(a); b = sp.csr_matrix(b)
        a.sort_indices(); b.sort_indices()
        if a.shape != b.shape or a.nnz != b.nnz or not np.array_equal(a.indices, b.indices):
            return "structure differs"
        a, b = a.data, b.data
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if a.dtype.kind in "OUSb" or b.dtype.kind in "OUSb":
        nd = int((a.astype(str) != b.astype(str)).sum())
        return "identical" if nd == 0 else f"{nd} differ"
    a = a.astype(np.float64); b = b.astype(np.float64)
    both = np.isnan(a) & np.isnan(b)
    d = np.where(both, 0, np.abs(a - b))
    if np.all(d == 0):
        return "identical"
    rel = d / np.maximum(np.abs(a), 1e-12)
    return f"max|d|={np.nanmax(d):.3g} rel={np.nanmax(rel):.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("outdir")
    ap.add_argument("--oom", action="store_true")
    ap.add_argument("--n-cells", type=int, default=0)
    ap.add_argument("--groupby", default="cell_type")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    A, tA = workflow(a.path, "python", a.oom, a.n_cells, a.groupby)
    B, tB = workflow(a.path, "cpp", a.oom, a.n_cells, a.groupby)
    rows = []
    for slot in ("obs", "var"):
        da, db = getattr(A, slot), getattr(B, slot)
        for k in sorted(set(da.columns) | set(db.columns)):
            if k not in da or k not in db:
                rows.append((slot, k, "missing in " + ("cpp" if k not in db else "python")))
                continue
            dt = "" if str(da[k].dtype) == str(db[k].dtype) else f" dtype {da[k].dtype} vs {db[k].dtype}"
            rows.append((slot, k, compare_arrays(da[k].values, db[k].values) + dt))
    for slot in ("obsm", "varm", "obsp", "layers"):
        ma, mb = getattr(A, slot), getattr(B, slot)
        for k in sorted(set(ma.keys()) | set(mb.keys())):
            if k not in ma or k not in mb:
                rows.append((slot, k, "missing in " + ("cpp" if k not in mb else "python")))
                continue
            va, vb = ma[k], mb[k]
            if not hasattr(va, "shape") or not hasattr(vb, "shape"):
                rows.append((slot, k, f"{type(va).__name__} vs {type(vb).__name__}"))
                continue
            note = "" if describe(va) == describe(vb) else f" {describe(va)} vs {describe(vb)}"
            if k in ("X_pca", "scaled|original|X_pca", "X_umap"):
                va2, vb2 = np.asarray(va, float), np.asarray(vb, float)
                cos = np.abs((va2 * vb2).sum(0)) / (np.linalg.norm(va2, axis=0) * np.linalg.norm(vb2, axis=0))
                rows.append((slot, k, f"|cos| per column min={cos.min():.6f}" + note))
            else:
                rows.append((slot, k, compare_arrays(va, vb) + note))
    rows.append(("uns", "keys", "same" if set(A.uns.keys()) == set(B.uns.keys())
                 else f"python-only {set(A.uns)-set(B.uns)} cpp-only {set(B.uns)-set(A.uns)}"))
    for k in ("pca", "neighbors", "umap", "rank_genes_groups"):
        if k in A.uns and k in B.uns:
            ka, kb = set(A.uns[k].keys()), set(B.uns[k].keys())
            rows.append(("uns", k, "keys same" if ka == kb else f"keys {ka ^ kb}"))
    from sklearn.metrics import adjusted_rand_score
    for r in RES:
        rows.append(("leiden", str(r), f"ARI={adjusted_rand_score(A.obs[f'leiden_{r}'], B.obs[f'leiden_{r}']):.4f}"))
    R1, R2 = A.uns["rank_genes_groups"], B.uns["rank_genes_groups"]
    same_lists = np.mean([np.array_equal(np.asarray(R1['names'][g]).astype(str), np.asarray(R2['names'][g]).astype(str))
                          for g in R1['names'].columns])
    rows.append(("markers", "top-50 lists identical", f"{same_lists:.3f}"))
    df = pd.DataFrame(rows, columns=["slot", "key", "result"])
    pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 120)
    print(df.to_string(index=False))
    tag = "oom" if a.oom else "memory"
    df.to_csv(os.path.join(a.outdir, f"parity_{tag}.csv"), index=False)
    with open(os.path.join(a.outdir, f"timing_{tag}.json"), "w") as f:
        json.dump({"python": tA, "cpp": tB}, f, indent=1)
    print("timing python:", {k: round(v, 2) for k, v in tA.items()}, "total", round(sum(tA.values()), 1))
    print("timing cpp   :", {k: round(v, 2) for k, v in tB.items()}, "total", round(sum(tB.values()), 1))


if __name__ == "__main__":
    main()
