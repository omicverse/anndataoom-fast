"""omicverse standard single-cell workflow on an out-of-core AnnData, timed per step.

    python ov_standard_workflow.py INPUT.h5ad OUT_PREFIX --impl {orig,fast} [--threads N]

Steps (Fig. 4 rows of the brisc paper, omicverse's own algorithms):
  Load data       anndataoom.read
  Quality control ov.pp.qc(mode='seurat', doublets=False)
  Normalization + Feature selection
                  ov.pp.preprocess(mode='shiftlog|pearson', n_HVGs=2000)
  PCA             ov.pp.scale(max_value=10) + ov.pp.pca(n_pcs=50, layer='scaled')
  Nearest neighbors ov.pp.neighbors(n_neighbors=15, use_rep='scaled|original|X_pca')
  Clustering      ov.pp.leiden at resolutions 0.25, 0.5, 1, 1.5, 2
  Embedding       ov.pp.umap
  Find markers    ov.single.find_markers(groupby='cell_type', method='cosg')
Per-step wall time and peak RSS are written to OUT_PREFIX.json, outputs to OUT_PREFIX.npz.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

RESOLUTIONS = [0.25, 0.5, 1.0, 1.5, 2.0]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--impl", choices=["orig", "fast"], required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--groupby", default="cell_type")
    a = ap.parse_args()

    if a.impl == "fast":
        import anndataoom_fast
        anndataoom_fast.install(omicverse=True)
        if a.threads:
            anndataoom_fast.set_num_threads(a.threads)
    import anndataoom as oom
    import omicverse as ov
    import omicverse.single  # noqa: F401
    ov.settings.cpu_init()

    steps = {}

    def step(name, fn):
        t = time.perf_counter()
        r = fn()
        steps[name] = {"seconds": time.perf_counter() - t, "peak_rss_gb": rss_gb()}
        print(f"[{a.impl}] {name}: {steps[name]['seconds']:.2f}s", file=sys.stderr, flush=True)
        return r

    adata = step("Load data", lambda: oom.read(a.input))
    step("Quality control", lambda: ov.pp.qc(adata, mode="seurat", doublets=False))
    step("Normalization + feature selection",
         lambda: ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000, batch_key=None))

    def pca():
        ov.pp.scale(adata, max_value=10, use_implicit_centering=False)
        ov.pp.pca(adata, n_pcs=50, layer="scaled")
    step("PCA", pca)
    step("Nearest neighbors",
         lambda: ov.pp.neighbors(adata, n_neighbors=15, use_rep="scaled|original|X_pca"))

    def cluster():
        for r in RESOLUTIONS:
            ov.pp.leiden(adata, resolution=r, key_added=f"leiden_{r}")
    step("Clustering", cluster)
    step("Embedding", lambda: ov.pp.umap(adata))
    step("Find markers",
         lambda: ov.single.find_markers(adata, groupby=a.groupby, method="cosg", n_genes=50))

    out = {"X_pca": np.asarray(adata.obsm["X_pca"]), "X_umap": np.asarray(adata.obsm["X_umap"]),
           "obs_names": np.asarray(adata.obs_names, dtype=str)}
    for r in RESOLUTIONS:
        out[f"leiden_{r}"] = np.asarray(adata.obs[f"leiden_{r}"].astype(int))
    conn = adata.obsp["connectivities"]
    out["conn_indptr"], out["conn_indices"], out["conn_data"] = conn.indptr, conn.indices, conn.data
    np.savez(a.out + ".npz", **out)
    meta = {"impl": a.impl, "input": os.path.abspath(a.input), "steps": steps,
            "total_seconds": sum(v["seconds"] for v in steps.values()),
            "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars)}
    if a.impl == "fast":
        import anndataoom_fast
        meta["threads"] = anndataoom_fast.get_num_threads()
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
