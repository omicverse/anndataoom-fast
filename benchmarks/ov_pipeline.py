"""Run the anndataoom benchmark pipeline (config ``ov-oom``) and save outputs.

    python ov_pipeline.py INPUT.h5ad OUT_PREFIX --impl {orig,fast} [--threads N]

Stages and arguments follow anndata-oom/benchmark/scripts/bench.py:
    oom.read -> ov.pp.qc(mode='seurat', doublets=False)
             -> ov.pp.preprocess(mode='shiftlog|pearson', n_HVGs=2000)
             -> ov.pp.scale(max_value=10) -> ov.pp.pca(n_pcs=50, layer='scaled')
Writes OUT_PREFIX.npz (every obs/var column, PCA outputs) and OUT_PREFIX.json
(per-stage wall time and peak RSS).
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--impl", choices=["orig", "fast"], required=True)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    if a.impl == "fast":
        import anndataoom_fast
        anndataoom_fast.install()
        if a.threads:
            anndataoom_fast.set_num_threads(a.threads)
    import anndataoom as oom
    import omicverse as ov
    ov.settings.cpu_init()

    times = {}

    def stage(name, fn):
        t = time.perf_counter()
        out = fn()
        times[name] = {"seconds": time.perf_counter() - t, "peak_rss_mb": rss_mb()}
        print(f"[{a.impl}] {name}: {times[name]['seconds']:.2f}s", file=sys.stderr, flush=True)
        return out

    adata = stage("load", lambda: oom.read(a.input))
    stage("qc", lambda: ov.pp.qc(adata, mode="seurat", doublets=False))
    stage("preprocess", lambda: ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000,
                                                 batch_key=None))
    stage("scale", lambda: ov.pp.scale(adata, max_value=10, use_implicit_centering=False))
    stage("pca", lambda: ov.pp.pca(adata, n_pcs=50, layer="scaled"))

    out = {}
    for c in adata.obs.columns:
        col = adata.obs[c]
        if col.dtype.kind in "biuf":
            out[f"obs/{c}"] = col.values
    for c in adata.var.columns:
        col = adata.var[c]
        if col.dtype.kind in "biuf":
            out[f"var/{c}"] = col.values
    out["obs_names"] = np.asarray(adata.obs_names, dtype=str)
    out["var_names"] = np.asarray(adata.var_names, dtype=str)
    out["X_pca"] = np.asarray(adata.obsm["X_pca"])
    out["PCs"] = np.asarray(adata.varm["PCs"])
    out["variance_ratio"] = np.asarray(adata.uns["pca"]["variance_ratio"])
    pre = adata.uns.get("_pearson_precompute", {})
    for k in ("sums_cells", "sums_genes", "sq_sums_genes"):
        if k in pre:
            out[f"pre/{k}"] = np.asarray(pre[k])
    sc = adata.layers["scaled"]
    out["scale_mean"] = np.asarray(sc._scale_mean)
    out["scale_std"] = np.asarray(sc._scale_std)
    np.savez(a.out + ".npz", **out)
    meta = {"impl": a.impl, "input": os.path.abspath(a.input), "stages": times,
            "total_seconds": sum(v["seconds"] for v in times.values())}
    if a.impl == "fast":
        import anndataoom_fast
        meta["threads"] = anndataoom_fast.get_num_threads()
        meta["fast_stats"] = dict(anndataoom_fast.STATS)
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
