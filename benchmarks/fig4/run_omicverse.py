"""omicverse standard workflow on an out-of-core AnnData, timed like run_brisc.py.

    python run_omicverse.py DATA_NAME DATA_PATH NUM_THREADS OUT_CSV REP --backend {cpp,python}

Uses brisc-benchmarking's MemoryTimer so time and memory are measured exactly
as for brisc.  Steps follow omicverse's documented workflow:
  qc(seurat) -> preprocess(shiftlog|pearson, 2000 HVGs) -> scale -> pca(50)
  -> neighbors(15) -> leiden at 5 resolutions -> localmap (cpp) and umap
  -> find_markers(cosg).
"""
import argparse
import os
import sys

sys.path.insert(0, os.environ.get(
    "BRISC_BENCH", "/scratch/users/steorra/analysis/brisc_probe/brisc-benchmarking"))
from utils_local import MemoryTimer, system_info  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data_name")
ap.add_argument("data_path")
ap.add_argument("num_threads", type=int)
ap.add_argument("out_csv")
ap.add_argument("rep", type=int)
ap.add_argument("--backend", choices=["cpp", "python"], required=True)
ap.add_argument("--mode", choices=["cpu", "mixed", "gpu"], default="cpu",
                help="omicverse settings mode: cpu; mixed = CPU preprocessing (out of core) "
                     "with PCA/neighbors/UMAP/Leiden on the GPU; gpu = RAPIDS, data in GPU memory")
ap.add_argument("--groupby", default="cell_type")
a = ap.parse_args()

if __name__ == "__main__":
    system_info()
    import anndataoom as oom
    import omicverse as ov
    import omicverse.single  # noqa: F401
    if a.mode == "gpu":
        ov.settings.gpu_init()
    elif a.mode == "mixed":
        ov.settings.cpu_gpu_mixed_init()
    else:
        ov.settings.cpu_init()
    ov.settings.backend = a.backend
    label = f"omicverse-{a.backend}" if a.mode == "cpu" else f"omicverse-{a.mode}"
    if a.backend == "cpp":
        import anndataoom_fast
        anndataoom_fast.set_num_threads(a.num_threads)
    timers = MemoryTimer(silent=False, csv_path=a.out_csv,
                         csv_columns={"library": label, "test": "basic",
                                      "dataset": a.data_name, "num_threads": a.num_threads,
                                      "rep": a.rep})
    with timers("Load data"):
        if a.mode == "gpu":
            # RAPIDS needs the whole matrix in (GPU) memory.
            import anndata
            adata = anndata.read_h5ad(a.data_path)
            ov.pp.anndata_to_GPU(adata)
        else:
            adata = oom.read(a.data_path)
    with timers("Quality control"):
        out = ov.pp.qc(adata, mode="seurat", doublets=False)
        adata = adata if out is None else out
    with timers("Normalization + feature selection"):
        ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000, batch_key=None)
    with timers("PCA"):
        ov.pp.scale(adata, max_value=10)
        ov.pp.pca(adata, n_pcs=50, layer="scaled")
    with timers("Nearest neighbors"):
        ov.pp.neighbors(adata, n_neighbors=15, use_rep="scaled|original|X_pca")
    with timers("Clustering"):
        # cpp: brisc's Leiden (default flavor).  python: leidenalg with
        # n_iterations=2, scanpy's announced default (-1 takes hours here).
        # One call with five resolutions in every mode (parallel with
        # backend='cpp'; the GPU modes run them one after another).
        kw = {"n_iterations": 2} if a.mode == "cpu" else {}
        ov.pp.leiden(adata, resolution=[0.25, 0.5, 1.0, 1.5, 2.0], key_added="leiden", **kw)
    if a.backend == "cpp" and a.mode != "gpu":
        with timers("Embedding (LocalMAP)"):
            ov.pp.localmap(adata, use_rep="scaled|original|X_pca")
    with timers("Embedding (UMAP)"):
        ov.pp.umap(adata)
    with timers("Find markers"):
        ov.single.find_markers(adata, groupby=a.groupby, method="cosg", n_genes=50)
    timers.shutdown()
    print("--- Completed successfully ---")
