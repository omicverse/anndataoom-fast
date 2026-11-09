"""brisc 'basic' workflow (brisc-benchmarking/test_basic_brisc.py), timed steps only.

    python run_brisc.py DATA_NAME DATA_PATH NUM_THREADS OUT_CSV REP [BATCH_COLUMN]

Same steps, arguments and MemoryTimer (50 ms RSS polling of the process and its
children) as the brisc paper's Fig. 4; the excluded extra embeddings (PaCMAP,
UMAP, UMAP hogwild) and the output files are skipped.
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "BRISC_BENCH", "/scratch/users/steorra/analysis/brisc_probe/brisc-benchmarking"))
from brisc import SingleCell  # noqa: E402
from utils_local import MemoryTimer, system_info  # noqa: E402

DATA_NAME, DATA_PATH, NUM_THREADS, OUT_CSV, REP = sys.argv[1:6]
BATCH = sys.argv[6] if len(sys.argv) > 6 else "donor"
NUM_THREADS, REP = int(NUM_THREADS), int(REP)

if __name__ == "__main__":
    system_info()
    timers = MemoryTimer(silent=False, csv_path=OUT_CSV,
                         csv_columns={"library": "brisc", "test": "basic", "dataset": DATA_NAME,
                                      "num_threads": NUM_THREADS, "rep": REP})
    with timers("Load data"):
        data = SingleCell(DATA_PATH, num_threads=NUM_THREADS)
    with timers("Quality control"):
        data = data.qc(allow_float=True)
    with timers("Feature selection"):
        data = data.hvg(batch_column=BATCH)
    with timers("Normalization"):
        data = data.normalize()
    with timers("PCA"):
        data = data.pca()
    with timers("Nearest neighbors"):
        data = data.neighbors().shared_neighbors()
    with timers("Clustering"):
        data = data.cluster(resolution=[0.25, 0.5, 1.0, 1.5, 2.0])
    with timers("Embedding"):
        data = data.localmap()
    with timers("Find markers"):
        markers = data.find_markers("cell_type", all_genes=True)
    timers.shutdown()
    print("--- Completed successfully ---")
