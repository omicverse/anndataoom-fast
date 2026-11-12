"""Headline figure for the READMEs: total runtime and peak memory of the
standard Fig. 4 workflow on SEA-AD (1.38M cells) for omicverse, brisc, Scanpy
and Seurat + BPCells.

    python plot_headline.py RESULTS_DIR OUT

omicverse and brisc: our runs on the same AMD EPYC 9754 node, each at its
fastest thread count (RESULTS_DIR/cpu128/SEAAD_{ovcpp,brisc}_t*.csv).
Scanpy and Seurat + BPCells: the numbers reported in the brisc paper
(RESULTS_DIR/brisc_paper/seaad_paper_baselines.csv, 192-core node)."""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

from plot_fig4 import fmt_time

# validated categorical order (validate_palette.js, light surface, all pairs)
ROWS = [("omicverse", "#7058be"), ("brisc", "#e16032"), ("Scanpy", "#269d69"),
        ("Seurat + BPCells", "#4a9fd0")]
EMBED_UMAP = "Embedding (UMAP)"


def fastest(results, tag):
    """(total s, peak GB, threads) of the fastest run; LocalMAP is the embedding."""
    best = None
    for f in glob.glob(os.path.join(results, "cpu128", f"SEAAD_{tag}_t*.csv")):
        d = pd.read_csv(f)
        d = d[d.operation != EMBED_UMAP]
        r = (d.duration.sum(), d.memory.max(), int(d.num_threads.iloc[0]))
        best = r if best is None or r[0] < best[0] else best
    return best


def paper(results, lib):
    d = pd.read_csv(os.path.join(results, "brisc_paper", "seaad_paper_baselines.csv"))
    d = d[d.library == f"{lib} (paper)"]
    return d.duration.sum(), d.memory.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    a = ap.parse_args()

    ov, br = fastest(a.results, "ovcpp"), fastest(a.results, "brisc")
    vals = {"omicverse": ov[:2], "brisc": br[:2],
            "Scanpy": paper(a.results, "Scanpy"),
            "Seurat + BPCells": paper(a.results, "Seurat + BPCells")}
    sub = {"omicverse": f"2.4.2, {ov[2]} threads, storage='disk'",
           "brisc": f"{br[2]} threads, whole matrix in RAM",
           "Scanpy": "from the brisc paper (192-core node)",
           "Seurat + BPCells": "from the brisc paper (192-core node)"}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, (axt, axm) = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True,
                                   gridspec_kw={"wspace": 0.08})
    y = range(len(ROWS))
    tmax = max(v[0] for v in vals.values())
    mmax = max(v[1] for v in vals.values())
    for i, (lib, col) in enumerate(ROWS):
        t, m = vals[lib]
        axt.barh(i, t, color=col, height=0.62)
        axt.text(t * 1.12, i, f"{t:.1f} s" if t < 100 else fmt_time(t), va="center", fontsize=8.5, color="#222222")
        axm.barh(i, m, color=col, height=0.62)
        axm.text(m + mmax * 0.02, i, f"{m:.1f} GB" if m < 20 else f"{m:.0f} GB", va="center", fontsize=8.5, color="#222222")
    axt.set_xscale("log")
    axt.set_xlim(10, tmax * 6)
    axm.set_xlim(0, mmax * 1.18)
    axt.set_yticks(list(y))
    axt.set_yticklabels([f"{lib}\n{sub[lib]}" for lib, _ in ROWS], fontsize=8)
    for t in axt.get_yticklabels():
        t.set_linespacing(1.3)
    axt.set_ylim(len(ROWS) - 0.5, -0.5)
    for ax, title, xl in ((axt, "Total runtime (lower is better)", "seconds, log scale"),
                          (axm, "Peak memory (lower is better)", "GB")):
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel(xl, fontsize=8, color="#555555")
        ax.tick_params(axis="x", labelsize=7.5, colors="#555555")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
    fig.text(0.0, -0.06,
             "SEA-AD MTG, 1.38M cells × 36,601 genes; brisc Fig. 4 workflow (load, QC, normalize + HVG, "
             "PCA, kNN, Leiden ×5, embedding, markers).\n"
             "omicverse and brisc: same AMD EPYC 9754 node (128 cores), fastest of 32/64/128 threads, "
             "LocalMAP embedding. Scanpy and Seurat + BPCells:\n"
             "values reported in the brisc paper on a 192-core node, UMAP embedding (not rerun here).",
             fontsize=7, color="#555555", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
