"""Headline figure for the READMEs (light and dark): total runtime and peak memory of the
standard Fig. 4 workflow on SEA-AD (1.38M cells), CPU tools above GPU tools.

    python plot_headline.py RESULTS_DIR OUT   # writes OUT_light.png, OUT_dark.png

omicverse and brisc: our runs on the same AMD EPYC 9754 node, each at its
fastest thread count (RESULTS_DIR/cpu128/SEAAD_{ovcpp,brisc}_t*.csv).
Scanpy, Seurat + BPCells and rapids-singlecell: the numbers reported in the
brisc paper (RESULTS_DIR/brisc_paper/seaad_paper_baselines.csv; 192-core node,
rapids-singlecell on a 4x H100 node, no host-memory figure).
omicverse[gpu] / omicverse[mixed]: our runs on one H100 (RESULTS_DIR/gpu_h100/,
UMAP embedding like rapids-singlecell; memory is host RAM).
Seurat v6: our run of Seurat 6.0.0 alone (in-memory layers, setThreads(64),
RESULTS_DIR/seurat6/); RunUMAP segfaulted, so its runtime is a lower bound
(without our own h5ad loader)."""
import math
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

# validated categorical order (validate_palette.js, light and dark surfaces, all pairs)
# (one colour per tool; its variants share it)
CPU = [("omicverse[cpu]", "#7058be"), ("brisc", "#e16032"), ("Scanpy", "#269d69"),
       ("Seurat + BPCells", "#4698cc"), ("Seurat v6", "#4698cc")]
GPU = [("omicverse[gpu]", "#7058be"), ("omicverse[mixed]", "#7058be"),
       ("rapids-singlecell", "#c2185b")]
GROUPS = {"CPU": "CPU · 64 threads", "GPU": "GPU · 1× H100 + 16 threads"}
PAPER = {"Scanpy", "Seurat + BPCells", "rapids-singlecell"}  # numbers from the brisc paper
GAP = 0.6  # extra space between the CPU and GPU groups
HEAD = 0.8  # space for a group header above each group
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


def gpu_run(results, tag):
    """One-H100 omicverse run with the UMAP embedding (as rapids-singlecell)."""
    d = pd.read_csv(os.path.join(results, "gpu_h100", f"SEAAD_{tag}_t16.csv"))
    d = d[d.operation != "Embedding (LocalMAP)"]
    return d.duration.sum(), d.memory.max(), False


def paper(results, lib):
    d = pd.read_csv(os.path.join(results, "brisc_paper", "seaad_paper_baselines.csv"))
    d = d[d.library == f"{lib} (paper)"]
    return d.duration.sum(), d.memory.max(), False


def seurat6(results):
    """Seurat 6.0.0 run that crashed in RunUMAP: time up to the crash, without
    our h5ad loader (Seurat has none)."""
    d = pd.read_csv(os.path.join(results, "seurat6", "SEAAD_seurat6native_t64.csv"))
    d = d[d.operation != "Load data"]
    return d.duration.sum(), d.memory.max(), bool(d.aborted.any())


def fmt_t(t, failed):
    s = f"{t:.0f} s" if t < 100 else f"{t / 60:.1f} min" if t < 3600 else f"{t / 3600:.1f} h"
    return f"> {s} · crashed" if failed else s


THEMES = {"light": {"bg": "#ffffff", "ink": "#1f2328", "muted": "#8c959f"},
          "dark": {"bg": "#0d1117", "ink": "#e6edf3", "muted": "#7d8590"}}


def draw(vals, theme, out):
    th = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    y_gpu = HEAD + len(CPU) + GAP + HEAD
    rows = [(lib, col, HEAD + i) for i, (lib, col) in enumerate(CPU)]
    rows += [(lib, col, y_gpu + i) for i, (lib, col) in enumerate(GPU)]
    ymax = rows[-1][2]
    fig, (axt, axm) = plt.subplots(1, 2, figsize=(8.6, 0.42 * (ymax + 1) + 0.9), sharey=True,
                                   gridspec_kw={"wspace": 0.25})
    tmax = max(v[0] for v in vals.values())
    mmax = max(v[1] for v in vals.values() if not math.isnan(v[1]))
    for lib, col, y in rows:
        t, m, failed = vals[lib]
        axt.barh(y, t, color=col, height=0.6)
        axt.text(t * 1.15, y, fmt_t(t, failed), va="center", color=th["ink"], fontsize=9.5)
        if math.isnan(m):
            axm.text(0, y, "n/a", va="center", color=th["muted"], fontsize=9.5)
            continue
        axm.barh(y, m, color=col, height=0.6)
        axm.text(m + mmax * 0.03, y, f"{m:.1f} GB" if m < 20 else f"{m:.0f} GB", va="center", color=th["ink"], fontsize=9.5)
    axt.set_xscale("log")
    axt.set_xlim(10, tmax * 25)
    axm.set_xlim(0, mmax * 1.25)
    axt.set_yticks([y for _, _, y in rows])
    axt.set_yticklabels([lib + ("*" if lib in PAPER else "") for lib, _, _ in rows],
                        color=th["ink"], fontsize=10.5)
    axt.set_ylim(ymax + 0.5, -0.5 + HEAD - 0.9)
    for ax, title in ((axt, "Runtime (log scale)"), (axm, "Peak memory")):
        ax.set_facecolor("none")
        ax.set_title(title, loc="left", color=th["ink"], fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    # divider between the groups, across both panels and the labels
    ysep = HEAD + len(CPU) - 0.5 + GAP / 2
    y_fig = fig.transFigure.inverted().transform(axt.transData.transform((1, ysep)))[1]
    fig.add_artist(plt.Line2D([0.0, 0.98], [y_fig, y_fig], color=th["muted"], linewidth=0.8,
                              linestyle=(0, (4, 3)), transform=fig.transFigure))
    for label, y in ((GROUPS["CPU"], HEAD - 0.95), (GROUPS["GPU"], y_gpu - 0.95)):
        axt.text(-0.015, y, label, transform=axt.get_yaxis_transform(), ha="right",
                 va="center", color=th["muted"], fontsize=9, fontweight="bold")
    fig.text(0.0, 0.0, "SEA-AD, 1.38M cells · full scRNA-seq workflow · "
             "* brisc paper: 192-core node; rapids-singlecell on 4× H100", color=th["muted"],
             fontsize=8.5, va="top")
    # dark: transparent, so it sits on any dark page (GitHub, omicverse.org tiles)
    fig.savefig(f"{out}_{theme}.png", dpi=200, bbox_inches="tight",
                facecolor="none" if theme == "dark" else th["bg"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    a = ap.parse_args()
    vals = {"omicverse[cpu]": fastest(a.results, "ovcpp")[:2] + (False,),
            "omicverse[gpu]": gpu_run(a.results, "ovgpu"),
            "omicverse[mixed]": gpu_run(a.results, "ovmixed"),
            "brisc": fastest(a.results, "brisc")[:2] + (False,),
            "rapids-singlecell": paper(a.results, "rapids-singlecell"),
            "Scanpy": paper(a.results, "Scanpy"),
            "Seurat + BPCells": paper(a.results, "Seurat + BPCells"),
            "Seurat v6": seurat6(a.results)}
    for theme in THEMES:
        draw(vals, theme, a.out)


if __name__ == "__main__":
    main()
