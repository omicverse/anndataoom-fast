"""Headline figure for the READMEs (light and dark): total runtime and peak memory of the
standard Fig. 4 workflow on SEA-AD (1.38M cells) for omicverse, brisc, Scanpy
and Seurat + BPCells.

    python plot_headline.py RESULTS_DIR OUT   # writes OUT_light.png, OUT_dark.png

omicverse and brisc: our runs on the same AMD EPYC 9754 node, each at its
fastest thread count (RESULTS_DIR/cpu128/SEAAD_{ovcpp,brisc}_t*.csv).
Scanpy and Seurat + BPCells: the numbers reported in the brisc paper
(RESULTS_DIR/brisc_paper/seaad_paper_baselines.csv, 192-core node)."""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

# validated categorical order (validate_palette.js, light and dark surfaces, all pairs)
ROWS = [("omicverse", "#7058be"), ("brisc", "#e16032"), ("Scanpy", "#269d69"),
        ("Seurat", "#4698cc")]
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


THEMES = {"light": {"bg": "#ffffff", "ink": "#1f2328", "muted": "#8c959f"},
          "dark": {"bg": "#0d1117", "ink": "#e6edf3", "muted": "#7d8590"}}


def draw(vals, theme, out):
    th = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (axt, axm) = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True,
                                   gridspec_kw={"wspace": 0.12})
    tmax = max(v[0] for v in vals.values())
    mmax = max(v[1] for v in vals.values())
    for i, (lib, col) in enumerate(ROWS):
        t, m = vals[lib]
        axt.barh(i, t, color=col, height=0.6)
        axt.text(t * 1.15, i, f"{t:.0f} s" if t < 100 else f"{t / 3600:.1f} h", va="center",
                 color=th["ink"], fontsize=9.5)
        axm.barh(i, m, color=col, height=0.6)
        axm.text(m + mmax * 0.03, i, f"{m:.3g} GB", va="center", color=th["ink"], fontsize=9.5)
    axt.set_xscale("log")
    axt.set_xlim(10, tmax * 8)
    axm.set_xlim(0, mmax * 1.25)
    axt.set_yticks(range(len(ROWS)))
    axt.set_yticklabels([lib for lib, _ in ROWS], color=th["ink"], fontsize=10.5)
    axt.set_ylim(len(ROWS) - 0.5, -0.5)
    for ax, title in ((axt, "Runtime (log scale)"), (axm, "Peak memory")):
        ax.set_facecolor("none")
        ax.set_title(title, loc="left", color=th["ink"], fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.0, 0.0, "SEA-AD, 1.38M cells · full scRNA-seq workflow", color=th["muted"],
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
    vals = {"omicverse": fastest(a.results, "ovcpp")[:2],
            "brisc": fastest(a.results, "brisc")[:2],
            "Scanpy": paper(a.results, "Scanpy"),
            "Seurat": paper(a.results, "Seurat + BPCells")}
    for theme in THEMES:
        draw(vals, theme, a.out)


if __name__ == "__main__":
    main()
