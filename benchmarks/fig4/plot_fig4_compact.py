"""Compact Fig. 4: per-step runtime as a heatmap (methods x steps, log colour)
next to bars of total runtime and peak memory.

    python plot_fig4_compact.py RESULTS_DIR DATASET OUT --libs a,b,c \
        [--note LIB:OP:TEXT ...] [--footnote TEXT] [--title TEXT]

Same inputs and ``--note`` syntax as plot_fig4.py (``LIB:*:TEXT`` fills every
missing step of LIB)."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from plot_fig4 import COLORS, LABELS, fmt_time, load

STEPS = ["Load data", "Quality control", "Normalization + feature selection", "PCA",
         "Nearest neighbors", "Clustering", "Embedding (LocalMAP)", "Embedding (UMAP)",
         "Find markers"]
STEP_LAB = {"Load data": "Load", "Quality control": "QC",
            "Normalization + feature selection": "Norm +\nHVG", "PCA": "PCA",
            "Nearest neighbors": "kNN", "Clustering": "Leiden\n(×5)",
            "Embedding (LocalMAP)": "LocalMAP", "Embedding (UMAP)": "UMAP",
            "Find markers": "Markers"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--libs", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--footnote", default=None)
    ap.add_argument("--note", action="append", default=[])
    a = ap.parse_args()

    d = load(a.results, a.dataset)
    libs = a.libs.split(",")
    notes = {}
    for n in a.note:
        lib, op, text = n.split(":", 2)
        notes[(lib, op)] = text
    steps = [s for s in STEPS if s in set(d["operation"])]
    T = np.full((len(libs), len(steps)), np.nan)
    for i, lib in enumerate(libs):
        for j, s in enumerate(steps):
            r = d[(d.library == lib) & (d.operation == s)]
            if len(r):
                T[i, j] = r.duration.iloc[0]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    h = 0.42 * len(libs) + 1.6
    fig = plt.figure(figsize=(11, h))
    gs = fig.add_gridspec(1, 3, width_ratios=[len(steps) * 0.78, 2.2, 1.5], wspace=0.06)
    ax, axt, axm = (fig.add_subplot(gs[k]) for k in range(3))

    # --- heatmap of per-step runtime -------------------------------------
    finite = T[np.isfinite(T)]
    norm = LogNorm(vmin=max(finite.min(), 0.5), vmax=finite.max())
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f2f2f2")
    ax.imshow(np.ma.masked_invalid(T), cmap=cmap, norm=norm, aspect="auto")
    for i, lib in enumerate(libs):
        for j, s in enumerate(steps):
            v = T[i, j]
            if np.isfinite(v):
                dark = norm(v) > 0.6
                ax.text(j, i, fmt_time(v), ha="center", va="center", fontsize=7,
                        color="white" if dark else "black")
            else:
                txt = notes.get((lib, s), notes.get((lib, "*"), "–"))
                short = {"not reached": "–", "not available": "n/a"}.get(txt, txt)
                if short.startswith("out of"):
                    short = "OOM"
                ax.text(j, i, short, ha="center", va="center", fontsize=6.5, color="#777777",
                        style="italic")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([STEP_LAB[s] for s in steps], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(libs)))
    ax.set_yticklabels([LABELS.get(l, l).replace("\n", " ") for l in libs], fontsize=7.5)
    for t, lib in zip(ax.get_yticklabels(), libs):
        t.set_color(COLORS.get(lib, "black"))
    ax.set_xticks(np.arange(-0.5, len(steps)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(libs)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Per-step runtime", fontsize=9, pad=24)

    # --- total runtime and peak memory -----------------------------------
    tot = {lib: d[(d.library == lib) & (d.operation == "Total")] for lib in libs}
    ypos = np.arange(len(libs))
    tmax = max(r.duration.iloc[0] for r in tot.values() if len(r))
    mmax = max(r.memory.iloc[0] for r in tot.values() if len(r) and np.isfinite(r.memory.iloc[0]))
    for i, lib in enumerate(libs):
        r = tot[lib]
        if not len(r):
            axt.text(1.1, i, notes.get((lib, "Total"), notes.get((lib, "*"), "–")).replace(
                "not reached", "failed"), va="center", fontsize=7, color="#777777", style="italic")
            continue
        t, m = r.duration.iloc[0], r.memory.iloc[0]
        done = set(d[(d.library == lib) & (d.operation.isin(steps))].operation)
        missing = set(steps) - done - {"Embedding (LocalMAP)", "Embedding (UMAP)"}
        axt.barh(i, t, color=COLORS.get(lib, "gray"), height=0.7)
        axt.text(t * 1.15, i, ("> " if missing else "") + fmt_time(t), va="center", fontsize=7)
        if np.isfinite(m):
            axm.barh(i, m, color="#bdbdbd", height=0.7)
            axm.text(m + mmax * 0.04, i, f"{m:.0f} GB", va="center", fontsize=7)
    axt.set_xscale("log")
    axt.set_xlim(1, tmax * 12)
    axm.set_xlim(0, mmax * 1.45)
    for x, title in ((axt, "Total runtime"), (axm, "Peak memory")):
        x.set_ylim(len(libs) - 0.5, -0.5)
        x.set_yticks([])
        x.set_title(title, fontsize=9, pad=24)
        for sp in ("top", "right", "left"):
            x.spines[sp].set_visible(False)
        x.tick_params(axis="x", labelsize=7)
    axt.set_xlabel("seconds (log)", fontsize=7)
    axm.set_xlabel("GB", fontsize=7)

    if a.title:
        fig.suptitle(a.title, fontsize=10.5, y=1.08, x=0.45)
    if a.footnote:
        fig.text(0.5, -0.02, a.footnote, ha="center", va="top", fontsize=6.5, color="#333333",
                 wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
