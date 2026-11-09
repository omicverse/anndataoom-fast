"""Per-step runtime and peak memory, in the style of the brisc paper's Fig. 4.

    python plot_fig4.py RESULTS_DIR DATASET OUT_PREFIX [--title "1.4M cells (SEA-AD)"]

Reads RESULTS_DIR/DATASET_*.csv written by run_brisc.py / run_omicverse.py
(brisc-benchmarking MemoryTimer format).  omicverse's preprocess() normalizes
and selects features in one call, so brisc's "Normalization" and "Feature
selection" are summed into one row (memory: max of the two).  Multipliers are
relative to brisc, as in the paper.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"brisc": "#e16032", "omicverse-cpp": "#7058be", "omicverse-python": "#269d69"}
LABELS = {"brisc": "brisc", "omicverse-cpp": "omicverse (backend='cpp')",
          "omicverse-python": "omicverse (backend='python')"}
OPS = ["Load data", "Quality control", "Normalization + feature selection", "PCA",
       "Nearest neighbors", "Clustering", "Embedding (LocalMAP)", "Embedding (UMAP)",
       "Find markers", "Total"]
OP_LAB = {"Normalization + feature selection": "Normalization +\nfeature selection",
          "Clustering": "Cluster\n(5 resolutions)",
          "Embedding (LocalMAP)": "Embedding\n(LocalMAP)", "Embedding (UMAP)": "Embedding\n(UMAP)"}
REF = "brisc"


def fmt_time(s):
    if s >= 3600:
        return f"{s / 3600:.1f}h"
    if s >= 60:
        return f"{s / 60:.1f}m"
    if s >= 0.1:
        return f"{s:.1f}s"
    return f"{s * 1000:.0f}ms"


def load(results, dataset):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(results, f"{dataset}_*.csv"))])
    df["operation"] = df["operation"].replace({"Normalization": "Normalization + feature selection",
                                               "Feature selection": "Normalization + feature selection"})
    g = df.groupby(["library", "rep", "operation"], as_index=False).agg(
        duration=("duration", "sum"), memory=("memory", "max"))
    # Total = the paper's standard workflow, embedding by LocalMAP where the
    # library has it (UMAP otherwise).
    has_lm = set(g.loc[g["operation"] == "Embedding (LocalMAP)", "library"])
    keep = ~((g["operation"] == "Embedding (UMAP)") & g["library"].isin(has_lm))
    tot = g[keep].groupby(["library", "rep"], as_index=False).agg(duration=("duration", "sum"),
                                                                  memory=("memory", "max"))
    tot["operation"] = "Total"
    g = pd.concat([g, tot])
    return g.groupby(["library", "operation"], as_index=False).agg(
        duration=("duration", "mean"), memory=("memory", "max"),
        n=("duration", "size"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    d = load(a.results, a.dataset)
    libs = [l for l in COLORS if l in set(d["library"])]
    ops = [o for o in OPS if o in set(d["operation"])]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(len(ops), 2, figsize=(7.2, 0.62 * len(ops) * len(libs) / 3 + 1.2),
                             gridspec_kw={"width_ratios": [4, 1.3], "wspace": 0.04, "hspace": 0.0},
                             squeeze=False)
    mem_max = d["memory"].max()
    for i, op in enumerate(ops):
        ax, axm = axes[i]
        sub = d[d["operation"] == op].set_index("library")
        ref = sub["duration"].get(REF)
        vmax = sub["duration"].max()
        for j, lib in enumerate(reversed(libs)):
            if lib not in sub.index:
                continue
            t, m = sub.at[lib, "duration"], sub.at[lib, "memory"]
            ax.barh(j, max(t, vmax * 0.02), height=0.8, color=COLORS[lib])
            lab = fmt_time(t)
            if ref is not None and lib != REF and ref > 0:
                mult = t / ref
                lab += f" ({mult:.1f}x)" if mult < 10 else f" ({mult:.0f}x)"
            ax.text(max(t, vmax * 0.02) + vmax * 0.03, j, lab, va="center", ha="left", fontsize=7.5)
            axm.barh(j, m, height=0.8, color="#d9d9d9")
            axm.text(mem_max * 0.08, j, f"{m:.0f} GB" if m >= 1 else f"{m:.1f} GB",
                     va="center", ha="left", fontsize=7.5)
        ax.set_xlim(0, vmax * 1.75)
        axm.set_xlim(0, mem_max * 1.5)
        for x in (ax, axm):
            x.set_ylim(-0.6, len(libs) - 0.4)
            x.set_xticks([])
            x.set_yticks([])
        ax.set_ylabel(OP_LAB.get(op, op), rotation=0, ha="right", va="center", fontsize=9,
                      fontweight="bold" if op == "Total" else "normal")
    axes[-1, 0].set_xlabel("Per-step runtime")
    axes[-1, 1].set_xlabel("Peak\nmemory")
    if a.title:
        fig.suptitle(a.title, fontsize=11, y=0.995)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in libs]
    fig.legend(handles, [LABELS[l] for l in libs], loc="lower center", ncol=len(libs),
               frameon=False, bbox_to_anchor=(0.55, -0.02), fontsize=8.5)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.95, bottom=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")
    d.to_csv(f"{a.out}_summary.csv", index=False)
    print(d.pivot(index="operation", columns="library", values=["duration", "memory"]).round(2))


if __name__ == "__main__":
    main()
