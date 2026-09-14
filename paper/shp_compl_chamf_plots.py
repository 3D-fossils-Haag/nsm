#!/usr/bin/env python3
"""
Accuracy-vs-cost and generalization-gap figures, styled to match chamfer_boxplots.py.

Usage:
    python shp_compl_chamf_plots.py --datadir paper/shape_completion_eval/ --outdir paper/shape_completion_eval/ 
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Style — mirrors chamfer_boxplots.py exactly
# --------------------------------------------------------------------------- #

CONDITIONS = {
    "Base":             "v72_{split}_chamfer.csv",
    "Encoder":          "v72_encoder_{split}_chamfer.csv",
    "Encoder\n+refine": "v72_encoder_{split}_chamfer_refine.csv",
    "Hierarchy":        "v73h_{split}_chamfer.csv",
    "Contrastive":      "v73c_{split}_chamfer.csv",
}
ORDER  = ["Base", "Encoder", "Encoder\n+refine", "Hierarchy", "Contrastive"]
SPLITS = {"train": "TRAIN", "val": "VALIDATION", "test": "TEST"}

COLORS = {
    "Base":             "#FF9500",
    "Encoder":          "#BF0603",
    "Encoder\n+refine": "#FF187C",
    "Hierarchy":        "#25998F",
    "Contrastive":      "#B056FF",
}

RUNTIME = {                  # mean inference time per mesh (s), from grid-search workbook
    "Base":             198.45,
    "Encoder":            6.53,
    "Encoder\n+refine":  26.71,
    "Hierarchy":         133.30,
    "Contrastive":       167.07,
}

SCALE    = 1e3
FONTSIZE = 15

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Liberation Sans", "DejaVu Sans"],
    "font.size":        FONTSIZE,
    "pdf.fonttype":     42,
})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_all(datadir):
    frames = []
    for cond, template in CONDITIONS.items():
        for split in SPLITS:
            path = os.path.join(datadir, template.format(split=split))
            if not os.path.exists(path):
                print(f"  missing {os.path.basename(path)} -- skipping")
                continue
            has_header = open(path).readline().startswith("mesh,")
            df = (pd.read_csv(path) if has_header
                  else pd.read_csv(path, header=None,
                                   names=["mesh", "chamfer", "gt_path"]))
            df = df[["mesh", "chamfer"]].copy()
            df["mesh"] = df["mesh"].str.replace("_partial.ply", "", regex=False)
            if df["mesh"].duplicated().any():
                df = df.groupby("mesh", as_index=False)["chamfer"].mean()
            df["condition"] = cond
            df["split"]     = split
            frames.append(df)
    if not frames:
        sys.exit(f"No chamfer CSVs found in {datadir}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Figure 1: generalization gap slopegraph
# --------------------------------------------------------------------------- #

LABEL_OFFSET_Y = {
    "Base":             -12,
    "Encoder":            0,
    "Encoder\n+refine":   0,
    "Hierarchy":          0,
    "Contrastive":       12,
}

def figure_gap(d, outdir):
    """Median chamfer across TRAIN → VALIDATION → TEST, one line per condition."""
    xs = list(range(len(SPLITS)))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for cond in ORDER:
        y = [d[(d.condition == cond) & (d.split == s)]["chamfer"].median() * SCALE
             for s in SPLITS]
        if any(np.isnan(v) for v in y):
            continue
        ax.plot(xs, y, "-o", color=COLORS[cond], lw=2.2, ms=8,
                label=cond.replace("\n", " ").upper(), zorder=3)
        ax.annotate(
            cond.upper(),
            (xs[-1], y[-1]),
            xytext=(8, LABEL_OFFSET_Y[cond]), textcoords="offset points",
            fontsize=FONTSIZE - 4, color=COLORS[cond],
            va="center", fontweight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(list(SPLITS.values()))
    ax.set_xlim(-0.25, len(SPLITS) - 0.4)
    ax.set_ylabel("MEDIAN CHAMFER ($\\times10^{-3}$)")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, outdir, "fig_generalization_gap")


# --------------------------------------------------------------------------- #
# Figure 2: accuracy vs inference cost
# --------------------------------------------------------------------------- #

TRADEOFF_OFFSET_Y = {
    "Base":             13,
    "Encoder":          -13,
    "Encoder\n+refine": 13,
    "Hierarchy":        13,
    "Contrastive":     -13,
}

TRADEOFF_VA = {
    "Base":             "bottom",
    "Encoder":          "top",
    "Encoder\n+refine": "bottom",
    "Hierarchy":        "bottom",
    "Contrastive":      "top",
}

def figure_tradeoff(d, outdir):
    """Scatter: median test Chamfer (y) vs mean inference time per mesh (x)."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    for cond in ORDER:
        v = d[(d.condition == cond) & (d.split == "test")]["chamfer"]
        if not len(v) or cond not in RUNTIME:
            continue
        med = v.median() * SCALE
        ax.scatter(RUNTIME[cond], med,
                   s=160, color=COLORS[cond],
                   edgecolor="black", linewidths=0.8,
                   zorder=3)
        ax.annotate(
            cond.replace("\n", "").upper(),
            (RUNTIME[cond], med),
            xytext=(0, TRADEOFF_OFFSET_Y[cond]), textcoords="offset points",
            ha="center", fontsize=FONTSIZE - 5,
            color=COLORS[cond], fontweight="bold",
            va=TRADEOFF_VA[cond],
        )
    ax.set_xlabel("INFERENCE TIME PER MESH (s)")
    ax.set_ylabel("MEDIAN TEST CHAMFER ($\\times10^{-3}$)")
    ax.set_xlim(-12, 225)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, outdir, "fig_accuracy_vs_cost")


# --------------------------------------------------------------------------- #

def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {name}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datadir", default=".")
    ap.add_argument("--outdir",  default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = load_all(args.datadir)
    figure_gap(d, args.outdir)
    figure_tradeoff(d, args.outdir)


if __name__ == "__main__":
    main()