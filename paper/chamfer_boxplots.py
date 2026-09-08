#!/usr/bin/env python3
"""
Plot Chamfer distance from shape_completion_eval.py CSVs.

Colour = condition, shade = split. All styling lives in the config block below.

Example use:
python chamfer_boxplots.py --datadir paper/shape_completion_eval/ --outdir paper/shape_completion_eval/ 
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = {"Base":           "v72_{split}_chamfer.csv",
              "Encoder":        "v72_encoder_{split}_chamfer.csv",
              "Encoder+refine": "v72_encoder_{split}_chamfer_refine.csv",
              "Hierarchy":      "v73h_{split}_chamfer.csv",
              "Contrastive":    "v73c_{split}_chamfer.csv"}

SPLITS = {"train": "TRAIN", "test": "TEST", "val": "VALIDATION"}

COLORS = {"Base":           "#FF9500",
          "Encoder":        "#BF0603",
          "Encoder+refine": "#FF187C",
          "Hierarchy":      "#25998F",
          "Contrastive":    "#B056FF"}

TINTS = {"train": 0.0, "test": 0.5, "val": 0.8}   # 0 = full colour, 1 = white

LEGEND_GREY = "#555555"                           # neutral swatch for the split key

SCALE = 1e3             # plot chamfer as x10^-3
FIGSIZE = (9.0, 4.5)
WIDTH = 0.9             # box / violin width
ALPHA = 0.5
WHIS = (5, 95)          # percentile whiskers; 1.5 would give Tukey
MEDIAN_LW = 1.0
POINT_SIZE, POINT_ALPHA, JITTER = 2.8, 0.6, 0.055
GROUP_GAP = 1.0
FONTSIZE = 11
YLABEL = "CHAMFER DISTANCE ($\\times10^{-3}$)"

matplotlib.rcParams.update({"font.family": "sans-serif",
                            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                            "font.size": FONTSIZE,
                            "pdf.fonttype": 42})

def load(datadir, conditions, splits):
    frames = []
    for cond in conditions:
        for split in splits:
            path = os.path.join(datadir, CONDITIONS[cond].format(split=split))
            if not os.path.exists(path):
                print(f"  missing {os.path.basename(path)} -- skipping")
                continue
            has_header = open(path).readline().startswith("mesh,")
            df = pd.read_csv(path) if has_header else pd.read_csv(
                path, header=None, names=["mesh", "chamfer", "gt_path"])
            df = df[["mesh", "chamfer"]].copy()
            df["mesh"] = df["mesh"].str.replace("_partial.ply", "", regex=False)
            if df["mesh"].duplicated().any():        # run twice and appended
                df = df.groupby("mesh", as_index=False)["chamfer"].mean()
            df["condition"], df["split"] = cond, split
            frames.append(df)
    if not frames:
        sys.exit(f"No chamfer CSVs found in {datadir}")
    return pd.concat(frames, ignore_index=True)

def shade(base, split):
    rgb = np.array(matplotlib.colors.to_rgb(COLORS.get(base, base)))
    return rgb + (1 - rgb) * TINTS.get(split, 0.0)

def plot(d, outbase, group_by="split", style="box", log=False, ylim=None, pdf=False):
    conds = [c for c in CONDITIONS if c in set(d.condition)]
    splits = [s for s in SPLITS if s in set(d.split)]
    if group_by == "split":
        groups, labels, inner = splits, [SPLITS[s] for s in splits], conds
        pair = lambda g, i: (i, g)                   # -> (condition, split)
    else:
        groups, labels, inner = conds, [c.upper() for c in conds], splits
        pair = lambda g, i: (g, i)
    step = len(inner) + GROUP_GAP
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for gi, g in enumerate(groups):
        for ii, i in enumerate(inner):
            cond, split = pair(g, i)
            v = d[(d.condition == cond) & (d.split == split)].chamfer.to_numpy() * SCALE
            if not len(v):
                continue
            pos, col = gi * step + ii, shade(cond, split)

            x = np.random.default_rng(0).normal(pos, JITTER, len(v))
            ax.scatter(x, v, s=POINT_SIZE, color=col, alpha=POINT_ALPHA,
                       zorder=0, linewidths=0, rasterized=True)

            if style == "violin":
                parts = ax.violinplot([v], positions=[pos], widths=WIDTH,
                                      showextrema=False, showmedians=True)
                parts["bodies"][0].set(facecolor=col, alpha=ALPHA, linewidth=0)
                parts["cmedians"].set(color="black", linewidth=MEDIAN_LW)
            else:
                bp = ax.boxplot([v], positions=[pos], widths=WIDTH, whis=WHIS,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", lw=MEDIAN_LW),
                                whiskerprops=dict(color=col),
                                capprops=dict(color=col))
                bp["boxes"][0].set(facecolor=col, alpha=ALPHA, linewidth=0)

            ax.scatter([pos], [v.mean()], marker="D", s=14, color="white",
                       edgecolor="black", linewidths=0.6, zorder=5)

    ax.set_xticks([gi * step + (len(inner) - 1) / 2 for gi in range(len(groups))])
    ax.set_xticklabels(labels)
    ax.set_xlim(-1, (len(groups) - 1) * step + len(inner))
    ax.set_ylabel(YLABEL)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    if log:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)

    patch = matplotlib.patches.Patch
    ax.add_artist(ax.legend(
        handles=[patch(facecolor=shade(LEGEND_GREY, s),
                       label=f"{SPLITS[s]}\n(N={d[d.split == s].mesh.nunique():,})")
                 for s in splits],
        fontsize=FONTSIZE - 1, ncol=len(splits), frameon=False, loc="upper center"))

    line = matplotlib.lines.Line2D
    ax.add_artist(ax.legend(
        handles=[line([], [], marker="D", linestyle="none", markerfacecolor="white",
                      markeredgecolor="black", markeredgewidth=0.6, markersize=5,
                      label="MEAN"),
                 line([], [], color="black", lw=MEDIAN_LW, label="MEDIAN")],
        fontsize=FONTSIZE - 1, frameon=False, loc="upper right"))

    fig.tight_layout()
    fig.savefig(outbase + ".png", dpi=300)
    if pdf:
        fig.savefig(outbase + ".pdf")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default=".")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--group-by", choices=["split", "condition"], default="condition")
    ap.add_argument("--style", choices=["box", "violin"], default="violin")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--ylim", nargs=2, type=float)
    ap.add_argument("--name", default="chamfer_boxplot")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    bad = [c for c in args.conditions if c not in CONDITIONS]
    if bad:
        sys.exit(f"Unknown condition(s) {bad}. Known: {', '.join(CONDITIONS)}")

    os.makedirs(args.outdir, exist_ok=True)
    d = load(args.datadir, args.conditions, args.splits)

    print("\nmedian [IQR] chamfer (x10^-3):")
    for (c, s), g in d.groupby(["condition", "split"]):
        v = g.chamfer.to_numpy() * SCALE
        print(f"  {c:<15} {s:<6} n={len(v):5d}  {np.median(v):6.2f} "
              f"[{np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f}]")

    out = os.path.join(args.outdir, args.name)
    plot(d, out, args.group_by, args.style, args.log, args.ylim, args.pdf)
    print(f"\nWrote {out}.png" + (" and .pdf" if args.pdf else ""))

if __name__ == "__main__":
    main()