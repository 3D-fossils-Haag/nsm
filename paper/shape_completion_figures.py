"""
Shape completion figures: boxplot, generalization gap, and accuracy-vs-cost.
Usage: 
python shape_completion_figures.py --datadir shape_completion_eval/ --outdir shape_completion_eval/
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────────

CONDITIONS = {
    "Base":             "v72_{split}_chamfer.csv",
    "Encoder":          "v72_encoder_{split}_chamfer.csv",
    "Encoder+refine":   "v72_encoder_{split}_chamfer_refine.csv",
    "Hierarchy":        "v73h_{split}_chamfer.csv",
    "Contrastive":      "v73c_{split}_chamfer.csv"}

ORDER  = list(CONDITIONS)
SPLITS = {"train": "TRAIN", "val": "VALIDATION", "test": "TEST"}
COLORS = {"Base":           "#FF9500", "Encoder":        "#BF0603",
          "Encoder+refine": "#FF187C", "Hierarchy":      "#25998F",
           "Contrastive":    "#B056FF"}
RUNTIME = {"Base": 198.45, "Encoder": 6.53, "Encoder+refine": 26.71,
           "Hierarchy": 133.30, "Contrastive": 167.07}
TINTS   = {"train": 0.0, "test": 0.5, "val": 0.8}
SCALE, FONTSIZE = 1e3, 12

matplotlib.rcParams.update({"font.family": "sans-serif",
                            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                            "font.size": FONTSIZE, "pdf.fonttype": 42})

# Helper funcs
def load(datadir):
    """Load all chamfer CSVs into a single DataFrame with condition/split columns."""
    frames = []
    for cond, tmpl in CONDITIONS.items():
        for split in SPLITS:
            path = os.path.join(datadir, tmpl.format(split=split))
            if not os.path.exists(path):
                print(f"  missing {os.path.basename(path)} -- skipping"); continue
            has_header = open(path).readline().startswith("mesh,")
            df = pd.read_csv(path) if has_header else pd.read_csv(
                path, header=None, names=["mesh", "chamfer", "gt_path"])
            df = df[["mesh", "chamfer"]].copy()
            df["mesh"] = df["mesh"].str.replace("_partial.ply", "", regex=False)
            if df["mesh"].duplicated().any():
                df = df.groupby("mesh", as_index=False)["chamfer"].mean()
            df["condition"], df["split"] = cond, split
            frames.append(df)
    if not frames:
        sys.exit(f"No chamfer CSVs found in {datadir}")
    return pd.concat(frames, ignore_index=True)

def tinted(cond, split):
    rgb = np.array(matplotlib.colors.to_rgb(COLORS.get(cond, cond)))
    return rgb + (1 - rgb) * TINTS.get(split, 0.0)

def save(fig, outdir, name, pdf=False):
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {name}.png" + (" + .pdf" if pdf else ""))

def fig_boxplot(d, outdir, group_by="condition", style="violin", log=False,
                ylim=None, pdf=False):
    """Boxplot / violin of chamfer distributions, grouped by condition or split."""
    conds  = [c for c in ORDER  if c in set(d.condition)]
    splits = [s for s in SPLITS if s in set(d.split)]
    if group_by == "split":
        groups, labels, inner = splits, [SPLITS[s] for s in splits], conds
        pair = lambda g, i: (i, g)
    else:
        groups, labels, inner = conds, [c.upper() for c in conds], splits
        pair = lambda g, i: (g, i)

    step = len(inner) + 1.0
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for gi, g in enumerate(groups):
        for ii, i in enumerate(inner):
            cond, split = pair(g, i)
            v = d[(d.condition == cond) & (d.split == split)].chamfer.to_numpy() * SCALE
            if not len(v): continue
            pos, col = gi * step + ii, tinted(cond, split)
            ax.scatter(np.random.default_rng(0).normal(pos, 0.055, len(v)), v,
                       s=2.8, color=col, alpha=0.6, zorder=0, linewidths=0, rasterized=True)
            if style == "violin":
                parts = ax.violinplot([v], positions=[pos], widths=0.9,
                                      showextrema=False, showmedians=True)
                parts["bodies"][0].set(facecolor=col, alpha=0.5, linewidth=0)
                parts["cmedians"].set(color="black", linewidth=1.0)
            else:
                bp = ax.boxplot([v], positions=[pos], widths=0.9, whis=(5, 95),
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", lw=1.0),
                                whiskerprops=dict(color=col), capprops=dict(color=col))
                bp["boxes"][0].set(facecolor=col, alpha=0.5, linewidth=0)
            ax.scatter([pos], [v.mean()], marker="D", s=14, color="white",
                       edgecolor="black", linewidths=0.6, zorder=5)

    ax.set_xticks([gi * step + (len(inner) - 1) / 2 for gi in range(len(groups))])
    ax.set_xticklabels(labels)
    ax.set_xlim(-1, (len(groups) - 1) * step + len(inner))
    ax.set_ylabel("CHAMFER DISTANCE ($\\times10^{-3}$)")
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    if log: ax.set_yscale("log")
    if ylim: ax.set_ylim(*ylim)

    P, L = matplotlib.patches.Patch, matplotlib.lines.Line2D
    ax.add_artist(ax.legend(
        handles=[P(facecolor=tinted("#555555", s),
                   label=f"{SPLITS[s]}\n(N={d[d.split==s].mesh.nunique():,})") for s in splits],
        fontsize=FONTSIZE-1, ncol=len(splits), frameon=False, loc="upper center"))
    ax.add_artist(ax.legend(
        handles=[L([], [], marker="D", linestyle="none", markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=0.6, markersize=5, label="MEAN"),
                 L([], [], color="black", lw=1.0, label="MEDIAN")],
        fontsize=FONTSIZE-1, frameon=False, loc="upper right"))

    fig.tight_layout()
    save(fig, outdir, "fig_chamfer_boxplot", pdf)

def fig_gap(d, outdir, pdf=False):
    """Median chamfer TRAIN → VAL → TEST slopegraph, one line per condition."""
    OFFSETS = {"Base": -12, "Encoder": 0, "Encoder+refine": 0,
               "Hierarchy": 0, "Contrastive": 12}
    xs = list(range(len(SPLITS)))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for cond in ORDER:
        y = [d[(d.condition == cond) & (d.split == s)]["chamfer"].median() * SCALE
             for s in SPLITS]
        if any(np.isnan(v) for v in y): continue
        ax.plot(xs, y, "-o", color=COLORS[cond], lw=2.2, ms=8)
        ax.annotate(cond.upper(), (xs[-1], y[-1]),
                    xytext=(8, OFFSETS.get(cond, 0)), textcoords="offset points",
                    fontsize=FONTSIZE-3, color=COLORS[cond], va="center", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(list(SPLITS.values()))
    ax.set_xlim(-0.25, len(SPLITS) + 0.3)
    ax.set_ylabel("MEDIAN CHAMFER ($\\times10^{-3}$)")
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, outdir, "fig_generalization_gap", pdf)

def fig_tradeoff(d, outdir, pdf=False):
    """Scatter: median test chamfer vs mean inference time per mesh."""
    OFFSETS_Y = {"Base": 13, "Encoder": -13, "Encoder+refine": 13,
                 "Hierarchy": 13, "Contrastive": -13}
    OFFSET_VA = {"Base": "bottom", "Encoder": "top", "Encoder+refine": "bottom",
                 "Hierarchy": "bottom", "Contrastive": "top"}
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for cond in ORDER:
        v = d[(d.condition == cond) & (d.split == "test")]["chamfer"]
        if not len(v) or cond not in RUNTIME: continue
        ax.scatter(RUNTIME[cond], v.median() * SCALE, s=160, color=COLORS[cond],
                   edgecolor="black", linewidths=0.8, zorder=3)
        ax.annotate(cond.upper(), (RUNTIME[cond], v.median() * SCALE),
                    xytext=(0, OFFSETS_Y.get(cond, 0)), textcoords="offset points",
                    ha="center", fontsize=FONTSIZE-4, color=COLORS[cond],
                    fontweight="bold", va=OFFSET_VA.get(cond, "center"))
    ax.set_xlabel("INFERENCE TIME PER MESH (s)")
    ax.set_ylabel("MEDIAN CHAMFER ($\\times10^{-3}$)")
    ax.set_xlim(-12, 225)
    ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, outdir, "fig_accuracy_vs_cost", pdf)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datadir", default=".")
    ap.add_argument("--outdir",  default=".")
    ap.add_argument("--group-by", choices=["split", "condition"], default="condition")
    ap.add_argument("--style",   choices=["box", "violin"], default="violin")
    ap.add_argument("--log",     action="store_true")
    ap.add_argument("--ylim",    nargs=2, type=float)
    ap.add_argument("--pdf",     action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    d = load(args.datadir)
    fig_boxplot(d, args.outdir, args.group_by, args.style, args.log, args.ylim, args.pdf)
    fig_gap(d, args.outdir, args.pdf)
    fig_tradeoff(d, args.outdir, args.pdf)

if __name__ == "__main__":
    main()