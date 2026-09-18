"""Figure R classification panels.

Read the tree written by classification_eval.py:
 
    <run>/classification/evaluation/<split>/<split>_<eval_level>_<latents>/
        metrics_summary.csv   report_<category>.csv   metrics.json
        predictions.csv

By default, renders one condition -- the one the published figure reports:
LOSO (specimen) masking, optimised latents, train split, top-5 diagonal -- and
writes seven files:

    fig_r_A_position_20.png  fig_r_B_life_history.png
    fig_r_C_broad_taxon.png  fig_r_D_region.png
    fig_r_colorbar.png                      saved once, shared by all panels

Tables S4 and S5, the per-class recall and support correlations behind these
panels, are written by classif_tables.py.


--mode picks the diagonal criterion. top5 credits a query whose true label is
anywhere in its top 5 to the diagonal and charges a complete miss to its top-1
prediction, so rows still sum to 1, the diagonal is exactly top-5 recall, and
near-miss confusions drop out -- they are not errors under a top-5 criterion.
top1 gives each query its single nearest label. predictions.csv stores top-5 as
a boolean hit rather than five labels, so full top-5 neighbourhoods cannot be
rebuilt here; this is the strongest top-5 view the file supports.

Usage
-----
    python classif_figures.py --roots ../run_v72                    # the published set
    python classif_figures.py --roots ../run_v72 --split test --mode top1
    python classif_figures.py --roots ../run_v72 ../run_v73h --extras  # everything
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import encoder
from NSM.evaluation import SPLIT_ORDER, find_results, normalize_labels

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42})

# --- panel geometry and type ----------------------------------------------
REF_CLASSES = 5                  # class count whose proportions look right
REF_FONT = 8.5
FONT_BOOST = 1.5                 
                                 
FONT_BOOST_BY_CATEGORY = {       
    "broad_taxon": 1.1,          
    "life_history": 1.1,
    "region": 2}
CM_CELL_REL = 0.9                # in-cell numbers, relative to tick labels
CM_CELL_MAX_CLASSES = 15         # above this, cells are too small to label
CM_ROTATION = 30
CM_CMAP = "viridis"

REGION_ORDER = ["Cervical", "Thoracic", "Lumbar"]

# --- the published condition ----------------------------------------------
FIG_CATEGORIES = ["position_20", "life_history", "broad_taxon", "region"]  # A-D
FIG_SPLIT = "train"
FIG_EVAL = "specimen"
FIG_LATENTS = "latent_opt"
FIG_MODE = "top5"

# Define functions
def class_order_for(cat, values):
    """Anatomical order for region, numeric for position bins, else alphabetical."""
    values = list(values)
    if cat == "region":
        lut = {str(v).lower(): v for v in values}
        known = [lut[r.lower()] for r in REGION_ORDER if r.lower() in lut]
        return known + sorted(v for v in values if v not in known)
    if cat.startswith("position"):
        def first_number(v):
            m = re.match(r"(-?\d+)", str(v))
            return float(m.group(1)) if m else np.inf
        return sorted(values, key=first_number)
    return sorted(values)

def build_rows(df, cat, mode):
    """Scored queries as (y_true, rows-of-predicted-labels) for one category."""
    tcol, pcol, hcol = f"{cat}_true", f"{cat}_pred", f"{cat}_top5_hit"
    if tcol not in df.columns or pcol not in df.columns:
        return None, None
    sub = df[df[tcol].notna() & df[pcol].notna()]
    if sub.empty:
        return None, None

    yt = sub[tcol].astype(str).to_numpy(dtype=object)
    yp = sub[pcol].astype(str).to_numpy(dtype=object)
    if mode == "top1":
        return yt, [[p] for p in yp]
    if hcol not in sub.columns:
        return None, None
    hit = sub[hcol].astype(bool).to_numpy()
    return yt, [[t] if h else [p] for t, p, h in zip(yt, yp, hit)]

def plot_confusion(y_true, rows, out_png, class_order=None, title="",
                   normalize=True, boost=FONT_BOOST, also_pdf=False):
    """One matrix, sized for assembly into a multipanel figure.
    """
    present = set(y_true) | {p for r in rows for p in r}
    classes = [c for c in (class_order or []) if c in present]
    classes += sorted(present - set(classes))         
    ix = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((len(classes), len(classes)))
    for t, r in zip(y_true, rows):
        for p in r:
            cm[ix[t], ix[p]] += 1
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    size = max(4.0, 0.45 * len(classes) + 2)
    if len(classes) == 3:
        fs = REF_FONT * 1.3
    else:
        fs = REF_FONT * (size / max(4.0, 0.45 * REF_CLASSES + 2)) * boost

    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap=CM_CMAP, vmin=0, vmax=1 if normalize else None)
    labels = [str(c).replace("_", " ").upper() for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=CM_ROTATION, ha="right", fontsize=fs,
                       rotation_mode="anchor")
    ax.set_yticklabels(labels, fontsize=fs)
    if title:
        ax.set_title(title, fontsize=fs * 1.15)
    if len(classes) <= CM_CELL_MAX_CLASSES:
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] > 0:
                    ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                            ha="center", va="center", fontsize=fs * CM_CELL_REL,
                            color="white" if cm[i, j] < 0.6 else "black")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_png, dpi=300)
    if also_pdf:
        fig.savefig(os.path.splitext(out_png)[0] + ".pdf")
    plt.close(fig)
    return len(classes)

def save_colorbar(out_png, label="Row-normalised fraction", vmax=1.0,
                  fontsize=9, also_pdf=False):
    """Standalone colorbar, tick numbers rotated 90 degrees."""
    fig, ax = plt.subplots(figsize=(0.2, 4.0))
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=vmax), cmap=CM_CMAP)
    cb = fig.colorbar(sm, cax=ax, orientation="vertical")
    cb.set_label(label, fontsize=fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_rotation(90)
        t.set_rotation_mode("anchor")
        t.set_va("center")
        t.set_ha("center")
    cb.ax.tick_params(labelsize=fontsize, pad=10)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if also_pdf:
        fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)

def process_run(root, outdir, args):
    """Panels for one run, into that run's own paper directory.
    One directory per run is what keeps the panel filenames short: nothing has
    to be prefixed with the run to stop two runs overwriting each other.
    """
    results = list(find_results([root], "predictions.csv",
                                eval_level=args.eval, latents=args.latents))
    if not results:
        print(f"  no predictions.csv with eval_level={args.eval} "
              f"latents={args.latents} under {root}", file=sys.stderr)
        return []
    os.makedirs(outdir, exist_ok=True)
    normalize = not args.counts
    panel = {c: chr(ord("A") + i) for i, c in enumerate(args.categories)}

    made = []
    for res in sorted(results, key=lambda r: SPLIT_ORDER.get(r.split, 9)):
        if not (args.extras or res.split == args.split):
            continue
        df = normalize_labels(pd.read_csv(res.path), quiet=args.quiet)

        for cat in args.categories:
            yt, rows = build_rows(df, cat, args.mode)
            if yt is None:
                if not args.quiet and f"{cat}_true" in df.columns:
                    print(f"  {res.run} {res.split} {cat}: no scored queries "
                          f"for mode={args.mode}, skipping")
                continue
            present = set(yt) | {p for r in rows for p in r}
            acc = float(np.mean([t in r for t, r in zip(yt, rows)]))
            stem = f"fig_r_{panel[cat]}_{cat}"
            if args.extras:
                stem = f"{res.split}_{stem}"
            out = os.path.join(outdir, f"{stem}.png")
            n = plot_confusion(
                yt, rows, out, class_order=class_order_for(cat, present),
                title=f"{cat}  (n={len(yt)}, {args.mode}={acc:.2f})"
                      if args.titles else "",
                normalize=normalize, also_pdf=args.pdf,
                boost=FONT_BOOST_BY_CATEGORY.get(cat, FONT_BOOST))
            made.append(out)
            if not args.quiet:
                print(f"  {res.run} {res.split} {cat:<13} n={len(yt):5d} "
                      f"classes={n:3d} {args.mode}={acc:.3f} -> "
                      f"{os.path.basename(out)}")

    written = [os.path.basename(m) for m in made]
    if made and normalize:
        cb = os.path.join(outdir, "fig_r_colorbar.png")
        save_colorbar(cb, also_pdf=args.pdf)
        written.append(os.path.basename(cb))
    if not made:
        print(f"  no panels for {root} split={args.split}", file=sys.stderr)
    return written

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="run dirs, e.g. ../run_v72 ../run_v73h ../run_v73c")
    ap.add_argument("--outdir", default=None,
                    help="default: <root>/classification/evaluation/paper")
    ap.add_argument("--split", default=FIG_SPLIT, help="split shown in the panels")
    ap.add_argument("--eval", default=FIG_EVAL, choices=["loo", "specimen",
                                                         "species", "genus"],
                    help="masking level (specimen = LOSO)")
    ap.add_argument("--latents", default=FIG_LATENTS, choices=["latent_opt", "base"])
    ap.add_argument("--mode", default=FIG_MODE, choices=["top1", "top5"],
                    help="diagonal criterion")
    ap.add_argument("--categories", nargs="+", default=FIG_CATEGORIES,
                    help="panel categories, in panel order")
    ap.add_argument("--extras", action="store_true",
                    help="a panel set for every split, not just --split")
    ap.add_argument("--counts", action="store_true",
                    help="plot counts rather than row-normalised fractions")
    ap.add_argument("--titles", action="store_true",
                    help="write a title on each panel (off: label them in the layout)")
    ap.add_argument("--pdf", action="store_true", help="also write vector PDFs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    total = 0
    for root in args.roots:
        outdir = args.outdir or os.path.join(root, "classification", "evaluation", "paper")
        if args.outdir and len(args.roots) > 1:   # keep the runs from colliding
            outdir = os.path.join(args.outdir, os.path.basename(root.rstrip("/")))
        written = process_run(root, outdir, args)
        total += len(written)
        if written:
            print(f"\nWrote {len(written)} files to {outdir}: {', '.join(written)}")
    if not total:
        sys.exit("Nothing written; check --roots and the filters.")


if __name__ == "__main__":
    main()