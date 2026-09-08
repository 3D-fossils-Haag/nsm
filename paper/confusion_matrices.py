#!/usr/bin/env python3
"""
Figure-ready confusion matrices from classification_eval.py predictions.

Reads predictions.csv (columns: mesh, specimen, <cat>_true, <cat>_pred,
<cat>_top5_hit) and renders one matrix per category, sized and typeset for
assembly into a multipanel figure.

Panel size grows with class count, so a fixed font size ends up tiny on dense
panels once every panel is scaled to a common width. Font size is therefore
derived from the same ratio as the figure size, anchored at REF_CLASSES, which
keeps the apparent label size constant across panels. The colorbar is saved
once separately rather than attached to each panel -- an attached colorbar
steals a different fraction of the figure depending on tick label widths, which
leaves the plot areas subtly different sizes across panels.

Usage
-----
    # crawl runs and render everything
    python confusion_matrices.py --roots run_v72 run_v73h run_v73c

    # one file, chosen categories
    python confusion_matrices.py \
        --pred run_v72/classification/evaluation/train/train_specimen_base/predictions.csv \
        --categories family region position_20

    # counts instead of row-normalised fractions
    python confusion_matrices.py --roots run_v72 --no-normalize

Outputs <run>_<split>_<evallevel>_<category>.png (and .pdf with --pdf) plus a
single colorbar.png, into --outdir.

Note: predictions.csv stores <cat>_top5_hit as a boolean, not the top-5 labels,
so only top-1 matrices can be rebuilt here. For the top-5 neighbourhood matrix,
plot it inside classification_eval.py where the labels are still in memory.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42,          # embed TrueType; journals reject Type 3
})

REF_CLASSES = 5                  # class count whose proportions look right
REF_FONT = 8.5
CM_TITLE_REL = 1.15              # title size relative to tick labels
CM_CELL_REL = 0.7                # in-cell numbers relative to tick labels
CM_CELL_MAX_CLASSES = 15         # above this, cells are too small to label
CM_ROTATION = 30
CM_CMAP = "viridis"

REGION_ORDER = ["Cervical", "Thoracic", "Lumbar"]
DEFAULT_CATEGORIES = ["family", "broad_taxon", "region", "position_20",
                      "position_10", "life_history"]


def _cm_figsize(n):
    return max(4.0, 0.45 * n + 2)


def _fmt_label(s):
    """cordylidae -> CORDYLIDAE. Uppercase reads better at small panel sizes:
    uniform x-height, no descenders to collide when rotated."""
    return str(s).replace("_", " ").upper()


def class_order_for(cat, values):
    """Anatomical order for region, numeric order for position bins,
    alphabetical otherwise."""
    if cat == "region":
        return [c for c in REGION_ORDER if c in values]
    if cat.startswith("position"):
        return sorted(values, key=lambda s: float(re.match(r"(-?\d+)", str(s)).group(1))
                      if re.match(r"(-?\d+)", str(s)) else np.inf)
    return sorted(values)


def plot_confusion(y_true, y_pred, title, out_png, class_order=None,
                   normalize=True, also_pdf=False):
    rows = [p if isinstance(p, (list, tuple, set)) else [p] for p in y_pred]
    present = set(y_true) | {p for r in rows for p in r}
    classes = [c for c in class_order if c in present] if class_order else sorted(present)
    ix = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((len(classes), len(classes)))
    for t, r in zip(y_true, rows):
        for p in r:
            cm[ix[t], ix[p]] += 1
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    size = _cm_figsize(len(classes))
    scale = size / _cm_figsize(REF_CLASSES)
    if len(classes) == 3:
        scale = 1.3
    fs = REF_FONT * scale

    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap=CM_CMAP, vmin=0, vmax=1 if normalize else None)

    disp = [_fmt_label(c) for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(disp, rotation=CM_ROTATION, ha="right", fontsize=fs, rotation_mode="anchor")
    ax.set_yticklabels(disp, fontsize=fs)

    if len(classes) <= CM_CELL_MAX_CLASSES:
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = cm[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.2f}" if normalize else int(v),
                            ha="center", va="center", fontsize=fs * CM_CELL_REL,
                            color="white" if v < 0.6 else "black")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_png, dpi=300)
    if also_pdf:
        fig.savefig(os.path.splitext(out_png)[0] + ".pdf")
    plt.close(fig)
    return len(classes)


def save_colorbar(out_png, label="Row-normalised fraction", vmin=0.0, vmax=1.0,
                  cmap=CM_CMAP, vertical=True, figsize=(0.2, 4.0), fontsize=9,
                  also_pdf=False):
    """Standalone colorbar, tick numbers rotated 90 degrees counter-clockwise."""
    fig, ax = plt.subplots(figsize=figsize)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax, orientation="vertical" if vertical else "horizontal")
    cb.set_label(label, fontsize=fontsize)

    # anchor the rotation at the label centre, or the numbers drift off their
    # tick positions as they pivot
    for t in (cb.ax.get_yticklabels() if vertical else cb.ax.get_xticklabels()):
        t.set_rotation(90)
        t.set_rotation_mode("anchor")
        t.set_va("center")
        t.set_ha("center")
    cb.ax.tick_params(labelsize=fontsize, pad=10)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if also_pdf:
        fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def parse_suffix(suffix, split):
    """Recover eval_level from the output directory name
    ('<split>_<eval_level>_<base|latent_opt>')."""
    s = re.sub(rf"^{re.escape(split)}_", "", suffix)
    m = re.match(r"^(loo|specimen|species|genus)(?:_(base|latent_opt))?", s)
    return (m.group(1), m.group(2) or "?") if m else (s, "?")


def find_predictions(roots):
    """Yield (run, split, eval_level, latents, path) for every predictions.csv."""
    for root in roots:
        base = os.path.join(root, "classification", "evaluation")
        if not os.path.isdir(base):
            print(f"  no evaluation directory under {root}", file=sys.stderr)
            continue
        for path in sorted(glob.glob(os.path.join(base, "*", "*", "predictions.csv"))):
            d = os.path.dirname(path)
            split = os.path.basename(os.path.dirname(d))
            ev, lat = parse_suffix(os.path.basename(d), split)
            yield os.path.basename(root.rstrip("/")), split, ev, lat, path


def render(path, tag, categories, outdir, normalize=True, no_title=False,
           also_pdf=False, quiet=False):
    df = pd.read_csv(path)
    made = []
    for cat in categories:
        tcol, pcol = f"{cat}_true", f"{cat}_pred"
        if tcol not in df.columns or pcol not in df.columns:
            continue
        sub = df[[tcol, pcol]].dropna()
        if sub.empty:
            if not quiet:
                print(f"  {tag} {cat}: no scored queries, skipping")
            continue

        yt = sub[tcol].to_numpy(dtype=object)
        yp = sub[pcol].to_numpy(dtype=object)
        order = class_order_for(cat, set(yt) | set(yp))
        acc = float((yt == yp).mean())
        title = "" if no_title else f"{cat}  (n={len(yt)}, top1={acc:.2f})"
        out = os.path.join(outdir, f"{tag}_{cat}.png")

        n = plot_confusion(yt, yp, title, out, class_order=order,
                           normalize=normalize, also_pdf=also_pdf)
        made.append(out)
        if not quiet:
            print(f"  {tag} {cat:<13} n={len(yt):5d} classes={n:3d} "
                  f"top1={acc:.3f} -> {os.path.basename(out)}")
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--roots", nargs="+", help="run dirs, e.g. run_v72 run_v73h")
    src.add_argument("--pred", nargs="+", help="explicit predictions.csv paths")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                    help=f"default: {' '.join(DEFAULT_CATEGORIES)}")
    ap.add_argument("--outdir", default="./confusion_figs")
    ap.add_argument("--no-normalize", action="store_true",
                    help="plot counts rather than row-normalised fractions")
    ap.add_argument("--no-title", action="store_true",
                    help="omit panel titles (label them in the figure layout instead)")
    ap.add_argument("--pdf", action="store_true", help="also write vector PDFs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    normalize = not args.no_normalize
    made = []

    if args.roots:
        for run, split, ev, lat, path in find_predictions(args.roots):
            tag = f"{run}_{split}_{ev}" + (f"_{lat}" if lat != "?" else "")
            made += render(path, tag, args.categories, args.outdir,
                           normalize, args.no_title, args.pdf, args.quiet)
    else:
        for path in args.pred:
            d = os.path.dirname(path)
            split = os.path.basename(os.path.dirname(d))
            ev, lat = parse_suffix(os.path.basename(d), split)
            run = os.path.basename(d.split("/classification/")[0]) if "/classification/" in d else "run"
            tag = f"{run}_{split}_{ev}" + (f"_{lat}" if lat != "?" else "")
            made += render(path, tag, args.categories, args.outdir,
                           normalize, args.no_title, args.pdf, args.quiet)

    if not made:
        sys.exit("No matrices produced. Check --roots/--pred and --categories.")

    save_colorbar(os.path.join(args.outdir, "colorbar.png"),
                  label="Row-normalised fraction" if normalize else "Count",
                  vmax=1.0 if normalize else None, also_pdf=args.pdf)

    print(f"\nWrote {len(made)} matrices + colorbar.png to {args.outdir}")


if __name__ == "__main__":
    main()