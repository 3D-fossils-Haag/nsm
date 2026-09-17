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

Support statistics
------------------
Alongside the matrices, per-class recall is tabulated against how much data
each class had, and the two are correlated (Spearman). "How much data" is
counted two ways because they answer different questions:

    n_specimens   distinct individuals carrying that label
    n_vertebrae   meshes carrying that label

Both come from predictions.csv itself -- the specimen column is already there,
so no external metadata is needed. Under specimen masking (LOSO) the specimen
count is the one that can plausibly drive recall, since all vertebrae of the
held-out individual leave the gallery together; a class with many vertebrae
spread over few specimens has little left to retrieve. Reporting both makes
that distinction checkable rather than assumed.

Usage
-----
    # crawl runs and render everything, with support stats
    python confusion_matrices.py --roots run_v72 run_v73h run_v73c

    # one file, chosen categories
    python confusion_matrices.py \
        --pred run_v72/classification/evaluation/train/train_specimen_base/predictions.csv \
        --categories family region position_20

    # counts instead of row-normalised fractions
    python confusion_matrices.py --roots run_v72 --no-normalize

    # skip the correlation tables
    python confusion_matrices.py --roots run_v72 --no-support-stats

Outputs <run>_<split>_<evallevel>_<category>.png (and .pdf with --pdf) plus a
single colorbar.png, per_class_support.csv and support_correlations.csv, into
--outdir.

Note: predictions.csv stores <cat>_top5_hit as a boolean, not the top-5 labels,
so only top-1 matrices can be rebuilt here. Top-5 *recall* per class is still
recoverable from that boolean and is included in the support tables. For the
top-5 neighbourhood matrix, plot it inside classification_eval.py where the
labels are still in memory.
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    from scipy.stats import t as _scipy_t
    _HAVE_SCIPY = True
except ImportError:                                  # fall back to a normal approximation
    _HAVE_SCIPY = False

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

MIN_CLASSES_FOR_RHO = 4          # below this a rho is not worth reporting
SUPPORT_COLS = ["n_specimens", "n_vertebrae"]
RECALL_COLS = ["top1_recall", "top5_recall"]

# the one condition the supplemental file reports, so a reader gets a single
# table rather than the full crawl
SUPP_CATEGORY = "broad_taxon"
SUPP_EVAL = "specimen"
SUPP_LATENTS = "latent_opt"


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


def build_rows(df, cat, mode="top1"):
    """Scored queries as (y_true, y_pred_rows) for one category.

    mode="top1"  each query contributes its single nearest label. Rows sum to
                 1 and the diagonal is Top-1 recall.

    mode="top5"  a query whose true label appears anywhere in its top 5 is
                 credited to the diagonal; one that misses entirely is charged
                 to its top-1 prediction. Rows still sum to 1, the diagonal is
                 exactly Top-5 recall, and the off-diagonal now reads "the true
                 label was absent from the top 5, and the nearest neighbour was
                 this instead". Near-miss confusions -- where the right answer
                 was retrieved, just not first -- drop out, which is the point:
                 they are not errors under a Top-5 criterion.

    predictions.csv stores top-5 as a boolean hit, not as five labels, so a
    matrix of full top-5 neighbourhoods cannot be built here. This is the
    strongest top-5 view the file supports.
    """
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


# --------------------------------------------------------------------------
# support statistics
# --------------------------------------------------------------------------

def spearman(x, y):
    """Spearman rho, two-sided p and the n actually used.

    Returns (nan, nan, n) when the sample is too small or a variable is
    constant, which happens on three-class panels and on any category where
    every class has the same number of specimens.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, n

    if _HAVE_SCIPY:
        rho, p = _scipy_spearmanr(x, y)
        return float(rho), float(p), n

    # rank-Pearson with a normal approximation to the t distribution; adequate
    # from about n=8 upward, and flagged in the output so it is not mistaken
    # for an exact p
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if not np.isfinite(rho):
        return np.nan, np.nan, n
    if abs(rho) >= 1.0:
        return rho, 0.0, n
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    return rho, float(math.erfc(abs(t) / math.sqrt(2))), n


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z partialled out.

    Specimen count and vertebra count are themselves strongly correlated
    across classes, so their separate rhos are not independent evidence. This
    asks the sharper question: does x still track y once the shared variation
    with z is removed? Computed as a Pearson partial on ranks, with df = n-3.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    n = len(x)
    if n < 5 or np.ptp(x) == 0 or np.ptp(y) == 0 or np.ptp(z) == 0:
        return np.nan, np.nan, n

    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = math.sqrt(max(0.0, (1 - rxz ** 2) * (1 - ryz ** 2)))
    if denom == 0 or not np.isfinite(denom):
        return np.nan, np.nan, n

    rho = float((rxy - rxz * ryz) / denom)
    rho = max(-1.0, min(1.0, rho))
    if abs(rho) >= 1.0:
        return rho, 0.0, n

    df = n - 3
    tstat = rho * math.sqrt(df / (1 - rho ** 2))
    if _HAVE_SCIPY:
        p = float(2 * _scipy_t.sf(abs(tstat), df))
    else:
        p = float(math.erfc(abs(tstat) / math.sqrt(2)))
    return rho, p, n


def class_support(df, cat):
    """Per-class recall and the amount of data behind each class.

    n_vertebrae counts scored queries, not gallery entries, so it matches the
    denominator of the recall it sits next to.
    """
    tcol, pcol, hcol = f"{cat}_true", f"{cat}_pred", f"{cat}_top5_hit"
    if tcol not in df.columns or pcol not in df.columns:
        return None
    sub = df[df[tcol].notna() & df[pcol].notna()].copy()
    if sub.empty:
        return None

    sub["_hit1"] = (sub[tcol].astype(str) == sub[pcol].astype(str)).astype(float)
    g = sub.groupby(sub[tcol].astype(str))

    out = pd.DataFrame({"n_vertebrae": g.size(), "top1_recall": g["_hit1"].mean()})
    if hcol in sub.columns:
        out["top5_recall"] = g[hcol].apply(lambda s: s.astype(bool).mean())
    else:
        out["top5_recall"] = np.nan
    if "specimen" in sub.columns:
        out["n_specimens"] = g["specimen"].nunique()
    else:
        out["n_specimens"] = np.nan

    out.index.name = "class"
    return out.reset_index()[["class", "n_specimens", "n_vertebrae",
                              "top1_recall", "top5_recall"]]


def support_correlations(support, meta):
    """Every recall x support-measure pair for one (run, split, eval, cat).

    Each pair is reported twice: marginally, and with the other support
    measure partialled out. The partial rows are the ones that speak to
    whether specimens matter beyond raw mesh count.
    """
    rows = []
    for rec in RECALL_COLS:
        if rec not in support.columns or support[rec].isna().all():
            continue
        for sup in SUPPORT_COLS:
            if support[sup].isna().all():
                continue
            rho, p, n = spearman(support[rec], support[sup])
            rows.append({**meta, "recall": rec, "support": sup, "control": "",
                         "method": "spearman", "n_classes": n,
                         "rho": rho, "p": p, "approx_p": not _HAVE_SCIPY})

            other = [c for c in SUPPORT_COLS if c != sup]
            for ctl in other:
                if support[ctl].isna().all():
                    continue
                prho, pp, pn = partial_spearman(support[rec], support[sup],
                                                support[ctl])
                rows.append({**meta, "recall": rec, "support": sup,
                             "control": ctl, "method": "partial_spearman",
                             "n_classes": pn, "rho": prho, "p": pp,
                             "approx_p": not _HAVE_SCIPY})

    # how collinear the two support measures are, which is the context the
    # partials have to be read against
    if not support["n_specimens"].isna().all() and not support["n_vertebrae"].isna().all():
        rho, p, n = spearman(support["n_specimens"], support["n_vertebrae"])
        rows.append({**meta, "recall": "n_specimens", "support": "n_vertebrae",
                     "control": "", "method": "collinearity", "n_classes": n,
                     "rho": rho, "p": p, "approx_p": not _HAVE_SCIPY})
    return rows


def supplemental_table(cdf, pcdf, outdir, category=SUPP_CATEGORY,
                       eval_level=SUPP_EVAL, latents=SUPP_LATENTS,
                       run=None, quiet=False):
    """One wide table per split for a single condition, plus its class counts.

    The long files hold every run x split x eval x category combination, which
    is the right shape for checking things and the wrong shape for a
    supplement. This collapses one condition into a table a reader can take at
    face value: for each split and recall metric, both marginal rhos and both
    partials on one row.
    """
    sel = (cdf["category"] == category) & (cdf["eval_level"] == eval_level) & \
          (cdf["latents"] == latents)
    if run:
        sel &= cdf["run"] == run
    d = cdf[sel]
    if d.empty:
        if not quiet:
            print(f"  no rows for category={category} eval_level={eval_level} "
                  f"latents={latents}; skipping supplemental table", file=sys.stderr)
        return None

    rows = []
    keys = ["run", "split", "recall"]
    for (r, split, rec), g in d[d["recall"].isin(RECALL_COLS)].groupby(keys):
        def pick(method, sup, ctl=""):
            m = g[(g["method"] == method) & (g["support"] == sup) &
                  (g["control"] == ctl)]
            return (m["rho"].iloc[0], m["p"].iloc[0]) if len(m) else (np.nan, np.nan)

        rho_s, p_s = pick("spearman", "n_specimens")
        rho_v, p_v = pick("spearman", "n_vertebrae")
        prho_s, pp_s = pick("partial_spearman", "n_specimens", "n_vertebrae")
        prho_v, pp_v = pick("partial_spearman", "n_vertebrae", "n_specimens")

        coll = d[(d["run"] == r) & (d["split"] == split) &
                 (d["method"] == "collinearity")]
        rows.append({
            "run": r, "split": split, "metric": rec,
            "n_classes": int(g["n_classes"].max()),
            "rho_specimens": rho_s, "p_specimens": p_s,
            "rho_vertebrae": rho_v, "p_vertebrae": p_v,
            "partial_rho_specimens": prho_s, "partial_p_specimens": pp_s,
            "partial_rho_vertebrae": prho_v, "partial_p_vertebrae": pp_v,
            "rho_specimens_vs_vertebrae": coll["rho"].iloc[0] if len(coll) else np.nan,
        })

    supp = pd.DataFrame(rows)
    if supp.empty:
        return None
    order = {"train": 0, "val": 1, "test": 2}
    supp = supp.sort_values(["run", "metric", "split"],
                            key=lambda c: c.map(order) if c.name == "split" else c)
    for c in supp.columns:
        if supp[c].dtype.kind == "f":
            supp[c] = supp[c].round(4)
    supp.to_csv(os.path.join(outdir, "supplemental_support_stats.csv"), index=False)

    # the class-level numbers the correlations were computed from, so the
    # supplement is self-contained
    if pcdf is not None and not pcdf.empty:
        psel = (pcdf["category"] == category) & (pcdf["eval_level"] == eval_level) & \
               (pcdf["latents"] == latents)
        if run:
            psel &= pcdf["run"] == run
        pc = pcdf[psel].copy()
        if not pc.empty:
            cols = ["run", "split", "class", "n_specimens", "n_vertebrae",
                    "top1_recall", "top5_recall"]
            pc = pc[[c for c in cols if c in pc.columns]]
            pc = pc.sort_values(["run", "split", "class"],
                                key=lambda c: c.map(order) if c.name == "split" else c)
            for c in ("top1_recall", "top5_recall"):
                if c in pc.columns:
                    pc[c] = pc[c].round(3)
            pc.to_csv(os.path.join(outdir, "supplemental_per_class.csv"), index=False)

    if not quiet:
        print(f"\n{'=' * 92}")
        print(f"SUPPLEMENTAL: {category}, {eval_level} masking, {latents} latents")
        print("=" * 92)
        print(supp.to_string(index=False))
    return supp


def write_support_outputs(per_class, corr, outdir, quiet=False):
    pcdf = None
    if per_class:
        pcdf = pd.concat(per_class, ignore_index=True)
        pcdf.to_csv(os.path.join(outdir, "per_class_support.csv"), index=False)
    if not corr:
        return None, pcdf

    cdf = pd.DataFrame(corr)
    cdf = cdf.sort_values(["category", "split", "eval_level", "run",
                           "recall", "support", "method"]).reset_index(drop=True)
    cdf.to_csv(os.path.join(outdir, "support_correlations.csv"), index=False)

    # readable version of the rows that are worth quoting in text; markdown
    # needs tabulate, which is an optional pandas dependency, so fall back to
    # fixed-width text rather than losing the csv that was just written
    keep = cdf[cdf["n_classes"] >= MIN_CLASSES_FOR_RHO].copy()
    if not keep.empty:
        keep["rho"] = keep["rho"].round(3)
        keep["p"] = keep["p"].round(4)
        table = keep.drop(columns=["approx_p"])
        try:
            text, ext = table.to_markdown(index=False), ".md"
        except ImportError:
            text, ext = table.to_string(index=False), ".txt"
            if not quiet:
                print("  tabulate not installed; wrote support_correlations.txt "
                      "instead of .md", file=sys.stderr)
        with open(os.path.join(outdir, f"support_correlations{ext}"), "w") as fh:
            fh.write(text)

    if not quiet:
        print(f"\n{'=' * 92}")
        print("RECALL vs SUPPORT (Spearman)"
              + ("" if _HAVE_SCIPY else "  [scipy absent: p values are approximate]"))
        print("=" * 92)
        if keep.empty:
            print(f"no category had >= {MIN_CLASSES_FOR_RHO} classes")
        else:
            print(keep.to_string(index=False))
            print(f"\nrows with < {MIN_CLASSES_FOR_RHO} classes are in the csv but "
                  "omitted here: rho on three or four points is not interpretable.")
    return cdf, pcdf


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
           also_pdf=False, quiet=False, meta=None, per_class=None, corr=None,
           mode="top1"):
    """Render matrices for one predictions.csv, and collect its support stats.

    per_class and corr are appended to in place when support stats are wanted;
    pass None for either to skip that half.
    """
    df = pd.read_csv(path)
    made = []
    for cat in categories:
        yt, rows = build_rows(df, cat, mode=mode)
        if yt is None:
            if not quiet and f"{cat}_true" in df.columns:
                print(f"  {tag} {cat}: no scored queries for mode={mode}, skipping")
            continue

        present = set(yt) | {p for r in rows for p in r}
        order = class_order_for(cat, present)
        # diagonal rate under the chosen mode: top-1 accuracy, or top-5 recall
        acc = float(np.mean([t in r for t, r in zip(yt, rows)]))
        title = "" if no_title else f"{cat}  (n={len(yt)}, {mode}={acc:.2f})"
        out = os.path.join(outdir, f"{tag}_{cat}_{mode}.png")

        n = plot_confusion(yt, rows, title, out, class_order=order,
                           normalize=normalize, also_pdf=also_pdf)
        made.append(out)

        if per_class is not None or corr is not None:
            sup = class_support(df, cat)
            if sup is not None and not sup.empty:
                cmeta = {**(meta or {}), "category": cat}
                if per_class is not None:
                    per_class.append(sup.assign(**cmeta))
                if corr is not None:
                    corr.extend(support_correlations(sup, cmeta))

        if not quiet:
            print(f"  {tag} {cat:<13} n={len(yt):5d} classes={n:3d} "
                  f"{mode}={acc:.3f} -> {os.path.basename(out)}")
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
    ap.add_argument("--mode", choices=["top1", "top5", "both"], default="top1",
                    help="diagonal criterion: top1 (default), top5, or both")
    ap.add_argument("--no-support-stats", action="store_true",
                    help="skip per-class support tables and correlations")
    ap.add_argument("--supp-category", default=SUPP_CATEGORY,
                    help=f"category for the supplemental table (default: {SUPP_CATEGORY})")
    ap.add_argument("--supp-eval", default=SUPP_EVAL,
                    help=f"masking level for the supplemental table (default: {SUPP_EVAL})")
    ap.add_argument("--supp-latents", default=SUPP_LATENTS,
                    help=f"latent source for the supplemental table (default: {SUPP_LATENTS})")
    ap.add_argument("--supp-run", default=None,
                    help="restrict the supplemental table to one run (default: all)")
    ap.add_argument("--pdf", action="store_true", help="also write vector PDFs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    normalize = not args.no_normalize
    stats = not args.no_support_stats
    modes = ["top1", "top5"] if args.mode == "both" else [args.mode]
    per_class = [] if stats else None
    corr = [] if stats else None
    made = []

    if args.roots:
        for run, split, ev, lat, path in find_predictions(args.roots):
            tag = f"{run}_{split}_{ev}" + (f"_{lat}" if lat != "?" else "")
            meta = {"run": run, "split": split, "eval_level": ev, "latents": lat}
            for m in modes:
                made += render(path, tag, args.categories, args.outdir,
                               normalize, args.no_title, args.pdf, args.quiet,
                               meta=meta, per_class=per_class if m == modes[0] else None,
                               corr=corr if m == modes[0] else None, mode=m)
    else:
        for path in args.pred:
            d = os.path.dirname(path)
            split = os.path.basename(os.path.dirname(d))
            ev, lat = parse_suffix(os.path.basename(d), split)
            run = os.path.basename(d.split("/classification/")[0]) if "/classification/" in d else "run"
            tag = f"{run}_{split}_{ev}" + (f"_{lat}" if lat != "?" else "")
            meta = {"run": run, "split": split, "eval_level": ev, "latents": lat}
            for m in modes:
                made += render(path, tag, args.categories, args.outdir,
                               normalize, args.no_title, args.pdf, args.quiet,
                               meta=meta, per_class=per_class if m == modes[0] else None,
                               corr=corr if m == modes[0] else None, mode=m)

    if not made:
        sys.exit("No matrices produced. Check --roots/--pred and --categories.")

    save_colorbar(os.path.join(args.outdir, "colorbar.png"),
                  label="Row-normalised fraction" if normalize else "Count",
                  vmax=1.0 if normalize else None, also_pdf=args.pdf)

    cdf = pcdf = supp = None
    if stats:
        cdf, pcdf = write_support_outputs(per_class, corr, args.outdir, args.quiet)
        if cdf is not None:
            supp = supplemental_table(cdf, pcdf, args.outdir,
                                      category=args.supp_category,
                                      eval_level=args.supp_eval,
                                      latents=args.supp_latents,
                                      run=args.supp_run, quiet=args.quiet)

    print(f"\nWrote {len(made)} matrices + colorbar.png"
          + (", per_class_support.csv, support_correlations.csv" if cdf is not None else "")
          + (", supplemental_support_stats.csv, supplemental_per_class.csv"
             if supp is not None else "")
          + f" to {args.outdir}")


if __name__ == "__main__":
    main()