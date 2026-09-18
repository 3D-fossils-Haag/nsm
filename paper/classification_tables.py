"""Manuscript tables from the classification results.

Read the tree written by classification_eval.py:
 
    <run>/classification/evaluation/<split>/<split>_<eval_level>_<latents>/
        metrics_summary.csv   report_<category>.csv   metrics.json
        predictions.csv

Writes two files by default:

    Table_S3_classification_stats.csv   one row per run x split x eval_level x
                                        latents x level -- the minimum needed to
                                        recompute everything quoted in the paper
    Table_2_classif_eval.csv            top-1 / top-5 accuracy for genus and
                                        spinal position, LOO vs LOSO
    Table_S4_classif_by_spec_num.csv    per-class recall vs how much data each
                                        class had, from predictions.csv
    Table_S5_classif_by_spec_spearmans.csv  the correlations behind Table S4

S3 and Table 2 describe every run given, so a copy goes into each run's paper
directory. S4 and S5 describe one run, so each directory gets its own.

Reachable metrics are the ones tabulated: under specimen masking a class with a
single specimen has no same-label gallery entry left once that specimen is
hidden, so its recall is forced to 0. That also means LOO and LOSO choose among
different numbers of classes, so n and the reachable-class count are emitted in
the first row of the table and belong in the caption.

Usage
-----
    # the two published tables
    python classif_tables.py --roots ../run_v72 ../run_v73h ../run_v73c

    # plus the diagnostic files, into a chosen directory
    python classif_tables.py --roots ../run_v72 ../run_v73h ../run_v73c \
        --outdir ./cls_out --extras
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import encoder
from NSM.evaluation import (EVAL_LABELS, RUN_LABELS, SPLIT_ORDER, class_support,
                            correlation_rows, find_results, normalize_labels,
                            write_supplement)

# carried through from metrics_summary.csv, in report order
METRICS = ["n_eval", "n_classes", "n_eligible", "n_reachable_classes",
           "top1_accuracy", "top5_accuracy",
           "top1_accuracy_reachable", "top5_accuracy_reachable",
           "macro_f1", "macro_f1_reachable", "weighted_f1"]

LEVELS = ["family", "genus", "species", "region", "position_10", "position_20",
          "life_history"]

# the paper table: which levels, and how they are titled in it
TABLE_LEVELS = {"genus": "Genus", "position_20": "Spinal position (20% bins)"}
TABLE_SPLITS = ("train", "test", "val")
TABLE_EVALS = ("loo", "specimen")
TABLE_LATENTS = "latent_opt"

S3_NAME = "Table_S3_classification_stats.csv"
T2_NAME = "Table_2_classif_eval.csv"

# the condition Tables S4 and S5 report, matching the published figure
SUPP_CATEGORY = "broad_taxon"
SUPP_EVAL = "specimen"
SUPP_LATENTS = "latent_opt"

# Define functions
def load_summary(res, levels):
    """metrics_summary.csv for one evaluation, tagged with its condition."""
    df = pd.read_csv(res.path)
    if "category" not in df.columns:
        print(f"  skipping {res.path}: no 'category' column", file=sys.stderr)
        return None
    df = df[df["category"].isin(levels)].copy()
    if df.empty:
        return None
    eval_level = res.eval_level
    jpath = os.path.join(os.path.dirname(res.path), "metrics.json")
    if os.path.exists(jpath):
        try:
            eval_level = json.load(open(jpath)).get("eval_level", eval_level)
        except (json.JSONDecodeError, OSError):
            pass

    df = df.rename(columns={"category": "level"})
    df.insert(0, "run", res.run)
    df.insert(1, "split", res.split)
    df.insert(2, "eval_level", eval_level)
    df.insert(3, "latents", res.latents)
    for c in METRICS:
        if c not in df.columns:
            df[c] = np.nan
    return df[["run", "split", "eval_level", "latents", "level"] + METRICS]

def build_summary(roots, levels, quiet=False):
    frames = []
    for res in find_results(roots, "metrics_summary.csv"):
        if not quiet:
            print(f"found {res.run:10s} {res.split:6s} {res.eval_level}_{res.latents}")
        df = load_summary(res, levels)
        if df is not None:
            frames.append(df)
    if not frames:
        sys.exit("No metrics_summary.csv files found. Check --roots.")

    rank = {"split": SPLIT_ORDER, "level": {lv: i for i, lv in enumerate(levels)}}
    summary = pd.concat(frames, ignore_index=True)
    return summary.sort_values(
        ["split", "eval_level", "latents", "level", "run"],
        key=lambda c: c.map(rank[c.name]) if c.name in rank else c).reset_index(drop=True)


def paper_table(summary, levels=TABLE_LEVELS, latents=TABLE_LATENTS,
                splits=TABLE_SPLITS, evals=TABLE_EVALS):
    """Stacked 'top1 / top5' blocks, one per level, in the published layout.

    Rows are conditions, columns are split x masking level. The first row of
    each block carries the query count and the reachable-class count, which
    differ between LOO and LOSO and so cannot be folded into a single caption.
    """
    d = summary[(summary["latents"] == latents)
                & (summary["eval_level"].isin(evals))]
    if d.empty:
        print(f"  no rows with latents={latents}; skipping {T2_NAME}",
              file=sys.stderr)
        return None

    blocks = []
    for level, level_label in levels.items():
        sub = d[d["level"] == level]
        if sub.empty:
            continue
        sub = sub.assign(Condition=sub["run"].map(lambda r: RUN_LABELS.get(r, r)))
        TOP1, TOP5 = "top1_accuracy_reachable", "top5_accuracy_reachable"
        N, NCLS = "n_eligible", "n_reachable_classes"
        p = sub.pivot_table(index="Condition", columns=["split", "eval_level"],
                            values=[TOP1, TOP5, N, NCLS])
        cols = {f"{s.capitalize()} {EVAL_LABELS.get(e, e)}": (s, e)
                for s in splits for e in evals if (TOP1, s, e) in p.columns}

        known = [c for c in RUN_LABELS.values() if c in p.index]
        conditions = known + [c for c in p.index if c not in known]

        rows = [{"Level": level_label, "Condition": "(N, classes)",
                 **{name: f"n={int(p[(N, *k)].max())}, "
                          f"{int(p[(NCLS, *k)].max())} cls"
                    for name, k in cols.items()}}]
        rows += [{"Level": level_label, "Condition": c,
                  **{name: f"{p.loc[c, (TOP1, *k)] * 100:.1f} / "
                           f"{p.loc[c, (TOP5, *k)] * 100:.1f}"
                     for name, k in cols.items()}}
                 for c in conditions]
        blocks.append(pd.DataFrame(rows))
    return pd.concat(blocks, ignore_index=True) if blocks else None

def per_family(roots, outdir):
    """Per-class recall from report_family.csv, runs side by side."""
    frames = []
    for res in find_results(roots, "report_family.csv"):
        df = pd.read_csv(res.path, index_col=0)
        df = df[~df.index.isin(["accuracy", "macro avg", "weighted avg"])]
        frames.append(pd.DataFrame({
            "family": df.index, "run": res.run, "split": res.split,
            "eval_level": res.eval_level, "latents": res.latents,
            "precision": df["precision"].values, "recall": df["recall"].values,
            "f1": df["f1-score"].values,
            "support": df["support"].astype(int).values}))
    if not frames:
        return None
    fam = pd.concat(frames, ignore_index=True)
    fam.to_csv(os.path.join(outdir, "per_family_long.csv"), index=False)
    keys = ["split", "eval_level", "latents", "family"]
    wide = fam.pivot_table(index=keys, columns="run", values="recall")
    wide = fam.groupby(keys)["support"].max().rename("support").to_frame() \
              .join(wide).reset_index()
    runs = [c for c in wide.columns if c not in keys + ["support"]]
    wide["spread"] = wide[runs].max(axis=1) - wide[runs].min(axis=1)
    wide = wide.sort_values(
        keys[:3] + ["support"], ascending=[True] * 3 + [False],
        key=lambda c: c.map(SPLIT_ORDER) if c.name == "split" else c)
    wide.to_csv(os.path.join(outdir, "per_family.csv"), index=False)
    return wide

def compare_runs(summary, metric="top1_accuracy_reachable"):
    """Spread across runs within each comparable (split, eval, latents, level)."""
    keys = ["split", "eval_level", "latents", "level"]
    rows = []
    for k, g in summary.groupby(keys, dropna=False):
        if g["run"].nunique() < 2:
            continue
        row = dict(zip(keys, k))
        row.update({r["run"]: r[metric] for _, r in g.iterrows()})
        vals = g[metric].astype(float)
        row["spread"] = vals.max() - vals.min()
        p, n = vals.mean(), g["n_eligible"].astype(float).mean()
        row["~1 SE"] = np.sqrt(p * (1 - p) / n) if n and np.isfinite(p) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def build_supplement(roots, args, outdir, quiet=False):
    """Tables S4 and S5, from the predictions of one condition.

    Every split is included: the correlation between recall and specimen count
    is only interpretable read across train, val and test together.
    """
    per_class, corr = [], []
    for res in find_results(roots, "predictions.csv", eval_level=args.supp_eval,
                            latents=args.supp_latents):
        df = normalize_labels(pd.read_csv(res.path), quiet=quiet)
        sup = class_support(df, args.supp_category)
        if sup is None or sup.empty:
            continue
        meta = {"run": res.run, "split": res.split}
        per_class.append(sup.assign(**meta))
        corr.extend(correlation_rows(sup, meta))
    if not per_class:
        print(f"  no predictions.csv with eval_level={args.supp_eval} "
              f"latents={args.supp_latents}; skipping S4 and S5", file=sys.stderr)
    return write_supplement(per_class, corr, outdir, quiet)

def write_tables(summary, args, outdir, verbose=True):
    """The table set for one output directory."""
    os.makedirs(outdir, exist_ok=True)
    summary.to_csv(os.path.join(outdir, S3_NAME), index=False)
    written = [S3_NAME]

    levels = {lv: TABLE_LEVELS.get(lv, lv.replace("_", " ").capitalize())
              for lv in args.table_levels}
    t2 = paper_table(summary, levels, latents=args.table_latents)
    if t2 is not None:
        t2.to_csv(os.path.join(outdir, T2_NAME), index=False)
        written.append(T2_NAME)
        if verbose:
            print(f"\n{'=' * 96}\nTABLE 2 - top-1 / top-5 accuracy (%), "
                  f"reachable classes\n{'=' * 96}")
            print(t2.to_string(index=False))

    if args.extras:
        show = ["run", "split", "eval_level", "latents", "level", "n_eval",
                "n_classes", "n_reachable_classes", "top1_accuracy",
                "top1_accuracy_reachable", "macro_f1_reachable"]
        md = summary[show].round(3)
        with open(os.path.join(outdir, "taxonomy_summary.txt"), "w") as fh:
            fh.write(md.to_string(index=False))
        written.append("taxonomy_summary.txt")

        cmp = compare_runs(summary)
        if not cmp.empty:
            cmp.to_csv(os.path.join(outdir, "run_comparison.csv"), index=False)
            written.append("run_comparison.csv")
            if verbose:
                print(f"\n{'=' * 78}\nCROSS-RUN SPREAD\n{'=' * 78}")
                print(cmp.round(4).to_string(index=False))
                print("\nspread <= ~1 SE means the runs are indistinguishable "
                      "on that row.")
        if per_family(args.roots, outdir) is not None:
            written += ["per_family.csv", "per_family_long.csv"]
    return written

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="run directories, e.g. ../run_v72 ../run_v73h ../run_v73c")
    ap.add_argument("--outdir", default=None,
                    help="one directory for the set (default: a copy under each "
                         "<root>/classification/evaluation/paper)")
    ap.add_argument("--levels", nargs="+", default=LEVELS,
                    help=f"levels to summarise (default: {' '.join(LEVELS)})")
    ap.add_argument("--table-levels", nargs="+", default=list(TABLE_LEVELS),
                    help="levels to put in the paper table")
    ap.add_argument("--table-latents", default=TABLE_LATENTS,
                    help="latent source used in the paper table")
    ap.add_argument("--supp-category", default=SUPP_CATEGORY,
                    help="category Tables S4 and S5 report")
    ap.add_argument("--supp-eval", default=SUPP_EVAL,
                    help="masking level for Tables S4 and S5")
    ap.add_argument("--supp-latents", default=SUPP_LATENTS,
                    help="latent source for Tables S4 and S5")
    ap.add_argument("--no-supplement", action="store_true",
                    help="skip Tables S4 and S5")
    ap.add_argument("--extras", action="store_true",
                    help="also write per-family recall, cross-run spread and a "
                         "readable copy of the summary")
    ap.add_argument("--quiet", action="store_true", help="write files without printing")
    args = ap.parse_args()

    summary = build_summary(args.roots, args.levels, args.quiet)
    # both tables describe every run given, so the same set goes into each run's
    # paper directory rather than only the first one's
    outdirs = ([args.outdir] if args.outdir else
               [os.path.join(r, "classification", "evaluation", "paper")
                for r in args.roots])
    for i, outdir in enumerate(outdirs):
        written = write_tables(summary, args, outdir, not args.quiet and i == 0)
        if not args.no_supplement:
            # one directory per run: its S4 and S5 describe that run alone.
            # A single --outdir instead holds every run in the one pair.
            roots = [args.roots[i]] if len(outdirs) == len(args.roots) else args.roots
            written += build_supplement(roots, args, outdir, args.quiet)
        print(f"\nWrote {', '.join(written)} to {outdir}")

if __name__ == "__main__":
    main()