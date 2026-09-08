#!/usr/bin/env python3
"""
Collect and summarise latent-space classification results across runs and splits.

Crawls the output tree written by classification_eval.py:

    <run>/classification/evaluation/<split>/<suffix>/metrics_summary.csv
                                               ...  /report_<category>.csv
                                               ...  /metrics.json

Outputs
-------
    taxonomy_summary.csv    one row per run x split x eval_level x latents x level
    per_family.csv          per-class recall, runs side by side (family only)
    taxonomy_summary.md     the same summary as a readable table

Usage
-----
python collect_classif_eval_resul.py --roots run_v72 run_v73h run_v73c   --levels family genus species region position_10 position_20 life_history   --table-levels family genus region position_20   --outdir ./cls_out
Notes
-----
* Reachable metrics are the honest ones under specimen masking: a class with a
  single specimen has no same-label gallery entry left once that specimen is
  hidden, so its recall is forced to 0 and averaging it in understates every
  model equally. Both are reported; n_reachable_classes says how many of
  n_classes could actually be retrieved.
* Rows are only comparable within a matching (split, eval_level, latents)
  group. The summary is sorted so those rows sit together, and --compare
  prints the spread across runs within each group.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_LEVELS = ["family", "genus", "species"]

# columns carried through from metrics_summary.csv, in report order
METRICS = [
    "n_eval", "n_classes", "n_eligible", "n_reachable_classes",
    "top1_accuracy", "top5_accuracy",
    "top1_accuracy_reachable", "top5_accuracy_reachable",
    "macro_f1", "macro_f1_reachable",
    "weighted_f1",
]


def parse_suffix(suffix, split):
    """Recover eval_level and latent source from the output directory name.

    classification_eval.py builds "<split>_<eval_level>_<base|latent_opt>", but
    --suffix can override it, so anything unrecognised is passed through rather
    than guessed at.
    """
    s = re.sub(rf"^{re.escape(split)}_", "", suffix)
    m = re.match(r"^(loo|specimen|species|genus)_(base|latent_opt)$", s)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^(loo|specimen|species|genus)\b", s)
    return (m.group(1) if m else s), ("?" if not m else s[len(m.group(1)):].strip("_") or "?")


def find_results(roots):
    """Yield (run, split, suffix, directory) for every evaluation found."""
    for root in roots:
        base = os.path.join(root, "classification", "evaluation")
        if not os.path.isdir(base):
            print(f"  no evaluation directory under {root}", file=sys.stderr)
            continue
        for path in sorted(glob.glob(os.path.join(base, "*", "*", "metrics_summary.csv"))):
            d = os.path.dirname(path)
            suffix = os.path.basename(d)
            split = os.path.basename(os.path.dirname(d))
            yield os.path.basename(root.rstrip("/")), split, suffix, d


def load_summary(run, split, suffix, d, levels):
    df = pd.read_csv(os.path.join(d, "metrics_summary.csv"))
    if "category" not in df.columns:
        print(f"  skipping {d}: no 'category' column", file=sys.stderr)
        return None
    df = df[df["category"].isin(levels)].copy()
    if df.empty:
        return None

    eval_level, latents = parse_suffix(suffix, split)
    # metrics.json is authoritative for eval_level when present
    jpath = os.path.join(d, "metrics.json")
    if os.path.exists(jpath):
        try:
            eval_level = json.load(open(jpath)).get("eval_level", eval_level)
        except (json.JSONDecodeError, OSError):
            pass

    df.insert(0, "run", run)
    df.insert(1, "split", split)
    df.insert(2, "eval_level", eval_level)
    df.insert(3, "latents", latents)
    df = df.rename(columns={"category": "level"})

    for c in METRICS:
        if c not in df.columns:
            df[c] = np.nan
    return df[["run", "split", "eval_level", "latents", "level"] + METRICS]


def load_per_family(run, split, suffix, d):
    """Per-class rows from report_family.csv, minus sklearn's summary rows."""
    path = os.path.join(d, "report_family.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df = df[~df.index.isin(["accuracy", "macro avg", "weighted avg"])]
    eval_level, latents = parse_suffix(suffix, split)
    return pd.DataFrame({
        "family": df.index,
        "run": run, "split": split, "eval_level": eval_level, "latents": latents,
        "precision": df["precision"].values,
        "recall": df["recall"].values,
        "f1": df["f1-score"].values,
        "support": df["support"].astype(int).values,
    })


def compare_runs(summary, metric="top1_accuracy_reachable"):
    """Spread across runs within each comparable group."""
    keys = ["split", "eval_level", "latents", "level"]
    out = []
    for k, g in summary.groupby(keys, dropna=False):
        if g["run"].nunique() < 2:
            continue
        row = dict(zip(keys, k))
        for _, r in g.iterrows():
            row[r["run"]] = r[metric]
        vals = g[metric].astype(float)
        row["spread"] = vals.max() - vals.min()
        # binomial SE at the mean rate, as a yardstick for whether the
        # spread is larger than sampling noise on n_eligible queries
        p, n = vals.mean(), g["n_eligible"].astype(float).mean()
        row["~1 SE"] = np.sqrt(p * (1 - p) / n) if n and np.isfinite(p) else np.nan
        out.append(row)
    return pd.DataFrame(out)


# manuscript-ready display names, in presentation order
RUN_LABELS = {"run_v72": "Base", "run_v73h": "Hierarchy", "run_v73c": "Contrastive"}
EVAL_LABELS = {"loo": "LOO", "specimen": "LOSO"}


def manuscript_tables(summary, outdir, levels=("family", "genus"),
                      splits=("train", "test", "val"),
                      eval_levels=("loo", "specimen"),
                      latents="latent_opt", quiet=False):
    """Wide 'top1 / top5' tables, one per taxonomic level.

    Rows are conditions, column groups are splits, sub-columns are masking
    levels -- the same shape as the shape-completion table, with LOO/LOSO in
    place of mean/median.

    Reachable metrics are used because a family with a single specimen has no
    same-label gallery entry left once that specimen is hidden, so its recall
    is forced to 0. That also means LOO and LOSO choose among different numbers
    of classes, so n and the reachable-class count are emitted alongside and
    belong in the caption.
    """
    d = summary[summary["latents"] == latents]
    if d.empty:
        print(f"  no rows with latents={latents}; skipping manuscript tables", file=sys.stderr)
        return {}

    d = d.assign(Condition=d["run"].map(lambda r: RUN_LABELS.get(r, r)))
    order = [RUN_LABELS.get(r, r) for r in RUN_LABELS if r in set(d["run"])]
    order += [c for c in d["Condition"].unique() if c not in order]

    tables = {}
    for level in levels:
        sub = d[(d["level"] == level) & (d["eval_level"].isin(eval_levels))]
        if sub.empty:
            continue

        def grid(col):
            return sub.pivot_table(index="Condition", columns=["split", "eval_level"],
                                   values=col)

        t1, t5 = grid("top1_accuracy_reachable"), grid("top5_accuracy_reachable")
        n, ncls = grid("n_eligible"), grid("n_reachable_classes")
        keys = [(s, e) for s in splits for e in eval_levels if (s, e) in t1.columns]

        out = pd.DataFrame(index=[c for c in order if c in t1.index])
        header = {}
        for s, e in keys:
            name = f"{s.capitalize()} {EVAL_LABELS.get(e, e)}"
            out[name] = [f"{t1.loc[c, (s, e)] * 100:.1f} / {t5.loc[c, (s, e)] * 100:.1f}"
                         for c in out.index]
            header[name] = f"n={int(n[(s, e)].max())}, {int(ncls[(s, e)].max())} cls"

        out = pd.concat([pd.DataFrame([header], index=["(N, classes)"]), out])
        out.index.name = "Condition"
        tables[level] = out
        out.to_csv(os.path.join(outdir, f"table_{level}_top1_top5.csv"))

        if not quiet:
            print(f"\n{'=' * 96}")
            print(f"{level.upper()} - top-1 / top-5 accuracy (%), reachable classes")
            print("=" * 96)
            print(out.to_string())

    if tables:
        with open(os.path.join(outdir, "manuscript_tables.md"), "w") as fh:
            for level, t in tables.items():
                fh.write(f"**{level.capitalize()} - top-1 / top-5 accuracy (%), "
                         f"reachable classes**\n\n")
                fh.write(t.to_markdown() + "\n\n")
    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="run directories, e.g. run_v72 run_v73h run_v73c")
    ap.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS,
                    help=f"taxonomic levels to summarise (default: {' '.join(DEFAULT_LEVELS)})")
    ap.add_argument("--outdir", default=".", help="where to write the tables")
    ap.add_argument("--metric", default="top1_accuracy_reachable",
                    help="metric used for the cross-run comparison")
    ap.add_argument("--table-levels", nargs="+", default=["family", "genus"],
                    help="levels to emit manuscript tables for")
    ap.add_argument("--table-latents", default="latent_opt",
                    help="latent source used in the manuscript tables")
    ap.add_argument("--quiet", action="store_true", help="write files without printing")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    summaries, families = [], []
    for run, split, suffix, d in find_results(args.roots):
        if not args.quiet:
            print(f"found {run:10s} {split:6s} {suffix}")
        s = load_summary(run, split, suffix, d, args.levels)
        if s is not None:
            summaries.append(s)
        f = load_per_family(run, split, suffix, d)
        if f is not None:
            families.append(f)

    if not summaries:
        sys.exit("No metrics_summary.csv files found. Check --roots.")

    order = {lv: i for i, lv in enumerate(args.levels)}
    summary = pd.concat(summaries, ignore_index=True)
    summary = summary.sort_values(
        ["split", "eval_level", "latents", "level", "run"],
        key=lambda c: c.map(order) if c.name == "level" else c,
    ).reset_index(drop=True)
    summary.to_csv(os.path.join(args.outdir, "taxonomy_summary.csv"), index=False)

    if families:
        fam = pd.concat(families, ignore_index=True)
        fam.to_csv(os.path.join(args.outdir, "per_family_long.csv"), index=False)
        # support can differ by a mesh or two between runs (a query is dropped
        # when all its neighbours are excluded), so keep it out of the pivot
        # index or one family splits into several rows
        keys = ["split", "eval_level", "latents", "family"]
        wide = fam.pivot_table(index=keys, columns="run", values="recall")
        sup = fam.groupby(keys)["support"].max().rename("support")
        wide = sup.to_frame().join(wide).reset_index()
        runs = [c for c in wide.columns if c not in keys + ["support"]]
        wide["spread"] = wide[runs].max(axis=1) - wide[runs].min(axis=1)
        wide = wide.sort_values(keys[:3] + ["support"], ascending=[True] * 3 + [False])
        wide.to_csv(os.path.join(args.outdir, "per_family.csv"), index=False)

    cmp = compare_runs(summary, args.metric)
    if not cmp.empty:
        cmp.to_csv(os.path.join(args.outdir, "run_comparison.csv"), index=False)

    # readable markdown of the main summary
    show = ["run", "split", "eval_level", "latents", "level", "n_eval",
            "n_classes", "n_reachable_classes", "top1_accuracy",
            "top1_accuracy_reachable", "macro_f1_reachable"]
    md = summary[[c for c in show if c in summary.columns]].copy()
    for c in md.columns:
        if md[c].dtype.kind == "f":
            md[c] = md[c].round(3)
    with open(os.path.join(args.outdir, "taxonomy_summary.md"), "w") as fh:
        fh.write(md.to_markdown(index=False))

    if not args.quiet:
        print(f"\n{'=' * 78}\nTAXONOMY SUMMARY\n{'=' * 78}")
        print(md.to_string(index=False))
        if not cmp.empty:
            print(f"\n{'=' * 78}\nCROSS-RUN SPREAD ({args.metric})\n{'=' * 78}")
            print(cmp.round(4).to_string(index=False))
            print("\nspread <= ~1 SE means the runs are indistinguishable on that row.")

    tables = manuscript_tables(summary, args.outdir, levels=args.table_levels,
                               latents=args.table_latents, quiet=args.quiet)

    print(f"\nWrote taxonomy_summary.csv, taxonomy_summary.md"
          + (f", table_<level>_top1_top5.csv, manuscript_tables.md" if tables else "")
          + (", per_family.csv, per_family_long.csv" if families else "")
          + (", run_comparison.csv" if not cmp.empty else "")
          + f" to {args.outdir}")


if __name__ == "__main__":
    main()
