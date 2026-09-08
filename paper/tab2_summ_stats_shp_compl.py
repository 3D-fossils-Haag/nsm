#!/usr/bin/env python3
"""
Table 2 -- full descriptive statistics for the NSM shape-completion evaluation.

Supplementary companion to Table 1 (parameters + median [IQR]). Reports, for
every condition x split: n, mean +/- SD, median [IQR], range, 95th percentile,
and the fraction of gross failures. Optionally adds a bootstrap CI on the median.

Chamfer distances are reported x10^3 throughout. Because the distributions are
right-skewed, median [IQR] is the primary summary; mean +/- SD is given for
comparability with other work.

Usage
-----
    python make_table2.py --datadir /path/to/csvs
    python make_table2.py --datadir . --outdir ./tables --ci --format latex

Outputs table2_descriptives.csv, plus a .tex or .md rendering if --format is given.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration -- edit here to add or reorder conditions
# --------------------------------------------------------------------------- #

# display label -> filename template (formatted with the split name)
MODELS = {
    "Baseline":       "v72_{split}_chamfer.csv",
    "Encoder":        "v72_encoder_{split}_chamfer.csv",
    "Encoder+refine": "v72_encoder_{split}_chamfer_refine.csv",
    "Hierarchy":      "v73h_{split}_chamfer.csv",
    "Contrastive":    "v73c_{split}_chamfer.csv",
}
SPLITS = {"train": "Train", "val": "Validation", "test": "Test"}

SCALE = 1e3             # report chamfer as x10^-3
TAIL = 0.015            # "gross failure" threshold, raw units
N_BOOT = 10_000         # bootstrap draws for the median CI
SEED = 0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_chamfer(path):
    """Read one chamfer CSV, tolerating a missing header row.

    shape_completion_eval.py normally writes a `mesh,chamfer,gt_path` header,
    but not every export has one. Reading a headerless file with pandas defaults
    silently promotes the first specimen to column names and drops it, so sniff
    the first line and read positionally when the header is absent.
    """
    with open(path) as fh:
        first = fh.readline()
    if first.startswith("mesh,"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=["mesh", "chamfer", "gt_path"])
        print(f"  note: no header in {os.path.basename(path)}; read positionally")
    return df[["mesh", "chamfer"]]


def load_all(datadir):
    frames = []
    for label, template in MODELS.items():
        for split in SPLITS:
            path = os.path.join(datadir, template.format(split=split))
            if not os.path.exists(path):
                print(f"  WARNING: missing {os.path.basename(path)} -- skipping")
                continue

            df = load_chamfer(path)
            df["mesh"] = df["mesh"].str.replace("_partial.ply", "", regex=False)

            # Some runs were re-executed and appended rather than overwritten,
            # leaving a mesh with 2-3 rows whose values differ by a few percent
            # (refinement is stochastic). Average the repeats and report it.
            n_dup = int(df["mesh"].duplicated().sum())
            if n_dup:
                spread = df.groupby("mesh")["chamfer"].agg(lambda x: x.max() / x.min() - 1).max()
                print(f"  note: {os.path.basename(path)} has {n_dup} duplicate row(s) "
                      f"(max within-mesh spread {100 * spread:.1f}%); averaging repeats")
                df = df.groupby("mesh", as_index=False)["chamfer"].mean()

            df["model"], df["split"] = label, split
            frames.append(df)

    if not frames:
        sys.exit(f"No chamfer CSVs found in {datadir}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #

def median_ci(x, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap 95% CI for the median."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True)
    meds = np.median(draws, axis=1)
    return np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def build_table(d, with_ci=False):
    rows = []
    for split, split_label in SPLITS.items():
        for model in MODELS:
            g = d[(d.split == split) & (d.model == model)]
            if not len(g):
                continue
            c = np.sort(g["chamfer"].to_numpy() * SCALE)

            row = {
                "Split": split_label,
                "Condition": model,
                "n": len(c),
                "Mean ± SD": f"{c.mean():.2f} ± {c.std(ddof=1):.2f}",
                "Median [IQR]": (f"{np.median(c):.2f} "
                                 f"[{np.percentile(c, 25):.2f}–{np.percentile(c, 75):.2f}]"),
                "Range": f"{c.min():.2f}–{c.max():.2f}",
                "P95": f"{np.percentile(c, 95):.2f}",
                f"% > {TAIL * SCALE:g}": f"{100 * np.mean(c > TAIL * SCALE):.2f}",
            }
            if with_ci:
                lo, hi = median_ci(c)
                row["Median 95% CI"] = f"[{lo:.2f}–{hi:.2f}]"
            rows.append(row)

    tab = pd.DataFrame(rows)
    # keep CI beside the median rather than at the end
    if with_ci:
        cols = list(tab.columns)
        cols.insert(cols.index("Median [IQR]") + 1, cols.pop(cols.index("Median 95% CI")))
        tab = tab[cols]
    return tab


def to_latex(tab):
    """booktabs table, one \\midrule between splits."""
    cols = list(tab.columns)
    out = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Chamfer distance ($\times10^{-3}$) by condition and split. "
           r"Distributions are right-skewed, so median [IQR] is the primary summary. "
           r"``\% $>$ 15'' is the proportion of meshes exceeding "
           r"$15\times10^{-3}$, i.e.\ gross reconstruction failures.}",
           r"\label{tab:descriptives}",
           r"\begin{tabular}{ll" + "r" * (len(cols) - 2) + "}", r"\toprule",
           " & ".join(c.replace("%", r"\%").replace("±", r"$\pm$").replace(">", r"$>$")
                     for c in cols) + r" \\",
           r"\midrule"]
    prev = None
    for _, r in tab.iterrows():
        if prev is not None and r["Split"] != prev:
            out.append(r"\midrule")
        cells = [("" if c == "Split" and r["Split"] == prev else str(r[c]))
                 for c in cols]
        out.append(" & ".join(x.replace("±", r"$\pm$").replace("–", "--") for x in cells) + r" \\")
        prev = r["Split"]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def to_markdown(tab):
    cols = list(tab.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in tab.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default=".", help="directory holding the chamfer CSVs")
    ap.add_argument("--outdir", default=".", help="where to write the table")
    ap.add_argument("--ci", action="store_true",
                    help="add a bootstrap 95%% CI for the median")
    ap.add_argument("--format", choices=["latex", "markdown"], default=None,
                    help="also emit a typeset rendering")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading chamfer CSVs from {args.datadir}")
    d = load_all(args.datadir)

    tab = build_table(d, with_ci=args.ci)

    print("\n" + "=" * 78)
    print(f"TABLE 2  Chamfer distance (×10⁻³) by condition and split")
    print("=" * 78)
    print(tab.to_string(index=False))

    csv_path = os.path.join(args.outdir, "table2_descriptives.csv")
    tab.to_csv(csv_path, index=False)
    written = [csv_path]

    if args.format == "latex":
        p = os.path.join(args.outdir, "table2_descriptives.tex")
        open(p, "w").write(to_latex(tab))
        written.append(p)
    elif args.format == "markdown":
        p = os.path.join(args.outdir, "table2_descriptives.md")
        open(p, "w").write(to_markdown(tab))
        written.append(p)

    # Flag any split where the conditions were not evaluated on the same meshes,
    # since an unequal n is the first thing a reviewer will ask about.
    print()
    for split, label in SPLITS.items():
        counts = d[d.split == split].groupby("model", observed=True).size()
        if len(counts) and counts.nunique() > 1:
            odd = counts[counts != counts.max()].to_dict()
            print(f"  NOTE: {label} n differs across conditions "
                  f"(most have {counts.max()}): {odd} -- footnote this.")

    print("\nWrote " + ", ".join(os.path.basename(p) for p in written) + f" to {args.outdir}")


if __name__ == "__main__":
    main()