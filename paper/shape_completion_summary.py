#!/usr/bin/env python3
"""
Summary statistics, paired inferential tests, and manuscript figures for NSM
shape-completion evaluation.

Inputs (one CSV per model x split, columns: mesh, chamfer, gt_path), produced by
shape_completion_eval.py:

    {model}_{split}_chamfer.csv   for model in {v72, v72_encoder, v73h, v73c}
                                  and split in {train, val, test}

Plus the grid-search config workbook from shape_completion_grid_search.py:

    2phaseopt_best_cfg_all_models.ods

Outputs (written to --outdir):

    table1_gridsearch_config.csv   best hyperparameters per model
    table2_descriptives.csv        n, mean +/- SD, median [IQR], range, p95
    table3_paired_tests.csv        Friedman + Holm-corrected Wilcoxon vs baseline
    table4_generalization_gap.csv  median chamfer by split, train->test gap
    fig_shape_completion.{pdf,png}     boxplots + ECDFs by split
    fig_generalization_gap.{pdf,png}   median chamfer train->val->test

Usage:
    python shape_completion_summary.py --datadir /path/to/csvs --outdir ./out

Notes
-----
* Chamfer distances are reported x10^3 throughout for legibility. The raw values
  are in normalized model units (meshes unit-scaled before evaluation).
* All four models are evaluated on an identical set of fragmented meshes within
  each split, so every model-vs-model comparison is paired. Paired
  (signed-rank) tests are used rather than independent-sample tests; this is
  both more powerful and the correct error model.
* Chamfer distributions are right-skewed (see the p95 and max columns), so
  median [IQR] is the primary descriptive statistic and mean +/- SD is reported
  secondarily.
* MISSING-HEADER GUARD: at least one export (v72_train_chamfer.csv) was written
  without a header row, which causes a naive pd.read_csv to silently consume the
  first specimen as column names and drop it (2315 rows instead of 2316).
  load_chamfer() sniffs the first line and reads positionally when needed, so
  results are correct whether or not the header is present.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# run directory prefix -> label used in tables and figures
# label -> filename template (formatted with the split name)
MODELS = {
    "Baseline":       "v72_{split}_chamfer.csv",
    "Encoder":        "v72_encoder_{split}_chamfer.csv",         # pure feedforward, untuned
    "Encoder+refine": "v72_encoder_{split}_chamfer_refine.csv",  # + grid-searched refinement
    "Hierarchy":      "v73h_{split}_chamfer.csv",
    "Contrastive":    "v73c_{split}_chamfer.csv",
}
REFERENCE = "Baseline"          # model all others are compared against
ORDER = ["Baseline", "Encoder", "Encoder+refine", "Hierarchy", "Contrastive"]
SPLITS = ["train", "val", "test"]
COLORS = dict(zip(ORDER, ["#4C4C4C", "#C1666B", "#E08A3C", "#48A9A6", "#4059AD"]))
SCALE = 1e3                     # display chamfer as x10^-3
TAIL_THRESHOLD = 0.015          # "gross failure" cutoff in raw units
RNG = np.random.default_rng(0)  # jitter in the strip plot only

# Mean inference time per mesh (s), from the grid-search workbook.
RUNTIME = {"Baseline": 198.45, "Encoder": 6.53, "Encoder+refine": 26.71,
           "Hierarchy": 133.30, "Contrastive": 167.07}

# Grid-search mean_cd is a SELECTION CRITERION, computed on 30 randomly drawn
# validation meshes -- not a performance estimate, and not on the same footing
# as the full-split numbers in Table 2. Renamed in Table 1 to discourage the
# comparison; see GRIDSEARCH_N.
GRIDSEARCH_N = 30

# grid-search trial name -> evaluation label
TRIAL_TO_MODEL = {
    "base_best": "Baseline",
    "encoder_base": "Encoder",
    "encoder_refine": "Encoder+refine",
    "hierarchy_best": "Hierarchy",
    "contrastive_best": "Contrastive",
}

# parameter rows for the transposed main table, in presentation order
PARAM_ROWS = [
    ("top_k", "top_k"), ("iters1", "Phase 1 iterations"), ("iters2", "Phase 2 iterations"),
    ("lr1", "Phase 1 LR"), ("lr2", "Phase 2 LR"),
    ("lambda1", "λ1"), ("lambda2", "λ2"),
    ("clamp1", "Clamp 1"), ("clamp2", "Clamp 2"),
    ("latent_std", "Latent init SD"),
    ("sched_step", "Scheduler step"), ("sched_gamma1", "γ1"), ("sched_gamma2", "γ2"),
    ("batch_infer", "Inference batch"), ("gridN", "Grid resolution"),
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_chamfer(path):
    """Read one chamfer CSV, tolerating a missing header row.

    shape_completion_eval.py normally writes a `mesh,chamfer,gt_path` header,
    but not every export has one. Reading a headerless file with pandas
    defaults promotes the first data row to column names and loses that
    specimen, so sniff the first line and read positionally when needed.
    """
    with open(path) as fh:
        first = fh.readline()
    if first.startswith("mesh,"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=["mesh", "chamfer", "gt_path"])
        print(f"  note: no header in {os.path.basename(path)}; read positionally")
    return df


def load_all(datadir):
    """Assemble every model x split CSV into one long-format frame."""
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
            # leaving a mesh with 2-3 rows whose chamfer values differ by a few
            # percent (refinement is stochastic). Average the repeats and say so.
            n_dup = int(df["mesh"].duplicated().sum())
            if n_dup:
                spread = (df.groupby("mesh")["chamfer"].agg(lambda x: x.max() / x.min() - 1).max())
                print(f"  note: {os.path.basename(path)} has {n_dup} duplicate row(s) "
                      f"across {df['mesh'].duplicated(keep=False).sum()} rows "
                      f"(max within-mesh spread {100 * spread:.1f}%); averaging repeats")
                df = df.groupby("mesh", as_index=False)["chamfer"].mean()

            df["model"] = label
            df["split"] = split
            frames.append(df[["mesh", "chamfer", "model", "split"]])
    if not frames:
        sys.exit(f"No chamfer CSVs found in {datadir}")

    d = pd.concat(frames, ignore_index=True)
    d["model"] = pd.Categorical(d["model"], ORDER, ordered=True)
    return d


def check_integrity(d):
    """Verify the assumptions the paired tests rely on."""
    print("\nRows per model x split:")
    print(d.groupby(["split", "model"], observed=True).size().unstack().to_string())

    n_bad = int(d["chamfer"].isna().sum() + np.isinf(d["chamfer"]).sum())
    print(f"\nNon-finite chamfer values: {n_bad}")

    print("\nPairing check (identical mesh sets across models, no duplicates):")
    ok = True
    for split in SPLITS:
        sub = d[d.split == split]
        sets = {m: set(g["mesh"]) for m, g in sub.groupby("model", observed=True)}
        if not sets:
            continue
        ref = sets.get(REFERENCE) or next(iter(sets.values()))
        diffs = {m: len(v ^ ref) for m, v in sets.items()}
        dups = {m: int(g["mesh"].duplicated().sum())
                for m, g in sub.groupby("model", observed=True)}
        n_common = len(set.intersection(*sets.values())) if sets else 0
        bad = any(diffs.values()) or any(dups.values())
        ok &= not bad
        print(f"  {split:5s} union n={len(set.union(*sets.values())):5d}  "
              f"complete-case n={n_common:5d}  mismatches vs {REFERENCE}={diffs}"
              f"  {'OK' if not bad else '  <-- paired tests use complete cases only'}")
    if not ok:
        print("  Mesh sets differ across models; paired tests below will drop"
              " unmatched meshes.")
    return ok


def wide(d, split):
    """Mesh x model matrix for one split, restricted to complete cases.

    A model that is merely *incomplete* (e.g. Encoder+refine is missing 20 test
    meshes) must not be dropped from the comparison entirely -- instead drop the
    affected meshes, so every model is compared on an identical subset.
    """
    w = d[d.split == split].pivot(index="mesh", columns="model", values="chamfer")
    cols = [c for c in ORDER if c in w.columns and w[c].notna().any()]
    return w[cols].dropna(axis=0, how="any")


# --------------------------------------------------------------------------- #
# Table 1: grid-search configurations
# --------------------------------------------------------------------------- #

def table1_gridsearch(ods_path, outdir):
    """Tidy the two-phase grid-search workbook into a per-model config table."""
    if not os.path.exists(ods_path):
        print(f"  WARNING: {os.path.basename(ods_path)} not found -- skipping Table 1")
        return None

    raw = pd.read_excel(ods_path, engine="odf")
    # The sheet carries a repeated header row and free-text notes below the
    # data block; keep only rows with a parseable trial label and mean_cd.
    tab = raw[raw["trial"].notna() & pd.to_numeric(raw["mean_cd"], errors="coerce").notna()].copy()
    tab["mean_cd"] = pd.to_numeric(tab["mean_cd"])

    # latent_std is exported as the repr of a torch scalar, e.g. "tensor(0.4402)"
    if "latent_std" in tab.columns:
        tab["latent_std"] = (
            tab["latent_std"].astype(str)
            .str.extract(r"([\d.]+)")[0].astype(float)
        )

    tab[f"selection CD x1e3 (n={GRIDSEARCH_N} val)"] = (tab["mean_cd"] * SCALE).round(3)
    tab = tab.rename(columns={"mean_time": "time/mesh (s)"})
    keep = ["trial", f"selection CD x1e3 (n={GRIDSEARCH_N} val)", "time/mesh (s)", "top_k", "iters1", "iters2",
            "lr1", "lr2", "lambda1", "lambda2", "clamp1", "clamp2",
            "latent_std", "sched_step", "sched_gamma1", "sched_gamma2",
            "batch_infer", "gridN"]
    tab = tab[[c for c in keep if c in tab.columns]]

    path = os.path.join(outdir, "table1_gridsearch_config.csv")
    tab.to_csv(path, index=False)
    print("\n" + "=" * 78)
    print("TABLE 1  Best configuration per model (two-phase grid search)")
    print("=" * 78)
    print(tab.to_string(index=False))
    print(f"  NOTE: selection CD is the grid-search objective on {GRIDSEARCH_N} random val\n"
          f"        meshes. It selects configs; it does not estimate performance.\n"
          f"        Do not compare it to Table 2.")
    return tab


# --------------------------------------------------------------------------- #
# Table 2: descriptive statistics
# --------------------------------------------------------------------------- #

def table1_combined(d, ods_path, outdir):
    """Main table: optimization parameters (rows) x condition (columns), with
    grid-search selection and full-split evaluation appended as further rows.

    Transposing is what makes this fit: 15 parameters as columns is unusable in
    print, but as rows it is a normal portrait table, and appending three
    evaluation rows costs almost nothing. Blocks are kept visually separate so
    the n=30 selection criterion is never read as a performance estimate.
    """
    if not os.path.exists(ods_path):
        print(f"  WARNING: {os.path.basename(ods_path)} not found -- skipping combined table")
        return None

    raw = pd.read_excel(ods_path, engine="odf")
    raw = raw[raw["trial"].isin(TRIAL_TO_MODEL)].copy()
    raw["model"] = raw["trial"].map(TRIAL_TO_MODEL)
    if "latent_std" in raw.columns:
        raw["latent_std"] = (raw["latent_std"].astype(str)
                             .str.extract(r"([\d.]+)")[0].astype(float))
    cfg = raw.set_index("model")

    cols = [m for m in ORDER if m in cfg.index]

    def fmt(v):
        if pd.isna(v) or str(v).strip() in {"-", ""}:
            return "—"
        f = float(v)
        return str(int(f)) if f == int(f) and abs(f) >= 1 else f"{f:g}"

    rows = [("__block__", "Optimization parameters")]
    for key, label in PARAM_ROWS:
        if key in cfg.columns:
            rows.append((label, [fmt(cfg.loc[m, key]) for m in cols]))

    rows.append(("__block__", f"Grid search (n={GRIDSEARCH_N} validation meshes)"))
    rows.append((f"Selection CD (×10⁻³)",
                 [f"{cfg.loc[m, 'mean_cd'] * SCALE:.2f}" for m in cols]))
    rows.append(("Inference time / mesh (s)",
                 [f"{cfg.loc[m, 'mean_time']:.1f}" for m in cols]))

    rows.append(("__block__", "Evaluation — median [IQR] Chamfer (×10⁻³)"))
    for split, label in zip(SPLITS, ["Train", "Validation", "Test"]):
        cells = []
        for m in cols:
            v = d.loc[(d.split == split) & (d.model == m), "chamfer"].to_numpy() * SCALE
            cells.append("—" if not len(v) else
                         f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}–{np.percentile(v, 75):.2f}]")
        rows.append((f"{label} (n={d[(d.split == split)].groupby('model', observed=True).size().max()})", cells))

    out, printable = [], []
    for label, val in rows:
        if label == "__block__":
            out.append([val] + [""] * len(cols))
            printable.append((val, [""] * len(cols)))
        else:
            out.append([label] + list(val))
            printable.append((label, list(val)))

    tab = pd.DataFrame(out, columns=["Parameter"] + cols)
    tab.to_csv(os.path.join(outdir, "table1_params_and_summary.csv"), index=False)

    width = max(len(l) for l, _ in printable) + 1
    cw = max(max((len(c) for _, vals in printable for c in vals), default=8), 14) + 2
    print("\n" + "=" * 78)
    print("TABLE 1  Optimization parameters and evaluation summary")
    print("=" * 78)
    print(" " * width + "".join(c.rjust(cw) for c in cols))
    for label, vals in printable:
        if all(v == "" for v in vals):
            print(f"\n{label}")
        else:
            print(label.ljust(width) + "".join(v.rjust(cw) for v in vals))
    return tab


def table2_descriptives(d, outdir):
    rows = []
    for split in SPLITS:
        for model in ORDER:
            g = d[(d.split == split) & (d.model == model)]
            if not len(g):
                continue
            c = g["chamfer"].to_numpy() * SCALE
            rows.append({
                "Split": split,
                "Model": model,
                "n": len(c),
                "Mean ± SD": f"{c.mean():.2f} ± {c.std(ddof=1):.2f}",
                "Median [IQR]": (f"{np.median(c):.2f} "
                                 f"[{np.percentile(c, 25):.2f}–{np.percentile(c, 75):.2f}]"),
                "Range": f"{c.min():.2f}–{c.max():.2f}",
                "95th pct": f"{np.percentile(c, 95):.2f}",
                f"% > {TAIL_THRESHOLD * SCALE:g}":
                    f"{100 * np.mean(c > TAIL_THRESHOLD * SCALE):.2f}",
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, "table2_descriptives.csv"), index=False)
    print("\n" + "=" * 78)
    print(f"TABLE 2  Chamfer distance by model and split (x10^-3, n={len(d)} evaluations)")
    print("=" * 78)
    print(tab.to_string(index=False))
    return tab


# --------------------------------------------------------------------------- #
# Table 3: paired inferential tests
# --------------------------------------------------------------------------- #

def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    k = len(p)
    order = np.argsort(p)
    adj = np.empty(k)
    adj[order] = np.maximum.accumulate(p[order] * (k - np.arange(k)))
    return np.minimum(adj, 1.0)


def table3_paired(d, outdir):
    """Friedman omnibus per split, then Wilcoxon signed-rank vs the reference.

    Effect is reported as the median paired difference (a Hodges-Lehmann-style
    location shift) plus the proportion of meshes improved, which is more
    interpretable for a morphology readership than a standardized effect size.
    """
    rows = []
    for split in SPLITS:
        w = wide(d, split)
        if w.shape[1] < 2:
            continue
        ref = REFERENCE if REFERENCE in w.columns else w.columns[0]
        others = [c for c in w.columns if c != ref]

        if w.shape[1] > 2:
            chi2, p_omni = stats.friedmanchisquare(*[w[c] for c in w.columns])
        else:
            chi2, p_omni = np.nan, np.nan

        raw_p, staged = [], []
        for c in others:
            statistic, p = stats.wilcoxon(w[c], w[ref])
            diff = w[c] - w[ref]
            staged.append({
                "Split": split,
                "Comparison": f"{c} vs {ref}",
                "n pairs": len(w),
                "Δ median (×10⁻³)": round(float(np.median(diff)) * SCALE, 4),
                "Δ %": round(100 * float(np.median(diff)) / float(np.median(w[ref])), 1),
                "% meshes improved": round(100 * float((diff < 0).mean()), 1),
                "W": int(statistic),
                "p (raw)": p,
                "Friedman χ²": round(float(chi2), 1) if np.isfinite(chi2) else "",
                "Friedman p": f"{p_omni:.2e}" if np.isfinite(p_omni) else "",
            })
            raw_p.append(p)

        for row, p_adj in zip(staged, holm(raw_p)):
            row["p (Holm)"] = f"{p_adj:.2e}"
            row["p (raw)"] = f"{row['p (raw)']:.2e}"
            rows.append(row)

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, "table3_paired_tests.csv"), index=False)
    print("\n" + "=" * 78)
    print(f"TABLE 3  Paired comparisons vs {REFERENCE} (Wilcoxon signed-rank, Holm-corrected)")
    print("=" * 78)
    print(tab.to_string(index=False))

    # Mean rank and win rate: a compact companion to the pairwise tests.
    print("\nPer-split mean rank (1 = best) and share of meshes where each model wins:")
    for split in SPLITS:
        w = wide(d, split)
        if w.empty:
            continue
        ranks = w.rank(axis=1).mean()
        wins = w.idxmin(axis=1).value_counts(normalize=True) * 100
        print(f"  {split:5s} " + "  ".join(
            f"{c}: rank {ranks[c]:.2f} / best {wins.get(c, 0.0):4.1f}%" for c in w.columns))
    return tab


# --------------------------------------------------------------------------- #
# Table 4: generalization gap
# --------------------------------------------------------------------------- #

def table4_gap(d, outdir):
    """Median chamfer per split, with the train->test degradation."""
    med = (d.pivot_table(index="model", columns="split", values="chamfer",
                         aggfunc="median", observed=True)
           .reindex(index=ORDER, columns=SPLITS) * SCALE)
    med["test − train"] = med["test"] - med["train"]
    med["gap %"] = 100 * (med["test"] / med["train"] - 1)
    tab = med.round(2).reset_index()
    tab.to_csv(os.path.join(outdir, "table4_generalization_gap.csv"), index=False)
    print("\n" + "=" * 78)
    print("TABLE 4  Generalization gap (median chamfer, x10^-3)")
    print("=" * 78)
    print(tab.to_string(index=False))
    return tab


# --------------------------------------------------------------------------- #
# Figure 1: distributions and ECDFs
# --------------------------------------------------------------------------- #

def figure_distributions(d, outdir):
    """Panel A: box + strip plot per split. Panel B: ECDF per split.

    The ECDF row is the load-bearing panel: it shows whether a model dominates
    across the whole distribution or only near the median.
    """
    labels = {s: f"{name} (n={d[d.split == s].mesh.nunique():,})"
              for s, name in zip(SPLITS, ["Train", "Validation", "Test"])}

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6),
                             gridspec_kw=dict(height_ratios=[1.25, 1]))

    for j, split in enumerate(SPLITS):
        sub = d[d.split == split]
        cols = [m for m in ORDER if m in sub["model"].unique()]
        if not cols:
            continue

        # --- Panel A: boxplot with jittered points and a mean marker ---
        ax = axes[0, j]
        data = [sub.loc[sub.model == m, "chamfer"].to_numpy() * SCALE for m in cols]
        pos = [ORDER.index(m) for m in cols]
        bp = ax.boxplot(data, positions=pos, widths=0.6, showfliers=False,
                        patch_artist=True,
                        medianprops=dict(color="k", lw=1.4),
                        whiskerprops=dict(color="#666"),
                        capprops=dict(color="#666"))
        for box, m in zip(bp["boxes"], cols):
            box.set(facecolor=COLORS[m], alpha=0.55, edgecolor="#444")
        for p, vals, m in zip(pos, data, cols):
            ax.scatter(RNG.normal(p, 0.055, len(vals)), vals, s=1.6,
                       color=COLORS[m], alpha=0.18, zorder=0, rasterized=True)
            ax.scatter([p], [vals.mean()], marker="D", s=22, color="w",
                       edgecolor="k", lw=0.8, zorder=5)

        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, rotation=25, ha="right", fontsize=8.5)
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.set_ylim(2, 22)          # clips the extreme tail; noted in caption
        ax.set_title(labels[split], fontsize=10, pad=6)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Chamfer distance (×10⁻³)", fontsize=9.5)
        else:
            ax.set_yticklabels([])

        # --- Panel B: empirical CDF ---
        ax = axes[1, j]
        for m in cols:
            v = np.sort(sub.loc[sub.model == m, "chamfer"].to_numpy() * SCALE)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLORS[m],
                    lw=1.7, label=m)
        ax.set_xlim(2, 16)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Chamfer distance (×10⁻³)", fontsize=9.5)
        if j == 0:
            ax.set_ylabel("Cumulative proportion\nof meshes", fontsize=9.5)
        else:
            ax.set_yticklabels([])
        if j == len(SPLITS) - 1:
            ax.legend(fontsize=8, frameon=False, loc="lower right")

    for ax, letter in zip([axes[0, 0], axes[1, 0]], ["A", "B"]):
        ax.text(-0.26, 1.02, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_shape_completion.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: generalization gap
# --------------------------------------------------------------------------- #

def figure_gap(d, outdir):
    """Slopegraph of median chamfer across train -> val -> test."""
    med = (d.pivot_table(index="model", columns="split", values="chamfer",
                         aggfunc="median", observed=True)
           .reindex(index=ORDER, columns=SPLITS) * SCALE)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    xs = range(len(SPLITS))
    for m in ORDER:
        if m not in med.index or med.loc[m].isna().any():
            continue
        y = med.loc[m, SPLITS].to_numpy()
        ax.plot(xs, y, "-o", color=COLORS[m], lw=2, ms=6, label=m, zorder=3)
        ax.annotate(f"{y[-1]:.2f}", (len(SPLITS) - 1, y[-1]), xytext=(7, 0),
                    textcoords="offset points", fontsize=8.5,
                    color=COLORS[m], va="center", fontweight="bold")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Train", "Validation", "Test"], fontsize=10)
    ax.set_xlim(-0.25, len(SPLITS) - 0.45)
    ax.set_ylabel("Median Chamfer distance (×10⁻³)", fontsize=10)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title("Generalization across splits", fontsize=10.5, pad=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_generalization_gap.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def figure_tradeoff(d, outdir):
    """Accuracy vs. inference cost, using the grid-search timings."""
    med = (d[d.split == "test"].groupby("model", observed=True)["chamfer"].median() * SCALE)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for m in ORDER:
        if m not in RUNTIME or m not in med.index or not np.isfinite(med.get(m, np.nan)):
            continue
        ax.scatter(RUNTIME[m], med[m], s=110, color=COLORS[m], edgecolor="k",
                   lw=0.8, zorder=3, label=m)
        ax.annotate(m, (RUNTIME[m], med[m]), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xlabel("Mean inference time per mesh (s)", fontsize=10)
    ax.set_ylabel("Median test Chamfer distance (×10⁻³)", fontsize=10)
    ax.set_xlim(-8, 230)
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Accuracy vs. inference cost (test split)", fontsize=10.5, pad=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_accuracy_vs_cost.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default=".", help="directory holding the chamfer CSVs")
    ap.add_argument("--ods", default=None,
                    help="grid-search workbook (default: DATADIR/2phaseopt_best_cfg_all_models.ods)")
    ap.add_argument("--outdir", default="./summary_out", help="where to write tables and figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ods = args.ods or os.path.join(args.datadir, "2phaseopt_best_cfg_all_models.ods")

    print(f"Loading chamfer CSVs from {args.datadir}")
    d = load_all(args.datadir)
    check_integrity(d)

    table1_combined(d, ods, args.outdir)
    table1_gridsearch(ods, args.outdir)   # wide/raw version, kept for reference
    table2_descriptives(d, args.outdir)
    table3_paired(d, args.outdir)
    table4_gap(d, args.outdir)

    figure_distributions(d, args.outdir)
    figure_gap(d, args.outdir)
    figure_tradeoff(d, args.outdir)

    d.to_csv(os.path.join(args.outdir, "chamfer_long.csv"), index=False)
    print(f"\nWrote 4 tables, 3 figures (pdf + png), and chamfer_long.csv to {args.outdir}")


if __name__ == "__main__":
    main()