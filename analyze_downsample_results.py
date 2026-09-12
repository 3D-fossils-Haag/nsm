"""Aggregate repeated learning curves and identify practically equivalent sizes."""

import argparse
import csv
import math
import random
from collections import defaultdict


HIGHER_IS_BETTER = [
    "family_macro_f1", "genus_macro_f1", "species_macro_f1", "region_macro_f1",
    "position_10_accuracy", "position_20_accuracy",
]
LOWER_IS_BETTER = ["position_mae", "mean_chamfer", "median_chamfer", "p90_chamfer"]


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def bootstrap_ci(values, seed=52122, iterations=10000):
    if not values:
        return None, None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                   for _ in range(iterations))
    return (sum(values) / len(values), means[int(.025 * iterations)],
            means[min(iterations - 1, int(.975 * iterations))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="downsample_splits/results_summary.csv")
    ap.add_argument("--out_csv", default="downsample_splits/learning_curve_summary.csv")
    ap.add_argument("--accuracy_margin", type=float, default=.02,
                    help="Absolute practical-equivalence margin for accuracy/F1")
    ap.add_argument("--error_margin", type=float, default=.05,
                    help="Relative practical-equivalence margin for error metrics")
    ap.add_argument("--bootstrap_iterations", type=int, default=10000)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.results, newline="")))
    groups = defaultdict(list)
    for row in rows:
        if str(row.get("skipped", "")).lower() == "true":
            continue
        groups[(row["strategy"], row["requested_train_size"])].append(row)

    output = []
    for (strategy, size), members in groups.items():
        out = {"strategy": strategy, "requested_train_size": size,
               "n_repeats": len(members)}
        for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
            values = [v for v in (number(r.get(metric)) for r in members) if v is not None]
            mean, low, high = bootstrap_ci(values, iterations=args.bootstrap_iterations)
            out[f"{metric}_mean"] = "" if mean is None else mean
            out[f"{metric}_ci_low"] = "" if low is None else low
            out[f"{metric}_ci_high"] = "" if high is None else high
        output.append(out)

    # Compare conservative confidence bounds with the corresponding all-data mean.
    full = {row["strategy"]: row for row in output if row["requested_train_size"] == "all"}
    for row in output:
        reference = full.get(row["strategy"])
        passed = []
        if reference:
            for metric in HIGHER_IS_BETTER:
                bound, ref = number(row[f"{metric}_ci_low"]), number(reference[f"{metric}_mean"])
                if bound is not None and ref is not None:
                    passed.append(bound >= ref - args.accuracy_margin)
            for metric in LOWER_IS_BETTER:
                bound, ref = number(row[f"{metric}_ci_high"]), number(reference[f"{metric}_mean"])
                if bound is not None and ref is not None:
                    passed.append(bound <= ref * (1 + args.error_margin))
        row["equivalent_to_all"] = bool(passed) and all(passed)
        row["n_equivalence_metrics"] = len(passed)

    def size_key(row):
        return (row["strategy"], float("inf") if row["requested_train_size"] == "all"
                else int(row["requested_train_size"]))
    output.sort(key=size_key)
    fieldnames = list(output[0]) if output else []
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} aggregate rows to {args.out_csv}")


if __name__ == "__main__":
    main()
