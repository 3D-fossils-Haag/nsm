"""
Collect downsample experiment coverage, classification and Chamfer metrics.

This joins a split manifest from make_downsample_splits.py with completed run
outputs from classification_eval.py and shape_completion_eval.py.
"""

import argparse
import csv
import glob
import math
import os
import statistics


CLASSIFICATION_CATEGORIES = [
    "family",
    "genus",
    "species",
    "region",
    "position_10",
    "position_20",
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def latest_path(paths):
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def find_latest_classification(run_dir, eval_level):
    patterns = [
        os.path.join(run_dir, "classification", "evaluation", "test",
                     f"test_{eval_level}_*", "metrics_summary.csv"),
        os.path.join(run_dir, "classification", "evaluation", "*",
                     f"*_{eval_level}_*", "metrics_summary.csv"),
    ]
    return latest_path([p for pattern in patterns for p in glob.glob(pattern)])


def find_latest_chamfer(run_dir):
    patterns = [
        os.path.join(run_dir, "shape_completion", "test_eval", "results.csv"),
        os.path.join(run_dir, "shape_completion", "evaluation", "*_chamfer.csv"),
    ]
    return latest_path([p for pattern in patterns for p in glob.glob(pattern)])


def category_name(row):
    if row.get("category"):
        return row["category"]
    if row.get(""):
        return row[""]
    return next(iter(row.values()))


def read_classification_metrics(path):
    if not path or not os.path.exists(path):
        return {}
    metrics = {}
    for row in read_csv(path):
        cat = category_name(row)
        if cat in CLASSIFICATION_CATEGORIES:
            metrics[f"{cat}_accuracy"] = row.get("top1_accuracy", "")
            metrics[f"{cat}_macro_f1"] = row.get("macro_f1", "")
        elif cat == "normalized_position":
            metrics["position_mae"] = row.get("mae", "")
            metrics["position_median_ae"] = row.get("median_ae", "")
    return metrics


def read_chamfer(path):
    if not path or not os.path.exists(path):
        return {"mean": "", "median": "", "p90": "", "n": 0}
    values = []
    for row in read_csv(path):
        for column in ("opt_chamfer", "chamfer"):
            raw = (row.get(column) or "").strip()
            if raw:
                try:
                    value = float(raw)
                except ValueError:
                    break
                if not math.isnan(value):
                    values.append(value)
                break
    if not values:
        return {"mean": "", "median": "", "p90": "", "n": 0}
    values.sort()
    p90 = values[min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)]
    return {"mean": f"{statistics.mean(values):.6g}",
            "median": f"{statistics.median(values):.6g}",
            "p90": f"{p90:.6g}", "n": len(values)}


def c_t_l_balance(split_file):
    if not split_file or not os.path.exists(split_file):
        return ""
    import json
    with open(split_file) as f:
        split = json.load(f)
    regions = split.get("summary", {}).get("train", {}).get("regions", {})
    return "/".join(str(regions.get(k, 0)) for k in ["c", "t", "l"])


def main():
    parser = argparse.ArgumentParser(
        description="Create a final CSV table for downsample experiments."
    )
    parser.add_argument("--manifest", default="downsample_splits/manifest.csv")
    parser.add_argument("--out_csv", default="downsample_splits/results_summary.csv")
    parser.add_argument("--eval_level", default="specimen")
    args = parser.parse_args()

    manifest_rows = read_csv(args.manifest)
    output_rows = []
    for row in manifest_rows:
        run_dir = row["run_name"]
        class_path = find_latest_classification(run_dir, args.eval_level)
        chamfer_path = find_latest_chamfer(run_dir)
        metrics = read_classification_metrics(class_path)
        chamfer = read_chamfer(chamfer_path)
        out = {
            "run": run_dir,
            "strategy": row["strategy"],
            "requested_train_size": row["requested_train_size"],
            "train_meshes": row["train_meshes"],
            "train_specimens": row["train_specimens"],
            "train_families": row["train_families"],
            "train_genera": row["train_genera"],
            "train_species": row["train_species"],
            "c_t_l_balance": c_t_l_balance(row["split_file"]),
            "family_accuracy": metrics.get("family_accuracy", ""),
            "family_macro_f1": metrics.get("family_macro_f1", ""),
            "genus_accuracy": metrics.get("genus_accuracy", ""),
            "genus_macro_f1": metrics.get("genus_macro_f1", ""),
            "species_accuracy": metrics.get("species_accuracy", ""),
            "species_macro_f1": metrics.get("species_macro_f1", ""),
            "region_accuracy": metrics.get("region_accuracy", ""),
            "region_macro_f1": metrics.get("region_macro_f1", ""),
            "position_10_accuracy": metrics.get("position_10_accuracy", ""),
            "position_20_accuracy": metrics.get("position_20_accuracy", ""),
            "position_mae": metrics.get("position_mae", ""),
            "position_median_ae": metrics.get("position_median_ae", ""),
            "mean_chamfer": chamfer["mean"],
            "median_chamfer": chamfer["median"],
            "p90_chamfer": chamfer["p90"],
            "n_chamfer": chamfer["n"],
            "classification_metrics": class_path or "",
            "shape_completion_metrics": chamfer_path or "",
            "skipped": row["skipped"],
            "skipped_reason": row["skipped_reason"],
        }
        output_rows.append(out)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
