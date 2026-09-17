"""
Create reproducible train/val/test split files for dataset-size experiments.

The generated JSON files can be passed to train_model.py with --split_file.
They keep validation/test meshes fixed for a seed, then downsample the training
pool by either random or diversity-aware selection.
"""

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict


DEFAULT_SIZES = ["100", "300", "500", "800", "1000", "all"]
DEFAULT_STRATEGIES = ["diverse", "random"]
DEFAULT_SEEDS = [52122, 52123, 52124, 52125, 52126]
REGION_ORDER = {"c": 0, "t": 1, "l": 2}


def norm_name(path):
    return os.path.basename(path).lower().rsplit(".", 1)[0]


def read_mapping(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {norm_name(row["vtk_name"]): row for row in rows}


def discover_meshes(mesh_dir):
    paths = []
    for name in os.listdir(mesh_dir):
        if name.lower().endswith(".vtk"):
            paths.append(os.path.abspath(os.path.join(mesh_dir, name)))
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


def mesh_record(path, mapping):
    base = norm_name(path)
    row = mapping.get(base, {})
    family = clean(row.get("family")) or token(path, 0)
    genus = clean(row.get("genus")) or token(path, 1)
    species = clean(row.get("species")) or token(path, 2)
    specimen = clean(row.get("specimen")) or specimen_from_filename(base)
    region = clean(row.get("vertebra_type")) or region_from_filename(base)
    if region:
        region = region[0]
    return {
        "path": path,
        "mesh": os.path.basename(path),
        "family": family,
        "genus": genus,
        "species": species,
        "specimen": specimen,
        "region": region,
    }


def clean(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None


def token(path, idx):
    toks = norm_name(path).replace("-", "_").split("_")
    return toks[idx] if len(toks) > idx else None


def region_from_filename(base):
    match = re.search(r"[-_]([ctl])\d+$", base.lower())
    return match.group(1) if match else None


def specimen_from_filename(base):
    return re.sub(r"[_-]?\d*[-_][ctl]\d+$", "", base.lower())


def specimen_groups(records):
    """Return biological units; a specimen must never straddle data splits."""
    groups = defaultdict(list)
    for record in records:
        if not record.get("specimen"):
            raise ValueError(f"Missing specimen metadata for {record['mesh']}")
        groups[record["specimen"]].append(record)
    return groups


def stratified_holdout(records, fraction, seed, protected_specimens=None):
    """Select whole specimens, approximately stratified by family.

    The requested fraction is a mesh-count target. Whole-specimen assignment can
    overshoot it slightly, which is preferable to leaking adjacent vertebrae.
    """
    if fraction <= 0:
        return []
    protected_specimens = set(protected_specimens or [])
    rng = random.Random(seed)
    target = int(round(len(records) * fraction))
    by_specimen = specimen_groups(records)
    groups = defaultdict(list)
    for specimen, rows in by_specimen.items():
        if specimen not in protected_specimens:
            family = rows[0].get("family") or "unknown"
            groups[family].append((specimen, rows))
    for items in groups.values():
        rng.shuffle(items)

    selected = []
    selected_n = 0
    keys = sorted(groups)
    while selected_n < target and keys:
        next_keys = []
        for key in keys:
            items = groups[key]
            if items and selected_n < target:
                _, rows = items.pop()
                selected.extend(rows)
                selected_n += len(rows)
            if items:
                next_keys.append(key)
        keys = next_keys
    return sorted(selected, key=sort_key)


def random_sample(records, n, seed):
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    return records[:n]


def diverse_order(records, seed):
    """Create one diversity-aware ordering whose prefixes form nested subsets."""
    rng = random.Random(seed)
    pool = list(records)
    selected = []
    counts = {
        "specimen": Counter(),
        "family": Counter(),
        "genus": Counter(),
        "species": Counter(),
        "region": Counter(),
    }
    while pool:
        best_i = None
        best_score = None
        for i, record in enumerate(pool):
            specimen_count = counts["specimen"][record["specimen"]]
            score = (
                5.0 / (1 + specimen_count)
                + 3.0 / (1 + counts["family"][record["family"]])
                + 2.0 / (1 + counts["genus"][record["genus"]])
                + 2.0 / (1 + counts["species"][record["species"]])
                + 1.5 / (1 + counts["region"][record["region"]])
                + rng.random() * 0.01
            )
            # Strongly prefer broad coverage first, then balance contributions.
            score -= 2.0 * specimen_count
            if best_score is None or score > best_score:
                best_i = i
                best_score = score

        chosen = pool.pop(best_i)
        selected.append(chosen)
        for key in counts:
            counts[key][chosen[key]] += 1

    return selected


def assert_disjoint_specimens(train, val, test):
    sets = [{r["specimen"] for r in split} for split in (train, val, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("A specimen occurs in more than one data split")


def count_unique(records, key):
    return len({record[key] for record in records if record.get(key)})


def sort_key(record):
    return (
        record.get("family") or "",
        record.get("genus") or "",
        record.get("species") or "",
        record.get("specimen") or "",
        REGION_ORDER.get(record.get("region"), 99),
        record["mesh"],
    )


def summarize(records):
    return {
        "meshes": len(records),
        "specimens": count_unique(records, "specimen"),
        "families": count_unique(records, "family"),
        "genera": count_unique(records, "genus"),
        "species": count_unique(records, "species"),
        "regions": dict(Counter(r.get("region") or "unknown" for r in records)),
    }


def split_name(size, strategy, seed):
    return f"n{size}_{strategy}_seed{seed}"


def write_split(path, name, size, strategy, seed, train, val, test, skipped_reason=None):
    payload = {
        "name": name,
        "requested_train_size": size,
        "strategy": strategy,
        "seed": seed,
        "skipped": skipped_reason is not None,
        "skipped_reason": skipped_reason,
        "list_mesh_paths": [r["path"] for r in train],
        "val_paths": [r["path"] for r in val],
        "test_paths": [r["path"] for r in test],
        "summary": {
            "train": summarize(train),
            "val": summarize(val),
            "test": summarize(test),
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return payload


def parse_sizes(values):
    sizes = []
    for value in values:
        if str(value).lower() == "all":
            sizes.append("all")
        else:
            sizes.append(int(value))
    return sizes


def main():
    parser = argparse.ArgumentParser(
        description="Generate downsampled NSM training split JSON files."
    )
    parser.add_argument("--mesh_dir", default="vertebrae_meshes")
    parser.add_argument("--mapping_csv", default="vtk_name_to_mapping_v2.csv")
    parser.add_argument("--out_dir", default="downsample_splits")
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES,
                        choices=DEFAULT_STRATEGIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.05)
    parser.add_argument("--holdout_seed", type=int, default=9173,
                        help="Fixed specimen-level val/test split seed shared by every run")
    parser.add_argument("--ckpt", default="3000",
                        help="Checkpoint used in generated evaluation commands")
    args = parser.parse_args()

    mapping = read_mapping(args.mapping_csv)
    mesh_paths = discover_meshes(args.mesh_dir)
    records = [mesh_record(path, mapping) for path in mesh_paths]
    if not records:
        raise ValueError(f"No .vtk meshes found in {args.mesh_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest_rows = []

    # One external holdout is shared by all sizes, strategies, and repetitions.
    test = stratified_holdout(records, args.test_fraction, args.holdout_seed)
    test_specimens = {r["specimen"] for r in test}
    val = stratified_holdout(records, args.val_fraction, args.holdout_seed + 1,
                             protected_specimens=test_specimens)
    heldout_specimens = test_specimens | {r["specimen"] for r in val}
    train_pool = [r for r in records if r["specimen"] not in heldout_specimens]

    for seed in args.seeds:
        random_order = random_sample(train_pool, len(train_pool), seed)
        diversity_order = diverse_order(train_pool, seed)

        for size in parse_sizes(args.sizes):
            requested_n = len(train_pool) if size == "all" else size
            for strategy in args.strategies:
                name = split_name(size, strategy, seed)
                out_path = os.path.join(args.out_dir, f"{name}.json")
                skipped = None
                train = []
                if requested_n > len(train_pool):
                    skipped = (
                        f"requested {requested_n} training meshes, "
                        f"but only {len(train_pool)} are available after holdout"
                    )
                elif strategy == "random":
                    train = sorted(random_order[:requested_n], key=sort_key)
                else:
                    train = sorted(diversity_order[:requested_n], key=sort_key)

                assert_disjoint_specimens(train, val, test)

                payload = write_split(
                    out_path, name, size, strategy, seed, train, val, test, skipped
                )
                row = {
                    "name": name,
                    "split_file": out_path,
                    "run_name": f"run_{name}",
                    "requested_train_size": size,
                    "strategy": strategy,
                    "seed": seed,
                    "skipped": payload["skipped"],
                    "skipped_reason": skipped or "",
                    "train_meshes": payload["summary"]["train"]["meshes"],
                    "train_specimens": payload["summary"]["train"]["specimens"],
                    "train_families": payload["summary"]["train"]["families"],
                    "train_genera": payload["summary"]["train"]["genera"],
                    "train_species": payload["summary"]["train"]["species"],
                    "val_meshes": payload["summary"]["val"]["meshes"],
                    "test_meshes": payload["summary"]["test"]["meshes"],
                    "train_command": (
                        f"python train_model.py --run_name run_{name} "
                        f"--split_file {out_path}"
                    ),
                    "classification_command": (
                        f"python classification_eval.py --model_root run_{name} "
                        f"--ckpt {args.ckpt} --dataset_split test --encoded_latents "
                        "--eval_level specimen"
                    ),
                    "encode_test_command": (
                        f"python encode_latents_for_eval.py --model_root run_{name} "
                        "--output_dir classification/evaluation/encoded_latents "
                        "--dataset_split test"
                    ),
                    "build_encoder_data_command": (
                        f"python encoder/generate_latent_dataset.py "
                        f"--config run_{name}/model_params_config.json "
                        f"--model run_{name}/model/{args.ckpt}.pth "
                        f"--latent_codes run_{name}/latent_codes/{args.ckpt}.pth "
                        f"--out run_{name}/encoder/data/latent_surface_dataset.pt"
                    ),
                    "train_encoder_command": (
                        f"python encoder/train_encoder.py "
                        f"--data run_{name}/encoder/data/latent_surface_dataset.pt "
                        f"--out run_{name}/encoder/checkpoints/encoder.pt --seed {seed}"
                    ),
                    "shape_completion_command": (
                        f"python evaluate_shape_completion_split.py --config run_{name}/model_params_config.json "
                        f"--model run_{name}/model/{args.ckpt}.pth "
                        f"--latent_codes run_{name}/latent_codes/{args.ckpt}.pth "
                        f"--encoder run_{name}/encoder/checkpoints/encoder.pt "
                        f"--out_dir run_{name}/shape_completion/test_eval --dataset_split test --resume"
                    ),
                }
                manifest_rows.append(row)

    fieldnames = list(manifest_rows[0].keys())
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Found {len(records)} meshes.")
    print(f"Wrote {len(manifest_rows)} split specs to {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
