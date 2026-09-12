"""Evaluate encoder and full optimizer shape completion on a config split.

This uses real mesh names from model_params_config.json (for example the test
split), crops sampled SDF points into a synthetic partial input, runs both the
PointNet encoder and the 2-phase optimizer, and appends per-mesh timing/Chamfer
rows as it goes.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch

from NSM.datasets import SDFSamples
from NSM.helper_funcs import load_config, load_model_and_latents
from NSM.mesh import create_mesh
from NSM.optimization import get_top_k_pcs, optimize_latent_partial
from encoder.pointnet_encoder import PointNetEncoder


def basename(path):
    return os.path.basename(path)


def safe_stem(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", basename(path).rsplit(".", 1)[0])


def chamfer(a, b, n=3000):
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return float("nan")
    a = torch.from_numpy(a[np.random.choice(len(a), min(n, len(a)), replace=False)])
    b = torch.from_numpy(b[np.random.choice(len(b), min(n, len(b)), replace=False)])
    d = torch.cdist(a.float(), b.float())
    return (d.min(1).values.mean() + d.min(0).values.mean()).item() / 2


def recon_surface(model, z, res, device):
    mesh = create_mesh(
        decoder=model,
        latent_vector=z,
        n_pts_per_axis=res,
        scale_to_original_mesh=False,
        device=device,
        verbose=False,
    )
    if mesh is None:
        return None, None
    return np.asarray(mesh.point_coords, dtype=np.float32), mesh


def encode(enc, n_points, points, sdf, device, surf_thresh=0.01):
    abs_sdf = sdf.abs().squeeze()
    surf = points[abs_sdf < surf_thresh]
    if surf.shape[0] < 32:
        k = min(n_points, points.shape[0])
        surf = points[torch.topk(abs_sdf, k, largest=False).indices]
    sel = np.random.choice(surf.shape[0], n_points, replace=surf.shape[0] < n_points)
    with torch.no_grad():
        return enc(surf[sel].unsqueeze(0).to(device))


def crop_partial(points, sdf, keep=0.65, rng=None):
    rng = rng or np.random
    surf_mask = sdf.abs().squeeze().detach().cpu().numpy() < 0.01
    center = points[surf_mask].mean(0) if surf_mask.any() else points.mean(0)
    normal = torch.tensor(rng.randn(3), dtype=torch.float32, device=points.device)
    normal = normal / normal.norm()
    proj = (points - center) @ normal
    thresh = torch.quantile(proj.detach().cpu(), 1.0 - keep).to(points.device)
    mask = proj > thresh
    return points[mask], sdf[mask]


def build_sdf_dataset(mesh_path, config, n_pts=None):
    return SDFSamples(
        list_mesh_paths=[mesh_path],
        multiprocessing=False,
        subsample=config["samples_per_object_per_batch"],
        print_filename=False,
        n_pts=n_pts or config["n_pts_per_object"],
        p_near_surface=config["percent_near_surface"],
        p_further_from_surface=config["percent_further_from_surface"],
        sigma_near=config["sigma_near"],
        sigma_far=config["sigma_far"],
        rand_function=config["random_function"],
        center_pts=config["center_pts"],
        norm_pts=config["normalize_pts"],
        scale_method=config["scale_method"],
        scale_jointly=False,
        reference_mesh=None,
        verbose=False,
        save_cache=False,
        store_data_in_memory=True,
        equal_pos_neg=config["equal_pos_neg"],
        fix_mesh=config["fix_mesh"],
    )


def existing_meshes(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {row["mesh"] for row in csv.DictReader(f) if row.get("mesh")}


def existing_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def existing_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def local_mesh_index(search_dirs):
    index = {}
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for name in os.listdir(search_dir):
            if name.lower().endswith(".vtk"):
                index.setdefault(name, os.path.abspath(os.path.join(search_dir, name)))
    return index


def resolve_mesh_paths(mesh_paths, search_dirs):
    index = local_mesh_index(search_dirs)
    resolved = []
    missing = []
    for path in mesh_paths:
        if os.path.exists(path):
            resolved.append(path)
            continue
        replacement = index.get(basename(path))
        if replacement:
            resolved.append(replacement)
        else:
            missing.append(path)
    return resolved, missing


def write_summary(out_dir, rows, phase1_iters, phase2_iters, keep, res):
    if not rows:
        return
    def mean(key):
        vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    md = [
        "# Shape completion split evaluation",
        "",
        f"Rows summarized: {len(rows)}. Partial input keeps {keep:.0%}; reconstruction res {res}.",
        "",
        "| Method | Chamfer to GT surface samples (mean) | Time/mesh (mean) |",
        "|---|---|---|",
        f"| Encoder (fast) | {mean('enc_chamfer'):.4f} | {mean('enc_time'):.3f} s |",
        f"| Optimizer (phase1 {phase1_iters} + phase2 {phase2_iters}) | {mean('opt_chamfer'):.4f} | {mean('opt_time'):.1f} s |",
        "",
        f"Mean total time/mesh: {mean('total_time'):.1f} s.",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="run_v72/model_params_config.json")
    ap.add_argument("--model", default="run_v72/model/2500.pth")
    ap.add_argument("--latent_codes", default="run_v72/latent_codes/2500.pth")
    ap.add_argument("--encoder", default="run_v72/encoder/checkpoints/encoder.pt")
    ap.add_argument("--out_dir", default="run_v72/shape_completion/test_eval")
    ap.add_argument("--dataset_split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep", type=float, default=0.65)
    ap.add_argument("--n_samples", type=int, default=240)
    ap.add_argument("--sdf_n_pts", type=int, default=None)
    ap.add_argument("--phase1_iters", type=int, default=3000)
    ap.add_argument("--phase2_iters", type=int, default=8000)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--seed", type=int, default=52122)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mesh_search_dirs", nargs="+", default=["vertebrae_meshes", "vertebrae_meshes_2"])
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    config = load_config(args.config)
    config["device"] = device
    model, _, latent_codes = load_model_and_latents(args.model, args.latent_codes, config, device)
    model.eval()
    latent_codes = latent_codes.to(device)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    latent_std = float(latent_codes.std().mean().item())
    _, top_k_reg = get_top_k_pcs(latent_codes.cpu(), threshold=0.99)

    ckpt = torch.load(args.encoder, map_location=device, weights_only=True)
    enc = PointNetEncoder(latent_size=ckpt["latent_size"]).to(device)
    enc.load_state_dict(ckpt["model"])
    enc.eval()
    n_points = ckpt["n_points"]

    split_key = {"train": "list_mesh_paths", "val": "val_paths", "test": "test_paths"}[args.dataset_split]
    mesh_paths, missing_paths = resolve_mesh_paths(config[split_key], args.mesh_search_dirs)
    if missing_paths:
        preview = ", ".join(basename(p) for p in missing_paths[:3])
        raise FileNotFoundError(
            f"Cannot evaluate the fixed split: {len(missing_paths)} meshes are missing "
            f"({preview}). Add their directory with --mesh_search_dirs."
        )
    if missing_paths:
        preview = ", ".join(basename(p) for p in missing_paths[:3])
        raise FileNotFoundError(
            f"Could not resolve {len(missing_paths)} meshes from the fixed {args.dataset_split} "
            f"split (e.g. {preview}). Supply their directory with --mesh_search_dirs."
        )
    if args.limit is not None:
        mesh_paths = mesh_paths[:args.limit]

    csv_path = os.path.join(args.out_dir, "results.csv")
    fields = [
        "mesh", "enc_chamfer", "opt_chamfer", "enc_time", "opt_time",
        "total_time", "status", "error",
    ]
    done = existing_meshes(csv_path) if args.resume else set()
    write_header = not os.path.exists(csv_path) or not args.resume
    mode = "a" if args.resume else "w"

    print(f"Device: {device}")
    print(
        f"Split: {args.dataset_split} | local meshes: {len(mesh_paths)} | "
        f"missing: {len(missing_paths)} | already done: {len(done)}"
    )
    print(f"Output: {csv_path}")
    start_all = time.time()

    rows_for_summary = existing_rows(csv_path) if args.resume else []
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        for idx_mesh, mesh_path in enumerate(mesh_paths, start=1):
            mesh_name = basename(mesh_path)
            if mesh_name in done:
                continue
            t_mesh = time.time()
            row = {
                "mesh": mesh_name,
                "enc_chamfer": "",
                "opt_chamfer": "",
                "enc_time": "",
                "opt_time": "",
                "total_time": "",
                "status": "ok",
                "error": "",
            }
            try:
                ds = build_sdf_dataset(mesh_path, config, n_pts=args.sdf_n_pts)
                sample, _ = ds[0]
                pts_full = sample["xyz"].to(device).squeeze()
                sdf_full = sample["gt_sdf"].to(device).reshape(-1, 1)

                gt_mask = sdf_full.abs().squeeze() < 0.01
                gt_pts = pts_full[gt_mask].detach().cpu().numpy().astype(np.float32)
                part_pts, part_sdf = crop_partial(pts_full, sdf_full, keep=args.keep, rng=rng)

                t0 = time.time()
                z_enc = encode(enc, n_points, part_pts, part_sdf, device)
                enc_pts, enc_mesh = recon_surface(model, z_enc, args.res, device)
                row["enc_time"] = time.time() - t0

                opt_idx = torch.randperm(part_pts.shape[0], device=device)[:args.n_samples]
                opt_pts_in = part_pts[opt_idx]
                opt_sdf_in = part_sdf[opt_idx]

                t1 = time.time()
                z_opt, _ = optimize_latent_partial(
                    model, opt_pts_in, opt_sdf_in, config["latent_size"],
                    mean_latent=mean_latent, latent_init=latent_codes, top_k=top_k_reg,
                    iters=args.phase1_iters, lr=1e-4, lambda_reg=1e-3,
                    clamp_val=1.0, latent_std=latent_std, scheduler_step=800,
                    scheduler_gamma=0.9, batch_inference_size=32768,
                    multi_stage=False, verbose=False, device=device,
                )
                z_opt, _ = optimize_latent_partial(
                    model, opt_pts_in, opt_sdf_in, config["latent_size"],
                    latent_init=z_opt, top_k=top_k_reg, iters=args.phase2_iters,
                    lr=1e-5, lambda_reg=1e-5, clamp_val=None,
                    latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.7,
                    batch_inference_size=32768, multi_stage=True,
                    verbose=False, device=device,
                )
                opt_pts, opt_mesh = recon_surface(model, z_opt, args.res, device)
                row["opt_time"] = time.time() - t1

                row["enc_chamfer"] = chamfer(gt_pts, enc_pts)
                row["opt_chamfer"] = chamfer(gt_pts, opt_pts)

                stem = safe_stem(mesh_path)
                if enc_mesh is not None:
                    enc_mesh.save_mesh(os.path.join(args.out_dir, f"{stem}_encoder.vtk"))
                if opt_mesh is not None:
                    opt_mesh.save_mesh(os.path.join(args.out_dir, f"{stem}_optimizer.vtk"))
            except Exception as exc:
                row["status"] = "error"
                row["error"] = repr(exc)
                print(f"[{idx_mesh}/{len(mesh_paths)}] ERROR {mesh_name}: {exc}", file=sys.stderr)

            row["total_time"] = time.time() - t_mesh
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            rows_for_summary.append(row)

            elapsed = time.time() - start_all
            completed = len(rows_for_summary)
            remaining = len(mesh_paths) - idx_mesh
            eta_h = (elapsed / max(completed, 1)) * remaining / 3600
            print(
                f"[{idx_mesh}/{len(mesh_paths)}] {mesh_name} "
                f"enc={row['enc_chamfer']} opt={row['opt_chamfer']} "
                f"time={row['total_time']:.1f}s ETA={eta_h:.2f}h",
                flush=True,
            )

    write_summary(args.out_dir, rows_for_summary, args.phase1_iters, args.phase2_iters, args.keep, args.res)
    print(f"Done. Wrote {csv_path}")


if __name__ == "__main__":
    main()
