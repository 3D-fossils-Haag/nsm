# Utility functions for model evaluation
import pandas as pd
import numpy as np
import random
import os
from scipy.spatial import cKDTree
import re
import torch
import time
import pyvista as pv
from NSM.optimization import normalize_mesh, get_norm_params, get_top_k_pcs, build_sdf_dataset, encode_latent, encode_latent_pointnet, reconstruct_mesh_from_latent, optimize_latent_partial
from NSM.helper_funcs import convert_ply_to_vtk
import glob
import sys
from collections import namedtuple
import math
from scipy.stats import spearmanr, t as t_dist

# Strip _partial to match partial_mesh_path and ground_truth_path pairs
def strip_partial_mesh_name(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    if name.endswith("_partial"):
        name = name[:-8]
    return name

# Build partial - ground truth mesh pairs
def build_partial_gt_mesh_pairs(partial_mesh_summary, mesh_names, split_key):
    pairs = []
    skipped = 0
    for m in partial_mesh_summary["meshes"]:
        base_name = m["base_name"]
        if base_name in mesh_names:
            pairs.append((m["partial"], m["ground_truth"]))
        else:
            skipped += 1
    print(f"Built {len(pairs)} (partial, ground_truth) pairs")
    print(f"Skipped {skipped} meshes not in {split_key}")
    return pairs

# Accuracy Metrics
def uniform_surface_sample(poly, n):
    # Triangulate the mesh
    poly = poly.triangulate().extract_surface(algorithm=None)
    verts = np.asarray(poly.points)
    faces = poly.faces.reshape(-1, 4)[:, 1:]  # (T,3)
    # Calculate areas of each triangle
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    # Select triangles to sample from
    probs = areas / areas.sum()
    tri_idx = np.random.choice(len(faces), size=n, p=probs)
    a = v0[tri_idx]; b = v1[tri_idx]; c = v2[tri_idx]
    r1 = np.sqrt(np.random.rand(n))
    r2 = np.random.rand(n)
    # Barycentric sampling to find random points inside each triangle
    pts = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    return pts

# Calculate chamfer distance on partial-completed mesh vs original-ground truth mesh
def chamfer_distance(pm, gt, n_samples=20000):
    # Sample points across surface
    sp = uniform_surface_sample(pm, n_samples)
    sg = uniform_surface_sample(gt, n_samples)
    # Use KD-tree to find nearest neighbor distances of gt to predicted surface and vice versa
    t1 = cKDTree(sp); t2 = cKDTree(sg)
    d1 = t1.query(sg, k=1)[0].mean()
    d2 = t2.query(sp, k=1)[0].mean()
    return float(0.5*(d1 + d2)) # Return average distance (symmetric penalty)

# Load in best_cfg from shape_completion_grid_search.py (or manually enter)
def load_best_cfg_from_csv(csv_path, fast_mode, device):
    df = pd.read_csv(csv_path)
    best_row = df.loc[df["mean_cd"].idxmin()]
    # Parse latent_std whether stored as "tensor(0.4401)" or plain float
    raw_std = best_row["latent_std"]
    if isinstance(raw_std, str):
        if raw_std.startswith("tensor("):
            raw_std = raw_std[len("tensor("):]
            raw_std = raw_std.split(",")[0].rstrip(")")
    latent_std = torch.tensor(float(raw_std), device=device)
    print(f"Building config from grid search. Fast_mode={fast_mode}")
    if fast_mode:
        clamp = None if pd.isna(best_row["clamp"]) else best_row["clamp"]
        best_cfg = {'top_k': int(best_row["top_k"]),
                    'refine_iters': int(best_row["iters"]),
                    'lr': float(best_row["lr"]),
                    'lambda_reg': float(best_row["lambda_reg"]),
                    'clamp': clamp,
                    'latent_std': latent_std,
                    'sched_step': int(best_row["sched_step"]),
                    'sched_gamma': float(best_row["sched_gamma"]),
                    'batch_infer': int(best_row["batch_infer"]),
                    'gridN': int(best_row["gridN"])}

    else:
        clamp2 = None if pd.isna(best_row["clamp2"]) else best_row["clamp2"]
        best_cfg = {"top_k":       int(best_row["top_k"]),
                    "iters1":      int(best_row["iters1"]),
                    "iters2":      int(best_row["iters2"]),
                    "lr1":         float(best_row["lr1"]),
                    "lr2":         float(best_row["lr2"]),
                    "lambda1":     float(best_row["lambda1"]),
                    "lambda2":     float(best_row["lambda2"]),
                    "clamp1":      best_row["clamp1"],
                    "clamp2":      clamp2,
                    "latent_std":  latent_std,
                    "sched_step":  int(best_row["sched_step"]),
                    "sched_gamma1": float(best_row["sched_gamma1"]),
                    "sched_gamma2": float(best_row["sched_gamma2"]),
                    "batch_infer": int(best_row["batch_infer"]),
                    "gridN":       int(best_row["gridN"])}

    print(f"\nLoaded best_cfg from CSV (min mean_cd = {best_row['mean_cd']:.4f}):")
    return best_cfg

# Random search on a small validation subset to pick best cfg
def grid_search(pairs, model, config, mean_latent, latent_codes, device, out_dir, n_trials=15, valN=30, log_path_csv=None, log_path_json=None):
    # Set up directory for fine-tuning experiemnts
    os.makedirs(out_dir, exist_ok=True)
    val_subset = random.sample(pairs, min(valN, len(pairs)))
    best = {'score': float('inf'), 'cfg': None}
    rows = []
    # Define how many PCs describe X% of variance
    _, k95 = get_top_k_pcs(latent_codes, threshold=0.95)
    _, k99 = get_top_k_pcs(latent_codes, threshold=0.99)
    latent_std = latent_codes.std().mean()

    # Randomly pick optimization parameters from provided values
    for t in range(n_trials):
        trial_cfg = {
            'top_k': random.choice([k95, k99]),
            'iters1': random.choice([3000, 5000]),
            'iters2': random.choice([6000, 8000]),
            'lr1': random.choice([1e-4, 1e-3]),
            'lr2': random.choice([1e-5, 1e-4]),
            'lambda1': random.choice([1e-6, 1e-3, 1e-2]),
            'lambda2': random.choice([1e-7, 1e-5, 1e-3]),
            'clamp1': 1,
            'clamp2': random.choice([None, 1]),
            'latent_std': latent_std,
            'sched_step': random.choice([500, 800]),
            'sched_gamma1': random.choice([0.7, 0.9]),
            'sched_gamma2': random.choice([0.5, 0.7, 0.9]),
            'batch_infer': random.choice([32768]),
            'gridN': random.choice([128, 256]),}
        scores = []
        times = []
        # Set up directory for each trial
        trial_dir = os.path.join(out_dir, f"trial_{t:02d}")
        os.makedirs(trial_dir, exist_ok=True)
        # Run trial on randomly chosen config params and log chamfer score
        for i, (pred_path, gt_path) in enumerate(val_subset):
            start = time.time()
            if '.ply' in pred_path:
                _, pred_path = convert_ply_to_vtk(pred_path, save=True)
            # Build SDF dataset
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(pred_path, config, n_samples=240)
            # Encode latent via 2 stage optimization (auto-decoder framework)
            latent_opt = encode_latent(decoder=model, points=points.squeeze(), sdf_vals=sdf_vals, latent_dim=latent_codes.shape[1], 
                               mean_latent=mean_latent, latent_codes=latent_codes, top_k_reg=trial_cfg['top_k'], latent_std=trial_cfg['latent_std'],
                               iters1=trial_cfg['iters1'], iters2=trial_cfg['iters2'], lr1=trial_cfg['lr1'], lr2=trial_cfg['lr2'], 
                               lambda_reg1=trial_cfg['lambda1'], lambda_reg2=trial_cfg['lambda2'], clamp_val1=trial_cfg['clamp1'], clamp_val2=trial_cfg['clamp2'], 
                               scheduler_step1=trial_cfg['sched_step'], scheduler_step2=trial_cfg['sched_step'], scheduler_gamma1=trial_cfg['sched_gamma1'],
                               scheduler_gamma2=trial_cfg['sched_gamma2'], batch_inference_size=trial_cfg['batch_infer']) 
            # Reconstruction mesh from latent
            mesh_out = reconstruct_mesh_from_latent(pred_path, model, latent_opt, trial_cfg)
            
            # Normalize and scale output using model config training params
            center, max_radius = get_norm_params(sdf_dataset, sample_dict, pred_path)
            mesh_pv = normalize_mesh(mesh_out, pred_path, config, center, max_radius)
            
            # Save to file
            mesh_pv = mesh_pv.clean().triangulate()
            for arr in ['RegionId', 'vtkOriginalCellIds']:
                if arr in mesh_pv.cell_data.keys():
                    mesh_pv.cell_data.remove(arr)
            base_name = os.path.splitext(os.path.basename(pred_path))[0]
            new_filename = f"{base_name}_completed.vtk"
            completed_path = os.path.join(trial_dir, new_filename)
            mesh_pv.save(completed_path)

            # Calculate chamfer distance between predicted mesh and ground truth
            compl = pv.read(completed_path).triangulate().extract_surface(algorithm=None)
            gt = pv.read(gt_path).triangulate().extract_surface(algorithm=None)
            cd = chamfer_distance(compl, gt, n_samples=5000)
            mesh_time = time.time() - start
            scores.append(cd)
            times.append(mesh_time)

        # Get mean chamfer for all meshes from trial
        mean_cd = float(np.mean(scores))
        if mean_cd < best['score']:
            best = {'score': mean_cd, 'cfg': trial_cfg}
        # Append the current trial's results to the list
        mean_time = float(np.mean(times))
        rows.append({'trial': t, 'mean_cd': mean_cd, 'mean_time': mean_time, **trial_cfg})
        # Save results to csv
        if log_path_csv is not None:
            pd.DataFrame(rows).to_csv(log_path_csv, index=False)
    print(f"Best cfg: {best['cfg']} (mean Chamfer={best['score']:.4f})")
    # Save logs
    if log_path_csv:
        print(f"Finished. Final trial log with all trials saved to {log_path_csv}")
    return best['cfg'], rows

# Random search on a small validation subset to pick best cfg
def grid_search_pointnet(pairs, model, config, mean_latent, latent_codes, device, encoder_ckpt, out_dir, n_trials=15, valN=30, log_path_csv=None, log_path_json=None):
    # Set up directory for fine-tuning experiemnts
    os.makedirs(out_dir, exist_ok=True)
    val_subset = random.sample(pairs, min(valN, len(pairs)))
    best = {'score': float('inf'), 'cfg': None}
    rows = []
    # Define how many PCs describe X% of variance
    _, k95 = get_top_k_pcs(latent_codes, threshold=0.95)
    _, k99 = get_top_k_pcs(latent_codes, threshold=0.99)
    latent_std = latent_codes.std().mean()

    # Randomly pick optimization parameters from provided values
    for t in range(n_trials):
        trial_cfg = {
            'top_k': random.choice([k95, k99]),
            'iters': random.choice([0, 300, 500, 700]),
            'lr': random.choice([1e-5, 1e-4, 1e-3]),
            'lambda_reg': random.choice([1e-7, 1e-6, 1e-5]),
            'clamp': None,
            'latent_std': latent_std,
            'sched_step': random.choice([100, 200, 300]),
            'sched_gamma': random.choice([0.5, 0.7, 0.9]),
            'batch_infer': random.choice([32768]),
            'gridN': random.choice([128, 256])}
        scores = []
        times = []

        # Set up directory for each trial
        trial_dir = os.path.join(out_dir, f"trial_{t:02d}")
        os.makedirs(trial_dir, exist_ok=True)

        # Run trial on randomly chosen config params and log chamfer score
        for i, (pred_path, gt_path) in enumerate(val_subset):
            start = time.time()
            if '.ply' in pred_path:
                _, pred_path = convert_ply_to_vtk(pred_path, save=True)
            
            # Build SDF dataset
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(pred_path, config, n_samples=None) # Use all points instead of downsampling by n_samples
            
            # Fast mode: one-shot encoder instead of full latent optimization 
            print("\n-----Fast mode: encoding latent (single forward pass)----\n")
            latent_opt = encode_latent_pointnet(encoder_ckpt, points, sdf_vals, device)
            if trial_cfg['iters'] > 0:
                latent_opt, _ = optimize_latent_partial(decoder=model, partial_pts=points.squeeze(), sdfs=sdf_vals, latent_dim=latent_codes.shape[1], latent_init=latent_opt, top_k=trial_cfg['top_k'],
                                                        iters=trial_cfg['iters'], lr=trial_cfg['lr'], lambda_reg=trial_cfg['lambda_reg'], clamp_val=trial_cfg['clamp'], latent_std=trial_cfg['latent_std'], 
                                                        scheduler_step=trial_cfg['sched_step'], scheduler_gamma=trial_cfg['sched_gamma'], batch_inference_size=trial_cfg['batch_infer'], 
                                                        multi_stage=True, device=device) 

            # Reconstruction mesh from latent
            mesh_out = reconstruct_mesh_from_latent(pred_path, model, latent_opt, trial_cfg)
            
            # Normalize and scale output using model config training params
            center, max_radius = get_norm_params(sdf_dataset, sample_dict, pred_path)
            mesh_pv = normalize_mesh(mesh_out, pred_path, config, center, max_radius)
            
            # Save to file
            mesh_pv = mesh_pv.clean().triangulate()
            for arr in ['RegionId', 'vtkOriginalCellIds']:
                if arr in mesh_pv.cell_data.keys():
                    mesh_pv.cell_data.remove(arr)
            base_name = os.path.splitext(os.path.basename(pred_path))[0]
            new_filename = f"{base_name}_completed.vtk"
            completed_path = os.path.join(trial_dir, new_filename)
            mesh_pv.save(completed_path)

            # Calculate chamfer distance between predicted mesh and ground truth
            compl = pv.read(completed_path).triangulate().extract_surface(algorithm=None)
            gt = pv.read(gt_path).triangulate().extract_surface(algorithm=None)
            cd = chamfer_distance(compl, gt, n_samples=5000)
            mesh_time = time.time() - start
            scores.append(cd)
            times.append(mesh_time)

        # Get mean chamfer for all meshes from trial
        mean_cd = float(np.mean(scores))
        if mean_cd < best['score']:
            best = {'score': mean_cd, 'cfg': trial_cfg}
        # Append the current trial's results to the list
        mean_time = float(np.mean(times))
        rows.append({'trial': t, 'mean_cd': mean_cd, 'mean_time': mean_time, **trial_cfg})
        # Save results to csv
        if log_path_csv is not None:
            pd.DataFrame(rows).to_csv(log_path_csv, index=False)
    print(f"Best cfg: {best['cfg']} (mean Chamfer={best['score']:.4f})")
    # Save logs
    if log_path_csv:
        print(f"Finished. Final trial log with all trials saved to {log_path_csv}")
    return best['cfg'], rows

# Helpers for classif_tables.py and classif_figures.py
Result = namedtuple("Result", "run split eval_level latents path")

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
RUN_LABELS = {"run_v72": "Base", "run_v73h": "Hierarchy", "run_v73c": "Contrastive"}
EVAL_LABELS = {"loo": "LOO", "specimen": "LOSO"}
SUFFIX_RE = re.compile(r"^(loo|specimen|species|genus)(?:_(base|latent_opt))?$")
LABEL_FIXES = {"amphisbaenea": "amphisbaenia"}

def find_results(roots, filename, split=None, eval_level=None, latents=None,
                 run=None):
    """Yield a Result for every <filename> found under the given runs.

    Any of split/eval_level/latents/run narrows the crawl; left as None they
    match everything. Filtering here rather than after the fact is what keeps
    the scripts from writing one file per combination.
    """
    want = {"split": split, "eval_level": eval_level, "latents": latents,
            "run": run}
    for root in roots:
        base = os.path.join(root, "classification", "evaluation")
        if not os.path.isdir(base):
            print(f"  no evaluation directory under {root}", file=sys.stderr)
            continue
        run_name = os.path.basename(root.rstrip("/"))
        for path in sorted(glob.glob(os.path.join(base, "*", "*", filename))):
            d = os.path.dirname(path)
            sp = os.path.basename(os.path.dirname(d))
            tail = re.sub(rf"^{re.escape(sp)}_", "", os.path.basename(d))
            m = SUFFIX_RE.match(tail)
            ev, lat = (m.group(1), m.group(2) or "?") if m else (tail, "?")
            res = Result(run_name, sp, ev, lat, path)
            if all(v is None or getattr(res, k) == v for k, v in want.items()):
                yield res

def normalize_labels(df, quiet=False):
    """Rewrite known spelling variants in every <cat>_true/<cat>_pred column."""
    lut = {k.lower(): v for k, v in LABEL_FIXES.items()}
    seen = set()
    for c in [c for c in df.columns if c.endswith(("_true", "_pred"))]:
        s = df[c].astype("string")
        hit = s.str.lower().isin(lut)
        if hit.any():
            seen.update(s[hit].unique().tolist())
            df[c] = s.where(~hit, s.str.lower().map(lut))
    if seen and not quiet:
        print(f"  normalised label spelling: {', '.join(sorted(seen))}")
    return df

MIN_CLASSES_FOR_RHO = 4          # below this a rho is not worth reporting
S4_NAME = "Table_S4_classif_by_spec_num.csv"
S5_NAME = "Table_S5_classif_by_spec_spearmans.csv"
 
def class_support(df, cat):
    """Per-class recall and the amount of data behind each class (Table S4).
    Both counts come from predictions.csv.
    Under specimen masking the specimen count is the one that can plausibly
    drive recall, since all vertebrae of the held-out individual leave the
    gallery together.
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
    out["top5_recall"] = (g[hcol].apply(lambda s: s.astype(bool).mean())
                          if hcol in sub.columns else np.nan)
    out["n_specimens"] = g["specimen"].nunique() if "specimen" in sub.columns else np.nan
    out.index.name = "class"
    return out.reset_index()[["class", "n_specimens", "n_vertebrae",
                              "top1_recall", "top5_recall"]]
 
def _clean(*arrays):
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    ok = np.all([np.isfinite(a) for a in arrays], axis=0)
    return [a[ok] for a in arrays], int(ok.sum())
 
def spearman(x, y):
    """Spearman rho, two-sided p, and the n actually used.
    """
    (x, y), n = _clean(x, y)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, n
    rho, p = spearmanr(x, y)
    return float(rho), float(p), n
 
def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z partialled out.
    Specimen count and vertebra count are themselves strongly correlated across
    classes, so their separate rhos are not independent evidence. This asks
    whether x still tracks y once the shared variation with z is removed:
    a Pearson partial on ranks, with df = n - 3.
    """
    (x, y, z), n = _clean(x, y, z)
    if n < 5 or min(np.ptp(x), np.ptp(y), np.ptp(z)) == 0:
        return np.nan, np.nan, n
 
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in
                     ((rx, ry), (rx, rz), (ry, rz)))
    denom = math.sqrt(max(0.0, (1 - rxz ** 2) * (1 - ryz ** 2)))
    if denom == 0 or not np.isfinite(denom):
        return np.nan, np.nan, n
 
    rho = max(-1.0, min(1.0, float((rxy - rxz * ryz) / denom)))
    if abs(rho) >= 1.0:
        return rho, 0.0, n
    df = n - 3
    tstat = rho * math.sqrt(df / (1 - rho ** 2))
    return rho, float(2 * t_dist.sf(abs(tstat), df)), n
 
def correlation_rows(support, meta, recalls=("top1_recall", "top5_recall")):
    """One Table S5 row per recall metric for one (run, split) support table.
    """
    rows = []
    coll, _, _ = spearman(support["n_specimens"], support["n_vertebrae"])
    for rec in recalls:
        if rec not in support.columns or support[rec].isna().all():
            continue
        rho_s, p_s, n = spearman(support[rec], support["n_specimens"])
        rho_v, p_v, _ = spearman(support[rec], support["n_vertebrae"])
        prho_s, pp_s, _ = partial_spearman(support[rec], support["n_specimens"],
                                           support["n_vertebrae"])
        prho_v, pp_v, _ = partial_spearman(support[rec], support["n_vertebrae"],
                                           support["n_specimens"])
        rows.append({**meta, "metric": rec, "n_classes": n,
                     "rho_specimens": rho_s, "p_specimens": p_s,
                     "rho_vertebrae": rho_v, "p_vertebrae": p_v,
                     "partial_rho_specimens": prho_s, "partial_p_specimens": pp_s,
                     "partial_rho_vertebrae": prho_v, "partial_p_vertebrae": pp_v,
                     "rho_specimens_vs_vertebrae": coll})
    return rows
 
def write_supplement(per_class, corr, outdir, quiet=False):
    """Table S4 (per-class counts and recall) and Table S5 (its correlations)."""
    written = []
    if per_class:
        s4 = pd.concat(per_class, ignore_index=True)
        lead = [c for c in ("run", "split", "category") if c in s4.columns]
        s4 = s4[lead + [c for c in s4.columns if c not in lead]]
        s4[["top1_recall", "top5_recall"]] = s4[["top1_recall", "top5_recall"]].round(3)
        s4 = s4.sort_values(
            ["run", "split", "class"],
            key=lambda c: c.map(SPLIT_ORDER) if c.name == "split" else c)
        s4.to_csv(os.path.join(outdir, S4_NAME), index=False)
        written.append(S4_NAME)
 
    if corr:
        s5 = pd.DataFrame(corr)
        s5 = s5.sort_values(
            ["run", "metric", "split"],
            key=lambda c: c.map(SPLIT_ORDER) if c.name == "split" else c)
        s5[s5.select_dtypes("float").columns] = s5.select_dtypes("float").round(4)
        s5.to_csv(os.path.join(outdir, S5_NAME), index=False)
        written.append(S5_NAME)
        if not quiet:
            keep = s5[s5["n_classes"] >= MIN_CLASSES_FOR_RHO]
            print(f"\n{'=' * 92}\nRECALL vs SUPPORT (Spearman)\n{'=' * 92}")
            print(keep.to_string(index=False) if not keep.empty else
                  f"no condition had >= {MIN_CLASSES_FOR_RHO} classes")
    return written
