# Fine-tune parameters for shape completion of partial vertebrae
# Use partial vertebrae dataset made with create_partial_meshes.py for validation against ground truth meshes

import os
import torch
import numpy as np
import pandas as pd
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.mesh import get_sdfs
from NSM.reconstruct import reconstruct_latent
import torch.nn.functional as F
import json
import sys
import pyvista as pv
import pymskt.mesh.meshes as meshes
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from NSM.mesh import create_mesh
import vtk
import re
import random
import open3d as o3d
from NSM.helper_funcs import NumpyTransform, load_config, load_model_and_latents, convert_ply_to_vtk, get_sdfs, fixed_point_coords, safe_load_mesh_scalars 
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, sample_near_surface, downsample_partial_pointcloud, optimize_latent_partial
# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)
import time

# Define training directory
TRAIN_DIR = "run_v56" # TO DO: Choose training directory containing model ckpt and latent codes
CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH =  TRAIN_DIR + '/latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = TRAIN_DIR +  '/model' + '/' + CKPT + '.pth'
val_sum_path = TRAIN_DIR + "/shape_completion/meshes/partial_meshes" # TO DO: Choose to load validation_summary.json from (generated using create_partial_meshes.py)
val_sum_fn = val_sum_path + "/partial_meshing_summary.json"

# Load model config
config = load_config(config_path=TRAIN_DIR + '/model_params_config.json')
device = config.get("device", "cuda:0")

## Fine-tune shape completion steps

# 1) Build partial_mesh_path and ground_truth_path pairs
def strip_mesh_name(path):
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    if name.endswith("_partial"):
        name = name[:-8]
    return name

#  Load validation summary
with open(val_sum_fn, "r") as f:
    val = json.load(f)

# Build test mesh name set from config
test_mesh_names = {strip_mesh_name(p) for p in config["test_paths"]}
print(f"Found {len(test_mesh_names)} test meshes in config")

# Filter validation meshes
pairs = []
skipped = 0
for m in val["meshes"]:
    base_name = strip_mesh_name(m["ground_truth"])

    if base_name in test_mesh_names:
        pairs.append((m["partial"], m["ground_truth"]))
    else:
        skipped += 1

print(f"Built {len(pairs)} (partial, ground_truth) pairs")
print(f"Skipped {skipped} meshes not in test_paths")

# 2) Accuracy Metrics
def _uniform_surface_sample(poly, n):
    # Triangulate the mesh
    poly = poly.triangulate().extract_geometry()
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
def chamfer_distance(pred_path, gt_path, n_samples=20000):
    # Read in completed and ground truth meshes
    mp = pv.read(pred_path).triangulate().extract_geometry()
    gt = pv.read(gt_path).triangulate().extract_geometry()
    # Sample points across surface
    sp = _uniform_surface_sample(mp, n_samples)
    sg = _uniform_surface_sample(gt, n_samples)
    # Use KD-tree to find nearest neighbor distances of gt to predicted surface and vice versa
    t1 = cKDTree(sp); t2 = cKDTree(sg)
    d1 = t1.query(sg, k=1)[0].mean()
    d2 = t2.query(sp, k=1)[0].mean()
    return float(0.5*(d1 + d2)) # Return average distance (symmetric penalty)

# 3) Run trial (uses optimize_latent_partial and create_mesh)
def run_trial(partial_mesh_path, gt_path, partial_pts, sdfs, cfg, out_dir, model, mean_latent, latent_codes, device):
    # 2-phase optimization to "encode" partial mesh into latent space 
    print("\n-----Optimizing latents----\n")
    lat, _ = optimize_latent_partial(  # Phase 1: coarse reconstruction near mean
        decoder=model, partial_pts=partial_pts, sdfs=sdfs, latent_dim=latent_codes.shape[1],
        mean_latent=mean_latent, latent_init=latent_codes, top_k=cfg['top_k'],
        iters=cfg['iters1'], lr=cfg['lr1'], lambda_reg=cfg['lambda1'],
        clamp_val=cfg['clamp'], latent_std=cfg['latent_std'], scheduler_step=cfg['sched_step'],
        scheduler_gamma=cfg['sched_gamma'], batch_inference_size=cfg['batch_infer'],
        device=device, multi_stage=False) 
    lat, _ = optimize_latent_partial(  # Phase 2: refine surface details for specific specimen
        decoder=model, partial_pts=partial_pts, sdfs=sdfs, latent_dim=latent_codes.shape[1],
        latent_init=lat, iters=cfg['iters2'], lr=cfg['lr2'], lambda_reg=cfg['lambda2'],
        clamp_val=cfg['clamp'], latent_std=cfg['latent_std'], scheduler_step=cfg['sched_step'],
        scheduler_gamma=cfg['sched_gamma'], batch_inference_size=cfg['batch_infer'],
        device=device, multi_stage=True)  
    print("\nTranslated novel mesh into latent space!\n")
    # Create mesh from optimized latent
    with torch.no_grad():
        mesh_out = create_mesh( 
            decoder=model, latent_vector=lat, n_pts_per_axis=cfg['gridN'],
            voxel_origin=(-1.0,-1.0,-1.0), voxel_size=(2.0/(cfg['gridN']-1)),
            path_original_mesh=None, offset=np.array([0,0,0]), scale=1.0,
            icp_transform=None, objects=1, verbose=False, device=device, smooth=1.0)
    mp = mesh_out[0] if isinstance(mesh_out, list) else mesh_out
    if not isinstance(mp, pv.PolyData): mp = mp.extract_geometry()
    mp = mp.clean().triangulate()
    # Save to file
    base_name = os.path.splitext(os.path.basename(partial_mesh_path))[0]
    new_filename = f"{base_name}_partial.vtk"
    pred_path = os.path.join(out_dir, new_filename)
    mp.save(pred_path)
    # Calculate chamfer distance between partial-completed and original-ground truth mesh
    cd = chamfer_distance(pred_path, gt_path)
    return cd, pred_path

# 4) Random search on a small validation subset to pick best cfg
def random_search(pairs, model, mean_latent, latent_codes, device, out_dir, n_trials=15, valN=30, log_path_csv=None, log_path_json=None):
    # Set up directory for fine-tuning experiemnts
    os.makedirs(out_dir, exist_ok=True)
    subset = pairs[:valN]
    best = {'score': float('inf'), 'cfg': None}
    rows = []
    # Define how many PCs describe X% of variance
    _, k95 = get_top_k_pcs(latent_codes, threshold=0.95)
    _, k90 = get_top_k_pcs(latent_codes, threshold=0.90)
    _, k99 = get_top_k_pcs(latent_codes, threshold=0.99)
    latent_std = latent_codes.std().mean()

    # Randomly pick optimization parameters from provided values
    for t in range(n_trials):

        cfg = {
            'top_k': random.choice([k95, k90, k99]),
            'iters1': random.choice([3000, 5000, 7000]),
            'iters2': random.choice([6000, 8000, 10000]),
            'lr1': random.choice([1.0e-5, 1e-4, 1.0e-3]),
            'lr2': random.choice([1e-6, 1e-5, 1e-4]),
            'lambda1': random.choice([1e-4, 1e-3, 1e-3]),
            'lambda2': random.choice([1e-5, 0.7e-4, 1e-3]),
            'clamp': random.choice([None, 1, 2]),
            'latent_std': latent_std,
            'sched_step': random.choice([500, 800, 1000]),
            'sched_gamma': random.choice([0.7, 0.8, 0.9]),
            'batch_infer': random.choice([16384, 32768]),
            'gridN': random.choice([256, 320, 384]),
        }
        scores = []
        times = []
        # Set up directory for each trial
        trial_dir = os.path.join(out_dir, f"trial_{t:02d}")
        os.makedirs(trial_dir, exist_ok=True)
        # Run trial on randomly chosen config params and log chamfer score
        for i, (pm, gt) in enumerate(subset):
            start = time.time()
            partial_pts = downsample_partial_pointcloud(pm, 180)
            partial_pts = torch.tensor(partial_pts, dtype=torch.float32)
            partial_pts, sdfs = sample_near_surface(partial_pts, eps=0.005, fraction_nonzero=0.4, 
                                                    fraction_far=0.05, far_eps=0.05)
            partial_pts = partial_pts.clone().detach()
            cd, _ = run_trial(pm, gt, partial_pts, sdfs, cfg, trial_dir, model, mean_latent, latent_codes, device)
            mesh_time = time.time() - start
            scores.append(cd)
            times.append(mesh_time)
        # Get mean chamfer for all meshes from trial
        mean_cd = float(np.mean(scores))
        if mean_cd < best['score']:
            best = {'score': mean_cd, 'cfg': cfg}
        # Append the current trial's results to the list
        mean_time = float(np.mean(times))
        rows.append({'trial': t, 'mean_cd': mean_cd, 'mean_time': mean_time, **cfg})
        # Save results to csv
        if log_path_csv is not None:
            pd.DataFrame(rows).to_csv(log_path_csv, index=False)
    print(f"Best cfg: {best['cfg']} (mean Chamfer={best['score']:.4f})")
    # Save logs
    if log_path_csv:
        print(f"Finished. Final trial log with all trials saved to {log_path_csv}")
    return best['cfg'], rows

## Actual optimization

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)

# Find the best hyperparameters using random search
best_cfg, trial_rows = random_search(pairs, model, mean_latent, latent_codes, device,
                                    out_dir="shape_completion/fine_tuning",
                                    n_trials=5, valN=200,
                                    log_path_csv="shape_completion/fine_tuning/trial_scores.csv")

# Loop through meshes using best parameters
summary_log = []
for pm_path, gt_path in pairs[:20]:    
    print(f"\033[32m\n=== Processing {os.path.basename(pm_path)} ===\033[0m")
    # Make a new dir to save predictions
    vert_fname = pm_path
    outfpath = TRAIN_DIR + '/shape_completion/predictions/' + os.path.splitext(os.path.basename(vert_fname))[0] # TO DO: Adjust to desired outpath
    print("Making a new directory to save model predictions and outputs at: ", outfpath)
    os.makedirs(outfpath, exist_ok=True)

    # Convert plys to vtks
    if '.ply' in vert_fname:
        ply_fname = vert_fname
        mesh, vert_fname = convert_ply_to_vtk(ply_fname, save=True)

    # Setup your dataset with just one mesh
    sdf_dataset = SDFSamples(
        list_mesh_paths=[vert_fname],
        multiprocessing=False,
        subsample=config["samples_per_object_per_batch"],
        print_filename=True,
        n_pts=config["n_pts_per_object"],
        p_near_surface=config['percent_near_surface'],
        p_further_from_surface=config['percent_further_from_surface'],
        sigma_near=config['sigma_near'],
        sigma_far=config['sigma_far'],
        rand_function=config['random_function'], 
        center_pts=config['center_pts'],
        norm_pts=config['normalize_pts'],
        reference_mesh=None,
        verbose=config['verbose'],
        save_cache=config['cache'],
        equal_pos_neg=config['equal_pos_neg'],
        fix_mesh=config['fix_mesh'])

    # Get the point/SDF data
    print("Setting up dataset")
    sdf_sample = sdf_dataset[0]  # returns a dict
    sample_dict, _ = sdf_sample
    points = sample_dict['xyz'].to(device) # shape: [N, 3]
    sdf_vals = sample_dict['gt_sdf']  # shape: [N, 1]
    partial_pts = downsample_partial_pointcloud(vert_fname, 180)
    partial_pts = torch.tensor(partial_pts, dtype=torch.float32)
    partial_cloud = pv.PolyData(partial_pts.cpu().numpy())
    partial_cloud.save(outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_partial_input.vtk")

    # Sample points with a variety of SDF values (not all = 0 to allow smoother fit)
    partial_pts, sdfs = sample_near_surface(partial_pts, eps=0.005, fraction_nonzero=0.4, 
                                            fraction_far=0.05, far_eps=0.05)
    partial_pts = partial_pts.clone().detach()

    # Optimize latents
    cd, pred_path = run_trial(pm_path, gt_path, partial_pts, sdfs, best_cfg, outfpath, model, mean_latent, latent_codes, device)
    print(f"\n{os.path.basename(pm_path)} Chamfer={cd:.4f} → {pred_path}\n") 