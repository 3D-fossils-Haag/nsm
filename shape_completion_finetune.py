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
from NSM.optimization import pca_initialize_latent, get_top_k_pcs
# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

# Define training directory
TRAIN_DIR = "run_v44" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '2000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'
val_sum_path = "shape_completion/meshes/partial_meshes" # TO DO: Choose path to save validation_summary.json
val_sum_fn = val_sum_path + "/partial_meshing_summary.json"

# Load model config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")

## Fine-tune shape completion steps

# 1) Build partial_mesh_path and ground_truth_path pairs
with open(val_sum_fn,"r") as f:
    val = json.load(f)
pairs = [(m['partial'], m['ground_truth']) for m in val['meshes']]

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
def run_trial(partial_mesh_path, gt_path, cfg, out_dir, model, mean_latent, latent_codes, device):
    # Get partial points (use mesh vertices; you can swap in your sampler if you prefer)
    partial_pts = pv.read(partial_mesh_path).triangulate().points
    partial_pts = torch.tensor(partial_pts, dtype=torch.float32, device=device)
    # 2-phase optimization to "encode" partial mesh into latent space 
    lat, _ = optimize_latent_partial(  # Phase 1: coarse reconstruction near mean
        decoder=model, partial_pts=partial_pts, latent_dim=latent_codes.shape[1],
        mean_latent=mean_latent, latent_init=latent_codes, top_k=cfg['top_k'],
        iters=cfg['iters1'], lr=cfg['lr1'], lambda_reg=cfg['lambda1'],
        clamp_val=cfg['clamp'], scheduler_step=cfg['sched_step'],
        scheduler_gamma=cfg['sched_gamma'], batch_inference_size=cfg['batch_infer'],
        device=device, multi_stage=False) 
    lat, _ = optimize_latent_partial(  # Phase 2: refine surface details for specific specimen
        decoder=model, partial_pts=partial_pts, latent_dim=latent_codes.shape[1],
        latent_init=lat, iters=cfg['iters2'], lr=cfg['lr2'], lambda_reg=cfg['lambda2'],
        clamp_val=cfg['clamp'], scheduler_step=cfg['sched_step'],
        scheduler_gamma=cfg['sched_gamma'], batch_inference_size=cfg['batch_infer'],
        device=device, multi_stage=True)  
    # Create mesh from optimized latent
    with torch.no_grad():
        mesh_out = create_mesh( 
            decoder=model, latent_vector=lat, n_pts_per_axis=cfg['gridN'],
            voxel_origin=(-1.0,-1.0,-1.0), voxel_size=(2.0/(cfg['gridN']-1)),
            path_original_mesh=None, offset=np.array([0,0,0]), scale=1.0,
            icp_transform=None, objects=1, verbose=False, device=device)
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

    # Randomly pick optimization parameters from provided values
    for t in range(n_trials):

        cfg = {
            'top_k': random.choice([k95, k90, k99]),
            'iters1': random.choice([1500, 2500, 3000]),
            'iters2': random.choice([5000, 8000, 10000]),
            'lr1': random.choice([1.0e-3, 5e-3, 1.0e-4]),
            'lr2': random.choice([1e-4, 5.0e-5, 3e-4]),
            'lambda1': random.choice([5e-6, 1e-6, 5e-5]),
            'lambda2': random.choice([5e-5, 1e-4, 5e-4]),
            'clamp': random.choice([None, 0.5, 1.0]),
            'sched_step': random.choice([800, 1000]),
            'sched_gamma': random.choice([0.5, 0.7]),
            'batch_infer': random.choice([16384, 32768, 65536]),
            'gridN': random.choice([256, 320, 384, 448]),
        }
        scores = []
        # Set up directory for each trial
        trial_dir = os.path.join(out_dir, f"trial_{t:02d}")
        os.makedirs(trial_dir, exist_ok=True)
        # Run trial on randomly chosen config params and log chamfer score
        for i, (pm, gt) in enumerate(subset):
            cd, _ = run_trial(pm, gt, cfg, trial_dir, model, mean_latent, latent_codes, device)
            scores.append(cd)
        # Get mean chamfer for all meshes from trial
        mean_cd = float(np.mean(scores))
        if mean_cd < best['score']:
            best = {'score': mean_cd, 'cfg': cfg}
        # Append the current trial's results to the list
        rows.append({'trial': t, 'mean_cd': mean_cd, **cfg})
        # Save results to csv
        if log_path_csv is not None:
            pd.DataFrame(rows).to_csv(log_path_csv, index=False)
    print(f"Best cfg: {best['cfg']} (mean Chamfer={best['score']:.4f})")
    # Save logs
    if log_path_csv:
        print(f"Finished. Final trial log with all trials saved to {log_path_csv}")
    return best['cfg'], rows

## Shape completion functions

# Optimize latent from partial pointcloud (model has no encoder, so need to optimize before feeding in new data)
def optimize_latent_partial(decoder, partial_pts, latent_dim, mean_latent=None, latent_init=None, iters=2000,
                            lr=1e-4, lambda_reg=1e-4, clamp_val=None, scheduler_step=1000, scheduler_gamma=0.5,
                            top_k=10, batch_inference_size=32768, verbose=True, device='cuda', multi_stage=False):
    # Load model
    decoder = decoder.to(device)
    decoder.eval()
    # Prepare sampled points and target SDFs
    if isinstance(partial_pts, np.ndarray):
        base_pts = torch.tensor(partial_pts, dtype=torch.float32, device=device)
    elif isinstance(partial_pts, torch.Tensor):
        base_pts = partial_pts.to(device).to(torch.float32)
    else:
        raise TypeError("partial_pts must be a numpy array or torch tensor")
    sampled_pts, sdfs = sample_near_surface(base_pts, eps=0.005, fraction_nonzero=0.4, fraction_far=0.05, far_eps=0.05)
    sampled_pts = sampled_pts.to(device).to(torch.float32)
    target = sdfs.to(device).to(torch.float32)
    # Initialize latent
    if multi_stage: # If second pass, initialize from latent output of first pass
        mean_latent = latent_init.clone().detach()
        latent = latent_init.clone().detach().to(device).requires_grad_(True)
    else: # First/only pass, initialize from PCA mean
        mean_latent = mean_latent.clone().detach()
        latent = pca_initialize_latent(mean_latent, latent_init, top_k).clone().detach().to(device).requires_grad_(True)
    # Set up optimizer and schedule
    optimizer = torch.optim.Adam([latent], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # Optimization loop (w/loss logging)
    loss_log = []
    for step in range(iters):
        optimizer.zero_grad()
        # Predict SDFs in mini-batches to save memory
        preds = get_sdfs(decoder, sampled_pts, latent, batch_size=batch_inference_size, device=device)
        # Losses
        sdf_loss = F.l1_loss(preds, target)
        reg_loss = torch.mean((latent - mean_latent.to(device)) ** 2)
        loss = sdf_loss + lambda_reg * reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([latent], 1.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if clamp_val is not None:
                offset = latent - mean_latent.to(latent.device)
                offset[:] = torch.clamp(offset, -clamp_val, clamp_val)
                latent[:] = mean_latent.to(latent.device) + offset
        loss_log.append(float(loss.item()))
        if verbose and ((step % 100 == 0) or (step == iters - 1)):
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            print(f"[{step:4d}/{iters}] loss={loss.item():.6e} sdf={sdf_loss.item():.6e} reg={reg_loss.item():.6e} lr={lr_now:.2e}")
    return latent.detach(), loss_log

# Downsample partial mesh into pointcloud for calculating SDF's
def downsample_partial_pointcloud(mesh_path, n_points=5000, voxel_fraction=0.01, method='poisson'):
    # Read mesh with PyVista
    mesh_pv = pv.read(mesh_path)
    if not mesh_pv.is_all_triangles:
        mesh_pv = mesh_pv.triangulate()
    # Smooth mesh to denoise
    mesh_pv = mesh_pv.smooth(n_iter=50, relaxation_factor=0.01)
    # Convert to Open3D mesh
    vertices = np.asarray(mesh_pv.points)
    faces = np.asarray(mesh_pv.faces.reshape(-1, 4)[:, 1:])  # PyVista stores faces as [n, i, j, k]
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces))
    mesh_o3d.compute_vertex_normals()
    # Uniform or Poisson disk sampling
    if method == 'poisson':
        pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=n_points)
    else:
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_points)
    # Voxel downsample
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(np.array(bbox.get_extent()))
    voxel_size = max(diag * voxel_fraction, 1e-5)  # ensure nonzero voxel size
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.asarray(pcd_down.points)
    print(f"Sampled {n_points} → Downsampled to {len(pts)} points (voxel={voxel_size:.4f})")
    return pts

# Sample points around surface to calculate SDF's from (a mix of zero and non-zero values)
def sample_near_surface(surface_pts, eps=0.005, fraction_nonzero=0.3, 
                        fraction_far=0.05, far_eps=0.05):
    n_pts = surface_pts.shape[0]
    # Slightly perturbed points (near-surface)
    n_nonzero = int(n_pts * fraction_nonzero)
    idx_near = torch.randperm(n_pts)[:n_nonzero]
    base_near = surface_pts[idx_near]
    dirs_near = torch.randn_like(base_near)
    dirs_near = dirs_near / torch.norm(dirs_near, dim=1, keepdim=True)
    pts_out_near = base_near + eps * dirs_near
    pts_in_near  = base_near - eps * dirs_near
    sdf_out_near = eps  * torch.ones((n_nonzero, 1), device=surface_pts.device)
    sdf_in_near  = -eps * torch.ones((n_nonzero, 1), device=surface_pts.device)
    pts_nonzero = torch.cat([pts_out_near, pts_in_near], dim=0)
    sdf_nonzero = torch.cat([sdf_out_near, sdf_in_near], dim=0)
    # Farther-away points for regularization
    n_far = int(n_pts * fraction_far)
    idx_far = torch.randperm(n_pts)[:n_far]
    base_far = surface_pts[idx_far]
    dirs_far = torch.randn_like(base_far)
    dirs_far = dirs_far / torch.norm(dirs_far, dim=1, keepdim=True)
    pts_out_far = base_far + far_eps * dirs_far
    pts_in_far  = base_far - far_eps * dirs_far
    sdf_out_far = far_eps  * torch.ones((n_far, 1), device=surface_pts.device)
    sdf_in_far  = -far_eps * torch.ones((n_far, 1), device=surface_pts.device)
    pts_far = torch.cat([pts_out_far, pts_in_far], dim=0)
    sdf_far = torch.cat([sdf_out_far, sdf_in_far], dim=0)
    # Keep remaining points exactly on the surface (SDF=0)
    mask = torch.ones(n_pts, dtype=torch.bool)
    mask[idx_near] = False
    mask[idx_far] = False
    pts_zero = surface_pts[mask]
    sdf_zero = torch.zeros((pts_zero.shape[0], 1), device=surface_pts.device)
    # Combine everything
    pts = torch.cat([pts_zero, pts_nonzero, pts_far], dim=0)
    sdf = torch.cat([sdf_zero, sdf_nonzero, sdf_far], dim=0)
    return pts, sdf


## Actual optimization

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)

# Find the best hyperparameters using random search
best_cfg, trial_rows = random_search(pairs, model, mean_latent, latent_codes, device,
                                    out_dir="shape_completion/fine_tuning",
                                    n_trials=20, valN=24,
                                    log_path_csv="shape_completion/fine_tuning/trial_scores.csv")

# Loop through meshes using best parameters
summary_log = []
for pm_path, gt_path in pairs[:20]:    
    print(f"\033[32m\n=== Processing {os.path.basename(pm_path)} ===\033[0m")
    # Make a new dir to save predictions
    vert_fname = pm_path
    outfpath = 'shape_completion/predictions/' + os.path.splitext(os.path.basename(vert_fname))[0] # TO DO: Adjust to desired outpath
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
    partial_pts = downsample_partial_pointcloud(vert_fname, 1200)
    partial_pts = torch.tensor(partial_pts, dtype=torch.float32)
    partial_cloud = pv.PolyData(partial_pts.cpu().numpy())
    partial_cloud.save(outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_partial_input.vtk")

    # Optimize latents
    cd, pred_path = run_trial(pm_path, gt_path, best_cfg, outfpath, model, mean_latent, latent_codes, device)
    print(f"{os.path.basename(pm_path)} Chamfer={cd:.4f} → {pred_path}") 