# Shape completion for partial vertebrae

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
import pyvista as pv
import pymskt.mesh.meshes as meshes
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
sample_by_bbox = False  # TO DO: indicate if want to sample from manually placed bounding box from Slicer
sample_by_mrks = False # TO DO: indicate if want to sample from manually placed markups from Slicer

# Load model config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")

# Select paths to meshes for shape completion
mesh_dir = "path/to/your/shape_completion/partial_meshes" # TO: Update path to your partial meshes made with create_partial_meshes.py
mesh_list = random.sample([os.path.join(mesh_dir, name) for name in os.listdir(mesh_dir)], 20) # TO DO: Update how many to randomly sample
# Select corresponding bounding boxes with intact regions of specimens outlined (filenames should match meshes, but with .mrk.json extension)
bbox_list = [os.path.join(dir, fname) for fname in os.listdir(dir) if ".mrk.json" in fname] # TO DO: Enter paths here
#bbox_list = [os.path.splitext(mesh_list[0])[0] + ".mrk.json"]

# Define functions

def load_slicer_mrkup_pts(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    markups = data["markups"][0]              # first markup node
    points = markups["controlPoints"]        # list of control point dicts
    # Extract positions
    pts = np.array([p["position"] for p in points], dtype=np.float32)
    return pts

# Optimize latent from partial pointcloud (model has no encoder, so need to optimize before feeding in new data)
def optimize_latent_partial(decoder, partial_pts, latent_dim, mean_latent=None, latent_init=None, iters=2000, 
                            lr=1e-4, lambda_reg=1e-4, clamp_val=None, scheduler_step=1000, scheduler_gamma=0.5, 
                            top_k=10, batch_inference_size=32768, verbose=True, device=device, multi_stage=False):
    decoder = decoder.to(device)
    decoder.eval()
    if isinstance(partial_pts, np.ndarray):
        partial_pts = torch.tensor(partial_pts, dtype=torch.float32)
    partial_pts, sdfs = sample_near_surface(partial_pts, eps=0.005, fraction_nonzero=0.4, 
                                            fraction_far=0.05, far_eps=0.05)
    partial_pts = torch.tensor(partial_pts, dtype=torch.float32)
    partial_pts = partial_pts.to(device)
    target = torch.tensor(sdfs, dtype=torch.float32).to(device)
    # If multi-stage optimization, intialize from previous latent, not mean
    if multi_stage:
        mean_latent = latent_init.clone().detach()
        latent = latent_init.clone().detach().to(device).requires_grad_(True)
    # If single-stage, initialize from pca based mean of latent codes
    else:
        mean_latent = mean_latent.clone().detach()
        latent = pca_initialize_latent(mean_latent, latent_init, top_k).clone().detach()
        latent = latent.to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    loss_log = []
    for step in range(iters):
        optimizer.zero_grad()
        # Evaluate predicted SDFs in mini-batches to save memory
        preds = get_sdfs(decoder, partial_pts, latent, batch_size=batch_inference_size, device=device)  # (N,1)
        # surface loss (absolute SDF near 0)
        sdf_loss = F.l1_loss(preds.to(device), target)
        # latent prior: encourage closeness to mean_latent
        reg_loss = torch.mean((latent - mean_latent.to(device)) ** 2)
        loss = sdf_loss + lambda_reg * reg_loss
        loss.backward()
        # gradient clipping and step
        torch.nn.utils.clip_grad_norm_([latent], 1.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if clamp_val is not None:
                offset = latent - mean_latent.to(latent.device)   # shape (1, D)
                clamp_val = clamp_val.to(device)
                offset = offset.to(device)
                offset[:] = torch.clamp(offset, -clamp_val, clamp_val) # in-place to offset
                latent[:] = mean_latent.to(latent.device) + offset
        loss_log.append(float(loss.item()))
        if verbose and (step % 100 == 0 or step == iters-1):
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            print(f"[{step:4d}/{iters}] loss={loss.item():.6e} sdf={sdf_loss.item():.6e} reg={reg_loss.item():.6e} lr={lr_now:.2e}")
    return latent.detach(), loss_log

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

def load_slicer_roi_bbox(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    roi = data["markups"][0]
    center = np.array(roi["center"])
    size = np.array(roi["size"])
    orientation = np.array(roi["orientation"]).reshape(3, 3)
    # Compute half-axes in world coordinates
    half_size = size / 2.0
    axes = orientation * half_size[np.newaxis, :]
    local_corners = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [ 1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [-1,  1,  1],
        [ 1,  1,  1]]) * half_size
    world_corners = (orientation @ local_corners.T).T + center
    # Create PyVista box mesh
    bbox_params = (half_size, orientation, center)
    return bbox_params

def sample_points_in_bbox(mesh_path, bbox_params, n_points=500, method='poisson'):
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
    mesh_o3d.compute_vertex_normals()    # Convert to Open3D for Poisson disk sampling
    oversample_factor = 5
    n_sample = n_points * oversample_factor
    if method == 'poisson':
        pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=n_sample)
    else:
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_sample)
    pts = np.asarray(pcd.points)
    # Transform points to ROI's local coordinate system
    half_size, orientation, center = bbox_params
    rel_pts = pts - center
    local_pts = rel_pts @ orientation.T  # rotate into ROI frame
    # Create mask for points inside the box extents
    in_x = np.abs(local_pts[:, 0]) <= half_size[0]
    in_y = np.abs(local_pts[:, 1]) <= half_size[1]
    in_z = np.abs(local_pts[:, 2]) <= half_size[2]
    mask = in_x & in_y & in_z
    pts_inside = pts[mask]
    if len(pts_inside) < n_points:
        print(f"Warning: only {len(pts_inside)} surface points inside ROI (requested {n_points})")
        n_points_final = len(pts_inside)
    else:
        n_points_final = n_points
        idx = np.random.choice(len(pts_inside), n_points_final, replace=False)
        pts_inside = pts_inside[idx]
    return pts_inside

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

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)
_, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)

# Loop through meshes
summary_log = []
for i, vert_fname in enumerate(mesh_list):    
    print(f"\033[32m\n=== Processing {os.path.basename(vert_fname)} ===\033[0m")
    print(f"\033[32m\n=== Mesh {i+1} / {len(mesh_list)} ===\033[0m")
    # Make a new dir to save predictions
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
    # Load points from specified bounding box or randomly downsample mesh
    if sample_by_bbox:
        bbox = load_slicer_roi_bbox(bbox_list[i])
        partial_pts = sample_points_in_bbox(vert_fname, bbox, n_points=800)
    elif sample_by_mrks:
        partial_pts = load_slicer_mrkup_pts(bbox_list[i])
    else: # Downsample intact ground truth mesh
        partial_pts = downsample_partial_pointcloud(vert_fname, 180)
    partial_pts = torch.tensor(partial_pts, dtype=torch.float32)
    partial_cloud = pv.PolyData(partial_pts.cpu().numpy())
    partial_cloud.save(outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_partial_input.vtk")

    # Optimize latents
    print("Optimizing latents")
    # Phase 1 - Coarse Optimization - get a global shape in the right area of latent space (close to target specimen (far enough from mean); but not so far from mean that it is noisy or unrealistic)
    latent_partial, loss_log = optimize_latent_partial(model, partial_pts, config['latent_size'], mean_latent=mean_latent, latent_init=latent_codes, top_k=top_k_reg, 
                                                       iters=5000, lr=1e-4, lambda_reg=1e-3, clamp_val=2.0 * latent_codes.std().mean(), scheduler_step=800, scheduler_gamma=0.8, 
                                                       batch_inference_size=32768, multi_stage=False)
    # Phase 2 - Refinement - emphasis on local SDF samples and surface consistency to refine target specimen shape
    latent_partial, loss_log = optimize_latent_partial(model, partial_pts, config['latent_size'], latent_init=latent_partial, top_k=top_k_reg, 
                                                        iters=8000, lr=1.3e-5, lambda_reg=0.7e-4, clamp_val=None, scheduler_step=800, scheduler_gamma=0.7, 
                                                        batch_inference_size=32768, multi_stage=True) # True because second stage using already initialized latent
    print("Translated novel mesh into latent space!")
    
    # Reconstruction parameters
    recon_grid_origin = 1.0
    n_pts_per_axis = 256 # TO DO: Adjust resolution
    voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
    voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
    offset = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    icp_transform = NumpyTransform(np.eye(4))
    objects = 1

    # Reconstruct the novel mesh
    with torch.no_grad():
        mesh_out = create_mesh(decoder=model, latent_vector=latent_partial, n_pts_per_axis=n_pts_per_axis,
                                voxel_origin=voxel_origin, voxel_size=voxel_size, path_original_mesh=None,
                                offset=offset, scale=scale, icp_transform=icp_transform, objects=objects,
                                verbose=True, device=device, smooth=1.0)
        
    # Ensure it's PyVista PolyData
    if isinstance(mesh_out, list):
        mesh_out = mesh_out[0]
    if not isinstance(mesh_out, pv.PolyData):
        mesh_pv = mesh_out.extract_geometry()
    else:
        mesh_pv = mesh_out

    # Save mesh
    mesh_pv = mesh_pv.clean()
    mesh_pv = mesh_pv.triangulate()
    output_path = outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_shape_completion.vtk"
    # Set color: RGB in range 0–255 or 0–1
    color = np.array([112, 215, 222], dtype=np.uint8)  
    # Broadcast color to all points
    rgb = np.tile(color, (mesh_pv.n_points, 1))
    mesh_pv.point_data.clear()
    mesh_pv.point_data['Colors'] = rgb
    mesh_pv.save(output_path)
    print(f"Completed mesh from partial pointcloud saved to: {output_path}")