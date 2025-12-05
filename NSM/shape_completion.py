# Shape completion for partial vertebrae

import os
import torch
import numpy as np
import pandas as pd
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
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
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, sample_near_surface, downsample_partial_pointcloud, optimize_latent_partial, sample_points_in_bbox, load_slicer_mrkup_pts, load_slicer_roi_bbox
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
#mesh_dir = "path/to/your/shape_completion/partial_meshes" # TO: Update path to your partial meshes made with create_partial_meshes.py
#mesh_list = random.sample([os.path.join(mesh_dir, name) for name in os.listdir(mesh_dir)], 20) # TO DO: Update how many to randomly sample
#mesh_list = random.sample(config['test_paths'], 100)
#mesh_list = random.sample(config['val_paths'], 5) # TO DO: Choose val or test paths
#mesh_list = ["/home/k.wolcott/NSM/nsm/exclu/zzzzz_fossil_parviraptoraa_fillholes_smooth_hollow.vtk", "/home/k.wolcott/NSM/nsm/exclu/zzzzz_fossil_uf546657_fillholes_smooth_hollow.vtk"] # TO DO: Enter paths here
mesh_list = [fname for fname in os.listdir("/home/k.wolcott/NSM/nsm/exclu") if "_fillholes_smooth" in fname]
dir = "/home/k.wolcott/NSM/nsm/exclu"
mesh_list = [os.path.join(dir, fname) for fname in os.listdir(dir) if "_fillholes_smooth" in fname]   
#mesh_list = [dir + "/zzzz_dmnh-epv-142201-mad0720-1_fillholes_smooth.vtk"]
# Select corresponding bounding boxes with intact regions of specimens outlined (filenames should match meshes, but with .mrk.json extension)
bbox_list = [os.path.join(dir, fname) for fname in os.listdir(dir) if ".mrk.json" in fname] # TO DO: Enter paths here
#bbox_list = [os.path.splitext(mesh_list[0])[0] + ".mrk.json"]

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)
latent_std = latent_codes.std().mean()
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
    print("\n-----Setting up dataset-----\n")
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

    # Sample points with a variety of SDF values (not all = 0 to allow smoother fit)
    partial_pts, sdfs = sample_near_surface(partial_pts, eps=0.005, fraction_nonzero=0.4, 
                                            fraction_far=0.05, far_eps=0.05)
    
    # Optimize latents
    print("\n-----Optimizing latents----\n")
    # Phase 1 - Coarse Optimization - get a global shape in the right area of latent space (close to target specimen (far enough from mean); but not so far from mean that it is noisy or unrealistic)
    latent_partial, _ = optimize_latent_partial(model, partial_pts, sdfs, config['latent_size'], mean_latent=mean_latent, latent_init=latent_codes, top_k=top_k_reg, 
                                                       iters=5000, lr=1e-4, lambda_reg=1e-3, clamp_val=2.0, latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.8, 
                                                       batch_inference_size=32768, multi_stage=False, device=device)
    # Phase 2 - Refinement - emphasis on local SDF samples and surface consistency to refine target specimen shape
    latent_partial, _ = optimize_latent_partial(model, partial_pts, sdfs, config['latent_size'], latent_init=latent_partial, top_k=top_k_reg, 
                                                        iters=8000, lr=1.3e-5, lambda_reg=7e-5, clamp_val=None, latent_std=latent_std, scheduler_step=800, scheduler_gamma=0.7, 
                                                        batch_inference_size=32768, multi_stage=True, device=device) # True because second stage using already initialized latent
    print("\nTranslated novel mesh into latent space!\n")
    
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