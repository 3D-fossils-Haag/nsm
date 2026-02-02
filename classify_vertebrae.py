# Identify novel meshes from latent space
import os
import torch
import numpy as np
import pandas as pd
from NSM.datasets import SDFSamples
from NSM.models import TriplanarDecoder
from NSM.mesh import get_sdfs  
import torch.nn.functional as F
import json
import pyvista as pv
import pymskt.mesh.meshes as meshes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from NSM.mesh import create_mesh
import vtk
import re
import random
from pathlib import Path
from NSM.helper_funcs import NumpyTransform, load_config, load_model_and_latents, convert_ply_to_vtk, get_sdfs, fixed_point_coords, safe_load_mesh_scalars, extract_species_prefix, parse_labels_from_filepaths
from NSM.optimization import pca_initialize_latent, get_top_k_pcs, find_similar, find_similar_cos, optimize_latent
# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

# Define PC index and model checkpoint to use for analysis of novel mdeshes
TRAIN_DIR = "run_v56" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '3000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'

# Load config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")
train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# List of meshes to be classified
# Randomly select test paths
mesh_list = random.sample(config['val_paths'], 100) # TO DO: Choose val or test paths

# Define predictions

def plot_predictions(dim_reduced_coords, similar_coords, novel_coord, filepaths, out_fn):
        plt.figure(figsize=(8, 6))
        plt.scatter(dim_reduced_coords[:, 0], dim_reduced_coords[:, 1], color='gray', alpha=0.3, label='Training Meshes')
        # Plot most similar (1st one) in pink
        plt.scatter(similar_coords[0, 0], similar_coords[0, 1], color='hotpink', s=80, label='Most Similar')
        # Plot next 4 similar in blue
        if len(similar_coords) > 1:
            plt.scatter(similar_coords[1:, 0], similar_coords[1:, 1], color='blue', s=60, label='Other Top-5 Similar')
        # Plot novel mesh in red
        plt.scatter(*novel_coord, color='red', s=80, label='Novel Mesh')
        # Aannotate each of the top-5 similar meshes
        for idx, (x, y) in zip(similar_ids, similar_coords):
            plt.text(x, y, filepaths[idx].split('.')[0], fontsize=6, color='black')
        plt.title("Latent Space Visualization (PCA)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outfpath + "/" + out_fn, dpi=300)
        plt.close()

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)
_, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.95)

# Loop through meshes
summary_log = []
for i, vert_fname in enumerate(mesh_list): 
    print(f"\033[32m\n=== Processing {os.path.basename(vert_fname)} ===\033[0m")
    print(f"\033[32m\n=== Mesh {i} / {len(mesh_list)} ===\033[0m")
    # Make a new dir to save predictions
    outfpath = 'classify_vertebrae/predictions/' + os.path.splitext(os.path.basename(vert_fname))[0]
    print("Making a new directory to save model predictions and outputs at: ", outfpath)
    os.makedirs(outfpath, exist_ok=True)

    # --- Set up inference dataset ---

    # Convert plys to vtks
    if '.ply' in vert_fname:
        ply_fname = vert_fname
        _, vert_fname = convert_ply_to_vtk(ply_fname)

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
        fix_mesh=config['fix_mesh']
        )

    # Get the point/SDF data
    print("Setting up dataset")
    sdf_sample = sdf_dataset[0]  # returns a dict
    sample_dict, _ = sdf_sample
    points = sample_dict['xyz'].to(device) # shape: [N, 3]
    sdf_vals = sample_dict['gt_sdf']  # shape: [N, 1]

    # Optimize latents (DeepSDF has no encoder, so must use optimization to encode novel data)
    print("Optimizing latents")
    latent_novel = optimize_latent(model, points, sdf_vals, config['latent_size'], top_k_reg, mean_latent, latent_codes)
    print("Translated novel mesh into latent space!")

    # --- Classify vertebra ---

    # Find most similar latents (Compare to existing latents)
    similar_ids, distances = find_similar_cos(latent_novel, latent_codes, top_k=5, n_std=2, device=device)

    # Write most similar meshes to txt file
    sim_mesh_fpath = outfpath + '/' + 'similar_meshes_pca_regularized_95pct_cos.txt'
    with open(sim_mesh_fpath, "w") as f:
        print(f"Most similar mesh indices to file: {os.path.basename(vert_fname)}\n")
        f.write(f"Most similar mesh indices to file: {os.path.basename(vert_fname)}:\n")
        for i, d in zip(similar_ids, distances):
            # Now construct the line using the integer i
            line = f"Name: {all_vtk_files[i]}, Index: {i}, Distance: {d:.4f}"
            print(line)
            f.write(line + "\n")

    # --- Inspect novel latent using clustering analysis ---

    # PCA Plot
    # Data loading
    latents = latent_codes.cpu().numpy()
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(latents)
    novel_coord = pca.transform(latent_novel.cpu().numpy())[0]
    similar_coords = coords_2d[similar_ids]
    plot_predictions(coords_2d, similar_coords, novel_coord, all_vtk_files, out_fn="latent_space_pca_pca_regularized_95pct_cos.png")

    # t-SNE Plot
    # Data loading
    latent_novel_np = latent_novel.detach().cpu().numpy()
    latents_with_novel = np.vstack([latents, latent_novel_np])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    coords_with_novel = tsne.fit_transform(latents_with_novel)
    train_coords = coords_with_novel[:-1]
    novel_coord = coords_with_novel[-1]
    similar_coords = train_coords[similar_ids]
    plot_predictions(train_coords, similar_coords, novel_coord, all_vtk_files, "latent_space_tsne_pca_regularized_95pct_cos.png")

    # --- Reconstruct optimized latent into mesh to confirm it looks normal ---
    
    # Reconstruction parameters
    recon_grid_origin = 1.0
    n_pts_per_axis = 256
    voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
    voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
    offset = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    icp_transform = NumpyTransform(np.eye(4))
    objects = 1

    # Reconstruct the novel mesh 
    mesh_out = create_mesh(
        decoder=model,
        latent_vector=latent_novel,
        n_pts_per_axis=n_pts_per_axis,
        voxel_origin=voxel_origin,
        voxel_size=voxel_size,
        path_original_mesh=None,
        offset=offset,
        scale=scale,
        icp_transform=icp_transform,
        objects=objects,
        verbose=True,
        device=device,
        )

    # Ensure it's PyVista PolyData
    if isinstance(mesh_out, list):
        mesh_out = mesh_out[0]
    if not isinstance(mesh_out, pv.PolyData):
        mesh_pv = mesh_out.extract_geometry()
    else:
        mesh_pv = mesh_out

    # Save mesh
    output_path = outfpath + "/" + os.path.splitext(os.path.basename(vert_fname))[0] + "_decoded_novel_pca_regularized_95pct_cos.vtk"
    mesh_pv.save(output_path)
    print(f"Novel mesh saved to: {output_path}")

    # Save results to summary log
    # Get species prefix
    labels, _ = parse_labels_from_filepaths([os.path.basename(vert_fname)])
    gt_species, gt_position = labels[0]

    # Check top-1 match
    labels, _ = parse_labels_from_filepaths([all_vtk_files[similar_ids[0]]])
    top1_species_pred, top1_position_pred = labels[0]
    top1_species_match = "yes" if gt_species and gt_species == top1_species_pred else "no"
    top1_region_match = "yes" if gt_position and gt_position[0] == top1_position_pred[0] else "no"
    if top1_region_match == "yes":
        top1_position_error = abs(int(gt_position[1:]) - int(top1_position_pred[1:]))
    else:
        top1_position_error = "NA_region_mismatch"

    # Check top-5 matches
    labels, _ = parse_labels_from_filepaths([all_vtk_files[i] for i in similar_ids])
    top5_species_pred  = [s for s, _ in labels]
    top5_position_pred = [v for _, v in labels]
    top5_species_match = "yes" if (gt_species is not None and gt_species in top5_species_pred) else "no"
    top5_region_match = "yes" if (gt_position is not None and any(pred[0].lower() == gt_position[0].lower() for pred in top5_position_pred)) else "no"
    position_errors = []
    if top5_region_match == "yes":
        for pred in top5_position_pred:
            if pred[0].lower() == gt_position[0].lower():
                position_errors.append(abs(int(pred[1:]) - int(gt_position[1:])))
    top5_position_error = min(position_errors) if position_errors else "NA_region_mismatch"

    # Prepare summary log with top-5
    top_k_summary = {
    "mesh": os.path.basename(vert_fname),
    "output_mesh": output_path,
    "top1_species_match": top1_species_match,
    "top5_species_match": top5_species_match,
    "top1_region_match": top1_region_match,
    "top1_position_error": top1_position_error,
    "top5_region_match": top5_region_match,
    "top5_position_error": top5_position_error,}
    # Add top-5 similar mesh names and distances
    for rank, (i, dist) in enumerate(zip(similar_ids, distances), 1):
        top_k_summary[f"similar_{rank}_name"] = all_vtk_files[i]
        top_k_summary[f"similar_{rank}_distance"] = dist
    summary_log.append(top_k_summary)

# Export results to summary log
df = pd.DataFrame(summary_log)
df.to_csv("summary_matches_95pct_cos.csv", index=False)
print("Summary saved to summary_matches_95pct_cos.csv")