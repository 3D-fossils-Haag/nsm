# Utility functions for loading trained models and inspecting results

import os, json, torch, numpy as np, open3d as o3d, pyvista as pv, vtk
from NSM.models import TriplanarDecoder
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import re
import pymskt.mesh.meshes as meshes
import torch.nn.functional as F

# ICP transform
class NumpyTransform:
    def __init__(self, matrix):
        self.matrix = matrix
    def GetMatrix(self):
        vtk_mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_mat.SetElement(i, j, self.matrix[i, j])
        return vtk_mat

# Pyvista to Open3D    
def pv_to_o3d(mesh_pv):
    pts = np.asarray(mesh_pv.points)
    faces = np.asarray(mesh_pv.faces)
    tris = faces.reshape(-1,4)[:,1:4]
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(pts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tris)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

# Convert ply file to vtk
def convert_ply_to_vtk(input_file, output_file=None, save=False):
    if not input_file.lower().endswith('.ply'):
        raise ValueError("Input file must have a .ply extension.")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".vtk"
    mesh = pv.read(input_file)
    if save==True:
        mesh.save(output_file)
    print(f"Converted {input_file} → {output_file}")
    return mesh, output_file

# Load model config file
def load_config(config_path='model_params_config.json'):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\033[92mLoaded config from {config_path}\033[0m")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: model_params_config.json not found at {config_path}")

# Load model and latents
def load_model_and_latents(MODEL_PATH, LC_PATH, config, device):
    # Load model
    triplane_args = {
        'latent_dim': config['latent_size'],
        'n_objects': config['objects_per_decoder'],
        'conv_hidden_dims': config['conv_hidden_dims'],
        'conv_deep_image_size': config['conv_deep_image_size'],
        'conv_norm': config['conv_norm'], 
        'conv_norm_type': config['conv_norm_type'],
        'conv_start_with_mlp': config['conv_start_with_mlp'],
        'sdf_latent_size': config['sdf_latent_size'],
        'sdf_hidden_dims': config['sdf_hidden_dims'],
        'sdf_weight_norm': config['weight_norm'],
        'sdf_final_activation': config['final_activation'],
        'sdf_activation': config['activation'],
        'sdf_dropout_prob': config['dropout_prob'],
        'sum_sdf_features': config['sum_conv_output_features'],
        'conv_pred_sdf': config['conv_pred_sdf'],
    }
    model = TriplanarDecoder(**triplane_args)
    model_ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(model_ckpt['model'])
    model.to(device)
    model.eval()
    # Load latents
    latent_ckpt = torch.load(LC_PATH, map_location=device)
    latent_codes = latent_ckpt['latent_codes']['weight'].detach().cpu()
    return model, latent_ckpt, latent_codes

# begin monkey patch
def safe_load_mesh_scalars(self):
    try:
        if hasattr(self, 'mesh'):
            mesh = self.mesh
        elif hasattr(self, '_mesh'):
            mesh = self._mesh
        else:
            raise AttributeError("No mesh attribute found in Mesh object.")
        point_scalars = mesh.point_data
        cell_scalars = mesh.cell_data
        if point_scalars and len(point_scalars.keys()) > 0:
            self.mesh_scalar_names = list(point_scalars.keys())
            self.scalar_name = self.mesh_scalar_names[0]
        elif cell_scalars and len(cell_scalars.keys()) > 0:
            self.mesh_scalar_names = list(cell_scalars.keys())
            self.scalar_name = self.mesh_scalar_names[0]
        else:
            self.mesh_scalar_names = []
            self.scalar_name = None
            print("No scalar data found in mesh. Proceeding without scalars.")
    except Exception as e:
        print(f"Failed to load mesh scalars: {e}")
        self.mesh_scalar_names = []
        self.scalar_name = None

def fixed_point_coords(self):
    if self.n_points < 1:
        raise AttributeError(f"No points found in mesh '{self}'")
    return self.points

def get_sdfs(decoder, samples, latent_vector, batch_size=32**3, objects=1, device='cuda'):
    n_pts_total = samples.shape[0]
    current_idx = 0
    sdf_values = torch.zeros(samples.shape[0], objects, device=device) 
    if batch_size > n_pts_total:
        #print('WARNING: batch_size is greater than the number of samples, setting batch_size to the number of samples')
        batch_size = n_pts_total
    while current_idx < n_pts_total:
        current_batch_size = min(batch_size, n_pts_total - current_idx)
        sampled_pts = samples[current_idx : current_idx + current_batch_size, :3].to(device)
        sdf_values[current_idx : current_idx + current_batch_size, :] = decode_sdf(
            decoder, latent_vector, sampled_pts, device) 
        current_idx += current_batch_size
        #print(f"Processed {current_idx} / {n_pts_total} points")
    return sdf_values

def decode_sdf(decoder, latent_vector, queries, device='cuda'):
    num_samples = queries.shape[0]
    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1).to(device)
        inputs = torch.cat([latent_repeat, queries], dim=1)
    inputs = inputs.to(next(decoder.parameters()).device)  
    return decoder(inputs)
# end monkey patch

# Get species name using regex from filenames (ex: Scincidae_Tribolonotus_novaeguineae)
def extract_species_prefix(filename):
    match = re.match(r"([A-Za-z]+_[A-Za-z]+_[a-z]+)", filename.lower())
    if match:
        return match.group(1)
    else:
        return None