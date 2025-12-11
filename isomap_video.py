import os, json, torch, numpy as np, cv2, open3d as o3d, pyvista as pv, vtk, gc
from NSM.mesh import create_mesh
from NSM.models import TriplanarDecoder
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.manifold import Isomap
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import re
from scipy.signal import savgol_filter
from NSM.helper_funcs import NumpyTransform, pv_to_o3d, load_config, load_model_and_latents
from NSM.traverse_latents import sample_latent_grid, solve_tsp_nearest_neighbor, interpolate_latent_loop, project_to_isomap, resample_by_cumulative_distance, plot_latent_paths

# Define model parameters to use for video generation
TRAIN_DIR = "run_v44" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
CKPT = '2000' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'
USE_AVERAGES = True # TO DO: Use region averages or individual vertebrae?
vertebral_regions = ['C', 'T', 'L'] # TO DO: Define vertebral regions to inspect

# Load config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")
train_paths = config['list_mesh_paths']
all_vtk_files = [os.path.basename(f) for f in train_paths]

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)

# Define vertebral regions
latent_codes_subs = []
all_vtk_files_subs = []
# Match "_C1" to "_C40" or "-C1" to "-C40"
for vert_region in vertebral_regions:
    r_p = r'[_-]' + vert_region + r'([1-9]|[1-3][0-9]|40)(?!\d)'
    pattern = re.compile(r_p, re.IGNORECASE)
    # Subset indices from all paths
    matches = [
                (i, int(pattern.search(fname).group(1)))
                for i, fname in enumerate(all_vtk_files)
                if pattern.search(fname)
                ]
    indices = [i for i, _ in matches]
    #cervical_nums = [num for _, num in matches]

    # Filter latent codes and corresponding mesh paths
    vert_region_codes = latent_codes[indices]
    vert_region_files = [all_vtk_files[i] for i in indices]
    print(f"\n\nFound {len(vert_region_files)} latent codes for region: {vert_region}")
    print(f"Sample files: {vert_region_files[:5]}\n\n")

    # To average across vertebrae regions
    # Regex to extract specimen ID from filename by removing "_C1" or "-C1", etc.
    if USE_AVERAGES == True:
        r_p_specimen = r'^(.*?)(?:[-_]\d+[-_]' + vert_region + r'\d+)(?:.*)$'
        specimen_pattern = re.compile(r_p_specimen, re.IGNORECASE)
        specimen_latents = {}
        specimen_files = {}
        for fname, latent in zip(vert_region_files, vert_region_codes):
            match = specimen_pattern.match(fname)
            if match:
                specimen_id = match.group(1)
                if specimen_id not in specimen_latents:
                    specimen_latents[specimen_id] = []
                    specimen_files[specimen_id] = []
                specimen_latents[specimen_id].append(latent.numpy())
                specimen_files[specimen_id].append(fname)
            else:
                print(f"\033[93mWarning: could not extract specimen ID from {fname}\033[0m")

        # Average the latent codes per specimen
        avg_latent_codes = []
        avg_specimen_ids = []
        for specimen_id, latents in specimen_latents.items():
            avg_latent = np.mean(latents, axis=0)
            avg_latent_codes.append(avg_latent)
            avg_specimen_ids.append(specimen_id + '_' + vert_region)
        # Convert to NumPy array
        avg_latent_codes = np.array(avg_latent_codes)
        print(f"\nAveraged latent codes for {len(avg_specimen_ids)} specimens.\nSample specimen IDs: {avg_specimen_ids[:5]}")
        vert_region_codes = avg_latent_codes
        vert_region_files = avg_specimen_ids
    # Add to dictionary
    latent_codes_subs.extend(vert_region_codes)
    all_vtk_files_subs.extend(vert_region_files)

latent_codes_tensor = torch.stack([torch.tensor(latent) for latent in latent_codes_subs])
latent_codes_subs = latent_codes_tensor
print(f"Running analysis for : {len(all_vtk_files_subs)} averaged latent codes for vertebral regions {vertebral_regions}.")

# Mesh creation params
recon_grid_origin = 1.0
n_pts_per_axis = 384
voxel_origin = (-recon_grid_origin, -recon_grid_origin, -recon_grid_origin)
voxel_size = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
offset = np.array([0.0, 0.0, 0.0])
scale = 1.0
icp_transform = NumpyTransform(np.eye(4))
objects = 1

# Isomap: project latent codes to 2D manifold
latents_np = latent_codes_subs.numpy() if torch.is_tensor(latent_codes_subs) else latent_codes_subs
isomap = Isomap(n_neighbors=20, n_components=2)
isomap_2d = isomap.fit_transform(latents_np)  # Shape: (N, 2)

projections = isomap_2d[:, 0]
min_proj = isomap_2d[:, 0].min()
max_proj = isomap_2d[:, 0].max()

# Path params
n_rotations = 3
n_frames = 120 * n_rotations # TO DO: Adjust the number of frames to resample by

# 5. k-d Tree and Sampling
tree = cKDTree(isomap_2d)
NUM_GRIDS_X = 8
NUM_GRIDS_Y = 8
sampled_points = sample_latent_grid(isomap_2d, NUM_GRIDS_X, NUM_GRIDS_Y)
indices = tree.query(sampled_points, k=3)
weights = 1 / (indices[0] + 1e-5)  # Inverse distance weighting (avoid division by 0)
weights /= weights.sum(axis=1)[:, None]  # Normalize the weights

# Step 6: Interpolate between the nearest latent codes
latent_interp = []
for i, sampled_point in enumerate(sampled_points):
        neighbors = indices[1][i]
        row = (latents_np[neighbors] * weights[i][:, None]).sum(axis=0)
        latent_interp.append(row)
latent_interp = np.array(latent_interp)

# Generate a pairwise distance matrix of latent codes
dist_matrix = cdist(latent_interp, latent_interp, metric='cosine')

# Use travelling salesman to determine nearest neighbor and reorder latent_interp for smooth transitions
tsp_path = solve_tsp_nearest_neighbor(dist_matrix)
latent_interp_ordered = latent_interp[tsp_path]
steps_per_segment = 100 # TO DO: adjust steps per segment
dense_interp = interpolate_latent_loop(latent_interp_ordered, steps_per_segment=steps_per_segment)
smooth_latent_loop = resample_by_cumulative_distance(dense_interp, n_frames=n_frames)
# Apply temporal smoothing filter over latent trajectory
# window_length must be odd and <= length of array
window_length = min(31, len(smooth_latent_loop) - 1 if len(smooth_latent_loop) % 2 == 0 else len(smooth_latent_loop))
smooth_latent_loop = savgol_filter(smooth_latent_loop, window_length=window_length, polyorder=3, axis=0)
print("\n\n\nLength: ", len(smooth_latent_loop))

# Project interpolated path points back into isomap for plotting
loop_2d, _ = project_to_isomap(latent_interp, latents_np, isomap_2d)
tsp_2d, _ = project_to_isomap(latent_interp_ordered, latents_np, isomap_2d)
smooth_loop_2d, loop_idx = project_to_isomap(smooth_latent_loop, latents_np, isomap_2d)

# Get closest specimen labels for smooth latent loop
closest_specimens = [all_vtk_files_subs[i] for i in loop_idx]

# Plot path in latent space
plot_latent_paths(isomap_2d, loop_2d, tsp_2d, smooth_loop_2d, sampled_points, vertebral_regions, USE_AVERAGES)

# Setup Offscreen Renderers (4)
width, height = 640, 480
renderers = [o3d.visualization.rendering.OffscreenRenderer(width, height) for _ in range(4)]
for r in renderers:
    r.scene.set_background([0.0, 0.0, 0.0, 1.0])
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"
material.base_color = [1.0, 1.0, 1.0, 1.0]

# Video Writer (2x2 grid → 1280x960)
video_path = "isomap" + "_4way_splitscreen_" + "C-T-L"
if USE_AVERAGES == True:
    video_path = video_path + "_avg" + ".mp4"
else:
    video_path = video_path + ".mp4"
fps = 15
out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height * 2))

generated_mesh_count = 0
loop_sequence = np.concatenate([smooth_latent_loop, smooth_latent_loop[::-1][1:]], axis=0)
loop_sequence_names = np.concatenate([closest_specimens, closest_specimens[::-1][1:]])
loop_sequence = smooth_latent_loop
loop_sequence_names = closest_specimens
for i, latent_code in enumerate(loop_sequence):
    try:
        generated_mesh_count += 1
        print(f"\033[92m\nGenerating mesh {generated_mesh_count}/{len(loop_sequence)}\033[0m")
        print(f"Frame {i}: Closest to {loop_sequence_names[i]}")
        new_latent = torch.tensor(latent_code, dtype=torch.float32).unsqueeze(0).to(device)

        mesh_out = create_mesh(
            decoder=model, latent_vector=new_latent, n_pts_per_axis=n_pts_per_axis,
            voxel_origin=voxel_origin, voxel_size=voxel_size, path_original_mesh=None,
            offset=offset, scale=scale, icp_transform=icp_transform,
            objects=objects, verbose=False, device=device
        )
        mesh_out = mesh_out[0] if isinstance(mesh_out, list) else mesh_out
        mesh_pv = mesh_out if isinstance(mesh_out, pv.PolyData) else mesh_out.extract_geometry()
        mesh_pv = mesh_pv.compute_normals(cell_normals=False, point_normals=True, inplace=False)
        mesh_o3d = pv_to_o3d(mesh_pv)

        for r in renderers:
            r.scene.clear_geometry()
            r.scene.add_geometry("mesh", mesh_o3d, material)

        # Camera setup
        pts = np.asarray(mesh_o3d.vertices)
        center = pts.mean(axis=0)
        r = np.linalg.norm(pts - center, axis=1).max()
        distance = 2.5 * r
        elevation = np.deg2rad(30)

        # Define 4 camera positions
        angle_deg = (i /  (len(loop_sequence) - 1)) * 360 * n_rotations
        angle_rad = np.deg2rad(angle_deg)
        cam_positions = [
            center + np.array([  # Top Left: rotating
                distance * np.cos(angle_rad) * np.cos(elevation),
                distance * np.sin(angle_rad) * np.cos(elevation),
                distance * np.sin(elevation)
            ]),
            center + np.array([0, -distance, 0]),  # Top Right: side
            center + np.array([distance, 0, 0]),  # Bottom Left: back (90° CCW from side)
            center + np.array([0, 0, distance])    # Bottom Right: top-down (90° CCW from side)
        ]
        ups = [
            [0, 0, 1],  # rotating
            [1, 0, 0],  # front
            [0, 0, 1],  # side
            [0, 1, 0],  # top-down
        ]

        for idx, (rdr, pos, up) in enumerate(zip(renderers, cam_positions, ups)):
            rdr.setup_camera(60, center, pos, up)

        # Render images
        imgs = [np.asarray(r.render_to_image()) for r in renderers]
        imgs_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]

        # Compose 4 views into 2x2 grid (width=640, height=480)
        top = np.hstack([imgs_bgr[0], imgs_bgr[1]])
        bottom = np.hstack([imgs_bgr[2], imgs_bgr[3]])
        combined = np.vstack([top, bottom])

        # Overlay specimen name info onto each frame
        specimen_name = loop_sequence_names[i]
        parts = specimen_name.split("_")
        family = parts[0] if len(parts) > 0 else specimen_name
        genus = parts[1]  if len(parts) > 1 else ""
        region = parts[-1] if len(parts) > 2 else ""
        if 'C' in region:
            reg_full = 'Cervical'
        elif 'T' in region:
            reg_full = 'Thoracic'
        elif 'L' in region:
            reg_full = 'Lumbar'
        else:
            reg_full = ''
        text = f"Closest Specimen: \n{family}\n{genus}\n{reg_full}"
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # White
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        # Position: center of the frame
        center_x = combined.shape[1] // 2
        center_y = combined.shape[0] // 2
        text_x = center_x - 120
        text_y = center_y
        # Put the text
        for j, line in enumerate(text.split("\n")):
                y = text_y + j * (text_size[1] + 10)
                cv2.putText(combined, line, (text_x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        out_video.write(combined)
        print(f"Captured frame {i + 1}/{len(loop_sequence)}")

    except Exception as e:
        print(f"Error at frame {i}: {e}")
    finally:
        for var in ['mesh_out', 'mesh_pv', 'mesh_o3d', 'new_latent']:
            if var in locals():
                del locals()[var]
        gc.collect()

out_video.release()
print("Video saved as", video_path)
