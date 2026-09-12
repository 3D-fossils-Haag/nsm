# Save 10 evenly spaced snapshots along PC1 (forward direction only)
import os, json, torch, numpy as np, cv2, open3d as o3d, pyvista as pv, gc
from NSM.mesh import create_mesh
from NSM.helper_funcs import NumpyTransform, pv_to_o3d, load_config, load_model_and_latents, render_cameras, generate_and_render_mesh
from NSM.traverse_latents import generate_latent_path_plot
from pathlib import Path

# ── config (same as your video script) ──────────────────────────────────────
cwd = Path.cwd()
base_wd = cwd.parent 
TRAIN_DIR = base_wd / "run_v72" # TO DO: Choose training directory containing model ckpt and latent codes
os.chdir(TRAIN_DIR)
PC_idx      = 3
CKPT        = '2500'
LC_PATH     = f'latent_codes/{CKPT}.pth'
MODEL_PATH  = f'model/{CKPT}.pth'
N_SNAPSHOTS = 10
amplify     = 1.5
width, height = 640, 480
# Set background color
BASE_BG_COL     = np.array([0.38, 1, 0.98])    # aquamarine
MAX_TINT_BG_COL = np.array([0.03, 0.11, 0.1])  # dark teal
# 5 evenly spaced colors between base and max tint
bg_cols = np.linspace(BASE_BG_COL, MAX_TINT_BG_COL, 5)  # shape (5, 3)
bg_col  = bg_cols[PC_idx]                                 # row for this PC

config  = load_config(config_path='model_params_config.json')
device  = config.get("device", "cuda:0")
model, _, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)

# ── mesh params (same as video script) ──────────────────────────────────────
recon_grid_origin = 1.0
n_pts_per_axis    = 256
voxel_origin      = (-recon_grid_origin,) * 3
voxel_size        = (recon_grid_origin * 2) / (n_pts_per_axis - 1)
offset            = np.zeros(3)
scale             = 1.0
icp_transform     = NumpyTransform(np.eye(4))
objects           = 1

# ── PCA (same as video script) ───────────────────────────────────────────────
latents_np    = latent_codes.numpy()
latent_mean   = latents_np.mean(axis=0)
centered      = latents_np - latent_mean
_, _, Vt      = np.linalg.svd(centered, full_matrices=False)
pc1           = Vt[PC_idx]
projections   = centered.dot(pc1)

center_idx    = np.argmin(np.linalg.norm(latents_np - latent_mean, axis=1))
center_sample = latents_np[center_idx]
center_proj   = np.dot(center_sample - latent_mean, pc1)

max_proj = projections.max()
min_proj = projections.min()
high_delta = max_proj - center_proj
low_delta  = min_proj - center_proj

# ── build the FORWARD alpha sweep only ──────────────────────────────────────
# Your video goes: low→0, 0→high, high→0, 0→low  (4 segments).
# The first direction is just low→high (first TWO segments concatenated).
n_seg        = 15                     # same as video (without n_rotations)
alpha_forward = np.concatenate([
    np.linspace(low_delta,  0,          n_seg),
    np.linspace(0,          high_delta, n_seg),
])
total_forward = len(alpha_forward)    # 30 frames in the forward pass

# Pick 10 evenly spaced indices from that forward sweep
snapshot_indices = np.linspace(0, total_forward - 1, N_SNAPSHOTS, dtype=int)

# ── renderer (same material as video) ───────────────────────────────────────
renderers = [o3d.visualization.rendering.OffscreenRenderer(width, height)
             for _ in range(4)]
for r in renderers:
    r.scene.set_background([float(c) for c in bg_col] + [1.0])

material = o3d.visualization.rendering.MaterialRecord()
material.shader       = "defaultLit"
material.base_color   = [1.0, 1.0, 1.0, 1.0]

# ── output folder ────────────────────────────────────────────────────────────
out_dir = f"pc{PC_idx+1}_snapshots"
os.makedirs(out_dir, exist_ok=True)

# ── render & save ────────────────────────────────────────────────────────────
for snap_num, frame_idx in enumerate(snapshot_indices):
    alpha = alpha_forward[frame_idx]
    try:
        new_latent_np = center_sample + amplify * alpha * pc1
        proj_val      = np.dot(new_latent_np - latent_mean, pc1)

        # Generate mesh (reuse your helper; total_frames / frame idx are only
        # used for progress printing inside the helper)
        mesh_o3d = generate_and_render_mesh(
            new_latent_np, total_forward, frame_idx, device, model,
            n_pts_per_axis, voxel_origin, voxel_size,
            offset, scale, icp_transform, objects,
            generated_mesh_count=snap_num
        )

        # After generating mesh_o3d, rotate 90° around the vertical (z) axis
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, (np.deg2rad(13))])
        mesh_o3d.rotate(R, center=mesh_o3d.get_center())

        # 4‑way render (same as video)
        # Get full 4-way render then crop to top-right panel only
        combined = render_cameras(
            renderers, mesh_o3d, frame_idx, material,
            total_forward, n_rotations=1
        )

        # Top-right panel: first row, second column
        top_right = combined[:height, width:]

        # Save as PNG
        out_file = os.path.join(
            out_dir,
            f"pc{PC_idx+1}_snapshot_{snap_num+1:02d}_of_{N_SNAPSHOTS}"
            f"_alpha{alpha:.3f}.png"
        )
        cv2.imwrite(out_file, top_right)
        print(f"Saved snapshot {snap_num+1}/{N_SNAPSHOTS} → {out_file}")

    except Exception as e:
        print(f"Error at snapshot {snap_num+1} (frame {frame_idx}): {e}")
        import traceback; traceback.print_exc()
    finally:
        for var in ['mesh_o3d', 'new_latent_np', 'combined', 'latent_path_img']:
            if var in locals():
                del locals()[var]
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

print(f"\nDone – {N_SNAPSHOTS} snapshots saved to '{out_dir}/'")