# Save evenly spaced snapshots along one PC (forward direction only)
import os, json, torch, numpy as np, cv2, open3d as o3d, pyvista as pv, gc, argparse
from NSM.mesh import create_mesh
from NSM.helper_funcs import NumpyTransform, pv_to_o3d, load_config, load_model_and_latents, render_cameras, generate_and_render_mesh
from NSM.traverse_latents import generate_latent_path_plot
from pathlib import Path
import sys

# ── args ────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--pc", type=int, required=True, help="PC to traverse, 1-based")
ap.add_argument("--train-dir", default=str(Path.cwd().parent / "run_v72"), help="dir with model ckpt and latent codes")
ap.add_argument("--ckpt", default="2500")
ap.add_argument("--n-snapshots", type=int, default=4)
ap.add_argument("--amplify", type=float, default=1.5)
ap.add_argument("--view", type=str, default="side", choices=["front", "side"])
ap.add_argument("--out-dir", default="pc_snapshots/NSM/{view}/pc{N}")
args = ap.parse_args()

# ── config ──────────────────────────────────────
TRAIN_DIR = Path(args.train_dir)
os.chdir(TRAIN_DIR)
PC_idx      = args.pc - 1
CKPT        = args.ckpt
LC_PATH     = f'latent_codes/{CKPT}.pth'
MODEL_PATH  = f'model/{CKPT}.pth'
N_SNAPSHOTS = args.n_snapshots
amplify     = args.amplify
width, height = 640, 480

# z-rotation applied before the render, one angle per view
VIEW_ROT_DEG = {"side": 13.0, "front": 90.0 + 13.0}
ROT_DEG      = VIEW_ROT_DEG[args.view]

# Set background colors to interpolate between
BASE_BG_COL     = np.array([0.839, 1, 0.996])    # aquamarine
MAX_TINT_BG_COL = np.array([0, 0.58, 0.522])  # dark teal

# Evenly spaced colors between base and max tint
bg_cols = np.linspace(BASE_BG_COL, MAX_TINT_BG_COL, N_SNAPSHOTS)
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

alphas = np.linspace(high_delta, low_delta, N_SNAPSHOTS)

# ── renderer ───────────────────────────────────────
renderers = [o3d.visualization.rendering.OffscreenRenderer(width, height)
             for _ in range(4)]
for r in renderers:
    r.scene.set_background([float(c) for c in bg_col] + [1.0])

material = o3d.visualization.rendering.MaterialRecord()
material.shader       = "defaultLit"
material.base_color   = [1.0, 1.0, 1.0, 1.0]

# ── output folder ────────────────────────────────────────────────────────────
out_dir = args.out_dir.format(view=args.view, N=PC_idx+1)
os.makedirs(out_dir, exist_ok=True)

# ── render & save ────────────────────────────────────────────────────────────
for snap_num, alpha in enumerate(alphas):
    try:
        print(f"NSM traversal for {snap_num} of PC {PC_idx+1}")
        new_latent_np = center_sample + amplify * alpha * pc1
        proj_val      = np.dot(new_latent_np - latent_mean, pc1)

        mesh_o3d = generate_and_render_mesh(new_latent_np, N_SNAPSHOTS, snap_num, device, model,
                                            n_pts_per_axis, voxel_origin, voxel_size,
                                            offset, scale, icp_transform, objects,
                                            generated_mesh_count=snap_num)

        # After generating mesh_o3d, rotate around the vertical (z) axis
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.deg2rad(ROT_DEG)])
        mesh_o3d.rotate(R, center=mesh_o3d.get_center())

        # 4‑way render from video script - hacky but doesnt cost much compute-wise
        # Get full 4-way render then crop to top-right panel only
        combined = render_cameras(renderers, mesh_o3d, snap_num, material, N_SNAPSHOTS, n_rotations=1)
        top_right = combined[:height, width:]

        # Save as PNG
        out_file = os.path.join(out_dir, f"step{snap_num+1:02d}_of_{N_SNAPSHOTS}.png")
        cv2.imwrite(out_file, top_right)
        print(f"NSM  PC{PC_idx+1}  {snap_num+1}/{N_SNAPSHOTS}  "
              f"score={proj_val:.3f}  ✓", flush=True)

    except Exception as e:
        print(f"Error at snapshot {snap_num+1} (frame {snap_num}): {e}")
        import traceback; traceback.print_exc()
    finally:
        mesh_o3d = combined = top_right = new_latent_np = None
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

print(f"Done – {N_SNAPSHOTS} snapshots saved to '{out_dir}/'")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)