import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# ── args ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--train-dir", default=str(Path.cwd().parent / "run_v72"))
ap.add_argument("--n-pcs", type=int, default=4)
ap.add_argument("--n-snapshots", type=int, default=4)
ap.add_argument("--view", type=str, default="side", choices=["front", "side"])
ap.add_argument("--in-dir", default="pc_snapshots/NSM/{view}/pc{N}")
ap.add_argument("--flip_pc_ax", type=int, nargs="*", default=[1, 3], help="PCs to reverse order/flip, 1-based")
ap.add_argument("-o", "--output", default="pc1_to_pc4_snapshot_grid_{view}.png")
args = ap.parse_args()

# ── config ───────────────────────────────────────────────────────────────────
TRAIN_DIR = Path(args.train_dir)
os.chdir(TRAIN_DIR)
N_PCS       = args.n_pcs
N_SNAPSHOTS = args.n_snapshots    
OUTPUT_FILE = args.output.format(view=args.view)

# ── collect images ────────────────────────────────────────────────────────────
rows = []
for pc in range(1, N_PCS + 1):
    folder = args.in_dir.format(N=pc, view=args.view)
    # Sort so images are in order along the PC axis
    files  = sorted((f for f in os.listdir(folder) if f.endswith(".png")))
    files  = files[:N_SNAPSHOTS]
    if len(files) != N_SNAPSHOTS:
        print(f"[WARN] PC{pc}: expected {N_SNAPSHOTS} images, got {len(files)}")
    if pc in args.flip_pc_ax:
        imgs = [cv2.imread(os.path.join(folder, f)) for f in reversed(files)]
    else:
        imgs = [cv2.imread(os.path.join(folder, f)) for f in files]
    # Stitch this row horizontally
    row_img = np.hstack(imgs)
    rows.append(row_img)
    print(f"PC{pc}: stitched {len(imgs)} images → row shape {row_img.shape}")

# ── stitch rows vertically ────────────────────────────────────────────────────
grid = np.vstack(rows)
cv2.imwrite(OUTPUT_FILE, grid)
print(f"\nGrid saved → {OUTPUT_FILE}  (shape {grid.shape})")