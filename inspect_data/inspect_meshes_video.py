# Generate a video inspecting meshes — 4-panel view
# Top-left: left side | Top-right: back | Bottom-left: right side | Bottom-right: front

import os
import subprocess
import pyvista as pv
import numpy as np
from pathlib import Path

# --- Define parameters ---
cwd = Path.cwd()
base_wd = cwd.parent
MESH_DIR = base_wd / "vertebrae_meshes"
FPS = 15
OUT_VIDEO = "inspect_meshes_4panel.mp4"
DURATION_PER_MESH = 1.0  # seconds per mesh
PANEL_W, PANEL_H = 640, 480
WIDTH  = PANEL_W * 2
HEIGHT = PANEL_H * 2
TEXT_COLOR = (0, 0, 0)
FONT_SCALE = 0.7

# Camera positions: (position, focal_point, view_up)
CAMERAS = {
    "top_left_side":      [(0,  2.0, 0.2), (0, 0, 0), (0, 0, 1)],   # left side  (+Y)
    "top_right_back":     [(0,  0,  -2.0), (0, 0, 0), (0, 1, 0)],   # back       (-Z)
    "bottom_left_side":   [(0, -2.0, 0.2), (0, 0, 0), (0, 0, 1)],   # right side (-Y)
    "bottom_right_front": [(0,  0,   2.0), (0, 0, 0), (0, 1, 0)],   # front      (+Z)
}

# --- Collect all VTK files ---
mesh_files = sorted(f for f in os.listdir(MESH_DIR) if f.endswith(".vtk"))

# --- ffmpeg subprocess: reads raw RGB frames from stdin, writes H.264 mp4 ---
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-pix_fmt", "rgb24",
    "-r", str(FPS),
    "-i", "pipe:0",
    "-vcodec", "libopenh264",
    "-pix_fmt", "yuv420p",
    "-b:v", "1M",
    "-movflags", "+faststart",
    OUT_VIDEO
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# --- One offscreen plotter per panel ---
plotters = {key: pv.Plotter(off_screen=True, window_size=(PANEL_W, PANEL_H)) for key in CAMERAS}
for p in plotters.values():
    p.show(auto_close=False)
    p.background_color = "white"
    p.remove_all_lights()

def make_camera_lights():
    return [pv.Light(light_type="headlight", color='white', intensity=0.8),
            pv.Light(position=(-1, 1, 0), focal_point=(0, 0, 0), light_type="camera light", color='white', intensity=0.4),
            pv.Light(position=(0, -1, 0), focal_point=(0, 0, 0), light_type="camera light", color='white', intensity=0.2)]

def render_panel(plotter, mesh, cam_pos, label=""):
    plotter.clear()
    plotter.add_mesh(mesh, color=(0.8, 0.8, 0.8), smooth_shading=True)
    if label:
        plotter.add_text(label, position='lower_left', font_size=8, color='black')
    plotter.camera_position = cam_pos
    for light in make_camera_lights():
        plotter.add_light(light)
    plotter.render()
    frame = plotter.screenshot(return_img=True)
    frame = np.ascontiguousarray(frame[:, :, :3])
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return frame

# --- Render loop ---
for fname in mesh_files:
    mesh_path = os.path.join(MESH_DIR, fname)
    mesh = pv.read(mesh_path)
    mesh.translate(-np.array(mesh.center), inplace=True)
    mesh.compute_normals(inplace=True)

    n_frames = int(DURATION_PER_MESH * FPS)
    for _ in range(n_frames):
        panels = {}
        for key, cam in CAMERAS.items():
            # Add filename label via pyvista
            plotters[key].add_text(fname, position='lower_left', font_size=8, color='black')
            panel = render_panel(plotters[key], mesh, cam, label=fname)
            panels[key] = panel

        top    = np.hstack([panels["top_left_side"],       panels["top_right_back"]])
        bottom = np.hstack([panels["bottom_left_side"],    panels["bottom_right_front"]])
        full_frame = np.vstack([top, bottom])

        ffmpeg_proc.stdin.write(full_frame.tobytes())

    print(f"Rendered {fname}")

# --- Cleanup ---
ffmpeg_proc.stdin.close()
ffmpeg_proc.wait()
for p in plotters.values():
    p.close()
print(f"Video saved as {OUT_VIDEO}")