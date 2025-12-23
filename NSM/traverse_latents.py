# Helper functions for latent space exploration videos and figures
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


# Sample latent grid aross num_x and num_y values - Isomap
def sample_latent_grid(latent_2d, num_x, num_y):
    x_min, y_min = latent_2d.min(axis=0)
    x_max, y_max = latent_2d.max(axis=0)
    x_vals = np.linspace(x_min, x_max, num_x)
    y_vals = np.linspace(y_min, y_max, num_y)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    grid_samples = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return grid_samples

# Solve travelling salesman nearest neghbor path - Isomap
def solve_tsp_nearest_neighbor(dist_matrix):
    N = dist_matrix.shape[0]
    visited = np.zeros(N, dtype=bool)
    path = [0]  # Start at point 0
    visited[0] = True
    for _ in range(1, N):
        last = path[-1]
        # Mask visited nodes
        dists = dist_matrix[last]
        dists[visited] = np.inf
        next_idx = np.argmin(dists)
        path.append(next_idx)
        visited[next_idx] = True
    return path

# Interpolate latent path as loop - Isomap
def interpolate_latent_loop(latents, steps_per_segment=10):
    loop_latents = []
    for i in range(len(latents) - 1):
        start = latents[i]
        end = latents[i + 1]
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            interp = (1 - t) * start + t * end
            loop_latents.append(interp)
    return np.array(loop_latents)

def resample_by_cumulative_distance(latents, n_frames):
    diffs = np.linalg.norm(np.diff(latents, axis=0), axis=1)
    dists = np.concatenate([[0], np.cumsum(diffs)])
    dists /= dists[-1]  # Normalize to [0, 1]
    new_steps = np.linspace(0, 1, n_frames)
    new_latents = np.array([
        np.interp(new_steps, dists, latents[:, i]) for i in range(latents.shape[1])]).T
    return new_latents

def project_to_isomap(latents_query, latents_all, isomap_2d):
    tree = cKDTree(latents_all)
    _, indices = tree.query(latents_query, k=1)
    return isomap_2d[indices], indices

def plot_latent_paths(isomap_2d, loop_2d, tsp_2d, smooth_loop_2d, sampled_points, vertebral_regions, use_averages=False, save_dir="."):
    # Plot settings
    line_width = 2
    alpha = 0.7
    start_marker_size = 50
    end_marker_size = 50
    path_dict = {"loop": {"data": loop_2d, "color": "violet", "title": "Latent Interpolation Path"},
                "tsp": {"data": tsp_2d, "color": "olivedrab", "title": "TSP-Ordered Path"},
                "smooth": {"data": smooth_loop_2d, "color": "deepskyblue", "title": "Smoothed TSP Path"}}
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, (key, path_info) in zip(axs, path_dict.items()):
        path = path_info["data"]
        color = path_info["color"]
        title = path_info["title"]
        ax.set_title(title)
        ax.scatter(isomap_2d[:, 0], isomap_2d[:, 1], c='lightgray', marker='o', s=10, label='All codes')
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], marker='x', s=20, color='dimgrey', label='Sample Grid')
        # Path line
        ax.plot(path[:, 0], path[:, 1], '-', lw=line_width, color=color, alpha=alpha, label=f'Path ({key})')
        # Start and end points
        ax.scatter(*path[0], color=color, edgecolor='black', marker='o', s=start_marker_size, alpha=alpha, label='Start')
        ax.scatter(*path[-1], color=color, edgecolor='black', marker='X', s=end_marker_size, alpha=alpha, label='End')
        ax.set_aspect('equal', adjustable='box')
    # Legend on the last subplot
    axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.0, fontsize='small')
    # Title and save
    plt.suptitle("Latent Interpolation Paths in Isomap 2D", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    # Save figure
    figpath = os.path.join(save_dir, f"latent_space_path_overlay_isomap_video_{'_'.join(vertebral_regions)}") # TO DO: define file path
    if use_averages == True:
        figpath = figpath + '_avg' + '.png'
    else:
        figpath = figpath + '.png'
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\033[92mSaved latent space path overlay to {figpath}\033[0m")

def generate_latent_path_plot(projections, proj_val, min_proj, max_proj, width=200, height=80):
    print(f"projections shape: {projections.shape}")
    print(f"proj_val: {proj_val}")
    # Ensure projections is a 1D array
    projections = projections.flatten()
    # Plot latent points
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=200)
    # Set the background color to black
    fig.patch.set_facecolor('black')  # Black background for the figure
    ax.set_facecolor('black')  # Black background for the axes
    # Plot latent points
    ax.set_xlim(min_proj, max_proj)
    ax.plot([min_proj, max_proj], [0, 0], color='paleturquoise', alpha=0.2, linewidth=1)
    # Plot the current latent point (use proj_val directly and ensure it's scalar)
    ax.scatter(proj_val, 0, color='deeppink', s=10)
    # Customize the plot
    ax.set_yticks([])  # Hide y-axis ticks
    ax.set_xticks([0])
    ax.set_xticklabels(['0'])
    ax.grid(False)
    ax.legend().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Latent Path (PC{str((PC_idx+1))})", fontsize=10, color='white')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=False)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)[..., :3]  # Drop alpha if any
    return img_np