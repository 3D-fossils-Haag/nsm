# Helper functions for latent space exploration videos and figures

import numpy as np

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