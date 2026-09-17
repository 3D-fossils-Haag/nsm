# Helpers for plotting PC's
import colorsys
import gc
import json
import os
import re
import traceback
from collections import defaultdict
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.multivariate.manova import MANOVA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import is_color_like
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import open3d as o3d
from NSM.helper_funcs import get_region, pv_to_o3d, render_cameras
from NSM.morphometrics import tps_fit, tps_apply, pc_shape

# Dictionary for species mapping to family, family-specific attributes, and colors
family_info = {
    'Iguania': {
        'species_keywords': ['chameleo', 'iguana', 'agamidae', 'anolidae', 'corytophanidae', 
                             'crotaphytidae', 'hoplocercidae', 'leiocephalidae', 'leiosauridae', 
                             'phrynosomatidae', 'tropiduridae'],  # Iguania family
        'family_name': 'Iguania',
        'color': (0.78, 0.16, 0.16)},  # auburn
    'Anguimorpha': {
        'species_keywords': ['anguidae', 'lanthonotus', 'varanus', 'shinosaurus', 'heloderma'],  # Anguimorpha family
        'family_name': 'Anguimorpha',
        'color': (1.00, 0.79, 0.20)},  # saffron
    'Cordylidae': {
        'species_keywords': ['scincus', 'scincidae'],  # Scincidae family
        'family_name': 'Cordylidae',
        'color': (0.10, 0.51, 0.40)},  # dark teal
    'Cordylidae_Ouroborus': {
        'species_keywords': ['ouroborus'],  # Cordylidae_Ouroborus family
        'family_name': 'Cordylidae_Ouroborus',
        'color': (0.082, 0.76, 0.92)},  # turquoise - blue
    'Gekkota': {
        'species_keywords': ['gecko', 'tarentola', 'eublepharis', 'aristelliger', 'phyllurus', 'lialis'],  # Gekkota family
        'family_name': 'Gekkota',
        'color': (0.41, 0.227, 0.6)},  # dark lilac
    'Gymnophthalmoidea': {
        'species_keywords': ['gymnopthalmidae', 'teiidae'],  # Gymnophthalmoidea family
        'family_name': 'Gymnophthalmoidea',
        'color': (0.73, 0.14, 0.5)},  # dark hot pink
    'Gerrhosauridae': {
        'species_keywords': ['gerrhosaurus', 'gerrho'],  # Gerrhosauridae family
        'family_name': 'Gerrhosauridae',
        'color': (0.98, 0.39, 0.14)},  # orange
    'Scincidae': {
        'species_keywords': ['scincus', 'scincidae'],  # Scincidae family
        'family_name': 'Scincidae',
        'color': (0.65, 0.69, 0.12)},  # apple green
    'Amphisbaenea': {
        'species_keywords': ['bipes', 'rhineura', 'dibamus'],  # Amphisbaenea family
        'family_name': 'Amphisbaenea',
        'color': (0.60, 0.50, 0.46)},  # warm slate
    'Lacertidae': {
        'species_keywords': ['lacertidae', 'lacerta'],  # Lacertidae family
        'family_name': 'Lacertidae',
        'color': (0.145, 0.39, 0.075)},  # forest green
    'Snake': {
        'species_keywords': ['eryx', 'homalopsis', 'aniolios'],  # Snake family
        'family_name': 'Snake',
        'color': (0.60, 0.50, 0.46)},  # slate dirty carrot
    'Xantusiidae': {
        'species_keywords': ['xantusiidae'],  # Xantusiidae family
        'family_name': 'Xantusiidae',
        'color': (0.88, 0.74, 0.59)}  # emoji white
}

def get_family(species_label):
    species_label = species_label.lower()
    for family, info in family_info.items():
        # Check if any keyword from the family matches the species label
        if any(keyword in species_label for keyword in info['species_keywords']):
            return family, info['color']
    # If no match is found, return a default family (e.g., 'Unknown') with a default color
    return 'Unknown', (0.52, 0.52, 0.52)  # Grey for unknown family

# Add a gradient by species within family
def make_species_cmap(family_info, species_groups, max_shift=0.4):
    species_colors = {}
    family_species_map = defaultdict(list)
    # Group species by family
    for species in species_groups:
        family = get_family(species)
        family_species_map[family].append(species)
    for family, species_list in family_species_map.items():
        base_rgb = family_info.get(family, {}).get('color', np.array([0.7, 0.7, 0.7]))  # fallback: gray
        base_hls = colorsys.rgb_to_hls(*base_rgb)
        sorted_species = sorted(species_list)
        n = len(sorted_species)
        center_idx = n // 2
        for i, sp in enumerate(sorted_species):
            if i == center_idx:
                # Middle species gets base color
                new_rgb = base_rgb
            else:
                # Shift lightness slightly (lighter or darker)
                shift_direction = -1 if i < center_idx else 1
                shift_amount = (abs(i - center_idx) / (n - 1)) * max_shift
                new_lightness = np.clip(base_hls[1] + shift_direction * shift_amount, 0, 1)
                new_rgb = colorsys.hls_to_rgb(base_hls[0], new_lightness, base_hls[2])
            species_colors[sp] = tuple(np.clip(new_rgb, 0, 1)) + (1.0,)  # Add alpha channel
    return species_colors

# Function to generate the legend for family colors
def plot_family_cmap(family_info):
    # Create a list of patches and labels for the legend
    patches = []
    labels = []
    for family, info in family_info.items():
        color = info['color']
        patch = mpatches.Patch(color=color, label=family)
        patches.append(patch)
        labels.append(family)
    # Create the legend
    plt.legend(handles=patches, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Family Colors")
    plt.axis('off')  # Turn off the axis since we only want the legend
    plt.show()

def plot_species_cmap(species_colors):
    plt.figure(figsize=(12, len(species_colors) * 0.25))
    for i, (species, color) in enumerate(species_colors.items()):
        plt.fill_between([0, 10], i, i + 1, color=color)
        text_color = 'black' if np.mean(color[:3]) > 0.6 else 'white'  # Choose text color based on background lightness
        plt.text(0.5, i + 0.5, species, va='center', fontsize=8,
                 color='black' if np.mean(color[:3]) > 0.6 else 'white')
    plt.ylim(0, len(species_colors))
    plt.axis('off')
    plt.title("Species Colors", fontsize=14)
    plt.tight_layout()
    plt.show()

# Dictionary for species mapping to life history, plotting symbols and colors
life_history_info = {
    'v': {
        'species_keywords': ['ouroborus'], # triangle (down)
        'life_history': 'Bites tail to front',
        'color': (0.32, 0.24, 0.56)},  # dark lilac
    'P': {
        'species_keywords': ['chalcides', 'tetradactylus', 'chamaesaura'], # plus filled
        'life_history': 'Grass swimmer',
        'color': (0.65, 0.69, 0.12)},  # pea soup
    '+': {
        'species_keywords': ['skoog', 'eremiascincus', '_scincus'], # plus regular
        'life_history': 'Sand swimmer',
        'color': (0.84, 0.65, 0.23)},  # dark mustard
    's': {
        'species_keywords': ['acontias', 'mochlus', 'rhineura', 'dibamus', 'lanthonotus', 
                             'bipes', 'diplometopon', 'pseudopus', 'amphisbaen', "bachia", "polychrous"],  # square
        'life_history': 'Burrowers',
        'color': (0.72, 0.44, 0.22)},  # dirty carrot
    'd': {
        'species_keywords': ['jonesi', 'corucia', 'gecko', 'chamaeleo', 'iguana', 'brookesia', 
                             'dracaena', 'anolis', 'basiliscus', 'dracaena', 'aristelliger', 
                             'sceloporus', 'lialis', 'phyllurus', 'polychrous'],  # thin diamond
        'life_history': 'Arboreal',
        'color': (0.36, 0.557, 0.68)},  # grey sky blue
    'X': {
        'species_keywords': ['elgaria', 'smaug_giganteus', 'broadleysaurus', 'ateuchosaurus', 
                             'alopoglossus', 'heloderma', 'tupinambis', 'carlia', 'lipinia', 
                             'tiliqua', 'tribolonotus', 'leiolepis', 'eublepharis', 'oreosaurus', 
                             'baranus', 'callopistes', 'cricosaura', 'lepidophyma', 'sphenodon', 
                             'lacerta', 'enyaloides', 'crocodilurus', 'varanus', 'egernia', 
                             'tropidurus', 'phrynosoma', 'leiosaurus', 'leiocephalus', 'gallotia'], # x filled
        'life_history': 'Terrestrial',
        'color': (0.10, 0.51, 0.40)},  # dark seafoam
    '2': {
        'species_keywords': ['eryx', 'homalopsis', 'aniolios'], # antibody/upside down y
        'life_history': 'Snake',
        'color': (0.25, 0.22, 0.2)},  # slate dirty carrot
    'o': {
        'species_keywords': [],  # circle (default) # Saxicolous/rock dwelling is the default
        'life_history': 'Saxicolous',
        'color': (0.60, 0.50, 0.46)}}  # slate (default)

# Function to get life history marker and color based on species
def get_life_history_marker(species, show_life_history_dict=False):
    species = species.lower().replace('_', ' ')  # Normalize species name: lowercase and replace underscores with spaces
    matched = False # Default state
    for marker, info in life_history_info.items():
        for keyword in info['species_keywords']:
            if keyword.lower() in species:  # Partial match, case-insensitive
                matched = True
                return marker, info['color']
    #if not matched:
        #print(f"Species name '{species}' not found in life history dictionary. Run again with show_life_history_dict=True to debug.")
    # If no match is found, print the dictionary if the option is set to True
    if show_life_history_dict:
        print("life_history_info dictionary:")
        print(life_history_info)
    # Return default 'o' if no match found
    return 'o', life_history_info['o']['color']

# Function to plot the legend for life history strategies
def plot_life_history_legend(life_history_legend, title='Symbol Key for Species Life History Strategies', outfpath=None):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    # Plot dummy points for the legend
    for i, (marker, label) in enumerate(life_history_legend):
        ax.plot([], [], marker=marker, linestyle='None', markersize=10, label=label, color='black')
    # Customize and display the legend
    ax.legend(loc='center left', frameon=False)
    ax.axis('off')
    plt.title(title)
    plt.tight_layout()
    # Save the plot if an output file path is provided
    if outfpath:
        plt.savefig(outfpath, dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()

def calculate_region_percentages(species_groups):
    region_percentages = defaultdict(list)
    for species, vertebrae in species_groups.items():
        total_vertebrae = len(vertebrae)
        # Initialize counts for each region
        cervical_count = 0
        thoracic_count = 0
        lumbar_count = 0
        # Find the region for each vertebra
        for vertebra_label, _ in vertebrae:
            region = get_region(vertebra_label)
            if region == 'Cervical':
                cervical_count += 1
            elif region == 'Thoracic':
                thoracic_count += 1
            elif region == 'Lumbar':
                lumbar_count += 1
        # Calculate the normalized percentages
        cervical_percentage = (cervical_count / total_vertebrae) * 100
        thoracic_percentage = (thoracic_count / total_vertebrae) * 100
        lumbar_percentage = (lumbar_count / total_vertebrae) * 100
        # Store the percentages for each species
        region_percentages[species] = {
            'cervical_count': cervical_count,
            'thoracic_count': thoracic_count,
            'lumbar_count': lumbar_count,
            'cervical_percentage': cervical_percentage,
            'thoracic_percentage': thoracic_percentage,
            'lumbar_percentage': lumbar_percentage}
    return region_percentages

# Function to calculate average percentages across species
def calculate_average_percentages(region_percentages):
    # Initialize sums for each region
    total_cervical = 0
    total_thoracic = 0
    total_lumbar = 0
    # Number of species
    num_species = len(region_percentages)
    # Sum the percentages for each region
    for counts in region_percentages.values():
        total_cervical += counts['cervical_percentage']
        total_thoracic += counts['thoracic_percentage']
        total_lumbar += counts['lumbar_percentage']
    # Calculate average percentages
    avg_cervical = total_cervical / num_species
    avg_thoracic = total_thoracic / num_species
    avg_lumbar = total_lumbar / num_species
    avg_total_vert = (total_cervical + total_thoracic + total_lumbar) / num_species
    return avg_cervical, avg_thoracic, avg_lumbar, avg_total_vert

def _resolve_color(species, marker, life_history_info, species_colors, ax):
    key = marker[0]
    color = None
    if life_history_info and key in life_history_info:
        color = life_history_info[key].get("color")
    if color is None and species_colors and species in species_colors:
        color = species_colors[species]
    if not (color is not None and is_color_like(color)):
        try:
            color = next(ax._get_lines.prop_cycler)["color"]
        except Exception:
            color = "C0"
    return color

def _savgol(x, window=21, poly=3):
    n = len(x)
    wl = min(window, n if n % 2 == 1 else n - 1)
    try:
        return savgol_filter(x, wl, poly)
    except Exception:
        return x

# Interpolate species' data over the grid
grid = np.linspace(0, 1, 100)

# Define the interpolation function
def interp_series(df, grid):
    f = interp1d(df["_std_pos"], df["PC1"], kind="linear", fill_value="extrapolate")
    return f(grid)

def compute_interpolated_trajs(normalized_species_groups, grid=grid, interp_series=interp_series, transform_pc1=None):
    """Return (trajs_array, species_list, markers_list). trajs_array shape = (n_species, len(grid))."""
    grid = np.asarray(grid)
    rows = []
    species_list = []
    markers = []
    for species, points in normalized_species_groups.items():
        df = pd.DataFrame(points, columns=["species", "vertebra_label", "_std_pos", "PC1", "marker"])
        vals = np.asarray(interp_series(df, grid), dtype=float)
        if vals.shape[0] != grid.shape[0]:
            raise ValueError(f"interp_series for {species} returned {vals.shape[0]} but grid length is {grid.shape[0]}")
        if transform_pc1 is not None:
            vals = np.asarray(transform_pc1(list(vals)), dtype=float)
        rows.append(vals)
        species_list.append(species)
        m = df["marker"].iloc[0] if not df.empty else None
        markers.append(m[0])
    if not rows:
        return np.empty((0, grid.shape[0])), species_list, markers
    return np.vstack(rows), species_list, markers

def plot_raw_species(ax, normalized_species_groups, pca, PC_idx, transform_pc1, life_history_info, species_colors, dim_alpha):
    for species, points in normalized_species_groups.items():
        pts_sorted = sorted(points, key=lambda x: x[2])
        _, _, x_vals, y_vals, markers = zip(*pts_sorted)
        y_vals = list(y_vals)
        if transform_pc1 is not None:
            y_vals = transform_pc1(y_vals)
        marker = markers[0]
        color = _resolve_color(species, marker, life_history_info, species_colors, ax)
        ax.plot(x_vals, y_vals, '-', alpha=dim_alpha, color=color)

def plot_overall_avg_std(ax, trajs, grid, avg_color='black', fill_alpha=0.2):
    if trajs.size == 0:
        avg = np.full(len(grid), np.nan)
        std = np.zeros(len(grid))
    else:
        avg = np.nanmean(trajs, axis=0)
        std = np.nanstd(trajs, axis=0)
    ax.plot(grid, avg, color=avg_color, linewidth=2, label='Average Trajectory')
    ax.fill_between(grid, avg - std, avg + std, color=avg_color, alpha=fill_alpha, label='±1 SD')

def plot_grouped_by_lifehistory(ax, trajs, markers_list, life_history_info=None, species_colors=None, grid=grid,
                                peaks_and_valleys=False, show_region_boundaries=False,
                                avg_cervical=None, avg_thoracic=None, plt_std=False):
    groups = defaultdict(list)
    for m, row in zip(markers_list, trajs):
        groups[m].append(row)
    for marker, rows in groups.items():
        arr = np.vstack(rows) if rows else np.empty((0, grid.size))
        if arr.size == 0:
            continue
        avg_y = np.nanmean(arr, axis=0)

        if life_history_info is not None:
            color = life_history_info.get(marker, {}).get('color')
        else:
            color = (0.60, 0.50, 0.46)
        if peaks_and_valleys:
            safe_avg = np.nan_to_num(avg_y, nan=0.0, posinf=0.0, neginf=0.0)
            smoothed = _savgol(safe_avg, window=min(21, len(safe_avg)), poly=3)
            peak_idx = int(np.nanargmax(smoothed)) if smoothed.size else None
            valley_idx = int(np.nanargmin(smoothed)) if smoothed.size else None
            ax.plot(grid, avg_y, color=color, linewidth=2, label=f"Original {marker}")
            ax.plot(grid, smoothed, color=color, linestyle='--', linewidth=1, alpha=0.7, label=f"Smoothed {marker}")
            if peak_idx is not None:
                ax.scatter(grid[peak_idx], avg_y[peak_idx], s=120, color=color, edgecolor='black', zorder=5, marker='o', alpha=0.8)
            if valley_idx is not None:
                ax.scatter(grid[valley_idx], avg_y[valley_idx], s=120, color=color, edgecolor='black', zorder=5, marker='o', alpha=0.8)
            if show_region_boundaries and avg_cervical is not None and avg_thoracic is not None:
                ax.axvline(x=avg_cervical * 0.01, color='gray', linestyle=':', linewidth=1)
                ax.axvline(x=(avg_cervical + avg_thoracic) * 0.01, color='gray', linestyle=':', linewidth=1)
        else:
            if plt_std and arr.shape[0] > 1:
                std_y = np.nanstd(arr, axis=0)
                ax.fill_between(grid, avg_y - std_y, avg_y + std_y, color=color, alpha=0.1)
            ax.plot(grid, avg_y, color=color, linewidth=2, label=f'{marker} Average')

def plot_species_groups(normalized_species_groups, pca, PC_idx=0, life_history_info=None, 
                        species_colors=None, figsize=(10,6), save=True, out_prefix=None, 
                        suffix="", transform_pc1=None, dpi=300, show_legend=False,
                        plt_avg_std=False, interp_series=interp_series, grid=grid, show=True,
                        group_by_life_hist=False, peaks_and_valleys=False, 
                        show_region_boundaries=False, avg_thoracic=None, avg_cervical=None, 
                        plt_std=False):
    fig, ax = plt.subplots(figsize=figsize)

    # 1) raw per-species lines (dim if avg requested)
    dim_alpha = 0.2 if plt_avg_std else 0.7
    if not group_by_life_hist:
        plot_raw_species(ax, normalized_species_groups, pca, PC_idx, transform_pc1, life_history_info, species_colors, dim_alpha)

    # 2) overall average ±1SD
    if plt_avg_std:
        if interp_series is None or grid is None:
            raise ValueError("plt_avg_std=True requires interp_series and grid.")
        grid = np.asarray(grid)
        trajs, species_list, markers_list = compute_interpolated_trajs(normalized_species_groups, grid, interp_series, transform_pc1)
        if group_by_life_hist:
            for i, s in enumerate(species_list):
                y = trajs[i]
                if np.all(np.isnan(y)):
                    continue
                if life_history_info: 
                    color = life_history_info.get(markers_list[i], {}).get('color')
                else:
                    color = (0.60, 0.50, 0.46)
                ax.plot(grid, y, '-', alpha=0.15, color=color)
        plot_overall_avg_std(ax, trajs, grid)

    # 3) grouped life_history plotting (averages, peaks, std)
    if group_by_life_hist:
        if interp_series is None or grid is None:
            raise ValueError("group_by_life_hist=True requires interp_series and grid.")
        trajs, species_list, markers_list = compute_interpolated_trajs(normalized_species_groups, grid, interp_series, transform_pc1)
        plot_grouped_by_lifehistory(
            ax, trajs, markers_list, grid=grid,
            life_history_info=life_history_info, species_colors=species_colors,
            peaks_and_valleys=peaks_and_valleys,
            show_region_boundaries=show_region_boundaries,
            avg_thoracic=avg_thoracic, avg_cervical=avg_cervical,
            plt_std=plt_std)

    # finalize
    ax.set_xlabel("Normalized Vertebra Number (%)")
    ax.set_ylabel(f"PC{PC_idx+1}: {(pca.explained_variance_ratio_[PC_idx]) * 100:.2f}%")
    ax.set_title(f"PC{PC_idx+1} vs Normalized Vertebra Number {suffix}".strip())
    if show_legend:
        handles, labels = _build_legend_handles(life_history_info)
        if handles:
            ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    if save:
        prefix = out_prefix or os.path.split(os.getcwd())[1]
        outfpath = f"{prefix}_pca_pc{PC_idx+1}_vs_normalized_vertebra{suffix}.png"
        fig.savefig(outfpath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


# helper legend (unchanged)
def _build_legend_handles(life_history_info):
    handles = []
    labels = []
    if life_history_info is None:
        return handles, labels
    for marker, info in life_history_info.items():
        h = plt.Line2D([0], [0], marker=marker, linestyle="None",
                       markerfacecolor=info['color'], markeredgecolor=info['color'],
                       markersize=10)
        handles.append(h)
        labels.append(info['life_history'])
    return handles, labels

# Plot closest matches
def plot_predictions(dim_reduced_coords, similar_ids, similar_coords, novel_coord, filepaths, outfpath, out_fn):
        if "tsne" in out_fn:
            plot_type = "TSNE"
        else:
            plot_type = "PCA"
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
        plt.title(f"Latent Space Visualization {plot_type}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outfpath + "/" + out_fn, dpi=300)
        plt.close()

def match(species_name, sdf):
    s = species_name.lower().strip().replace(' ', '_')
    key = '_'.join(s.split('_')[:3])
    hit = sdf[sdf.index.str.contains(key, regex=False)]
    if hit.empty:
        parts = s.split('_')
        genus = parts[1] if len(parts) > 1 else parts[0]
        hit = sdf[sdf.index.str.contains(f'_{genus}_', regex=False)]
    return hit.iloc[0] if not hit.empty else None

def get_marker(species_name, sdf):
    row = match(species_name)
    return row['marker'] if row is not None else 'o'

def get_color(species_name, sdf):
    row = match(species_name, sdf)
    return row['color'] if row is not None else (0.5, 0.5, 0.5)

def get_trait(species_name, sdf):
    row = match(species_name)
    return row['trait'] if row is not None else None

# Function to plot the legend for life history strategies
def plot_life_history_legend(legend_items, title='Symbol Key for Species Life History Strategies', outfpath=None):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    # Plot dummy points for the legend
    for i, (marker, label) in enumerate(legend_items):
        ax.plot([], [], marker=marker, linestyle='None', markersize=10, label=label, color='black')
    # Customize and display the legend
    ax.legend(loc='center left', frameon=False)
    ax.axis('off')
    plt.title(title)
    plt.tight_layout()
    # Save the plot if an output file path is provided
    if outfpath:
        plt.savefig(outfpath, dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()

# Function to generate the legend for family colors
def plot_family_color_legend(family_colors):
    # Create a list of patches and labels for the legend
    patches = []
    labels = []
    for family, color in family_colors.items():
        patch = mpatches.Patch(color=color, label=family)
        patches.append(patch)
        labels.append(family)
    # Create the legend
    plt.figure(figsize=(8, 6))
    plt.legend(handles=patches, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Family Colors")
    plt.axis('off')  # Turn off the axis since we only want the legend
    plt.show()
    
# Define a sort key that orders vertebrae by region (C < T < L) then by the numeric part.
def sort_key(item):
    region_order = {'C': 0, 'T': 1, 'L': 2}
    v = item[0]
    return (region_order.get(v[0], 99), int(v[1:]))

def generate_species_cmap_gradient(family_base_colors, species_groups, sdf, max_shift=0.4):
    species_colors = {}
    family_species_map = defaultdict(list)
    for species in species_groups:
        row = match(species)
        family = row['broad_taxon_for_plotting'] if row is not None else 'unknown'
        family_species_map[family].append(species)
    for family, species_list in family_species_map.items():
        base_rgb = family_base_colors.get(family, np.array([0.7, 0.7, 0.7]))
        base_hls = colorsys.rgb_to_hls(*base_rgb)
        sorted_species = sorted(species_list)
        n = len(sorted_species)
        center_idx = n // 2
        for i, sp in enumerate(sorted_species):
            if i == center_idx:
                new_rgb = base_rgb
            else:
                shift_direction = -1 if i < center_idx else 1
                shift_amount = (abs(i - center_idx) / (n - 1)) * max_shift
                new_lightness = np.clip(base_hls[1] + shift_direction * shift_amount, 0, 1)
                new_rgb = colorsys.hls_to_rgb(base_hls[0], new_lightness, base_hls[2])
            species_colors[sp] = tuple(np.clip(new_rgb, 0, 1)) + (1.0,)
    return species_colors

def trait_to_label(trait):
    if pd.isna(trait):
        return 'SNAKE'
    overrides = {'grass-swimmer': 'GRASS SWIMMER', 'burrowing': 'BURROWER'}
    return overrides.get(trait, trait.upper())

# Convert matplotlib style colors to plotly
def plotly_color(c):
    if isinstance(c, tuple) and len(c) in (3, 4):
        r, g, b = [int(255 * v) for v in c[:3]]
        return f'rgb({r},{g},{b})'
    return c

# Load landmarks file (.mrk.json)
def load_mrk_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    markups = data.get("markups", [])
    if not markups:
        raise ValueError(f"No 'markups' found in {path}")
    control_points = markups[0].get("controlPoints", [])
    points = []
    labels = []
    for cp in control_points:
        pos = cp.get("position")
        if pos is None:
            continue
        points.append(pos)
        labels.append(cp.get("label"))
    points = np.asarray(points, dtype=np.float32)
    return points, labels
 
 
def bg_colors(n_pcs, base_col, max_tint_col):
    """One background colour per PC row, base → max tint."""
    return np.linspace(base_col, max_tint_col, n_pcs)

def view_rotation(rot_deg):
    """z-rotation matrix, degrees about the vertical axis."""
    return o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.deg2rad(rot_deg)])

def make_renderers(width, height, n=4):
    return [o3d.visualization.rendering.OffscreenRenderer(width, height) for _ in range(n)]

def make_material():
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader     = "defaultLit"
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    return mat

def warp_mesh(mesh, ref_lms, target_lms, mag=1.0):
    """geomorph::warpRefMesh -- deform a surface with the TPS fitted to the landmarks."""
    tgt = ref_lms + mag * (np.asarray(target_lms) - ref_lms)
    tps = tps_fit(ref_lms, tgt)
    warped = mesh.copy()
    warped.points = tps_apply(tps, np.asarray(mesh.points))
    return warped, tps
 
def crop_top_right(combined, width, height):
    return combined[:height, width:]
 
def build_warp_grid(pca, mean_lms, atlas_mesh, out_dir, label, renderers, rot_matrix,
                    width, height, n_pcs=4, n_steps=4, bg_cols=None):
    """Render each cell and save as individual PNG — no in-memory assembly."""
    os.makedirs(out_dir, exist_ok=True)
    mat = make_material()
    if bg_cols is None:
        bg_cols = bg_colors(n_pcs)
 
    for pc_idx in range(n_pcs):
        pc_dir   = os.path.join(out_dir, f"pc{pc_idx + 1}")
        os.makedirs(pc_dir, exist_ok=True)
 
        observed = pca["x"][:, pc_idx]
        scores   = np.linspace(observed.max(), observed.min(), n_steps)
        bg_color = bg_cols[pc_idx]
 
        for r in renderers:
            r.scene.set_background(list(bg_color) + [1.0])
 
        for step_idx, score in enumerate(scores):
            img     = np.full((height, width, 3),
                              (bg_color * 255).astype(np.uint8), dtype=np.uint8)
            tps_obj = None
            try:
                target             = pc_shape(pca, pc_idx, score=score)
                warped_pv, tps_obj = warp_mesh(atlas_mesh, mean_lms, target)
                del target, tps_obj;  tps_obj = None
 
                pv_clean = warped_pv.extract_surface(algorithm='dataset_surface').triangulate()
                del warped_pv
                pv_clean = pv_clean.compute_normals(cell_normals=False, point_normals=True,
                                                    inplace=False, auto_orient_normals=True)
                o3d_mesh = pv_to_o3d(pv_clean)
                del pv_clean
                o3d_mesh.compute_vertex_normals()
                o3d_mesh.rotate(rot_matrix, center=o3d_mesh.get_center())
 
                combined = render_cameras(renderers, o3d_mesh, step_idx,
                                          mat, n_steps, n_rotations=1)
                del o3d_mesh
                img = crop_top_right(combined, width, height)
                del combined
 
            except Exception as e:
                print(f"  Error {label} PC{pc_idx+1} step {step_idx+1}: {e}")
                import traceback; traceback.print_exc()
            finally:
                if tps_obj is not None:
                    del tps_obj
                gc.collect()
 
            fname = os.path.join(pc_dir, f"step{step_idx+1:02d}_of_{n_steps}.png")
            cv2.imwrite(fname, img)
            del img
            print(f"  {label}  PC{pc_idx+1}  {step_idx+1}/{n_steps}  score={score:.3f}  ✓",
                  flush=True)
            gc.collect()
 
    print(f"Done — PNGs saved to {out_dir}")

def stitch_grid(out_dir, label, grid_path=None, flip_pcs=(), n_pcs=4, n_steps=4):
    out_dir  = Path(out_dir)
    flip_pcs = set(flip_pcs)
    rows = []
    for pc_idx in range(n_pcs):
        pc_dir = out_dir / f"pc{pc_idx + 1}"
        files  = sorted(pc_dir.glob("step*.png"))
        if len(files) != n_steps:
            print(f"  Warning: PC{pc_idx+1} has {len(files)} files, expected {n_steps}")
        if pc_idx + 1 in flip_pcs:
            imgs = [cv2.imread(str(f)) for f in reversed(files)]
        else:
            imgs = [cv2.imread(str(f)) for f in files]
        rows.append(np.hstack(imgs))
 
    grid      = np.vstack(rows)
    grid_path = Path(grid_path) if grid_path else out_dir.parent / f"pc_grid_{label}.png"
    ok = cv2.imwrite(str(grid_path), grid)
    if not ok:
        raise OSError(f"cv2.imwrite failed for {grid_path.resolve()}")
    print(f"Grid saved → {grid_path.resolve()}  ({grid.shape[1]}×{grid.shape[0]} px)")
    return grid_path
 
def load_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ── Spearman correlations between PC blocks ──────────────────────────────────
_pc_num = lambda c: int(re.search(r"\d+$", c).group())

def load_pc_block(csv_path, prefix, id_cols):
    df = pd.read_csv(csv_path)
    return df[list(id_cols) + sorted([c for c in df.columns if c.startswith(prefix)], key=_pc_num)]

def pc_block(df, prefix):
    cols = sorted([c for c in df.columns if c.startswith(prefix)], key=_pc_num)
    return cols, df[cols].to_numpy()

def spearman_matrix(A, B):
    """r and Bonferroni-corrected p for every column pair."""
    R = np.empty((A.shape[1], B.shape[1]))
    P = np.empty_like(R)
    for i, j in product(range(A.shape[1]), range(B.shape[1])):
        R[i, j], P[i, j] = spearmanr(A[:, i], B[:, j])
    return R, np.clip(P * R.size, 0, 1)

def plot_spearman_heatmaps(results, out_path=None, alpha=0.05,
                           label_size=35, tick_size=30, cell_size=30, cb_size=30,
                           figsize=(28, 8), dpi=300, cmap_name="PuBuGn",
                           cell_fmt="{:+.2f}", tick_rotation=45):
    """results: list of (label_a, label_b, R, P). Colour encodes |r|."""
    plt.rcParams.update({"font.weight": "normal", "axes.labelweight": "normal",
                         "font.size": label_size})
    fig, axes = plt.subplots(1, len(results), figsize=figsize)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.15, wspace=0.4)
    cax = fig.add_axes([0.905, 0.15, 0.018, 0.77])

    for ax, (lbl_a, lbl_b, R, P) in zip(axes, results):
        n_a, n_b = R.shape
        im = ax.imshow(abs(R), vmin=0, vmax=1, cmap=plt.get_cmap(cmap_name).reversed(), aspect="auto")
        ax.set_yticks(range(n_a), [f"PC{i+1}" for i in range(n_a)], fontsize=tick_size)
        ax.set_xticks(range(n_b), [f"PC{j+1}" for j in range(n_b)],
                      rotation=tick_rotation, ha="right", fontsize=tick_size)
        ax.set_ylabel(lbl_a, fontsize=label_size, labelpad=10)
        ax.set_xlabel(lbl_b, fontsize=label_size, labelpad=10)
        for i, j in product(range(n_a), range(n_b)):
            ax.text(j, i, cell_fmt.format(R[i, j]), ha="center", va="center", fontsize=cell_size,
                    color="black", fontweight="bold" if P[i, j] < alpha else "normal")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|SPEARMAN'S R|", fontsize=cb_size, labelpad=12)
    cb.ax.tick_params(labelsize=cb_size)
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig

def spearman_tables(lbl_a, lbl_b, R, P, out_dir=None, alpha=0.05):
    rows = [f"{lbl_a} PC{i+1}" for i in range(R.shape[0])]
    cols = [f"{lbl_b} PC{j+1}" for j in range(R.shape[1])]
    df_r = pd.DataFrame(R.round(3), index=rows, columns=cols)
    df_p = pd.DataFrame(P.round(4), index=rows, columns=cols)
    for row in rows:
        best = df_r.loc[row].abs().idxmax()
        flag = "  *" if df_p.loc[row, best] < alpha else ""
        print(f"    {row:24s}  →  {best:26s}  r = {df_r.loc[row, best]:+.3f}{flag}")
    if out_dir:
        slug = f"{lbl_a}_vs_{lbl_b}".replace(" ", "_")
        df_r.to_csv(Path(out_dir) / f"spearman_r_{slug}.csv")
        df_p.to_csv(Path(out_dir) / f"spearman_p_{slug}.csv")
    return df_r, df_p

# LDA plotting funcs

def n_pcs_for_variance(cum, threshold):
    """Number of PCs needed to reach `threshold` cumulative variance."""
    if threshold >= 1.0:
        return len(cum)
    return int(np.searchsorted(cum, threshold) + 1)
 
def fmt_p(val, significant=False):
    """Format a p-value in compact scientific notation, starred if significant."""
    if pd.isna(val):
        return "NaN"
    if val == 0:
        s = "< 1e-300"
    else:
        exp = int(np.floor(np.log10(abs(val))))
        s = f"{val / 10**exp:.2f}e{exp:+03d}"
    return s + ("*" if significant else "")
 
def manova_stats(X_lda, ys):
    """Overall MANOVA on the LD scores. Returns Wilks' lambda, F, p, and partial eta-squared."""
    df = pd.DataFrame(X_lda, columns=[f"LD{i+1}" for i in range(X_lda.shape[1])])
    df["group"] = ys
    formula = " + ".join(c for c in df.columns if c != "group")
    stat = MANOVA.from_formula(f"{formula} ~ group", data=df).mv_test().results["group"]["stat"]
    wilks = stat.loc["Wilks' lambda"]
    lam = float(wilks["Value"])
    return {"wilks_lambda": lam,
            "manova_F": float(wilks["F Value"]),
            "manova_num_df": float(wilks["Num DF"]),
            "manova_den_df": float(wilks["Den DF"]),
            "manova_p": float(wilks["Pr > F"]),
            "partial_eta_sq": 1 - lam ** (1 / min(X_lda.shape[1], len(set(ys)) - 1))}
 
def lda_threshold_grid(reps_by_threshold, labels, group_colors, thresholds, col_order, col_titles,
                       row_labels=None, min_n=10, use_shrinkage=True, show_accuracy=False,
                       width=2100, height=2100, font_size=45, marker_size=12,
                       legend_marker_size=60, outstem=None, out_dir=None, show=True):
    """LDA scatter grid: rows = variance thresholds, cols = representations.
 
    out_dir : directory for the .html/.png written when `outstem` is given. Required
              with `outstem` (the notebook used to supply this via a global OUT_DIR).
    show    : call fig.show() before returning.
    """
    if outstem and out_dir is None:
        raise ValueError("out_dir is required when outstem is given")
 
    labels = pd.Series(labels).reset_index(drop=True)
    fig = make_subplots(rows=len(thresholds), cols=len(col_order),
                        horizontal_spacing=0.055, vertical_spacing=0.11)
 
    # Row selection depends only on `labels`, so it is identical for every panel:
    # drop NaNs, then drop classes with fewer than min_n specimens.
    rows = labels.notna().values.copy()
    kept = labels[rows].reset_index(drop=True)
    rows[rows] = ~kept.isin(kept.value_counts()[lambda v: v < min_n].index)
    ys = labels[rows].values
    levels = sorted(set(ys))
 
    clf = LDA(solver="eigen", shrinkage="auto") if use_shrinkage else LDA(solver="svd")
    if show_accuracy:
        cv = StratifiedKFold(n_splits=min(5, pd.Series(ys).value_counts().min()),
                             shuffle=True, random_state=42)
 
    subtitles = {}
    for r, thr in enumerate(thresholds, start=1):
        for c, rep in enumerate(col_order, start=1):
            Xs = np.asarray(reps_by_threshold[(thr, rep)])[rows]
            X_lda = clf.fit_transform(Xs, ys)
 
            subtitles[(r, c)] = f"{Xs.shape[1]} PCs"
            if show_accuracy:
                acc = cross_val_score(clf, Xs, ys, cv=cv,
                                      scoring=make_scorer(balanced_accuracy_score)).mean()
                subtitles[(r, c)] += f"<br>ACC = {acc:.3f}"
 
            for lev in levels:
                idx = ys == lev
                fig.add_trace(go.Scatter(
                    x=X_lda[idx, 0], y=X_lda[idx, 1], mode="markers",
                    name=str(lev).upper(), legendgroup=str(lev), showlegend=False,
                    marker=dict(color=plotly_color(group_colors.get(lev, (.5, .5, .5))),
                                size=marker_size, symbol="circle"),
                    hovertemplate=f"{lev}<extra></extra>"), row=r, col=c)
 
    fig.update_layout(width=width, height=height, plot_bgcolor="white",
                      font=dict(size=font_size),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05,
                                  yanchor="top", font=dict(size=font_size)),
                      margin=dict(t=170, b=200, l=170, r=60))
    axis_style = dict(showline=True, linewidth=2, linecolor="black", mirror=True,
                      ticks="outside", showticklabels=False, showgrid=False)
    fig.update_xaxes(title_text="LD1", **axis_style)
    fig.update_yaxes(title_text="LD2", **axis_style)
 
    def ann(x, y, text, **kw):
        return dict(x=x, y=y, text=text, xref="paper", yref="paper", showarrow=False, **kw)
 
    anns = [ann(sum(fig.get_subplot(1, c).xaxis.domain) / 2, 1.1, ct,
                xanchor="center", font=dict(size=font_size * 1.15))
            for c, ct in enumerate(col_titles, start=1)]
    anns += [ann(-0.075, sum(fig.get_subplot(r, 1).yaxis.domain) / 2, rl,
                 textangle=-90, yanchor="middle", font=dict(size=font_size * 1.15))
             for r, rl in enumerate(row_labels or [f"{100*t:.0f}% VAR" for t in thresholds],
                                    start=1)]
    anns += [ann(sum(fig.get_subplot(r, c).xaxis.domain) / 2,
                 fig.get_subplot(r, c).yaxis.domain[1] + 0.004, st,
                 xanchor="center", yanchor="bottom", font=dict(size=font_size * 0.9))
             for (r, c), st in subtitles.items()]
    fig.update_layout(annotations=anns)
 
    for lev in levels:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 name=str(lev).upper(), legendgroup=str(lev), showlegend=True,
                                 marker=dict(color=plotly_color(group_colors.get(lev, (.5, .5, .5))),
                                             size=legend_marker_size, symbol="circle")))
    if outstem:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_dir / f"{outstem}.html"), include_plotlyjs="cdn")
        fig.write_image(str(out_dir / f"{outstem}.png"))
    if show:
        fig.show()
    return fig
