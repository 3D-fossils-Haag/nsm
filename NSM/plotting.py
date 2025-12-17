# Helpers for plotting PC's
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from matplotlib.colors import is_color_like
import matplotlib.patches as mpatches
import colorsys
from NSM.helper_funcs import get_region

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
                             'bipes', 'diplometopon', 'pseudopus'],  # square
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
        'color': (0.60, 0.50, 0.46)},  # slate dirty carrot
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

def plot_grouped_by_lifehistory(ax, trajs, markers_list, life_history_info, species_colors, grid=grid,
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
        color = life_history_info.get(marker, {}).get('color') if life_history_info else None
        if color is None:
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
        trajs, species_list, markers_list = compute_interpolated_trajs(normalized_species_groups, interp_series, grid, transform_pc1)
        # if group_by_life_hist True we will draw group averages later; still optionally show faint per-species
        if group_by_life_hist:
            for i, s in enumerate(species_list):
                y = trajs[i]
                if np.all(np.isnan(y)):
                    continue
                color = life_history_info.get(markers_list[i], {}).get('color') if life_history_info else None
                if color is None:
                    color = (0.60, 0.50, 0.46)
                ax.plot(grid, y, '-', alpha=0.15, color=color)
        plot_overall_avg_std(ax, trajs, grid)

    # 3) grouped life_history plotting (averages, peaks, std)
    if group_by_life_hist:
        if interp_series is None or grid is None:
            raise ValueError("group_by_life_hist=True requires interp_series and grid.")
        trajs, species_list, markers_list = compute_interpolated_trajs(normalized_species_groups, interp_series, grid, transform_pc1)
        plot_grouped_by_lifehistory(
            ax, trajs, markers_list, grid,
            life_history_info, species_colors,
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