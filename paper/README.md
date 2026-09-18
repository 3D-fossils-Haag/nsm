## Paper analysis and figure code
Code for all figures and tables in the manuscript. Each script/notebook reads
processed data and writes outputs into a designated directory.

## Requirements

Run these analyses in an environment with the `NSM` package and its dependencies installed, as described in the repository-level `README.md`.

## At a Glance

| File | Manuscript output |
| --- | --- |
| `shape_completion_table.py` | Table 1 |
| `shape_completion_figures.py` | Figure L |
| `classification_tables.py` | Table 2; Tables S3–S5 |
| `classification_figures.py` | Figure R |
| `life_hist_grid_fig.ipynb` | Figure F |
| `PCA_tSNE_UMAP_paper_figs.ipynb` | Figure Y |
| `LDA_paper_fig.ipynb` | Figure S |
| `GMM_spearman_mesh_grid.ipynb` | Figures K and M |
| `morphol_disparity_lms_latents.ipynb` | Figures T and Su |
| `latent_interp_figs.ipynb` | Figures W and Sd |
| `conf_matr_schaema.ipynb` | Figure N schematic element |
| `classify_vertebrae_inference_fig.py` | Figure N inference panel(s) |

## Analysis Files

---
### `shape_completion_table.py`

**Produces:** Table 1

Computes mean ± SD and median [IQR] Chamfer distance for five shape-completion models across train, validation, and test splits.

**Inputs**

- `shape_completion_eval/v72_{split}_chamfer.csv`
- `shape_completion_eval/v72_encoder_{split}_chamfer.csv`
- `shape_completion_eval/v72_encoder_{split}_chamfer_refine.csv`
- `shape_completion_eval/v73h_{split}_chamfer.csv`
- `shape_completion_eval/v73c_{split}_chamfer.csv`

**Output:** `Table_2_shp_compl_eval.csv`

**Run**

    python shape_completion_table.py --datadir shape_completion_eval/ --outdir shape_completion_eval/
---

### `shape_completion_figures.py`

**Produces:** Figure L (A–C)

Generates Chamfer-distance distributions, a train/validation/test generalization-gap plot, and an accuracy–speed tradeoff plot for the Baseline, Encoder, Encoder + refinement, Hierarchy, and Contrastive models.

**Inputs:** The same Chamfer-distance CSV files used by `shape_completion_table.py`.

**Outputs**

- `fig_chamfer_boxplot.png` — Chamfer-distance distributions
- `fig_generalization_gap.png` — Generalization across dataset splits
- `fig_chamfer_tradeoff.png` — Accuracy–speed tradeoff

**Run**

    python shape_completion_figures.py --datadir shape_completion_eval/ --outdir shape_completion_eval/

---

### `classification_tables.py`

**Produces:** Table 2 and Tables S3–S5

Reads evaluation results from `classification_eval.py` and summarizes Top-1 and Top-5 classification performance for genus and normalized spinal position (20% bins) under leave-one-vertebra-out (LOO) and leave-one-specimen-out (LOSO) masking. It also produces per-class recall and sample-size correlation tables.

**Inputs**

- `<run>/classification/evaluation/<split>/<split>_<eval_level>_<latents>/predictions.csv`
- `<run>/classification/evaluation/<split>/<split>_<eval_level>_<latents>/metrics_summary.csv`
- `<run>/classification/evaluation/<split>/<split>_<eval_level>_<latents>/metrics.json`

**Outputs**

- `Table_S3_classification_stats.csv` — metrics across runs, splits, masking schemes, latent conditions, and levels
- `Table_2_classif_eval.csv` — Top-1 and Top-5 accuracy for genus and `position_20`
- `Table_S4_classif_by_spec_num.csv` — per-class recall and sample size
- `Table_S5_classif_by_spec_spearmans.csv` — correlations between performance and support

**Run**

    # Published tables
    python classification_tables.py --roots ../run_v72 ../run_v73h ../run_v73c

    # All splits and diagnostic outputs
    python classification_tables.py --roots ../run_v72 ../run_v73h ../run_v73c --outdir ./cls_out --extras

---

### `classification_figures.py`

**Produces:** Figure R (A–D)

Renders Top-5 LOSO confusion matrices for normalized spinal position, life-history strategy, broad taxonomic group, and spinal region. The default configuration uses optimized latents, the training split, specimen-level masking, and Top-5 recall on the diagonal.

**Input**

- `<run>/classification/evaluation/<split>/<split>_<eval_level>_<latents>/predictions.csv`

**Outputs**

- `fig_r_A_position_20.png`
- `fig_r_B_life_history.png`
- `fig_r_C_broad_taxon.png`
- `fig_r_D_region.png`
- `fig_r_colorbar.png`
- Optional PDF versions with `--pdf`

**Run**

    python classification_figures.py --roots ../run_v72

---

### `life_hist_grid_fig.ipynb`

**Produces:** Figure F

Renders exemplar vertebrae for six life-history strategies—arboreal, burrowing, grass-swimmer, saxicolous, terrestrial, and snake—in lateral and posterior views. Sparse landmark locations are overlaid as red spheres and renders are assembled into a 4 × 3 panel.

**Inputs**

- VTK mesh files specified in the model configuration
- Sparse landmark files in `LM_DIR` (`.mrk.json`)
- `lizard_species_list.csv`
- Model configuration and checkpoint

**Output:** `fig_f_exemplar_lifehist_vert.png` (configured as `OUT_PANEL`)

---

### `PCA_tSNE_UMAP_paper_figs.ipynb`

**Produces:** Figure Y

Computes and visualizes PCA (PC1–PC2 and PC3–PC4), t-SNE, and UMAP for sparse landmarks, dense correspondences, and NSM latent codes. Points are colored by broad taxonomic group; coordinates are also exported for downstream analyses.

**Inputs**

- VTK mesh files
- Sparse landmark files (`LM_DIR`)
- Dense correspondence files (`DENSE_LM_DIR`)
- NSM latent codes from the checkpoint in `TRAIN_DIR`
- `lizard_species_list.csv`

**Outputs**

- `{RUN}_pca_1v2_comparison.png`
- `{RUN}_pca_3v4_comparison.png`
- `{RUN}_tsne_comparison.png`
- `{RUN}_umap_comparison.png`
- Combined publication panel
- `*_points_for_stats.csv` files containing PCA, t-SNE, and UMAP coordinates

---

### `LDA_paper_fig.ipynb`

**Produces:** Figure S

Applies linear discriminant analysis to sparse landmarks, dense correspondences, and NSM latents at 90%, 95%, and 99% variance thresholds. Produces a 3 × 3 life-history visualization grid and calculates MANOVA statistics.

**Inputs**

- VTK mesh files
- Sparse-landmark, dense-correspondence, and NSM-latent representations
- `lizard_species_list.csv`

**Outputs**

- `{RUN}_lda_trait_3x3_thresholds.png`
- MANOVA summary CSV

---

### `GMM_spearman_mesh_grid.ipynb`

**Produces:** Figures K and M

Computes pairwise Spearman rank correlations between principal-component scores from sparse landmarks, dense correspondences, and NSM latents. It also generates PC1–PC4 traversal grids for landmark-based warps and NSM reconstructions, shown in front and side views.

**Inputs**

- VTK mesh files
- Sparse landmark files (`.mrk.json`)
- NSM model checkpoint and latent codes
- PC-score exports from `PCA_tSNE_UMAP_paper_figs.ipynb`
- Outputs from `save_pc_snapshots.py` and `pc_snapshot_grid.py` for NSM traversals

**Outputs**

- `spearman_pc_correlations.png`
- Spearman correlation tables (CSV)
- Landmark and NSM PC traversal PNGs
- Combined traversal panel

---

### `morphol_disparity_lms_latents.ipynb`

**Produces:** Figure T and Figure Su

Compares morphological disparity across lizard families and life-history categories using sparse landmarks and NSM latents reduced to 84 principal components. Ranks are compared between representations with dumbbell plots.

**Inputs**

- VTK mesh files
- Sparse landmark files in `LM_DIR`
- Atlas mean landmarks (`MEAN_LMS_FN`)
- NSM latent codes from the model checkpoint
- `lizard_species_list.csv`

**Outputs**

- `disparity_shift_by_trait_specimen.csv`
- `disparity_dumbbell_shift_by_trait.png` — family-level comparison
- `disparity_dumbbell_shift_by_trait_specimen.png` — life-history comparison

---

### `latent_interp_figs.ipynb`

**Produces:** Figure W and Figure Sd

Interpolates linearly between mean latent codes or PointNet-encoded mesh endpoints, decodes each interpolation step to a surface mesh, renders lateral and posterior views, and assembles the images into a figure. Absolute mesh paths can be used as endpoints to encode fossil specimens.

**Inputs**

- NSM model checkpoint (`TRAIN_DIR`, `CKPT`)
- VTK or PLY mesh files, or training-set latent-code selections

**Outputs**

- Stitched interpolation figure in the configured `OUT_DIR`
- Per-step renderings in `OUT_DIR/row{N}_{lbl_a}_to_{lbl_b}/`

**Run:** Set `PAIRS`, `TRAIN_DIR`, `CKPT`, and `N_STEPS` in the configuration cell; then run `run_render()` followed by `run_stitch()`.

---

### `conf_matr_schaema.ipynb`

**Produces:** Figure N schematic element

Creates a synthetic five-class, row-normalized confusion-matrix schematic for use in a methods or inference illustration.

**Input:** None; the synthetic matrix is generated within the notebook.

**Output:** `conf_matrix_schematic.png`

---

### `classify_vertebrae_inference_fig.py`

**Produces:** Figure N inference panel(s)

Encodes a novel vertebral mesh into NSM latent space through latent optimization, identifies the five nearest training latents using cosine similarity, and displays the query and nearest neighbors in PCA or t-SNE latent space.

**Inputs**

- Novel mesh (`.vtk` or `.ply`)
- NSM model checkpoint and training latent codes
- Model configuration JSON

**Outputs**

- `latent_space_pca_pca_regularized_95pct_cos.png`
- `latent_space_tsne_pca_regularized_95pct_cos.png`
- `similar_meshes_pca_regularized_95pct_cos.txt`

---

## Notes

- `classification_tables.py` and `classification_figures.py` require the result-tree files written by the upstream `classification_eval.py` workflow.
- Shape-completion scripts require Chamfer-distance CSV files produced by the upstream shape-completion evaluation workflow.
- Model weights and configuration files for reported models are available at https://huggingface.co/BioVisionLab/models.
- Interactive 3D renderings are available at https://3d-fossils-haag.github.io/vert-nsm-figs/.
