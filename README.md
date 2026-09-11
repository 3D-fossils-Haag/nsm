## Introduction
This code uses generative deep learning models to understand the skeletal anatomy of lizards and some snakes (Squamata). This code is forked and modified from [gattia/NSM](https://github.com/gattia/NSM) following the terms of the [GNU Affero GPL 3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html). See [Original NSM Documentation](http://anthonygattiphd.com/NSM/). 

![Isomap GIF](https://github.com/aubricot/nsm/blob/main/images/isomap_4way_splitscreen_C-T-L_avg.gif)
*Figure 1: Traversing an isomap of the NSM trained model latent space using travelling salesman and k-nearest neighbors. Video animation made using [isomap_video.py](https://github.com/aubricot/nsm/blob/main/isomap_video.py)*

## Installation

```bash
# Create and activate conda environment
conda create -n NSM python=3.10
conda activate NSM

# Install pytorch and dependencies
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge -c defaults

# Install NSM
mkdir NSM
cd NSM
git clone https://github.com/3D-fossils-Haag/nsm.git
cd nsm
python -m pip install -r requirements.txt
pip install -e .

```

## Usage
Please refer to the project [Wiki](https://github.com/3D-fossils-Haag/nsm/wiki) for detailed instructions on using NSM.

### Demos
Check out our demos to build NSM fully in the Google Colab runtime environment and interactively evaluate our tools with demo data in under 10 minutes, no need to connect to your Google Drive!

:arrow_right: :lizard: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/nsm/blob/main/demos/classification_demo.ipynb) Click here to try out classification of unknown fossils.


:arrow_right: :lizard: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/nsm/blob/main/demos/shape_completion_demo.ipynb) Click here to try out shape completion for partial fossils.

:arrow_right: :mirror_ball: Interactively explore our latent space and dataset on [3d-fossils-haag.github.io/vert-nsm-figs/](https://3d-fossils-haag.github.io/vert-nsm-figs/).

### Training
Update sections of [train_model.py]() commented with # TO DO: to update PROJECT_NAME, ENTITY_NAME, RUN_NAME, folder_vtk, N_TRAIN, N_TEST, N_VAL. These variables point to where your data was collected, where it is saved, and where outputs should go. Adjust model training hyperparameters in [vertebrae_config.json](). See python script and config files for details and save before running using commands below. 
```
conda activate NSM
cd NSM/nsm
python train_model.py
```

### Model Loading

NSM provides a convenient model loader that simplifies loading pre-trained Neural Shape Models. For **trained models**, you'll have:

- `experiment_dir/model_params_config.json` - Configuration saved during training
- `experiment_dir/model/2000.pth` - Model weights at epoch 2000
- `experiment_dir/latent_codes/2000.pth` - Latent codes at epoch 2000

```python
import json, torch
from NSM.models import TriplanarDecoder

# Load config file
with open(config_path, 'r') as f:
     config = json.load(f)

# Get model weights and latent codes
latent_ckpt = torch.load(LC_PATH, map_location=device)
latent_codes = latent_ckpt['latent_codes']['weight'].detach().cpu()

# Build model
triplane_args = {
    'latent_dim': config['latent_size'],
    'n_objects': config['objects_per_decoder'],
    'conv_hidden_dims': config['conv_hidden_dims'],
    'conv_deep_image_size': config['conv_deep_image_size'],
    'conv_norm': config['conv_norm'], 
    'conv_norm_type': config['conv_norm_type'],
    'conv_start_with_mlp': config['conv_start_with_mlp'],
    'sdf_latent_size': config['sdf_latent_size'],
    'sdf_hidden_dims': config['sdf_hidden_dims'],
    'sdf_weight_norm': config['weight_norm'],
    'sdf_final_activation': config['final_activation'],
    'sdf_activation': config['activation'],
    'sdf_dropout_prob': config['dropout_prob'],
    'sum_sdf_features': config['sum_conv_output_features'],
    'conv_pred_sdf': config['conv_pred_sdf'],
}
model = TriplanarDecoder(**triplane_args)
model_ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(model_ckpt['model'])
device = config.get("device", "cuda:0")
model.to(device)
model.eval()
```

### Create Meshes

After loading a trained model, you can generate meshes from manipulated/new latent vectors. The example below generates the mean mesh shape based on model training data.

```python
from NSM.mesh import create_mesh
import pyvista as pv

# Get the mean of the latent codes
latents_np = latent_codes.numpy()
latent_mean = np.mean(latents_np, axis=0)

# Convert the mean latent code to a pytorch tensor
new_latent = torch.tensor(new_latent_np, dtype=torch.float32).unsqueeze(0).to(device)

# Create a mesh from the latent tensor
mesh_out = create_mesh(
            decoder=model, latent_vector=new_latent, n_pts_per_axis=n_pts_per_axis,
            voxel_origin=voxel_origin, voxel_size=voxel_size, path_original_mesh=None,
            offset=offset, scale=scale, icp_transform=icp_transform,
            objects=objects, verbose=False, device=device
)

# Ensure mesh is PyVista Polydata (.vtk) 
if isinstance(mesh_out, list):
     mesh_out = mesh_out[0]

if not isinstance(mesh_out, pv.PolyData):
     mesh_pv = mesh_out.extract_geometry()
else:
     mesh_pv = mesh_out

# Write to file
mesh_pv.save(output_path)
```

### Classification validation metrics

[classification_eval.py](classification_eval.py) scores the latent space as a nearest neighbour classifier and writes precision/recall/F1 tables and confusion matrices for taxonomy, spinal region and spinal position. A query vertebra is given the labels of the training vertebra with the most similar latent code (cosine).

```
venv/bin/python classification_eval.py --model_root run_v44 --ckpt 3000
venv/bin/python classification_eval.py --model_root run_v44 --ckpt 3000 --eval_level genus
```

Outputs land under `<model_root>/classification/evaluation/<dataset_split>/<suffix>/`: `metrics_summary.csv`, per-class `report_<category>.csv`, `confusion_<category>.png`, plus `query_labels.csv`, `predictions.csv` and `metrics.json`.

Parameters:

| flag | default | description |
| --- | --- | --- |
| `--model_root` | required | run directory holding `model_params_config.json` and `latent_codes/`. |
| `--ckpt` | required | checkpoint epoch to load. |
| `--dataset_split` | `train` | query split; use `test --encoded_latents` for the learning-curve experiment. |
| `--encoded_latents` | off | load separately optimized validation/test latents instead of evaluating memorized train codes. |
| `--eval_level` | `specimen` | what to hide when classifying a vertebra: `loo` (only itself, optimistic upper bound), `specimen`, `species`, or `genus`. |
| `--species_list` | `lizard_species_list.csv` | per specimen sheet with `broad_taxon_for_plotting` and life history `trait`. |
| `--cm_max_classes` | `40` | skip the confusion matrix figure for categories with more classes than this. |
| `--suffix` | `<eval_level>_<timestamp>` | name of the output subfolder. |

Categories scored: `family`, `genus`, `species`, `broad_taxon`, `region` (cervical/thoracic/lumbar), `position_10` and `position_20` (normalized position in 10% and 20% bins), and `life_history`.

### Downsampled dataset-size experiments

[make_downsample_splits.py](make_downsample_splits.py) creates reproducible random and diversity-aware train/val/test split files for learning-curve experiments such as 100, 300, 500, 800 and 1000 training meshes. Validation and test contain whole specimens and are fixed across all sizes, strategies and repetitions. Training subsets are nested within each strategy/seed; the diversity ordering prioritizes specimen, family, genus, species and spinal-region coverage.

Example:

```bash
python make_downsample_splits.py \
  --mesh_dir vertebrae_meshes \
  --mapping_csv vtk_name_to_mapping_v2.csv \
  --out_dir downsample_splits \
  --sizes 100 300 500 800 1000 all \
  --strategies diverse random \
  --seeds 52122 52123 52124 52125 52126 \
  --holdout_seed 9173 \
  --ckpt 3000
```

Train one generated split:

```bash
python train_model.py \
  --run_name run_n300_diverse_seed52122 \
  --split_file downsample_splits/n300_diverse_seed52122.json
```

After training, optimize latent codes for the same held-out test meshes and evaluate them against the training gallery:

```bash
python encode_latents_for_eval.py \
  --model_root run_n300_diverse_seed52122 \
  --output_dir classification/evaluation/encoded_latents \
  --dataset_split test

python classification_eval.py \
  --model_root run_n300_diverse_seed52122 \
  --ckpt 3000 \
  --dataset_split test \
  --encoded_latents \
  --eval_level specimen

python collect_downsample_results.py \
  --manifest downsample_splits/manifest.csv \
  --out_csv downsample_splits/results_summary.csv \
  --eval_level specimen

python analyze_downsample_results.py \
  --results downsample_splits/results_summary.csv \
  --out_csv downsample_splits/learning_curve_summary.csv \
  --accuracy_margin 0.02 \
  --error_margin 0.05
```

The analysis reports bootstrap 95% confidence intervals across repetitions. A size is marked `equivalent_to_all` only when every available accuracy/F1 lower bound is within 0.02 of the all-data mean and every available error upper bound is within 5% of it. Choose these practical margins before inspecting results.

The generated `downsample_splits/manifest.csv` records each run name, split file, coverage counts, and commands for training, held-out encoding, classification, per-run encoder training, and shape-completion evaluation. Run those columns in order. Skipped rows indicate requested training sizes larger than the locally available mesh pool. Do not reuse an encoder from another dataset-size run.

### Shape completion split evaluation

[evaluate_shape_completion_split.py](evaluate_shape_completion_split.py) evaluates shape completion on a trained model's saved split (`train`, `val` or `test`). It uses the real mesh paths from `<run>/model_params_config.json`, creates a synthetic partial input by cropping SDF samples, then compares:

- encoder fast mode: one PointNet forward pass plus reconstruction
- full optimizer mode: two-stage latent optimization plus reconstruction

The script streams one row per completed mesh to `results.csv`, so it can be stopped and resumed with `--resume`.

Full optimizer test-set run for `run_v72`:

```bash
evaluate_shape_completion_split.py \
  --config run_v72/model_params_config.json \
  --model run_v72/model/2500.pth \
  --latent_codes run_v72/latent_codes/2500.pth \
  --encoder run_v72/encoder/checkpoints/encoder.pt \
  --dataset_split test \
  --out_dir run_v72/shape_completion/test_eval \
  --resume \
  --device cuda \
  --mesh_search_dirs vertebrae_meshes
```

Important runtime knobs:

| flag | full/default value | quick test value | effect |
| --- | ---: | ---: | --- |
| `--phase1_iters` | `3000` | `300` | first optimizer phase iterations |
| `--phase2_iters` | `8000` | `800` | second optimizer phase iterations |
| `--res` | `128` | `64` | marching-cubes reconstruction resolution |
| `--sdf_n_pts` | config `n_pts_per_object` | `50000` | SDF samples generated per input mesh |
| `--n_samples` | `240` | `240` | partial SDF points used by optimizer |

Quick calibration run:

```bash
evaluate_shape_completion_split.py \
  --config run_v72/model_params_config.json \
  --model run_v72/model/2500.pth \
  --latent_codes run_v72/latent_codes/2500.pth \
  --encoder run_v72/encoder/checkpoints/encoder.pt \
  --dataset_split test \
  --out_dir run_v72/shape_completion/test_eval_quick \
  --resume \
  --device cuda \
  --mesh_search_dirs vertebrae_meshes \
  --phase1_iters 300 \
  --phase2_iters 800 \
  --res 64 \
  --sdf_n_pts 50000
```

For a full test-set run, the mesh filenames from `model_params_config.json` must
exist somewhere locally. The absolute paths inside that config may point to a
different machine, so the evaluator does not require those exact directories.
Instead, it first tries the stored path, then searches the directories passed to
`--mesh_search_dirs` and remaps by basename, for example
`/old/path/teiidae_x_01.vtk` -> `vertebrae_meshes/teiidae_x_01.vtk`. Meshes
whose filenames cannot be found locally are skipped before evaluation starts.

If your complete mesh dataset lives somewhere else, pass that directory instead:

```bash
--mesh_search_dirs /path/to/full/vertebrae_meshes
```

## License

This code is forked and modified from [https://github.com/gattia/NSM](https://github.com/gattia/NSM) following the terms of the [GNU Affero GPL 3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html) and [NSM License](https://github.com/gattia/nsm/blob/main/LICENSE). See [NOTICE](https://github.com/3D-fossils-Haag/nsm/blob/main/NOTICE).

## Citation
If you use this code or the trained models in your research, please cite this repository
```
Wolcott et al. 2026. “Squamate NSM” GitHub repository. https://github.com/3D-fossils-Haag/nsm (accessed YYYY-MM-DD).
```
