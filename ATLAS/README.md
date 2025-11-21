# Use ATLAS in 3D Slicer to generate data for validation of shape_completion.py using ground truth's and partial meshes
**Last edited 21 Nov 2025**

Use [ATLAS](https://github.com/agporto/ATLAS/tree/main) in [3D Slicer](https://www.slicer.org/) to generate a validataion dataset to evaluate shape_completion.py performance and optimize parameters for reconstruction. ATLAS will split meshes into n-segments and save them as .PLY's, which can then be systematically removed from ground truth meshes to quanitfy shape completion accuracy. The original ATLAS segmentation code returns a .VTP mesh with painted segments as scalars. Replace ATLAS/SEGMENTATION/SEGMENTATION.py with the script provided here to export N-segments per mesh as .ply files. After generating ground truth and partial mesh datasets, return to NSM and use create_partial_meshes.py and shape_completion_finetune.py.      

[LICENSE](https://github.com/agporto/ATLAS/blob/main/LICENSE) for original ATLAS code. 
