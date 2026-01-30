# Create partial meshes to use for shape completion validation against original/ground truth meshes
# Subtract segment PLY files generated using ATLAS from original/ground truth meshes

import os
import vtk
import json
import argparse
from glob import glob

def load_ply(path):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def load_polydata(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        reader = vtk.vtkPLYReader()
    elif ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def save_ply(polydata, path, binary=True):
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    writer.Write()

def clean_triangulate(pd):
    cl = vtk.vtkCleanPolyData()
    cl.SetInputData(pd); cl.ConvertStripsToPolysOn(); cl.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cl.GetOutputPort()); tri.PassLinesOff(); tri.PassVertsOff(); tri.Update()
    return tri.GetOutput()

def fast_subtract(original_pd, segment_pd, eps=0.0):
    # distance field of segment, clip original to keep outside
    dist = vtk.vtkImplicitPolyDataDistance()
    dist.SetInput(segment_pd)
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(original_pd)
    clip.SetClipFunction(dist)
    clip.SetValue(eps)          # keep points with distance >= eps (outside)
    clip.InsideOutOff()
    clip.GenerateClippedOutputOff()
    clip.Update()
    return clean_triangulate(clip.GetOutput())

def create_partial_mesh(original_ply, segment_ply, output_ply):
    print(f"  Loading original: {os.path.basename(original_ply)}")
    original = clean_triangulate(load_polydata(original_ply))
    print(f"  Loading segment: {os.path.basename(segment_ply)}")
    segment  = clean_triangulate(load_polydata(segment_ply))
    print("  Subtracting via implicit distance clip (fast)...")
    partial = fast_subtract(original, segment, eps=0.0)
    print(f"  Original: {original.GetNumberOfPoints()} vertices")
    print(f"  Segment:  {segment.GetNumberOfPoints()} vertices")
    print(f"  Partial:  {partial.GetNumberOfPoints()} vertices")
    save_ply(partial, output_ply)
    print(f"  ✓ Saved: {output_ply}")
    return partial.GetNumberOfPoints()

def create_validation_dataset(original_dir, segments_dir, output_dir, segment_to_remove=5):
    # Create output directory for partial meshes
    partial_dir = os.path.join(output_dir, "partial_meshes")
    os.makedirs(partial_dir, exist_ok=True)
    # Find all segment files for the target segment
    segment_pattern = os.path.join(segments_dir, f"*_seg_{segment_to_remove:02d}.ply")
    segment_files = glob(segment_pattern)
    if not segment_files:
        raise RuntimeError(f"No segment files found matching: {segment_pattern}")
    print(f"Found {len(segment_files)} meshes with segment {segment_to_remove}")
    results = []
    for segment_file in segment_files:
        # Extract base name
        basename = os.path.basename(segment_file)
        base_name = basename.replace(f"_seg_{segment_to_remove:02d}.ply", "")
        print(f"\n{'='*60}")
        print(f"Processing: {base_name}") 
        # Find corresponding original mesh (ground truth)
        original_ply = os.path.join(original_dir, f"{base_name}.ply")
        if not os.path.exists(original_ply):
            vtk_path = os.path.join(original_dir, f"{base_name}.vtk")
            if os.path.exists(vtk_path):
                original_ply = vtk_path
            else:
                print(f"Original mesh not found: {base_name}.ply or .vtk")
                continue

        try:
            # Create partial mesh
            partial_path = os.path.join(partial_dir, f"{base_name}_partial.ply")
            n_vertices = create_partial_mesh(original_ply, segment_file, partial_path) 
            results.append({
                'base_name': base_name,
                'ground_truth': original_ply,  # Just reference the original
                'partial': partial_path,
                'removed_segment': segment_file,  # Just reference existing segment
                'partial_vertices': int(n_vertices)
            })
            print(f"  ✓ Success!")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    # Save summary
    summary = {
        'segment_removed': segment_to_remove,
        'n_meshes': len(results),
        'original_dir': os.path.abspath(original_dir),
        'segments_dir': os.path.abspath(segments_dir),
        'meshes': results
    }
    summary_path = os.path.join(output_dir, "partial_meshing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*70}")
    print(f"✓ Created {len(results)} partial meshes")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"\nDirectories:")
    print(f"  - Partial meshes (input to shape completion): {partial_dir}")
    print(f"  - Ground truth (compare with output):        {original_dir}")
    print(f"  - Removed segments (for reference):          {segments_dir}")
    print(f"{'='*70}")
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Create partial meshes for shape completion validation")
    parser.add_argument("original_dir", 
                       help="Directory with original meshes (ground truth)")
    parser.add_argument("segments_dir",
                       help="Directory with segment PLYs (*_seg_XX.ply)")
    parser.add_argument("output_dir",
                       help="Output directory for partial meshes")
    parser.add_argument("--segment", type=int, default=6,
                       help="Segment number to remove (default: 6)")
    args = parser.parse_args()
    create_validation_dataset(
        args.original_dir,
        args.segments_dir,
        args.output_dir,
        segment_to_remove=args.segment)

if __name__ == "__main__":
    main()