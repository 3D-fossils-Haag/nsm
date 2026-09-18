"""
Table 2: shape completion chamfer descriptives by model and split.
Usage: 
python shape_completion_table.py --datadir shape_completion_eval/ --outdir shape_completion_eval/
"""
import argparse, os, sys
import numpy as np
import pandas as pd

MODELS = {
    "Baseline":       "v72_{split}_chamfer.csv",
    "Encoder":        "v72_encoder_{split}_chamfer.csv",
    "Encoder+refine": "v72_encoder_{split}_chamfer_refine.csv",
    "Hierarchy":      "v73h_{split}_chamfer.csv",
    "Contrastive":    "v73c_{split}_chamfer.csv"}

ORDER  = ["Baseline", "Encoder", "Encoder+refine", "Hierarchy", "Contrastive"]
SPLITS = ["train", "val", "test"]
SCALE  = 1e3

def load_all(datadir):
    frames = []
    for label, template in MODELS.items():
        for split in SPLITS:
            path = os.path.join(datadir, template.format(split=split))
            if not os.path.exists(path):
                print(f"  missing {os.path.basename(path)} -- skipping"); continue
            first = open(path).readline()
            df = pd.read_csv(path) if first.startswith("mesh,") else \
                 pd.read_csv(path, header=None, names=["mesh", "chamfer", "gt_path"])
            df["mesh"] = df["mesh"].str.replace("_partial.ply", "", regex=False)
            if df["mesh"].duplicated().any():
                df = df.groupby("mesh", as_index=False)["chamfer"].mean()
            df["model"], df["split"] = label, split
            frames.append(df[["mesh", "chamfer", "model", "split"]])
    if not frames:
        sys.exit(f"No chamfer CSVs found in {datadir}")
    return pd.concat(frames, ignore_index=True)

def table2(d, outdir):
    rows = []
    for split in SPLITS:
        for model in ORDER:
            c = d[(d.split == split) & (d.model == model)]["chamfer"].to_numpy() * SCALE
            if not len(c): continue
            rows.append({
                "Split": split, "Model": model, "n": len(c),
                "Mean ± SD": f"{c.mean():.2f} ± {c.std(ddof=1):.2f}",
                "Median [IQR]": f"{np.median(c):.2f} [{np.percentile(c,25):.2f}–{np.percentile(c,75):.2f}]",
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, "Table_2_shp_compl_eval.csv"), index=False)
    print(tab.to_string(index=False))
    return tab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=".")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    table2(load_all(args.datadir), args.outdir)

if __name__ == "__main__":
    main()