"""
Data-driven choice of lesion method + parameters PER SYMPTOM GROUP, on a validation dev set.

    python -m src.evaluation.tune --data-root data/dev_val --out outputs/tune

1. cache: preprocess + leaf mask (ExG+) + GT mask for every dev image (done once)
2. grid : for each symptom group, try every (method, params) candidate on that group's images
3. score: Dice (pixel agreement) and severity MAE (infected % error); objective = Dice - MAE/100
          healthy profile is scored on healthy images (false-positive %) instead
Only VALIDATION images are used here; the test split is reserved for the final report.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics import mask_metrics, severity_pcts
from src.imageproc.leaf_seg import segment_leaf
from src.imageproc.lesion_seg import segment_lesions
from src.imageproc.preprocess import preprocess, resize_mask_like
from src.severity.grade import get_profile, symptom_group



def build_cache(args):
    root, r, cache_dir = args
    f = Path(cache_dir) / (Path(r["image_path"]).stem + ".npz")
    if f.exists():
        return str(f)
    img = cv2.imread(str(Path(root) / r["image_path"]))
    pre, _ = preprocess(img)
    leaf, _ = segment_leaf(pre, "exg_plus")
    if isinstance(r.get("mask_path"), str) and r["mask_path"]:
        gt = resize_mask_like(cv2.imread(str(Path(root) / r["mask_path"]), 0), pre)
    else:
        gt = np.zeros(pre.shape[:2], np.uint8)
    np.savez_compressed(f, pre=pre, leaf=leaf, gt=gt)
    return str(f)


def eval_cfg(args):
    f, group, method, params = args
    d = np.load(f)
    prof = get_profile(group)
    key = "delta_e" if "delta_e" in method else method
    prof = {**prof, key: {**prof.get(key, {}), **params}}
    les, _, _ = segment_lesions(d["pre"], d["leaf"], method, prof)
    s = severity_pcts(les, d["gt"], d["leaf"])
    return {**mask_metrics(les, d["gt"]), "pred_pct": s["pred_pct"], "gt_pct": s["gt_pct"]}


GRIDS = {
    "delta_e": {"floor": [12, 16, 20, 25], "ratio": [0.5, 0.65, 0.8], "w_L": [0.3, 0.6, 1.0], "ref_frac": [0.4]},
    "hsv": {"green_s_min": [25, 40, 55], "green_h_min": [28, 33, 38]},
    "lab": {"floor": [126, 128, 132, 136], "offset": [-6, 0, 6]},
    "kmeans": {"k": [3, 4], "min_delta": [15, 25, 35]},
}
POWDERY_GRIDS = {
    "delta_e": {"floor": [10, 14, 18, 24], "ratio": [0.5, 0.65, 0.8], "w_L": [0.6, 1.0, 1.5], "ref_frac": [0.4]},
    "hsv": {"mode": ["powdery"], "s_max": [45, 60, 80], "v_min": [140, 160, 180]},
    "kmeans": {"k": [3, 4], "min_delta": [10, 20, 30]},
}


# v2: stricter ΔE and two-detector voting (the ΔE grid in v1 stopped at floor 25 and over-detected)
GRIDS_V2 = {
    "delta_e": {"floor": [25, 30, 35, 40, 50], "ratio": [0.6, 0.8], "w_L": [0.3, 0.6], "ref_frac": [0.4]},
    "lab&delta_e": {"floor": [10, 15, 20, 25], "ratio": [0.6, 0.8], "w_L": [0.3, 0.6], "ref_frac": [0.4]},
    "hsv&delta_e": {"floor": [10, 15, 20, 25], "ratio": [0.6, 0.8], "w_L": [0.3, 0.6], "ref_frac": [0.4]},
    "lab&hsv": {"floor": [0]},   # params unused (each detector keeps its group params)
}


def expand(grid):
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/dev_val")
    ap.add_argument("--out", default="outputs/tune")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--grids", default="v1", choices=["v1", "v2"])
    ap.add_argument("--groups", default="", help="comma list to restrict")
    a = ap.parse_args()
    out = Path(a.out); cache_dir = out / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(a.data_root) / "manifest.csv").fillna("")
    df["group"] = df.label.map(symptom_group)
    rows = df.to_dict("records")
    with Pool(a.workers) as pool:
        files = pool.map(build_cache, [(a.data_root, r, str(cache_dir)) for r in rows])
    df["cache"] = files
    print("cache ready:", df.group.value_counts().to_dict(), flush=True)

    results, best = [], {}
    with Pool(a.workers) as pool:
        for group, g in df.groupby("group"):
            if a.groups and group not in a.groups.split(","):
                continue
            grids = GRIDS_V2 if a.grids == "v2" else (POWDERY_GRIDS if group == "powdery" else GRIDS)
            eval_group = "generic" if group == "healthy" else group
            # healthy profile: tuned for low false positives on healthy leaves
            for method, grid in grids.items():
                for params in expand(grid):
                    rs = pool.map(eval_cfg, [(f, eval_group if group != "healthy" else "healthy", method, params)
                                             for f in g.cache])
                    r = pd.DataFrame(rs)
                    if group == "healthy":
                        score = -r.pred_pct.mean() / 100
                        rec = {"fp_pct": r.pred_pct.mean(), "graded_healthy": (r.pred_pct < 1).mean()}
                    else:
                        mae = (r.pred_pct - r.gt_pct).abs().mean()
                        score = r.dice.mean() - mae / 100
                        rec = {"dice": r.dice.mean(), "iou": r.iou.mean(), "recall": r.recall.mean(),
                               "precision": r.precision.mean(), "mae": mae, "bias": (r.pred_pct - r.gt_pct).mean()}
                    results.append({"group": group, "method": method, "params": json.dumps(params), "score": score,
                                    "n": len(g), **rec})
                    if score > best.get(group, {}).get("score", -1e9):
                        best[group] = {"method": method, "params": params, "score": score, **rec}
            print(f"[{group}] best: {best[group]}", flush=True)
    pd.DataFrame(results).sort_values(["group", "score"], ascending=[True, False]).to_csv(out / "grid_results.csv", index=False)
    (out / "best.json").write_text(json.dumps(best, indent=2, default=float))
    # best per (group, method) for the report table
    res = pd.DataFrame(results)
    top = res.loc[res.groupby(["group", "method"]).score.idxmax()]
    print(top.drop(columns=["params"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
