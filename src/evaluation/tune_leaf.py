"""
Tune leaf-segmentation settings (ExG+) with the per-group best lesion configs held fixed.

    python -m src.evaluation.tune_leaf --best outputs/tune_v1/best.json --out outputs/tune_leaf

For each leaf setting reports: share of GT lesion pixels inside the leaf (should stay high),
leaf share of image, severity MAE / Dice on diseased dev images, false-positive % on healthy ones.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import mask_metrics, severity_pcts
from src.imageproc.leaf_seg import segment_leaf
from src.imageproc.lesion_seg import segment_lesions
from src.severity.grade import get_profile, symptom_group

BEST: dict = {}   # set by init() in each worker process


def run(args):
    f, group, leaf_cfg = args
    d = np.load(f)
    leaf, _ = segment_leaf(d["pre"], "exg_plus", {"exg_plus": leaf_cfg})
    b = BEST[group]
    prof = get_profile(group)
    prof = {**prof, b["method"]: {**prof.get(b["method"], {}), **b["params"]}}
    les, _, _ = segment_lesions(d["pre"], leaf, b["method"], prof)
    s = severity_pcts(les, d["gt"], leaf)
    return {**mask_metrics(les, d["gt"]), **s}


def init(best):
    global BEST
    BEST = best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/dev_val")
    ap.add_argument("--cache", default="outputs/tune_v1/cache")
    ap.add_argument("--best", default="outputs/tune_v1/best.json")
    ap.add_argument("--out", default="outputs/tune_leaf")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    a = ap.parse_args()
    best = json.loads(Path(a.best).read_text())
    df = pd.read_csv(Path(a.data_root) / "manifest.csv").fillna("")
    df["group"] = df.label.map(symptom_group)
    df["cache"] = [str(Path(a.cache) / (Path(p).stem + ".npz")) for p in df.image_path]
    grid = {"min_green_border": [0.0, 0.3, 0.5, 0.7], "expand_frac": [0.02, 0.04]}
    rows = []
    with Pool(a.workers, initializer=init, initargs=(best,)) as pool:
        for vals in itertools.product(*grid.values()):
            cfg = dict(zip(grid, vals))
            res = pd.DataFrame(pool.map(run, [(f, g, cfg) for f, g in zip(df.cache, df.group)]))
            res["group"] = df.group.values
            dis, hl = res[res.group != "healthy"], res[res.group == "healthy"]
            rows.append({**cfg,
                         "gt_inside_leaf": dis.gt_inside_leaf.mean(), "leaf_img_pct": res.leaf_img_pct.mean(),
                         "dice": dis.dice.mean(), "mae": (dis.pred_pct - dis.gt_pct).abs().mean(),
                         "bias": (dis.pred_pct - dis.gt_pct).mean(),
                         "healthy_fp_pct": hl.pred_pct.mean(), "healthy_graded_healthy": (hl.pred_pct < 1).mean()})
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "leaf_grid.csv", index=False)


if __name__ == "__main__":
    main()
