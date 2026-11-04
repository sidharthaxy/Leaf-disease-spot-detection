"""
Error analysis for the classical pipeline on a local dev set (uses the tune.py cache).

    python -m src.evaluation.error_analysis --cache outputs/tune_v1/cache --out outputs/error_analysis

Uses the CURRENT configs/disease_profiles.yaml (leaf method + per-group lesion method/params).
Writes per_image.csv, an error-by-severity-bin table and a gallery of the worst N cases
(image | ground truth | prediction) for the report's failure analysis.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics import mask_metrics, severity_pcts
from src.imageproc import viz
from src.imageproc.leaf_seg import segment_leaf
from src.imageproc.lesion_seg import segment_lesions
from src.severity.grade import get_profile, load_yaml, symptom_group

CACHE: Path = Path("outputs/tune_v1/cache")   # set by _init() in each worker process


def _init(cache):
    global CACHE
    CACHE = Path(cache)


def masks_for(image_path: str, group: str):
    d = np.load(CACHE / (Path(image_path).stem + ".npz"))
    prof = get_profile(group)
    leaf, _ = segment_leaf(d["pre"], load_yaml("disease_profiles.yaml")["leaf"]["method"])
    les, _, _ = segment_lesions(d["pre"], leaf, prof["lesion_method"], prof)
    return d["pre"], d["gt"], leaf, les, prof["lesion_method"]


def run(args):
    image_path, group = args
    _, gt, leaf, les, method = masks_for(image_path, group)
    return {**mask_metrics(les, gt), **severity_pcts(les, gt, leaf), "lesion_method": method}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/dev_val")
    ap.add_argument("--cache", default="outputs/tune_v1/cache")
    ap.add_argument("--out", default="outputs/error_analysis")
    ap.add_argument("--worst", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    df = pd.read_csv(Path(a.data_root) / "manifest.csv").fillna("")
    df["group"] = df.label.map(symptom_group)
    with Pool(a.workers, initializer=_init, initargs=(a.cache,)) as p:
        res = pd.DataFrame(p.map(run, list(zip(df.image_path, df.group))))
    df = pd.concat([df.reset_index(drop=True), res], axis=1)
    df["err"] = df.pred_pct - df.gt_pct
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_image.csv", index=False)

    dis, hl = df[df.source == "plantseg"].copy(), df[df.source != "plantseg"]
    print(f"diseased n={len(dis)}  MAE={dis.err.abs().mean():.2f}  median AE={dis.err.abs().median():.2f}  "
          f"bias={dis.err.mean():.2f}  Dice={dis.dice.mean():.3f}  recall={dis.recall.mean():.3f}  precision={dis.precision.mean():.3f}")
    print(f"healthy  n={len(hl)}  mean FP={hl.pred_pct.mean():.2f}%  median FP={hl.pred_pct.median():.2f}%  "
          f"<1%: {(hl.pred_pct < 1).mean():.2f}")
    dis["gt_bin"] = pd.cut(dis.gt_pct, [-0.01, 5, 15, 30, 50, 101])
    t = dis.groupby("gt_bin", observed=True).agg(n=("err", "size"), mae=("err", lambda e: e.abs().mean()),
                                                 bias=("err", "mean"), dice=("dice", "mean")).round(2)
    print(t.to_string())
    print(dis.groupby("group").agg(n=("err", "size"), mae=("err", lambda e: e.abs().mean()), bias=("err", "mean"),
                                   dice=("dice", "mean")).round(2).to_string())
    t.to_csv(out / "error_by_severity_bin.csv")

    _init(a.cache)
    worst = dis.reindex(dis.err.abs().sort_values(ascending=False).index).head(a.worst)
    rows = []
    for r in worst.itertuples():
        pre, gt, leaf, les, method = masks_for(str(r.image_path), str(r.group))
        rows.append(viz.make_strip([(str(r.label)[:32], pre), (f"GT {r.gt_pct:.0f}%", viz.overlay(pre, leaf, gt)),
                                    (f"pred {r.pred_pct:.0f}% ({method})", viz.overlay(pre, leaf, les))], tile_h=170))
    W = max(x.shape[1] for x in rows)
    rows = [np.hstack([x, np.full((x.shape[0], W - x.shape[1], 3), 255, np.uint8)]) for x in rows]
    cv2.imwrite(str(out / "worst_cases.jpg"), np.vstack(rows))
    print("gallery:", out / "worst_cases.jpg")


if __name__ == "__main__":
    main()
