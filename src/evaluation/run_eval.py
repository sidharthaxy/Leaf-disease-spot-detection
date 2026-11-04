"""
Stage D runner: evaluate the classical pipeline on the test split against PlantSeg masks.

Runs on the 30-image sample or on the full validation / test split (multi-core CPU):
    python -m src.evaluation.run_eval --data-root data/sample --out reports/eval_sample
    python -m src.evaluation.run_eval --data-root data/processed --split test --predictions test_predictions.csv --out outputs/eval/test

Experiments (one row per image x experiment in per_image.csv):
  method_hsv / method_lab / method_kmeans : each lesion method with the disease-aware profile
                                            (oracle disease = ground-truth label)
  aware        : disease-aware profile, using that group's own chosen method   (oracle disease)
  generic      : the generic profile for every image (no disease knowledge)    -> ablation
  aware_noclahe: aware, but without CLAHE                                       -> ablation
  predicted    : profile from the CLASSIFIER's prediction (if --predictions CSV given)
Leaf-method comparison (no leaf GT exists): for exg / hsv / grabcut we record how much of the GT
lesion area falls inside the leaf mask (gt_inside_leaf) and the leaf's share of the image.
"""

from __future__ import annotations

import argparse
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
from src.severity.grade import get_profile, load_yaml, severity_label, symptom_group

ROOT: Path = Path(".")      # set by _init() in each worker process
PRED: dict = {}


def eval_one(row: dict) -> list[dict]:
    img = cv2.imread(str(ROOT / row["image_path"]))
    if img is None:
        return []
    gt = None
    if isinstance(row.get("mask_path"), str) and row["mask_path"]:
        gt = cv2.imread(str(ROOT / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    profiles = load_yaml("disease_profiles.yaml")
    true_group = symptom_group(row["label"], profiles)
    out = []

    pre, steps = preprocess(img)
    pre_noclahe, _ = preprocess(img, {"clahe": False})
    gt_r = resize_mask_like(gt, pre) if gt is not None else np.zeros(pre.shape[:2], np.uint8)

    # leaf masks (computed once per leaf method)
    leaves = {m: segment_leaf(pre, m)[0] for m in ("exg", "hsv", "grabcut", "exg_plus")}
    base = {"image_path": row["image_path"], "label": row["label"], "plant": row["plant"],
            "source": row["source"], "true_group": true_group,
            "non_leaf": row["label"] in set(profiles.get("non_leaf_labels") or [])}
    for m, L in leaves.items():
        s = severity_pcts(np.zeros_like(L), gt_r, L)
        out.append({**base, "experiment": f"leaf_{m}", "leaf_method": m,
                    "gt_inside_leaf": s["gt_inside_leaf"], "leaf_img_pct": s["leaf_img_pct"]})

    leaf_method = profiles["leaf"]["method"]
    L = leaves[leaf_method]

    def run(name, image, leaf, group, method=None):
        prof = get_profile(group, profiles)
        meth = method or prof["lesion_method"]
        les, _, _ = segment_lesions(image, leaf, meth, prof)
        s = severity_pcts(les, gt_r, leaf)
        rec = {**base, "experiment": name, "group_used": group, "lesion_method": meth, "leaf_method": leaf_method,
               **mask_metrics(les, gt_r), **s,
               "pred_sev": severity_label(s["pred_pct"], group), "gt_sev": severity_label(s["gt_pct"], true_group)}
        out.append(rec)

    for meth in ("hsv", "lab", "kmeans", "delta_e", "hsv&delta_e", "lab&delta_e"):
        run(f"method_{meth}", pre, L, true_group, meth)
    run("aware", pre, L, true_group)
    run("generic", pre, L, "generic")
    L_nc = segment_leaf(pre_noclahe, leaf_method)[0]
    run("aware_noclahe", pre_noclahe, L_nc, true_group)
    if row["image_path"] in PRED:
        label, conf, gname, gconf = PRED[row["image_path"]]
        thr = load_yaml("severity.yaml")["confidence"]["min_confidence"]
        run("predicted_top1rule", pre, L, symptom_group(label, profiles) if conf >= thr else "generic")
        run("predicted_grouprule", pre, L, gname if gconf >= thr else "generic")
        run("predicted_nofallback", pre, L, symptom_group(label, profiles))
        # features for the learned severity correction (predicted group's profile only)
        from src.evaluation.features import image_features
        s_true = severity_pcts(np.zeros_like(L), gt_r, L)
        out.append({**base, "experiment": "features", "pred_group": symptom_group(label, profiles),
                    "cnn_conf": conf, "cnn_group_conf": gconf, "gt_pct": s_true["gt_pct"],
                    **{f"f_{k}": v for k, v in image_features(pre, L, symptom_group(label, profiles)).items()}})
    return out


def _init(root, pred):
    global ROOT, PRED
    ROOT, PRED = Path(root), pred
    cv2.setNumThreads(1)  # we parallelise over images instead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--predictions", default=None, help="test_predictions.csv from training (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(Path(a.data_root) / a.manifest)
    df = df[df.split == a.split]
    if "mask_size_matches" in df.columns:  # 29 PlantSeg masks don't match their image size (likely EXIF rotation)
        bad = (df.source == "plantseg") & (df.mask_size_matches.astype(str).str.lower() != "true")
        print(f"skipping {int(bad.sum())} images whose mask size != image size", flush=True)
        df = df[~bad]
    if a.limit:
        df = df.head(a.limit)
    pred = {}
    if a.predictions and Path(a.predictions).exists():
        p = pd.read_csv(a.predictions)
        pred = dict(zip(p.image_path, zip(p.pred_label, p.confidence, p.pred_group, p.group_confidence)))
    rows = df.fillna("").to_dict("records")
    print(f"evaluating {len(rows)} images with {a.workers} workers", flush=True)
    res = []
    with Pool(a.workers, initializer=_init, initargs=(a.data_root, pred)) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_one, rows, chunksize=4)):
            res.extend(r)
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(res)
    res.to_csv(out / "per_image.csv", index=False)
    from src.evaluation.summarize import summarize
    summarize(res, out)


if __name__ == "__main__":
    main()
