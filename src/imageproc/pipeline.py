"""
Stage B end-to-end: one function that runs the whole classical pipeline on one image and keeps
every intermediate image for the figure strip.

    original -> preprocessed -> leaf mask -> lesion candidates -> cleaned lesion mask -> overlay

Usage (CLI, runs on a folder of images and saves strips):
    python -m src.imageproc.pipeline sample_data/images --out reports/figures/strips --group generic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.severity.grade import get_profile, load_yaml, severity_label

from . import viz
from .leaf_seg import segment_leaf
from .lesion_seg import segment_lesions
from .measure import measure
from .preprocess import preprocess


def run_pipeline(img_bgr: np.ndarray, group: str = "generic", lesion_method: str | None = None,
                 leaf_method: str | None = None, pre_cfg: dict | None = None,
                 leaf_cfg: dict | None = None) -> dict:
    """Run Stage B (+ severity label) on one BGR image.

    group         : symptom group (from the classifier via src.severity.grade); 'generic' if unknown
    lesion_method : override the group's method ('hsv' | 'lab' | 'kmeans'), used for comparisons
    leaf_method   : override the leaf method ('exg' | 'hsv' | 'grabcut')
    """
    profiles = load_yaml("disease_profiles.yaml")
    profile = get_profile(group, profiles)
    leaf_method = leaf_method or profiles["leaf"]["method"]
    lesion_method = lesion_method or profile["lesion_method"]

    pre, pre_steps = preprocess(img_bgr, pre_cfg)
    leaf, leaf_dbg = segment_leaf(pre, leaf_method, leaf_cfg)
    lesion, cand, les_dbg = segment_lesions(pre, leaf, lesion_method, profile)
    m = measure(leaf, lesion, les_dbg.get("spots"))
    m["severity"] = severity_label(m["infected_pct"], group)
    m.update({"group": group, "leaf_method": leaf_method, "lesion_method": lesion_method})
    if "threshold" in les_dbg:
        m["lesion_threshold"] = les_dbg["threshold"]
    if "scores" in les_dbg:
        m["kmeans"] = {k: les_dbg[k] for k in ("centres_Lab", "scores", "healthy_cluster", "lesion_clusters")}

    steps = {
        "original": pre_steps["resized"],
        "preprocessed": pre,
        "leaf mask": viz.mask_on_image(pre, leaf),
        "lesion candidates": viz.mask_on_image(pre, cand),
        "cleaned lesions": viz.mask_on_image(pre, lesion),
        "overlay": viz.overlay(pre_steps["resized"], leaf, lesion),
    }
    if "label_img" in les_dbg:
        steps["k-means clusters"] = viz.label_colors(les_dbg["label_img"])
    return {"measurements": m, "leaf_mask": leaf, "lesion_mask": lesion, "candidates": cand, "steps": steps}


def strip_for(result: dict, title_extra: str = "") -> np.ndarray:
    order = ["original", "preprocessed", "leaf mask", "lesion candidates", "cleaned lesions", "overlay"]
    m = result["measurements"]
    panels = [(k, result["steps"][k]) for k in order]
    panels[-1] = (f"overlay {m['infected_pct']:.1f}% {m['severity']} ({m['n_spots']} spots){title_extra}",
                  panels[-1][1])
    return viz.make_strip(panels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", help="folder of images")
    ap.add_argument("--out", default="reports/figures/strips")
    ap.add_argument("--group", default="generic")
    ap.add_argument("--lesion-method", default=None)
    ap.add_argument("--leaf-method", default=None)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    out = Path(args.out)
    rows = []
    for p in files[: args.limit]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = run_pipeline(img, args.group, args.lesion_method, args.leaf_method)
        viz.save(out / f"{p.stem}_strip.jpg", strip_for(r))
        m = {k: v for k, v in r["measurements"].items() if k not in ("spot_areas_px", "spot_centroids")}
        rows.append({"image": p.name, **m})
        print(f"{p.name:40s} {m['infected_pct']:6.2f}%  {m['severity']:9s} spots={m['n_spots']}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "measurements.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
