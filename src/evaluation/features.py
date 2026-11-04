"""
Per-image image-processing features for the learned severity correction (src/evaluation/learned_correction.py).

All features are classical measurements computed with the PREDICTED symptom group's profile (what the
deployed system knows), never with the ground truth:
  - infected % from every detector (Lab, HSV, k-means, ΔE+hysteresis) and the affected-zone %
  - spot statistics: count, density, largest spot, mean spot size (relative to leaf area)
  - convex-hull area of all lesions / leaf area (how spread out the disease is)
  - leaf share of the image
  - Lab colour statistics of the leaf and of the lesions
  - GLCM texture of the leaf (contrast, homogeneity, energy, correlation)
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from src.imageproc.lesion_seg import affected_zone, segment_lesions
from src.imageproc.measure import measure
from src.severity.grade import get_profile

METHODS = ("lab", "hsv", "kmeans", "delta_e")


def image_features(pre: np.ndarray, leaf: np.ndarray, group: str) -> dict:
    prof = get_profile(group)
    inside = leaf > 0
    leaf_px = max(1, int(inside.sum()))
    f = {"leaf_img_pct": 100 * inside.mean()}
    final = None
    for m in METHODS:
        les, _, info = segment_lesions(pre, leaf, m, {**prof, "zone_frac": None})
        f[f"pct_{m}"] = 100 * ((les > 0) & inside).sum() / leaf_px
    final, _, info = segment_lesions(pre, leaf, prof["lesion_method"], prof)
    spots = info.get("spots", final)
    meas = measure(leaf, final, spots)
    f["pct_final"] = meas["infected_pct"]
    for frac in (0.05, 0.10, 0.20):
        z = affected_zone(spots, leaf, frac)
        f[f"pct_zone{int(frac * 100)}"] = 100 * (z > 0).sum() / leaf_px
    areas = np.array(meas["spot_areas_px"] or [0])
    f.update({"n_spots": meas["n_spots"], "spot_density": 1e4 * meas["n_spots"] / leaf_px,
              "max_spot_frac": 100 * areas.max() / leaf_px, "mean_spot_frac": 100 * areas.mean() / leaf_px})
    cs, _ = cv2.findContours((spots > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        hull = cv2.convexHull(np.vstack(cs))
        f["lesion_hull_frac"] = 100 * cv2.contourArea(hull) / leaf_px
    else:
        f["lesion_hull_frac"] = 0.0
    lab = cv2.cvtColor(pre, cv2.COLOR_BGR2LAB).astype(np.float32)
    for i, ch in enumerate("Lab"):
        v = lab[..., i][inside]
        f[f"leaf_{ch}_mean"], f[f"leaf_{ch}_std"] = (float(v.mean()), float(v.std())) if v.size else (0.0, 0.0)
        w = lab[..., i][(spots > 0) & inside]
        f[f"les_{ch}_mean"] = float(w.mean()) if w.size else 0.0
    g = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (192, round(192 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA) // 8   # 32 levels
    glcm = graycomatrix(g, distances=[2], angles=[0, np.pi / 2], levels=32, symmetric=True, normed=True)
    for p in ("contrast", "homogeneity", "energy", "correlation"):
        f[f"glcm_{p}"] = float(graycoprops(glcm, p).mean())
    return f
