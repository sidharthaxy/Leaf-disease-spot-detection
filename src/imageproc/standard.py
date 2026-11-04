"""
The STANDARD colour-thresholding pipeline, used as the baseline in the demo's side-by-side comparison.

This is the textbook recipe most leaf-disease tutorials / student projects use:
    leaf    = Excess Green (2g - r - b) + Otsu threshold, morphology, hole filling, largest regions
    lesions = one fixed HSV rule for every disease: "inside the leaf and NOT healthy green"
              (green hue 35..85 in OpenCV's 0..179 scale, saturation >= 40)
It deliberately shares our preprocessing and our clean-up / measurement code, so the comparison isolates
the two things we changed: leaf segmentation (ExG+ vs ExG+Otsu) and disease-aware lesion detection
(CNN-chosen detector vs one fixed rule).
"""

from __future__ import annotations

import numpy as np

from . import viz
from .leaf_seg import segment_leaf
from .lesion_seg import postprocess_lesions, hsv_lesions
from .measure import measure
from .preprocess import preprocess

STANDARD_HSV = {"mode": "nongreen", "green_h_min": 35, "green_h_max": 85, "green_s_min": 40}


def run_standard(img_bgr: np.ndarray) -> dict:
    pre, steps = preprocess(img_bgr)
    leaf, _ = segment_leaf(pre, "exg")                     # plain ExG + Otsu (+ morphology, fill holes)
    cand = hsv_lesions(pre, leaf, STANDARD_HSV)            # one fixed rule for every disease
    lesion = postprocess_lesions(cand, leaf)
    m = measure(leaf, lesion)
    return {"pre": pre, "leaf_mask": leaf, "lesion_mask": lesion, "measurements": m,
            "overlay": viz.overlay(steps["resized"], leaf, lesion),
            "leaf_view": viz.mask_on_image(pre, leaf)}
