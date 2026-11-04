"""
"Is this a plant at all?" gate, run BEFORE the classifier and the lesion pipeline.

Why it is needed: the CNN is a closed-set classifier (it must pick one of its 125 plant classes for ANY
image) and the ExG + Otsu leaf mask is relative (it picks the "greener" part of whatever it is given).
So a photo of a hand, a face or a wall would otherwise be analysed as a leaf.

The test is absolute and classical: the share of the image that is genuinely green plant tissue,
    green = Excess Green (2g - r - b on chromatic coordinates) > exg_min  AND  HSV saturation >= s_min
Measured on 464 real plant photos (435 validation + 29 gallery images, every disease and healthy):
median 76 %, 99.5 % of photos above 5.3 %, lowest 1.0 % (plum pox on a fruit), next 2.5 % (carrot
cavity spot on a root). Synthetic non-plant scenes (skin on a white wall, a brown table): 0.0 %.
Thresholds live in configs/disease_profiles.yaml (plant_check).
"""

from __future__ import annotations

import cv2
import numpy as np

from .leaf_seg import exg_index

DEFAULTS = {"exg_min": 0.05, "s_min": 40, "reject_below_pct": 0.5, "warn_below_pct": 3.0}


def green_fraction(img_bgr: np.ndarray, exg_min: float = 0.05, s_min: int = 40) -> float:
    """Fraction (0..1) of pixels that are genuinely green plant tissue."""
    sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1]
    return float(((exg_index(img_bgr) > exg_min) & (sat >= s_min)).mean())


def check_plant(img_bgr: np.ndarray, cfg: dict | None = None) -> dict:
    """Returns {"status": "ok" | "low_green" | "not_plant", "green_pct": float, "message": str}."""
    c = {**DEFAULTS, **(cfg or {})}
    pct = 100.0 * green_fraction(img_bgr, float(c["exg_min"]), int(c["s_min"]))
    if pct < c["reject_below_pct"]:
        status, msg = "not_plant", (f"This does not look like a plant: only {pct:.1f} % of the image is green plant "
                                    f"tissue (every real leaf photo we measured has at least 1 %). Analysis skipped.")
    elif pct < c["warn_below_pct"]:
        status, msg = "low_green", (f"Very little green plant tissue ({pct:.1f} % of the image), e.g. a fruit, stem or "
                                    f"a mostly dead leaf: results may be unreliable.")
    else:
        status, msg = "ok", f"{pct:.1f} % of the image is green plant tissue."
    return {"status": status, "green_pct": round(pct, 2), "message": msg}
