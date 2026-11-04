"""
Stage B.1: preprocessing.

All functions take and return OpenCV-style BGR uint8 images.

Steps (each one is optional / configurable):
  1. resize so the longest side is `max_side`, keeping the aspect ratio
     (in-the-wild photos are often 3000+ px; we don't need that many pixels to find spots,
      and it keeps RAM and runtime low on the Mac)
  2. denoise: median (good for salt-and-pepper / JPEG speckle), Gaussian (generic smoothing)
     or bilateral (smooths flat regions but keeps edges, so lesion borders stay sharp)
  3. contrast enhancement with CLAHE on the L channel of Lab:
     - Lab separates lightness (L) from colour (a*, b*), so equalising L improves contrast
       WITHOUT shifting colours (equalising R, G, B separately would distort the hue of lesions)
     - CLAHE = Contrast Limited Adaptive Histogram Equalisation: equalises the histogram in small
       tiles, clipping the histogram at `clip_limit` so noise is not over-amplified
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_keep_aspect(img: np.ndarray, max_side: int = 768) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img.copy()
    new_size = (round(w * scale), round(h * scale))  # cv2 wants (width, height)
    # INTER_AREA is the recommended interpolation for shrinking (averages pixels, avoids aliasing)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def resize_mask_like(mask: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Resize a label/binary mask to match `like` (nearest neighbour so labels stay integers)."""
    h, w = like.shape[:2]
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def denoise(img: np.ndarray, method: str = "median", ksize: int = 5) -> np.ndarray:
    if method in (None, "none"):
        return img.copy()
    if method == "median":
        return cv2.medianBlur(img, ksize)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == "bilateral":
        # d = neighbourhood diameter; sigmaColor = how different colours can be and still be mixed;
        # sigmaSpace = how far pixels can be and still influence each other
        return cv2.bilateralFilter(img, d=ksize, sigmaColor=50, sigmaSpace=50)
    raise ValueError(f"unknown denoise method: {method}")


def clahe_lab(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2BGR)


def preprocess(img: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """Run the full preprocessing chain. Returns (preprocessed image, dict of intermediate images)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    steps = {}
    x = resize_keep_aspect(img, cfg["max_side"])
    steps["resized"] = x
    x = denoise(x, cfg["denoise"], cfg["denoise_ksize"])
    steps["denoised"] = x
    if cfg["clahe"]:
        x = clahe_lab(x, cfg["clahe_clip"], cfg["clahe_tile"])
    steps["preprocessed"] = x
    return x, steps


DEFAULTS = {
    "max_side": 768,
    "denoise": "median",
    "denoise_ksize": 5,
    "clahe": True,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
}
