"""
Synthetic damage for tuning/evaluating the palette branch (clearly labelled "synthetic" in all reports).

Real, licence-clean photos of DAMAGED ornamental leaves are scarce (a Wikimedia Commons title search
found ~2 for our 10 species; iNaturalist descriptions almost never mention damage). To still measure
whether the branch finds off-palette tissue, and where, we paste REAL lesions onto healthy leaves:

  1. lesion bank: connected lesion regions cut out of PlantSeg VALIDATION images using their expert
     masks (brown/black necrosis, yellow chlorosis, white mildew: real lesion colours and textures)
  2. paste 1-6 lesions inside the healthy leaf (rough leaf prior = GrabCut with a centre rectangle),
     each scaled to 3-12 % of the image side, with a feathered alpha edge
  3. ground truth = the pasted pixels (alpha >= 0.5)

Limits (stated in the README): pasted crop lesions are not the same as sun scorch on a croton; lesions
whose colour happens to be normal for the species (yellow on croton) are expected to be missed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def rough_leaf_prior(img_bgr: np.ndarray, margin: float = 0.05, iters: int = 3) -> np.ndarray:
    """GrabCut with a centre rectangle: a generic, species-agnostic leaf/plant prior (0/255)."""
    h, w = img_bgr.shape[:2]
    rect = (int(margin * w), int(margin * h), int((1 - 2 * margin) * w), int((1 - 2 * margin) * h))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def build_lesion_bank(plantseg_root: str | Path, max_images: int = 200, seed: int = 0,
                      min_area: int = 150, max_patches: int = 600) -> list[tuple[np.ndarray, np.ndarray]]:
    """(patch_bgr, patch_mask) pairs cut out of PlantSeg val images with their lesion masks."""
    root = Path(plantseg_root)
    m = pd.read_csv(root / "manifest.csv")
    m = m[(m.split == "val") & m.mask_path.notna() & (m.source == "plantseg")]
    m = m.sample(min(len(m), max_images), random_state=seed)
    bank = []
    for r in m.itertuples():
        img = cv2.imread(str(root / r.image_path))
        msk = cv2.imread(str(root / r.mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        if msk.shape != img.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        n, lab, st, _ = cv2.connectedComponentsWithStats((msk > 0).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < min_area or w < 8 or h < 8 or a > 0.25 * msk.size:
                continue
            bank.append((img[y:y + h, x:x + w].copy(), ((lab[y:y + h, x:x + w] == i) * 255).astype(np.uint8)))
    rng = np.random.default_rng(seed)
    rng.shuffle(bank)
    return bank[:max_patches]


def paste_lesions(img_bgr: np.ndarray, prior: np.ndarray, bank: list, rng: np.random.Generator,
                  n_range=(1, 6), size_range=(0.03, 0.12)) -> tuple[np.ndarray, np.ndarray]:
    """Returns (damaged image, ground-truth mask 0/255). Each lesion lies >= 80 % inside the eroded prior."""
    out = img_bgr.astype(np.float32).copy()
    gt = np.zeros(img_bgr.shape[:2], np.uint8)
    h, w = gt.shape
    inner = cv2.erode(prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if inner.sum() == 0 or not bank:
        return img_bgr.copy(), gt
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        patch, pm = bank[int(rng.integers(len(bank)))]
        side = rng.uniform(*size_range) * max(h, w)
        s = side / max(patch.shape[:2])
        ph, pw = max(4, int(patch.shape[0] * s)), max(4, int(patch.shape[1] * s))
        if ph >= h - 2 or pw >= w - 2:
            continue
        p = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
        a = cv2.resize(pm, (pw, ph), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        a = cv2.GaussianBlur(a, (0, 0), 1.2)
        for _try in range(60):
            y0, x0 = int(rng.integers(0, h - ph)), int(rng.integers(0, w - pw))
            region = inner[y0:y0 + ph, x0:x0 + pw]
            if (region[a > 0.5] > 0).mean() >= 0.8:   # (almost) entirely on the plant
                out[y0:y0 + ph, x0:x0 + pw] = a[..., None] * p + (1 - a[..., None]) * out[y0:y0 + ph, x0:x0 + pw]
                gt[y0:y0 + ph, x0:x0 + pw] |= ((a >= 0.5) * 255).astype(np.uint8)
                break
    return np.clip(out, 0, 255).astype(np.uint8), gt


def paste_to_target(img_bgr: np.ndarray, prior: np.ndarray, bank: list, rng: np.random.Generator,
                    target_frac: float, size_range=(0.04, 0.15), max_lesions: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Paste lesions one at a time until they cover >= target_frac of the rough plant prior (a controlled
    damage level, e.g. 0.05 / 0.10 / 0.20). Returns (damaged image, ground-truth mask)."""
    out, gt = img_bgr.copy(), np.zeros(img_bgr.shape[:2], np.uint8)
    area = max(1, int((prior > 0).sum()))
    for _ in range(max_lesions):
        if (gt > 0).sum() >= target_frac * area:
            break
        out, g = paste_lesions(out, prior, bank, rng, n_range=(1, 1), size_range=size_range)
        gt |= g
    return out, gt
