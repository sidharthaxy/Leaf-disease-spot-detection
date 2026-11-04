"""
Stage D metrics: compare our classical lesion mask P with the PlantSeg ground-truth lesion mask G.

  IoU       = |P ∩ G| / |P ∪ G|
  Dice      = 2|P ∩ G| / (|P| + |G|)          (= F1 score at pixel level)
  precision = |P ∩ G| / |P|                   (how much of what we flagged is really diseased)
  recall    = |P ∩ G| / |G|                   (how much of the real disease we found)

Severity (infected % of leaf). PlantSeg has no leaf masks, so we use ONE shared denominator for
prediction and ground truth:   D = |L ∪ G|   (our classical leaf mask L, plus any GT lesion pixels
the leaf mask missed, because lesions are by definition on the leaf)
  pred_pct = |P| / D * 100,   gt_pct = |G| / D * 100
=> any difference between pred_pct and gt_pct comes from the LESION step, not from leaf segmentation.
Image-level ratios (|P| / image, |G| / image) are also reported; they do not depend on L at all.

Edge cases: healthy images have G = ∅. IoU/Dice are undefined there (0/0); we report NaN and
judge healthy images by pred_pct only (false-positive area).
"""

from __future__ import annotations

import numpy as np


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred > 0, gt > 0
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    union = tp + fp + fn
    nan = float("nan")
    return {
        "iou": tp / union if union else nan,
        "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan,
        "precision": tp / (tp + fp) if (tp + fp) else nan,
        "recall": tp / (tp + fn) if (tp + fn) else nan,
    }


def severity_pcts(pred: np.ndarray, gt: np.ndarray, leaf: np.ndarray) -> dict:
    p, g, l = pred > 0, gt > 0, leaf > 0
    denom = int((l | g).sum())
    return {
        "pred_pct": 100 * p.sum() / denom if denom else 0.0,
        "gt_pct": 100 * g.sum() / denom if denom else 0.0,
        "pred_img_pct": 100 * p.mean(),
        "gt_img_pct": 100 * g.mean(),
        "leaf_img_pct": 100 * l.mean(),
        "gt_inside_leaf": float((g & l).sum() / g.sum()) if g.sum() else float("nan"),
    }
