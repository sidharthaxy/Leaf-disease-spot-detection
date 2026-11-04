"""
Stage B.5: measurements from the final masks.

  infected_pct = lesion_px / leaf_px * 100
  spots        = connected components of the lesion mask (8-connectivity)
"""

from __future__ import annotations

import cv2
import numpy as np

from .morphology import components


def measure(leaf: np.ndarray, lesion: np.ndarray, spots: np.ndarray | None = None) -> dict:
    """infected % from `lesion` (lesion area, or affected zone for rust); spot statistics from
    `spots` (individual lesions) if given."""
    leaf_px = int((leaf > 0).sum())
    lesion_px = int(((lesion > 0) & (leaf > 0)).sum())
    n, _, stats, centroids = components(lesion if spots is None else spots)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(int) if n > 1 else np.array([], int)
    return {
        "leaf_px": leaf_px,
        "lesion_px": lesion_px,
        "infected_pct": 100.0 * lesion_px / leaf_px if leaf_px else 0.0,
        "leaf_pct_of_image": 100.0 * leaf_px / leaf.size,
        "n_spots": int(len(areas)),
        "spot_areas_px": areas.tolist(),
        "mean_spot_area_px": float(areas.mean()) if len(areas) else 0.0,
        "median_spot_area_px": float(np.median(areas)) if len(areas) else 0.0,
        "max_spot_area_px": int(areas.max()) if len(areas) else 0,
        "spot_centroids": centroids[1:].round(1).tolist() if n > 1 else [],
    }
