"""
Shared binary-mask utilities (morphology + connected components).

Masks are uint8 arrays with values {0, 255} (OpenCV convention) unless stated otherwise.

Morphology recap (for the viva):
  - erosion  : pixel stays ON only if the whole structuring element fits -> shrinks objects
  - dilation : pixel turns ON if the structuring element touches any ON pixel -> grows objects
  - opening  = erosion then dilation  -> removes small specks / thin bridges, keeps big shapes
  - closing  = dilation then erosion  -> fills small gaps / cracks, joins nearby fragments
"""

from __future__ import annotations

import cv2
import numpy as np


def to_u8(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def ellipse_kernel(k: int) -> np.ndarray:
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def open_close(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    m = to_u8(mask)
    if open_k and open_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse_kernel(open_k))
    if close_k and close_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse_kernel(close_k))
    return m


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes: flood-fill the background from the image border; whatever background
    the flood could NOT reach is enclosed by the object, so it is a hole -> set it ON.

    We pad by 1 px first so the flood can go around objects that touch the image border.
    """
    m = to_u8(mask)
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)  # pixels the flood never reached
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def components(mask: np.ndarray, connectivity: int = 8):
    """Connected-component labelling. Returns (n_labels_incl_bg, labels, stats, centroids).
    stats columns: x, y, width, height, area (cv2.CC_STAT_*)."""
    return cv2.connectedComponentsWithStats(to_u8(mask), connectivity=connectivity)


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = components(mask)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return to_u8(keep[labels])


def keep_largest(mask: np.ndarray, top_k: int = 1, min_frac_of_largest: float = 0.0) -> np.ndarray:
    """Keep the `top_k` largest components (optionally only those >= frac x the largest one)."""
    n, labels, stats, _ = components(mask)
    if n <= 1:
        return to_u8(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1][:top_k]
    largest = areas[order[0]]
    keep = np.zeros(n, bool)
    for idx in order:
        if areas[idx] >= min_frac_of_largest * largest:
            keep[idx + 1] = True
    return to_u8(keep[labels])
