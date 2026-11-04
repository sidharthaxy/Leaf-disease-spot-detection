"""
Visualisation helpers: overlays and the step-by-step figure strip for the report / demo.

  original -> preprocessed -> leaf mask -> lesion candidates -> cleaned lesion mask -> final overlay
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .morphology import to_u8

LESION_FILL = (40, 40, 230)     # BGR red
LESION_EDGE = (0, 255, 255)     # yellow
LEAF_EDGE = (60, 220, 60)       # green


def overlay(img: np.ndarray, leaf: np.ndarray | None, lesion: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img.copy()
    les = lesion > 0
    tint = out.copy()
    tint[les] = LESION_FILL
    out = cv2.addWeighted(tint, alpha, out, 1 - alpha, 0)
    if leaf is not None:
        cs, _ = cv2.findContours(to_u8(leaf), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, LEAF_EDGE, 2)
    cs, _ = cv2.findContours(to_u8(lesion), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, LESION_EDGE, 1)
    return out


def mask_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Show only the masked pixels (background darkened) - easier to judge than a pure b/w mask."""
    out = (img * 0.25).astype(np.uint8)
    out[mask > 0] = img[mask > 0]
    return out


def to_bgr(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    return x


def label_colors(label_img: np.ndarray) -> np.ndarray:
    """Colour a k-means label image (0 = background)."""
    palette = np.array([[0, 0, 0], [60, 180, 75], [230, 25, 75], [255, 225, 25],
                        [0, 130, 200], [245, 130, 48], [145, 30, 180]], np.uint8)[:, ::-1]
    return palette[np.clip(label_img, 0, len(palette) - 1)]


def make_strip(panels: list[tuple[str, np.ndarray]], tile_h: int = 260) -> np.ndarray:
    tiles = []
    for title, im in panels:
        im = to_bgr(im)
        h, w = im.shape[:2]
        t = cv2.resize(im, (max(1, round(w * tile_h / h)), tile_h), interpolation=cv2.INTER_AREA)
        bar = np.full((28, t.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, t]))
        tiles.append(np.full((tile_h + 28, 6, 3), 255, np.uint8))
    return np.hstack(tiles[:-1])


def save(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
