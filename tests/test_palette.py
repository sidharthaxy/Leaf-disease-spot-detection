"""Palette branch on synthetic two-colour leaves, plus the 'crop route unchanged' guard."""

import hashlib

import cv2
import numpy as np

from src.palette.fit_palettes import robust_gmm, to_dict
from src.palette.palette import INFER, Palette, hysteresis, lab_features, run_palette


def two_colour_leaf(spots=(), seed=0):
    """Grey background, elliptical leaf: left half red-purple, right half green (a 'normal' bicolour
    ornamental). Optional brown spots. Returns (BGR image, leaf mask, spot mask)."""
    rng = np.random.default_rng(seed)
    h, w = 300, 400
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (128, 128, 128)
    leaf = np.zeros((h, w), np.uint8)
    cv2.ellipse(leaf, (200, 150), (160, 100), 0, 0, 360, (255,), -1)
    left = leaf.copy()
    left[:, 200:] = 0
    img[(leaf > 0) & (left > 0)] = (90, 30, 150)        # red-purple (BGR)
    img[(leaf > 0) & (left == 0)] = (50, 150, 60)       # green
    sp = np.zeros((h, w), np.uint8)
    for x, y, r in spots:
        cv2.circle(sp, (x, y), r, (255,), -1)
    sp &= leaf
    img[sp > 0] = (40, 90, 160)                         # tan/brown damage
    img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    return img, leaf, sp


def fitted_palette():
    X = []
    for s in range(4):
        img, leaf, _ = two_colour_leaf(seed=s)
        X.append(lab_features(img)[leaf > 0])
    X = np.concatenate(X)[::5]
    g = robust_gmm(X, 3, seed=0)
    pal = Palette(to_dict("test_plant", g, 3, g.score_samples(X), 0.005, 0.02))
    pal.cutoffs = [3.0, 10.0]
    return pal


def test_numpy_loglik_matches_sklearn():
    img, leaf, _ = two_colour_leaf()
    X = lab_features(img)[leaf > 0][:500]
    g = robust_gmm(X, 2, seed=0)
    pal = Palette(to_dict("t", g, 2, g.score_samples(X), 0.01, 0.05))
    assert np.allclose(pal.loglik(X), g.score_samples(X), atol=1e-6)


def test_healthy_bicolour_leaf_is_normal():
    pal = fitted_palette()
    img, leaf, _ = two_colour_leaf(seed=11)
    r = run_palette(img, pal, {"max_side": 400})
    assert r["measurements"]["abnormal_pct"] < 2.0
    assert r["measurements"]["label"] == "Normal"
    # the red half is NOT flagged: the crop pipeline would call it lesion
    inter = ((r["leaf_mask"] > 0) & (leaf > 0)).sum() / ((r["leaf_mask"] > 0) | (leaf > 0)).sum()
    assert inter > 0.9


def test_brown_spots_are_found():
    pal = fitted_palette()
    img, leaf, sp = two_colour_leaf(spots=((140, 130, 22), (260, 170, 18)), seed=12)
    r = run_palette(img, pal, {"max_side": 400})
    ab = r["abnormal_mask"] > 0
    dice = 2 * (ab & (sp > 0)).sum() / (ab.sum() + (sp > 0).sum())
    assert dice > 0.8
    true_pct = 100 * (sp > 0).sum() / (leaf > 0).sum()
    assert abs(r["measurements"]["abnormal_pct"] - true_pct) < 2.0
    assert r["measurements"]["n_spots"] == 2


def test_hysteresis_keeps_only_weak_regions_with_a_strong_seed():
    weak = np.zeros((20, 20), np.uint8)
    weak[2:6, 2:6] = 255
    weak[12:16, 12:16] = 255
    strong = np.zeros_like(weak)
    strong[3, 3] = 255
    out = hysteresis(strong, weak)
    assert out[2:6, 2:6].all() and not out[12:16, 12:16].any()


def test_swatches_are_valid_hex():
    sw = fitted_palette().swatches_bgr()
    assert len(sw) == 3 and all(s["hex"].startswith("#") and len(s["hex"]) == 7 for s in sw)


def test_crop_route_output_is_byte_identical_to_main():
    """The router/palette work must not change the crop pipeline. The reference hash was computed with the
    unchanged Stage B code from `main` (OpenCV 4.10, numpy 1.26)."""
    from test_imageproc import synthetic_leaf

    from src.imageproc.pipeline import run_pipeline

    img, _, _ = synthetic_leaf()
    r = run_pipeline(img, group="necrotic")
    h = hashlib.md5(r["leaf_mask"].tobytes() + r["lesion_mask"].tobytes()).hexdigest()
    assert h == "e8c8c289c3a6222df0e18980b0a6a3eb"
    assert r["measurements"]["n_spots"] == 3 and abs(r["measurements"]["infected_pct"] - 7.3548298321254935) < 1e-9
    assert INFER["max_side"] == 640  # palette branch has its own config; Stage B DEFAULTS untouched
