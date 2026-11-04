"""Unit tests for the classical pipeline on synthetic leaves with a known infected %."""

import cv2
import numpy as np
import pytest

from src.imageproc.leaf_seg import segment_leaf
from src.imageproc.lesion_seg import segment_lesions
from src.imageproc.measure import measure
from src.imageproc.morphology import fill_holes
from src.imageproc.pipeline import run_pipeline
from src.severity.grade import get_profile, resolve_from_prediction, severity_label, symptom_group


def synthetic_leaf(lesions=((150, 140, 22), (220, 200, 15), (260, 130, 18)), seed=0):
    """Soil-coloured background, green elliptical leaf, brown circular lesions.
    Returns (BGR image, leaf mask, lesion mask)."""
    rng = np.random.default_rng(seed)
    h, w = 300, 400
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (70, 95, 120)                                     # brownish-grey soil (BGR)
    leaf = np.zeros((h, w), np.uint8)
    cv2.ellipse(leaf, (200, 150), (150, 90), 15, 0, 360, (255,), -1)
    img[leaf > 0] = (45, 150, 60)                               # green leaf
    les = np.zeros((h, w), np.uint8)
    for x, y, r in lesions:
        cv2.circle(les, (x, y), r, (255,), -1)
    les &= leaf
    img[les > 0] = (35, 70, 120)                                # brown necrosis
    noise = rng.normal(0, 6, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, leaf, les


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max(1, (a | b).sum())


def test_fill_holes_fills_enclosed_hole_only():
    m = np.zeros((50, 50), np.uint8)
    cv2.rectangle(m, (10, 10), (40, 40), 255, -1)
    m[20:30, 20:30] = 0                      # enclosed hole
    m[0:5, 0:5] = 0
    out = fill_holes(m)
    assert out[25, 25] == 255
    assert out[2, 2] == 0                    # outside stays background


@pytest.mark.parametrize("method", ["exg", "hsv", "grabcut"])
def test_leaf_segmentation_recovers_leaf_including_lesions(method):
    img, leaf_gt, _ = synthetic_leaf()
    leaf, _ = segment_leaf(img, method)
    assert iou(leaf, leaf_gt) > 0.93, method


@pytest.mark.parametrize("method", ["hsv", "lab", "kmeans"])
def test_lesion_methods_estimate_infected_pct(method):
    img, leaf_gt, les_gt = synthetic_leaf()
    true_pct = 100 * (les_gt > 0).sum() / (leaf_gt > 0).sum()
    lesion, _, _ = segment_lesions(img, leaf_gt, method, get_profile("necrotic"))
    m = measure(leaf_gt, lesion)
    assert abs(m["infected_pct"] - true_pct) < 1.5, (method, m["infected_pct"], true_pct)
    assert m["n_spots"] == 3
    assert iou(lesion, les_gt) > 0.85


def test_healthy_leaf_is_near_zero():
    img, _, _ = synthetic_leaf(lesions=())
    r = run_pipeline(img, group="healthy")
    assert r["measurements"]["infected_pct"] < 1.0
    assert r["measurements"]["severity"] == "Healthy"


def test_full_pipeline_end_to_end():
    img, leaf_gt, les_gt = synthetic_leaf()
    true_pct = 100 * (les_gt > 0).sum() / (leaf_gt > 0).sum()
    r = run_pipeline(img, group="necrotic")
    assert abs(r["measurements"]["infected_pct"] - true_pct) < 2.0
    assert set(r["steps"]) >= {"original", "preprocessed", "leaf mask", "lesion candidates",
                               "cleaned lesions", "overlay"}


def test_symptom_group_mapping():
    assert symptom_group("Tomato___Early_blight") == "necrotic"
    assert symptom_group("apple rust") == "rust"
    assert symptom_group("Grape___powdery_mildew") == "powdery"
    assert symptom_group("tomato yellow leaf curl virus") == "chlorosis"
    assert symptom_group("Apple___healthy") == "healthy"
    assert symptom_group("something unknown") == "generic"


def test_severity_scale_and_uncertainty():
    assert severity_label(0.5) == "Healthy"
    assert severity_label(5) == "Mild"
    assert severity_label(15) == "Moderate"
    assert severity_label(60) == "Severe"
    assert severity_label(6, "rust") == "Mild"
    assert severity_label(8, "healthy") == "Healthy"   # CNN-healthy needs >= 10 % to overturn
    sev = {"confidence": {"min_confidence": 0.5, "top_k": 3, "rule": "top1"}, "default": [[101, "x"]]}
    r = resolve_from_prediction([("Tomato___Early_blight", 0.3), ("Tomato___Late_blight", 0.25)], sev)
    assert r["uncertain"] and r["group"] == "generic"
    sev["confidence"]["rule"] = "group"      # 0.30 + 0.25 = 0.55 necrotic -> confident group
    assert resolve_from_prediction([("Tomato___Early_blight", 0.3), ("Tomato___Late_blight", 0.25)], sev)["group"] == "necrotic"
    sev["confidence"]["rule"] = "none"
    r = resolve_from_prediction([("Tomato___Early_blight", 0.3), ("Apple___rust", 0.25)], sev)
    assert r["uncertain"] and r["group"] == "necrotic"


def test_plant_gate_rejects_non_plant_and_accepts_leaves():
    from src.imageproc.plant_check import check_plant
    # a hand-like skin-tone blob on a white wall: no green tissue at all
    wall = np.full((400, 300, 3), (225, 228, 230), np.uint8)
    cv2.ellipse(wall, (150, 200), (90, 150), 0, 0, 360, (60, 80, 120), -1)
    assert check_plant(wall)["status"] == "out_of_scope"
    # diseased and healthy synthetic leaves pass
    for lesions in (((150, 140, 22), (220, 200, 15)), ()):
        img, _, _ = synthetic_leaf(lesions=lesions)
        assert check_plant(img)["status"] == "ok"
