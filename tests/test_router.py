"""Routing decisions on mock router outputs (no model needed)."""

import numpy as np

from src.router.router import decide, preprocess, softmax_T

TH = {"palette": 0.8, "other_houseplant": 0.7, "non_plant": 0.9}


def mk(top, p, others=("crop", "croton", "other_houseplant", "non_plant")):
    rest = [c for c in others if c != top]
    return {top: p, **{c: (1 - p) / len(rest) for c in rest}}


def test_crop_top1_goes_to_crop_pipeline_at_any_confidence():
    assert decide(mk("crop", 0.3), 40.0, TH)["route"] == "crop"
    assert decide(mk("crop", 0.99), 40.0, TH)["route"] == "crop"


def test_confident_palette_species():
    r = decide(mk("croton", 0.95), 20.0, TH)
    assert r["route"] == "palette" and r["species"] == "croton"


def test_unconfident_non_crop_falls_back_to_crop_pipeline():
    for top in ("croton", "other_houseplant", "non_plant"):
        assert decide(mk(top, 0.6), 30.0, TH)["route"] == "crop_uncertain"


def test_confident_houseplant_and_non_plant():
    assert decide(mk("other_houseplant", 0.75), 30.0, TH)["route"] == "other_houseplant"
    assert decide(mk("non_plant", 0.95), 5.0, TH)["route"] == "out_of_scope"


def test_low_green_gate_kept_unless_confident_palette():
    assert decide(mk("crop", 0.99), 0.2, TH)["route"] == "out_of_scope"
    assert decide(mk("non_plant", 0.5), 0.2, TH)["route"] == "out_of_scope"
    assert decide(mk("croton", 0.5), 0.2, TH)["route"] == "out_of_scope"
    r = decide(mk("croton", 0.9), 0.2, TH)
    assert r["route"] == "palette" and r["species"] == "croton"


def test_threshold_boundary_is_inclusive():
    assert decide(mk("croton", 0.8), 10.0, TH)["route"] == "palette"


def test_softmax_temperature_keeps_argmax_and_sums_to_one():
    z = np.array([2.0, 0.5, -1.0])
    for T in (0.5, 1.0, 2.0):
        p = softmax_T(z, T)
        assert abs(p.sum() - 1) < 1e-9 and p.argmax() == 0
    assert softmax_T(z, 0.5).max() > softmax_T(z, 2.0).max()


def test_preprocess_shape_and_normalisation():
    img = np.full((300, 500, 3), 128, np.uint8)
    x = preprocess(img, 224, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    assert x.shape == (1, 3, 224, 224) and abs(float(x.mean()) - (128 / 255 - 0.5) / 0.5) < 1e-3
