"""
Router inference + routing decision (runs on the Mac CPU via ONNX Runtime).

    from src.router.router import Router
    r = Router()                      # loads models/router/{model.onnx, label_map.json, router_config.json}
    out = r.route(img_bgr, green_pct) # {"route": ..., "top3": [...], "probs": {...}, "reason": ...}

Routes (see the routing diagram in the README):
    "crop"              -> existing crop pipeline, unchanged (Stages A-D)
    "crop_uncertain"    -> existing crop pipeline + warning "the plant may be outside our crop classes"
    "palette"           -> palette branch for out["species"]
    "other_houseplant"  -> recognised ornamental houseplant that this system doesn't assess
    "out_of_scope"      -> stop (non-plant, or no green tissue and not a known palette species)

Design rule: crop images must never be pushed out of the pipeline that handles them today. Every
non-crop route therefore needs (a) the router's top-1 to be that class AND (b) a calibrated
probability above a threshold chosen on VALIDATION so that <= ~1% of val crop images leave the crop
pipeline. Anything not confident falls back to the crop pipeline (with a warning), exactly as today.
The one exception is the existing green-share gate: below 0.5% green the crop CNN was never run, and
that stays so (only a confident palette species can rescue such an image).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "router"
NON_PALETTE = {"crop", "other_houseplant", "non_plant"}


def softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def route_group(cls: str) -> str:
    return cls if cls in NON_PALETTE else "palette"


def decide(probs: dict, green_pct: float, thresholds: dict, reject_below_pct: float = 0.5) -> dict:
    """Pure routing rule (unit-tested with mock router outputs).

    probs: {class: calibrated probability}; thresholds: {"palette": t, "other_houseplant": t, "non_plant": t}.
    """
    top = max(probs, key=probs.get)
    p = float(probs[top])
    group = route_group(top)
    confident = group != "crop" and p >= thresholds[group]
    if green_pct < reject_below_pct:  # same gate as before: the crop CNN is not run on these
        if group == "palette" and confident:
            return {"route": "palette", "species": top, "reason": f"low green ({green_pct:.2f}%), recognised as {top} (p={p:.2f})"}
        return {"route": "out_of_scope", "species": None, "reason": f"almost no green leaf tissue ({green_pct:.2f}%)"}
    if group == "crop":
        return {"route": "crop", "species": None, "reason": f"router: crop (p={p:.2f})"}
    if confident:
        if group == "palette":
            return {"route": "palette", "species": top, "reason": f"router: {top} (p={p:.2f})"}
        if group == "other_houseplant":
            return {"route": "other_houseplant", "species": None, "reason": f"router: other houseplant (p={p:.2f})"}
        return {"route": "out_of_scope", "species": None, "reason": f"router: not a plant (p={p:.2f})"}
    return {"route": "crop_uncertain", "species": None,
            "reason": f"router not confident (top: {top}, p={p:.2f} < {thresholds[group]:.2f}); using the crop pipeline"}


def preprocess(img_bgr: np.ndarray, size: int, mean, std) -> np.ndarray:
    """Same as the training eval transform: Resize(short side = 1.14*size) -> CenterCrop(size) -> normalise."""
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))
    s = int(size * 1.14)
    w, h = img.size
    if w <= h:
        nw, nh = s, int(round(h * s / w))
    else:
        nw, nh = int(round(w * s / h)), s
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
    img = img.crop((left, top, left + size, top + size))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)
    return x.transpose(2, 0, 1)[None]


class Router:
    def __init__(self, model_dir: str | Path = MODEL_DIR):
        import onnxruntime as ort

        d = Path(model_dir)
        self.meta = json.loads((d / "label_map.json").read_text())
        self.cfg = json.loads((d / "router_config.json").read_text())
        self.classes = self.meta["classes"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])

    def probs(self, img_bgr: np.ndarray) -> dict:
        x = preprocess(img_bgr, self.meta["img_size"], self.meta["mean"], self.meta["std"])
        logits = self.sess.run(None, {"image": x})[0][0]
        p = softmax_T(logits, self.cfg["temperature"])
        return {c: float(v) for c, v in zip(self.classes, p)}

    def route(self, img_bgr: np.ndarray, green_pct: float) -> dict:
        pr = self.probs(img_bgr)
        out = decide(pr, green_pct, self.cfg["thresholds"], self.cfg.get("reject_below_pct", 0.5))
        out["top3"] = [{"class": c, "p": round(pr[c], 4)} for c in sorted(pr, key=pr.get, reverse=True)[:3]]
        out["probs"] = pr
        return out
