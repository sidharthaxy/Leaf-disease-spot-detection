"""
CPU inference for the demo / evaluation on the Mac: ONNX Runtime (no PyTorch needed for prediction).

    clf = OnnxClassifier("outputs/<job>/model")
    clf.predict(img_bgr, k=3) -> {"topk": [(label, prob), ...], "plant": (plant, prob), ...}
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prepare(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Same as the eval transform used in training: resize short side to 1.14*size, centre-crop."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = int(size * 1.14) / min(h, w)
    rgb = cv2.resize(rgb, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    rgb = rgb[y:y + size, x:x + size].astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]  # 1x3xHxW


class OnnxClassifier:
    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        self.maps = json.loads((model_dir / "label_maps.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2  # keep the Mac responsive
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.size = self.sess.get_inputs()[0].shape[-1]
        if not isinstance(self.size, int):
            self.size = 320

    def predict(self, img_bgr: np.ndarray, k: int = 3) -> dict:
        label_p, plant_p = self.sess.run(None, {"image": prepare(img_bgr, self.size)})
        label_p, plant_p = label_p[0], plant_p[0]
        top = np.argsort(-label_p)[:k]
        pl = int(plant_p.argmax())
        return {
            "topk": [(self.maps["labels"][i], float(label_p[i])) for i in top],
            "plant": (self.maps["plants"][pl], float(plant_p[pl])),
        }


class EnsembleClassifier:
    """The final classifier chosen by src/classifier/ensemble.py (configs/inference.yaml):
    average of one or more ONNX models, optional mirror-image TTA, optional plant-head re-weighting
        score(label) = p(label) * p(plant of label)^alpha
    """

    def __init__(self, cfg_path: str | Path):
        import yaml

        root = Path(cfg_path).resolve().parents[1]
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.models = [OnnxClassifier(root / d) for d in cfg["model_dirs"]]
        self.tta = bool(cfg.get("tta", False))
        self.alpha = float(cfg.get("plant_alpha", 0.0))
        self.temperature = float(cfg.get("temperature", 1.0))   # fitted on validation (calibrate.py)
        self.maps = self.models[0].maps
        self.l2p = np.array([self.maps["plants"].index(self.maps["label_to_plant"][l]) for l in self.maps["labels"]])
        self.model_dirs = cfg["model_dirs"]

    def _probs(self, m: OnnxClassifier, img_bgr: np.ndarray):
        views = [img_bgr, cv2.flip(img_bgr, 1)] if self.tta else [img_bgr]
        outs = [m.sess.run(None, {"image": prepare(v, m.size)}) for v in views]
        return np.mean([o[0][0] for o in outs], 0), np.mean([o[1][0] for o in outs], 0)

    def predict(self, img_bgr: np.ndarray, k: int = 3) -> dict:
        ps = [self._probs(m, img_bgr) for m in self.models]
        lp, pp = np.mean([p[0] for p in ps], 0), np.mean([p[1] for p in ps], 0)
        if self.alpha > 0:
            lp = lp * pp[self.l2p] ** self.alpha
            lp = lp / lp.sum()
        if self.temperature != 1.0:                              # p^(1/T), renormalised
            lp = np.power(np.clip(lp, 1e-12, 1), 1.0 / self.temperature)
            lp = lp / lp.sum()
        top = np.argsort(-lp)[:k]
        pl = int(pp.argmax())
        return {"topk": [(self.maps["labels"][i], float(lp[i])) for i in top],
                "plant": (self.maps["plants"][pl], float(pp[pl]))}
