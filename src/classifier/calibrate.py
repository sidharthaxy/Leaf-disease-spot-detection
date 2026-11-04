"""
Temperature scaling of the final ensemble (fitted on VALIDATION, reported on TEST).

    python -m src.classifier.calibrate

p_T(c) ∝ p(c)^(1/T): T > 1 softens over-confident probabilities, T < 1 sharpens them. One scalar, so
accuracy is unchanged; only the confidence values (and therefore the "uncertain" flag) change.
Expected calibration error (ECE, 15 bins) measures |confidence - accuracy| averaged over bins.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def ens_probs(dirs, split):
    d = [np.load(ROOT / x / f"probs_{split}.npz", allow_pickle=True) for x in dirs]
    return np.mean([x["label_probs_plain"].astype(np.float64) for x in d], 0), d[0]["y"]


def scale(p, T):
    q = np.power(np.clip(p, 1e-12, 1), 1.0 / T)
    return q / q.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(e)


def main():
    cfg_path = ROOT / "configs" / "inference.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    pv, yv = ens_probs(cfg["model_dirs"], "val")
    pt, yt = ens_probs(cfg["model_dirs"], "test")
    Ts = np.exp(np.linspace(np.log(0.3), np.log(3.0), 200))
    T = float(Ts[np.argmin([nll(scale(pv, t), yv) for t in Ts])])
    rep = {"T": T,
           "val": {"ece_before": ece(pv, yv), "ece_after": ece(scale(pv, T), yv)},
           "test": {"ece_before": ece(pt, yt), "ece_after": ece(scale(pt, T), yt),
                    "nll_before": nll(pt, yt), "nll_after": nll(scale(pt, T), yt),
                    "uncertain_before": float((pt.max(1) < 0.5).mean()), "uncertain_after": float((scale(pt, T).max(1) < 0.5).mean()),
                    "acc_when_confident_after": float((scale(pt, T).argmax(1) == yt)[scale(pt, T).max(1) >= 0.5].mean()),
                    "acc_when_uncertain_after": float((scale(pt, T).argmax(1) == yt)[scale(pt, T).max(1) < 0.5].mean())}}
    print(json.dumps(rep, indent=2))
    cfg["temperature"] = round(T, 4)
    cfg_path.write_text("# Final classifier, chosen on the validation split by src/classifier/ensemble.py\n"
                        "# temperature: fitted on validation by src/classifier/calibrate.py\n" + yaml.safe_dump(cfg))
    (ROOT / "outputs" / "ensemble" / "calibration.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
