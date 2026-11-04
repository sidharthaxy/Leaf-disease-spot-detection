"""
Choose the final classifier from saved probabilities (no GPU needed; runs on the Mac in seconds).

    python -m src.classifier.ensemble outputs/effv2 outputs/convnext --out outputs/ensemble --write-config --write-predictions

Options compared on the VALIDATION split, the winner reported once on TEST:
  - single model, plain vs test-time augmentation (TTA: average of image + mirror image)
  - ensembles (average of probabilities) of every subset of models
  - plant-head re-weighting:  score(label) = p(label) * p(plant of label)^alpha
    (alpha = 0: off; the plant head is trained on an easier 34-way task and is more accurate,
     so it can veto diseases of the wrong plant)
Writes configs/inference.yaml with the chosen setting, used by the demo.
"""

from __future__ import annotations

import argparse
import itertools
import json
from typing import Any
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]


def load(model_dir: Path, split: str):
    d = np.load(model_dir / f"probs_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def combine(runs, use_tta: bool, alpha: float, label_to_plant_idx: np.ndarray):
    key = "tta" if use_tta else "plain"
    lp = np.mean([r[f"label_probs_{key}"].astype(np.float32) for r in runs], axis=0)
    pp = np.mean([r[f"plant_probs_{key}"].astype(np.float32) for r in runs], axis=0)
    if alpha > 0:
        lp = lp * pp[:, label_to_plant_idx] ** alpha
        lp = lp / lp.sum(1, keepdims=True)
    return lp, pp


def metrics(lp, pp, y, yp):
    top5 = np.argsort(-lp, 1)[:, :5]
    return {"top1": float((lp.argmax(1) == y).mean()), "top5": float((top5 == y[:, None]).any(1).mean()),
            "macro_f1": float(f1_score(y, lp.argmax(1), average="macro", zero_division=0)),
            "plant_acc": float((pp.argmax(1) == yp).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default="outputs/ensemble")
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--write-predictions", action="store_true", help="write {val,test}_predictions.csv for run_eval")
    a = ap.parse_args()
    dirs = [Path(m) for m in a.models]
    maps = json.loads((dirs[0] / "label_maps.json").read_text())
    for d in dirs[1:]:
        assert json.loads((d / "label_maps.json").read_text())["labels"] == maps["labels"], "label maps differ"
    l2p = np.array([maps["plants"].index(maps["label_to_plant"][l]) for l in maps["labels"]])
    val = [load(d, "val") for d in dirs]
    test = [load(d, "test") for d in dirs]
    y_v, yp_v, y_t, yp_t = val[0]["y"], val[0]["yp"], test[0]["y"], test[0]["yp"]

    rows: list[dict[str, Any]] = []
    for n in range(1, len(dirs) + 1):
        for subset in itertools.combinations(range(len(dirs)), n):
            for tta, alpha in itertools.product([False, True], [0.0, 0.5, 1.0, 2.0]):
                lp, pp = combine([val[i] for i in subset], tta, alpha, l2p)
                rows.append({"models": [dirs[i].parent.name for i in subset], "subset": subset, "tta": tta,
                             "alpha": alpha, **{f"val_{k}": v for k, v in metrics(lp, pp, y_v, yp_v).items()}})
    rows.sort(key=lambda r: (r["val_macro_f1"], r["val_top1"]), reverse=True)
    print("top configurations on VALIDATION:")
    for r in rows[:8]:
        print(f"  {r['models']} tta={r['tta']} alpha={r['alpha']}  "
              f"top1={r['val_top1']:.4f} top5={r['val_top5']:.4f} macroF1={r['val_macro_f1']:.4f} plant={r['val_plant_acc']:.4f}")
    # reference rows: each single model, plain
    best = rows[0]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {"chosen_on_val": best, "test": {}}
    for name, subset, tta, alpha in [("chosen", best["subset"], best["tta"], best["alpha"])] + \
            [(f"single:{dirs[i].parent.name}", (i,), False, 0.0) for i in range(len(dirs))]:
        lp, pp = combine([test[i] for i in subset], tta, alpha, l2p)
        report["test"][name] = metrics(lp, pp, y_t, yp_t)
    print("\nTEST (reported once):")
    for k, v in report["test"].items():
        print(f"  {k:60s} " + "  ".join(f"{m}={x:.4f}" for m, x in v.items()))
    (out / "ensemble_report.json").write_text(json.dumps(report, indent=2, default=str))
    if a.write_predictions:
        # per-image predictions of the chosen classifier, consumed by src.evaluation.run_eval --predictions
        import pandas as pd

        from src.severity.grade import symptom_group

        groups = [symptom_group(l) for l in maps["labels"]]
        G = sorted(set(groups)); gi = np.array([G.index(g) for g in groups])
        for split, runs in (("val", val), ("test", test)):
            lp, _ = combine([runs[i] for i in best["subset"]], best["tta"], best["alpha"], l2p)
            gp = np.stack([lp[:, gi == k].sum(1) for k in range(len(G))], 1)
            pd.DataFrame({"image_path": runs[0]["image_path"], "label": [maps["labels"][i] for i in runs[0]["y"]],
                          "pred_label": [maps["labels"][i] for i in lp.argmax(1)], "confidence": lp.max(1),
                          "pred_group": [G[i] for i in gp.argmax(1)], "group_confidence": gp.max(1)}
                         ).to_csv(out / f"{split}_predictions.csv", index=False)
        print(f"wrote {out}/val_predictions.csv and test_predictions.csv")
    if a.write_config:
        cfg = {"model_dirs": [str(dirs[i].relative_to(ROOT)) if dirs[i].is_absolute() else str(dirs[i]) for i in best["subset"]],
               "tta": bool(best["tta"]), "plant_alpha": float(best["alpha"])}
        (ROOT / "configs" / "inference.yaml").write_text(
            "# Final classifier, chosen on the validation split by src/classifier/ensemble.py\n" + yaml.safe_dump(cfg))
        print("\nwrote configs/inference.yaml:", cfg)


if __name__ == "__main__":
    main()
