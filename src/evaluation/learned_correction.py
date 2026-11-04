"""
Learned severity correction (round 2): learn PlantSeg's annotation convention from classical features.

    python -m src.evaluation.learned_correction --val outputs/eval_v6/val/per_image.csv \
        --test outputs/eval_v6/test/per_image.csv --out outputs/learned

Why: many PlantSeg masks outline whole diseased REGIONS while colour methods find individual SPOTS, so
pure measurement under-estimates heavily infected leaves (and over-estimates lightly infected ones).
Inputs are only image-processing measurements (src/evaluation/features.py) + the CNN's predicted group.

Compared (all fitted on VALIDATION only; hyper-parameters by 5-fold CV on validation; TEST used once):
  A  classical      : the pipeline's infected % (no learning)
  B  constant       : validation median for every image (trivial baseline)
  C  xgb_regress    : XGBoost regressor  features -> annotated infected %
  D  xgb_spot_zone  : XGBoost classifier picks per image "spots" or "affected zone" (output is still a
                      direct measurement, just the right kind of one)
Metrics: MAE, median AE, severity-label agreement, within-one-class, Spearman rank correlation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor

from src.severity.grade import severity_label

ORDER = ["Healthy", "Mild", "Moderate", "Severe"]
GROUPS = ["necrotic", "chlorosis", "rust", "powdery", "generic", "healthy"]


def load(path):
    r = pd.read_csv(path)
    f = r[r.experiment == "features"].copy().reset_index(drop=True)
    for g in GROUPS:
        f[f"g_{g}"] = (f.pred_group == g).astype(int)
    # image-processing features + predicted symptom group only. CNN confidences are left out on purpose:
    # the demo uses temperature-calibrated probabilities, so they would differ between training and use.
    cols = [c for c in f.columns if c.startswith("f_") or c.startswith("g_")]
    return f, cols


def metrics(pred, gt, groups):
    p_lab = [severity_label(p, g) for p, g in zip(pred, groups)]
    g_lab = [severity_label(t, "generic") for t in gt]
    d = np.abs(np.array([ORDER.index(a) for a in p_lab]) - np.array([ORDER.index(b) for b in g_lab]))
    return {"MAE": float(np.mean(np.abs(pred - gt))), "medAE": float(np.median(np.abs(pred - gt))),
            "agreement": float(np.mean(d == 0)), "within_one": float(np.mean(d <= 1)),
            "spearman": float(spearmanr(pred, gt).correlation)}


def cv_predict(make, X, y, seed=0):
    out = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = make(); m.fit(X[tr], y[tr]); out[te] = m.predict(X[te])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="outputs/learned")
    a = ap.parse_args()
    v, cols = load(a.val)
    t, _ = load(a.test)
    Xv, yv, Xt, yt = v[cols].to_numpy(float), v.gt_pct.to_numpy(float), t[cols].to_numpy(float), t.gt_pct.to_numpy(float)
    print(f"val n={len(v)}  test n={len(t)}  features={len(cols)}")

    # hyper-parameters by 5-fold CV on validation (MAE objective)
    grid = list(itertools.product([2, 3, 4], [100, 300], [0.05]))
    # each candidate: (CV MAE, depth, n_estimators, learning_rate); the lowest CV MAE wins
    reg_scores = []
    for depth, n, lr in grid:
        mk = lambda: XGBRegressor(max_depth=depth, n_estimators=n, learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                                  objective="reg:absoluteerror", random_state=0, n_jobs=4)
        reg_scores.append((float(np.mean(np.abs(np.clip(cv_predict(mk, Xv, yv), 0, 100) - yv))), depth, n, lr))
    best_r = min(reg_scores)
    # spot-vs-zone target: which measurement is closer to the annotation
    zone_v, spot_v = v.f_pct_zone10.to_numpy(float), v.f_pct_final.to_numpy(float)
    zbetter = (np.abs(zone_v - yv) < np.abs(spot_v - yv)).astype(int)
    clf_scores = []
    for depth, n, lr in grid:
        mk = lambda: XGBClassifier(max_depth=depth, n_estimators=n, learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                                   random_state=0, n_jobs=4)
        pz = cv_predict(mk, Xv, zbetter) > 0.5
        clf_scores.append((float(np.mean(np.abs(np.where(pz, zone_v, spot_v) - yv))), depth, n, lr))
    best_c = min(clf_scores)
    print(f"CV on val: regressor MAE {best_r[0]:.2f} (depth={best_r[1]}, n={best_r[2]}); "
          f"spot/zone switch MAE {best_c[0]:.2f} (depth={best_c[1]}, n={best_c[2]})")

    reg = XGBRegressor(max_depth=best_r[1], n_estimators=best_r[2], learning_rate=best_r[3], subsample=0.8,
                       colsample_bytree=0.8, objective="reg:absoluteerror", random_state=0, n_jobs=4).fit(Xv, yv)
    clf = XGBClassifier(max_depth=best_c[1], n_estimators=best_c[2], learning_rate=best_c[3], subsample=0.8,
                        colsample_bytree=0.8, random_state=0, n_jobs=4).fit(Xv, zbetter)
    # trivial baseline: median infected % of the DISEASED validation images (16 %); including healthy
    # leaves (0 %) would lower it to ~12 % and give a weaker baseline, so we use the stronger one
    med = float(v.loc[v.source == "plantseg", "gt_pct"].median())
    preds = {
        "A classical (no learning)": t.f_pct_final.to_numpy(float),
        "B constant (val median)": np.full(len(t), med),
        "C xgb regression": np.clip(reg.predict(Xt), 0, 100),
        "D xgb spot/zone switch": np.where(clf.predict(Xt) == 1, t.f_pct_zone10.to_numpy(float), t.f_pct_final.to_numpy(float)),
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for subset, mask in [("all test (diseased + healthy)", np.ones(len(t), bool)),
                         ("diseased only", (t.source == "plantseg").to_numpy()),
                         ("healthy only", (t.source != "plantseg").to_numpy())]:
        for name, p in preds.items():
            rows.append({"subset": subset, "method": name, **metrics(p[mask], yt[mask], t.pred_group.to_numpy()[mask])})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv(out / "learned_correction_test.csv", index=False)
    imp = pd.Series(dict(zip(cols, reg.feature_importances_.tolist()))).sort_values(ascending=False)
    print("\nregressor feature importance (top 10):\n" + imp.head(10).round(3).to_string())
    imp.to_csv(out / "feature_importance.csv")
    reg.save_model(out / "xgb_regressor.json"); clf.save_model(out / "xgb_spot_zone.json")
    (out / "meta.json").write_text(json.dumps({"features": cols, "regressor": best_r, "switch": best_c, "val_median": med}))


if __name__ == "__main__":
    main()
