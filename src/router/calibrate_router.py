"""
Fit the router's temperature and routing thresholds on VALIDATION, then report routing.

    python -m src.router.calibrate_router --run outputs/router/<job>/model --manifest data/router/manifest.csv
    python -m src.router.calibrate_router --run ... --manifest ... --test     # routing-only TEST report (once)

1. Temperature T: minimises val NLL (one scalar; accuracy unchanged).
2. Thresholds per non-crop route group (palette / other_houseplant / non_plant): grid search on val.
   Constraint (primary): val crop images routed away from the crop pipeline <= --max-crop-misroute (1%).
   Objective (secondary): mean over the three groups of "non-crop val images routed correctly".
   Crop misrouting is additive over the groups, so the joint grid is cheap.
3. Writes models/router/router_config.json and reports/router_val.md (+ reports/router_test_routing.md with --test).
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.router.router import decide, route_group, softmax_T

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("palette", "other_houseplant", "non_plant")
EXPECTED = {"crop": ("crop", "crop_uncertain"), "other_houseplant": ("other_houseplant",), "non_plant": ("out_of_scope",)}


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(e)


def routes(P, classes, th):
    """green share is not used here: the <0.5 % gate is the existing behaviour, unchanged by the router."""
    return [decide(dict(zip(classes, p)), 100.0, th) for p in P]


def correct(r, label):
    if label in EXPECTED:
        return r["route"] in EXPECTED[label]
    return r["route"] == "palette" and r["species"] == label


def summarise(P, y, classes, th, srcs=None):
    lab = np.array(classes)[y]
    rs = routes(P, classes, th)
    ok = np.array([correct(r, l) for r, l in zip(rs, lab)])
    out_of_crop = np.array([r["route"] not in ("crop", "crop_uncertain") for r in rs])
    is_crop = lab == "crop"
    grp = np.array([route_group(l) for l in lab])
    s = {"n": int(len(y)), "crop_n": int(is_crop.sum()),
         "crop_misroute_pct": 100 * float(out_of_crop[is_crop].mean()) if is_crop.any() else float("nan"),
         "crop_misroute_n": int(out_of_crop[is_crop].sum()),
         "top1_acc": float((P.argmax(1) == y).mean()), "macro_f1": float(f1_score(y, P.argmax(1), average="macro")),
         "ece": ece(P, y)}
    for g in GROUPS:
        m = grp == g
        if m.any():
            s[f"{g}_routed_correctly_pct"] = 100 * float(ok[m].mean())
            s[f"{g}_kept_out_of_crop_pct"] = 100 * float(out_of_crop[m].mean())
    if srcs is not None:
        green = np.array([str(x).startswith("wikimedia") for x in srcs]) & (lab == "non_plant")
        if green.any():
            s["green_nonplant_n"] = int(green.sum())
            s["green_nonplant_stopped_pct"] = 100 * float(np.array([r["route"] == "out_of_scope" for r in rs])[green].mean())
    per = {}
    for c in classes:
        m = lab == c
        if m.any():
            per[c] = {"n": int(m.sum()), "routed_correctly_pct": round(100 * float(ok[m].mean()), 1),
                      "kept_out_of_crop_pct": round(100 * float(out_of_crop[m].mean()), 1)}
    return s, per, rs


def search(P, y, classes, max_mis):
    lab = np.array(classes)[y]
    top = P.argmax(1)
    pmax = P.max(1)
    tgrp = np.array([route_group(classes[t]) for t in top])
    is_crop = lab == "crop"
    grid = np.round(np.concatenate([np.arange(0.30, 0.90, 0.05), np.arange(0.90, 0.995, 0.01)]), 3)
    # per group & threshold: crop images that would leave, and correct routings of that group's images
    mis = {g: np.array([((tgrp == g) & (pmax >= t) & is_crop).sum() for t in grid]) for g in GROUPS}
    good = {}
    for g in GROUPS:
        m = np.array([route_group(l) == g for l in lab])
        if g == "palette":
            hit = (np.array(classes)[top] == lab)
        else:
            hit = tgrp == g
        good[g] = np.array([((hit & (pmax >= t) & m).sum() / max(1, m.sum())) for t in grid])
    best = None
    for idx in itertools.product(range(len(grid)), repeat=3):
        nmis = sum(mis[g][i] for g, i in zip(GROUPS, idx))
        if 100 * nmis / max(1, is_crop.sum()) > max_mis:
            continue
        score = float(np.mean([good[g][i] for g, i in zip(GROUPS, idx)]))
        key = (round(score, 6), -nmis, sum(idx))
        if best is None or key > best[0]:
            best = (key, {g: float(grid[i]) for g, i in zip(GROUPS, idx)})
    return best[1]


def md_table(per):
    lines = ["| class | n | routed correctly % | kept out of crop pipeline % |", "|---|---:|---:|---:|"]
    lines += [f"| {c} | {v['n']} | {v['routed_correctly_pct']} | {v['kept_out_of_crop_pct']} |" for c, v in per.items()]
    return "\n".join(lines)


def md_confusion(P, y, classes):
    cm = confusion_matrix(y, P.argmax(1), labels=range(len(classes)))
    short = [c[:10] for c in classes]
    lines = ["| true \\ pred | " + " | ".join(short) + " |", "|---|" + "---:|" * len(classes)]
    lines += [f"| {classes[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with logits_val.npz, logits_test.npz, label_map.json, model.onnx")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "router" / "manifest.csv"))
    ap.add_argument("--max-crop-misroute", type=float, default=1.0, help="percent of VAL crop images")
    ap.add_argument("--test", action="store_true", help="also write the routing-only TEST report")
    a = ap.parse_args()
    run = Path(a.run)
    classes = json.loads((run / "label_map.json").read_text())["classes"]
    man = pd.read_csv(a.manifest) if Path(a.manifest).exists() else None
    src_of = dict(zip(man.image_path, man.source)) if man is not None else {}

    v = np.load(run / "logits_val.npz", allow_pickle=True)
    Ts = np.exp(np.linspace(np.log(0.3), np.log(5.0), 300))
    T = float(Ts[np.argmin([nll(softmax_T(v["logits"], t), v["y"]) for t in Ts])])
    Pv = softmax_T(v["logits"], T)
    th = search(Pv, v["y"], classes, a.max_crop_misroute)
    sv, perv, _ = summarise(Pv, v["y"], classes, th, [src_of.get(p) for p in v["paths"]])
    sv["ece_before_T"] = ece(softmax_T(v["logits"], 1.0), v["y"])

    out = ROOT / "models" / "router"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("model.onnx", "label_map.json"):
        shutil.copy(run / f, out / f)
    cfg = {"temperature": round(T, 4), "thresholds": th, "reject_below_pct": 0.5,
           "chosen_on": "validation", "max_crop_misroute_pct_val": a.max_crop_misroute, "val": sv}
    (out / "router_config.json").write_text(json.dumps(cfg, indent=1))
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    (rep / "router_val.md").write_text(
        f"# Router, validation split (temperature and thresholds chosen here)\n\n"
        f"Temperature T = {T:.4f} (ECE {sv['ece_before_T']:.3f} -> {sv['ece']:.3f}). Thresholds: {th}.\n\n"
        f"```json\n{json.dumps(sv, indent=1)}\n```\n\n## Routing per class\n\n{md_table(perv)}\n\n"
        f"## Confusion matrix (top-1, before thresholds)\n\n{md_confusion(Pv, v['y'], classes)}\n")
    print(json.dumps(cfg, indent=1))

    if a.test:
        t = np.load(run / "logits_test.npz", allow_pickle=True)
        Pt = softmax_T(t["logits"], T)
        st, pert, rs = summarise(Pt, t["y"], classes, th, [src_of.get(p) for p in t["paths"]])
        lab = np.array(classes)[t["y"]]
        moved = [(p, r["route"], r["reason"]) for p, r, l in zip(t["paths"], rs, lab)
                 if l == "crop" and r["route"] not in ("crop", "crop_uncertain")]
        (rep / "router_test_routing.md").write_text(
            "# Router, TEST split: routing-only evaluation\n\n"
            "**Crop rows: PlantSeg/PlantDoc test split, third use, routing only, nothing tuned on it.** "
            "Temperature and thresholds come from validation (reports/router_val.md). The ornamental, "
            "houseplant and non-plant test images are held-out web images (first use).\n\n"
            f"```json\n{json.dumps(st, indent=1)}\n```\n\n## Routing per class\n\n{md_table(pert)}\n\n"
            f"## Confusion matrix (top-1, before thresholds)\n\n{md_confusion(Pt, t['y'], classes)}\n\n"
            f"## Crop test images moved out of the crop pipeline ({len(moved)})\n\n"
            "The crop pipeline itself is unchanged, so every other crop test image gets byte-identical results; "
            "only these images would differ from the published numbers.\n\n"
            + "\n".join(f"- `{p}` -> {r} ({why})" for p, r, why in moved) + "\n")
        print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
