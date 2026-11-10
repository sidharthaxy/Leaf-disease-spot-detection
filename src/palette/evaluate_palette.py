"""
Palette branch evaluation on the held-out TEST images of the palette species (first use of this split).

    python -m src.palette.evaluate_palette --data data/router --plantseg data/dev_val [--old-pipeline]

  6.4  healthy false-positive area per species: mean / median abnormal %, % labelled "Normal"
  6.5  damage detection on SYNTHETIC damage (real PlantSeg val lesions pasted onto the same test photos' normal
       tissue until they cover 5 / 10 / 20 % of the plant): image-level ROC-AUC (healthy vs damaged, score =
       abnormal %), accuracy at the val cut-off, pixel Dice
  6.6  old crop pipeline vs palette branch on the same healthy test images (variegated + non-green species):
       what the crop system would have graded (CNN + Stage B + severity) vs the palette abnormal %
  6.7  timing on this machine's CPU: router + palette branch per image
Writes reports/palette_test.md, reports/palette_test_per_image.csv and a review sheet
reports/figures/palette_review/<species>.jpg (healthy | synthetic damage, with the predicted %).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.palette.palette import Palette, run_palette
from src.palette.synthetic import build_lesion_bank, paste_to_target, rough_leaf_prior

ROOT = Path(__file__).resolve().parents[2]
LEVELS = (5, 10, 20)   # synthetic damage: % of the plant region covered by pasted lesions


def dice(a, b):
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def old_pipeline():
    """The unchanged crop system: ensemble CNN -> symptom group -> Stage B -> severity."""
    from src.classifier.infer import EnsembleClassifier
    from src.imageproc.pipeline import run_pipeline
    from src.severity.grade import resolve_from_prediction

    clf = EnsembleClassifier(ROOT / "configs" / "inference.yaml")

    def run(img):
        pred = clf.predict(img, k=len(clf.maps["labels"]))
        res = resolve_from_prediction(pred["topk"])
        m = run_pipeline(img, group=res["group"])["measurements"]
        return {"old_label": res["label"], "old_conf": round(float(res["confidence"]), 3),
                "old_infected_pct": round(float(m["infected_pct"]), 2), "old_severity": m["severity"]}
    return run


def tile(img, text, h=150):
    t = cv2.resize(img, (max(1, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)
    t = cv2.resize(t, (h, h)) if t.shape[1] > 1.6 * h else t
    bar = np.full((22, t.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, t])


def grid(tiles, cols=6):
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1] + 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    rows = [np.hstack(tiles[i:i + cols] + [np.full_like(tiles[0], 255)] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


VARIEGATED = ("golden_pothos", "dieffenbachia")


def old_vs_palette(df: pd.DataFrame) -> pd.DataFrame:
    """6.6: what the crop system graded vs the palette abnormal % on the same healthy test photos."""
    def row(name, d):
        sev = d.old_severity.str.lower()
        return {"healthy test photos": name, "n": len(d),
                "crop pipeline: mean infected %": round(d.old_infected_pct.mean(), 1),
                "crop pipeline: graded diseased": f"{100 * (sev != 'healthy').mean():.1f} %",
                "crop pipeline: graded Severe": f"{100 * (sev == 'severe').mean():.1f} %",
                "palette: mean abnormal %": round(d.healthy_abnormal_pct.mean(), 1),
                "palette: not 'Normal'": f"{100 * (d.healthy_label != 'Normal').mean():.1f} %"}
    var = df.species.isin(VARIEGATED)
    return pd.DataFrame([row("variegated green (pothos, dieffenbachia)", df[var]),
                         row(f"non-green ({df[~var].species.nunique()} species)", df[~var]),
                         row(f"all {df.species.nunique()} species", df)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "router"))
    ap.add_argument("--plantseg", default=str(ROOT / "data" / "dev_val"))
    ap.add_argument("--old-pipeline", action="store_true", help="also run the crop system on the same images (6.6)")
    ap.add_argument("--router", action="store_true", help="time the router too (needs models/router)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    data = Path(a.data)
    man = pd.read_csv(data / "manifest.csv")
    species = sorted(p.stem for p in (ROOT / "models" / "palettes").glob("*.json"))
    bank = build_lesion_bank(a.plantseg, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    old = old_pipeline() if a.old_pipeline else None
    router = None
    if a.router:
        from src.router.router import Router
        router = Router()
    rows, t_pal, t_router = [], [], []
    figs = ROOT / "reports" / "figures" / "palette_review"
    figs.mkdir(parents=True, exist_ok=True)
    for sp in species:
        pal = Palette.load(sp)
        tiles = []
        for p in man[(man.label == sp) & (man.split == "test")].image_path:
            img = cv2.imread(str(data / p))
            if img is None:
                continue
            if router is not None:
                t0 = time.perf_counter()
                router.probs(img)
                t_router.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            rh = run_palette(img, pal)
            t_pal.append(time.perf_counter() - t0)
            pre = rh["steps"]["preprocessed"]
            place = rough_leaf_prior(pre) & rh["normal"]          # lesions go on the plant's normal tissue
            row = {"species": sp, "image_path": p, "healthy_abnormal_pct": rh["measurements"]["abnormal_pct"],
                   "healthy_label": rh["measurements"]["label"]}
            rd10 = rh                                              # replaced by the 10 % damage run below
            for lvl in LEVELS:
                dmg, gt = paste_to_target(pre, place, bank, rng, lvl / 100)
                rd = run_palette(dmg, pal)
                row.update({f"d{lvl}_abnormal_pct": rd["measurements"]["abnormal_pct"], f"d{lvl}_label": rd["measurements"]["label"],
                            f"d{lvl}_gt_pct": round(100 * (gt > 0).sum() / max(1, (rd["leaf_mask"] > 0).sum()), 2),
                            f"d{lvl}_dice": round(dice(rd["abnormal_mask"], gt), 3)})
                if lvl == 10:
                    rd10 = rd
            if old is not None:
                row.update(old(img))
            rows.append(row)
            if len(tiles) < 24:
                tiles.append(tile(rh["steps"]["overlay"], f"{Path(p).stem[:14]} H {row['healthy_abnormal_pct']:.1f}%"))
                tiles.append(tile(rd10["steps"]["overlay"], f"synthetic {row['d10_abnormal_pct']:.1f}% (gt {row['d10_gt_pct']:.0f})"))
        g = grid(tiles)
        if g is not None:
            cv2.imwrite(str(figs / f"{sp}.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(sp, len([r for r in rows if r["species"] == sp]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "palette_test_per_image.csv", index=False)
    cut = Palette.load(species[0]).cutoffs[0]

    def det(d, lvl):
        y = np.r_[np.zeros(len(d)), np.ones(len(d))]
        s = np.r_[d.healthy_abnormal_pct, d[f"d{lvl}_abnormal_pct"]]
        return round(roc_auc_score(y, s), 3), round(float(((s >= cut) == y).mean()), 3)

    per = []
    for sp, d in df.groupby("species"):
        auc, acc = det(d, 10)
        r = {"species": sp, "n_test": len(d),
             "healthy_mean_pct": round(d.healthy_abnormal_pct.mean(), 2), "healthy_median_pct": round(d.healthy_abnormal_pct.median(), 2),
             "healthy_labelled_normal_pct": round(100 * (d.healthy_label == "Normal").mean(), 1),
             "synthetic10_auc": auc, "synthetic10_dice": round(d.d10_dice.mean(), 3)}
        if "old_infected_pct" in d:
            r.update({"old_pipeline_mean_infected_pct": round(d.old_infected_pct.mean(), 1),
                      "old_pipeline_graded_diseased_pct": round(100 * (d.old_severity.str.lower() != "healthy").mean(), 1)})
        per.append(r)
    per = pd.DataFrame(per)
    overall = {"n_test_images": len(df), "healthy_mean_pct": round(df.healthy_abnormal_pct.mean(), 2),
               "healthy_median_pct": round(df.healthy_abnormal_pct.median(), 2),
               "healthy_labelled_normal_pct": round(100 * (df.healthy_label == "Normal").mean(), 1), "normal_cutoff_pct": cut,
               "palette_ms_median": round(1000 * float(np.median(t_pal)), 1),
               "router_ms_median": round(1000 * float(np.median(t_router)), 1) if t_router else None}
    if "old_infected_pct" in df:
        overall.update({"old_pipeline_mean_infected_pct": round(df.old_infected_pct.mean(), 1),
                        "old_pipeline_graded_diseased_pct": round(100 * (df.old_severity.str.lower() != "healthy").mean(), 1)})
    levels = pd.DataFrame([{"damage level": f"{l} % of plant", "true % of leaf (median)": round(df[f"d{l}_gt_pct"].median(), 1),
                            "predicted % (median)": round(df[f"d{l}_abnormal_pct"].median(), 1),
                            "image ROC-AUC": det(df, l)[0], f"accuracy at {cut:g} %": det(df, l)[1],
                            "pixel Dice": round(df[f"d{l}_dice"].mean(), 3)} for l in LEVELS])
    md = ["# Palette branch, TEST split of the palette species (first use; nothing tuned here)", "",
          "Healthy = the web photos as labelled (not individually verified; see the review sheets).",
          "Damaged = SYNTHETIC: real PlantSeg val lesions pasted onto the same test leaves (src/palette/synthetic.py).", "",
          "## Overall", "", pd.DataFrame([overall]).T.rename(columns={0: "value"}).to_markdown(), "",
          "## Synthetic damage levels", "", levels.to_markdown(index=False), "",
          "## Per species (synthetic level: 10 %)", "", per.to_markdown(index=False), ""]
    if "old_infected_pct" in df:
        md += ["## Old crop pipeline vs palette branch on the same HEALTHY test images", "",
               "The crop system assumes healthy tissue is green, so red/purple/white normal tissue is graded as lesion.", "",
               old_vs_palette(df).to_markdown(index=False), ""]
    (ROOT / "reports" / "palette_test.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
