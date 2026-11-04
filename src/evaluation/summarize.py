"""Aggregate per_image.csv into the report tables (markdown + csv). Plots live in src/evaluation/figures.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SEV_ORDER = ["Healthy", "Mild", "Moderate", "Severe"]


def summarize(res: pd.DataFrame, out: Path) -> dict:
    out = Path(out)
    lines = []
    les = res[~res.experiment.str.startswith("leaf_") & (res.experiment != "features")]
    ps = les[les.source == "plantseg"]
    hl = les[les.source != "plantseg"]

    # 1. lesion segmentation quality per experiment (diseased images only)
    t1 = ps.groupby("experiment")[["iou", "dice", "precision", "recall"]].mean().round(3)
    t1["n"] = ps.groupby("experiment").size()
    lines += ["## Lesion segmentation vs PlantSeg masks (mean over diseased test images)", t1.to_markdown(), ""]

    # 2. per symptom group x method
    t2 = ps[ps.experiment.str.startswith("method_")].pivot_table(index="true_group", columns="lesion_method",
                                                                  values="dice", aggfunc="mean").round(3)
    t2["n_images"] = ps[ps.experiment == "aware"].groupby("true_group").size()
    lines += ["## Dice per symptom group and lesion method", t2.to_markdown(), ""]

    # 3. severity
    def sev_row(d):
        err = (d.pred_pct - d.gt_pct).abs()
        return pd.Series({
            "MAE_leaf_pct": err.mean(), "median_AE": err.median(),
            "MAE_image_pct": (d.pred_img_pct - d.gt_img_pct).abs().mean(),
            "bias(pred-gt)": (d.pred_pct - d.gt_pct).mean(),
            "label_agreement": (d.pred_sev == d.gt_sev).mean(),
            "within_one_class": (d.pred_sev.map(SEV_ORDER.index) - d.gt_sev.map(SEV_ORDER.index)).abs().le(1).mean(),
            "n": len(d)})
    t3 = ps.groupby("experiment").apply(sev_row).round(3)
    lines += ["## Severity (infected % of leaf, shared denominator) on diseased images", t3.to_markdown(), ""]
    if "non_leaf" in ps.columns:
        nl = ps.non_leaf.astype(str).str.lower() == "true"
        t3b = ps[~nl].groupby("experiment").apply(sev_row).round(3)
        t3c = ps[nl].groupby("experiment").apply(sev_row).round(3)
        lines += ["## Severity on LEAF diseases only", t3b.to_markdown(), "",
                  "## Severity on NON-LEAF diseases (fruit / heads / stems / roots)", t3c.to_markdown(), ""]

    if len(hl):
        t4 = hl.groupby("experiment").agg(mean_pred_pct=("pred_pct", "mean"),
                                          graded_healthy=("pred_sev", lambda s: (s == "Healthy").mean()),
                                          n=("pred_pct", "size")).round(3)
        lines += ["## Healthy leaves (PlantDoc): false-positive infected %", t4.to_markdown(), ""]

    # 4. severity confusion for the main system
    main_exp = next((e for e in ("predicted_grouprule", "predicted_top1rule", "aware") if (ps.experiment == e).any()), "aware")
    main = ps[ps.experiment == main_exp]
    cm = pd.crosstab(main.gt_sev, main.pred_sev).reindex(index=SEV_ORDER, columns=SEV_ORDER, fill_value=0)
    lines += [f"## Severity confusion ({main.experiment.iloc[0] if len(main) else '-'}; rows = GT, cols = predicted)",
              cm.to_markdown(), ""]

    # 5. leaf method proxy
    lf = res[res.experiment.str.startswith("leaf_")]
    lfp = lf[lf.source == "plantseg"]
    t5 = lfp.groupby("leaf_method").agg(gt_lesion_inside_leaf=("gt_inside_leaf", "mean"),
                                        leaf_share_of_image=("leaf_img_pct", "mean"), n=("leaf_img_pct", "size")).round(3)
    lines += ["## Leaf segmentation proxy (no leaf GT): share of GT lesion pixels inside our leaf mask",
              t5.to_markdown(), ""]

    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    for name, t in [("lesion_methods", t1), ("dice_by_group", t2), ("severity", t3), ("severity_confusion", cm),
                    ("leaf_methods", t5)]:
        t.to_csv(out / f"{name}.csv")
    print(text)
    return {"lesion": t1, "severity": t3}


if __name__ == "__main__":
    import sys

    d = Path(sys.argv[1])
    summarize(pd.read_csv(d / "per_image.csv"), d)
