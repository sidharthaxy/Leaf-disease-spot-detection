"""
Pick the lesion method per symptom group from a VALIDATION evaluation (run_eval per_image.csv).

    python -m src.evaluation.select_methods outputs/eval_val_v5/val/per_image.csv

Objective (fixed before looking at test): mean Dice - severity MAE / 100 over the group's diseased
validation images, i.e. 1 percentage point of severity error costs as much as 0.01 Dice.
Prints the table and the winners; configs/disease_profiles.yaml is then updated by hand (with a comment).
"""

from __future__ import annotations

import sys

import pandas as pd


def main():
    r = pd.read_csv(sys.argv[1])
    ps = r[(r.source == "plantseg") & r.experiment.str.startswith("method_")]
    rows = []
    for (g, e), x in ps.groupby(["true_group", "experiment"]):
        mae = (x.pred_pct - x.gt_pct).abs().mean()
        rows.append({"group": g, "method": e.replace("method_", ""), "n": len(x), "dice": x.dice.mean(),
                     "precision": x.precision.mean(), "recall": x.recall.mean(), "mae": mae,
                     "bias": (x.pred_pct - x.gt_pct).mean(), "score": x.dice.mean() - mae / 100})
    t = pd.DataFrame(rows).sort_values(["group", "score"], ascending=[True, False])
    print(t.round(3).to_string(index=False))
    print("\nwinners (validation):")
    for g, x in t.groupby("group"):
        b = x.iloc[0]
        print(f"  {g:10s} -> {b.method:14s} score={b.score:.3f} dice={b.dice:.3f} mae={b.mae:.2f}  (n={b.n})")


if __name__ == "__main__":
    main()
