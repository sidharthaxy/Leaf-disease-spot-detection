"""
README / report figures (matplotlib, light theme).

    python -m src.evaluation.figures --eval outputs/eval_final --out reports/figures

Charts (committed):        training_curves.png, confusion_symptom_group.png, confusion_plant.png,
                           lesion_methods.png, severity_scatter.png, severity_ablation.png
Image figures (NOT committed - contain PlantSeg photos, see .gitignore): pipeline_strips.jpg

Style: palette slots 1-3 of the reference categorical palette (validated for CVD), single-hue
sequential ramp for matrices, thin marks, recessive grid, legend + direct labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (blue, orange, aqua)
BLUES = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cfe0f6", "#7fb0ea", "#2a78d6", "#123f7a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False,
})

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "EfficientNet-B0 @320": "models/effnetb0_320",
    "EfficientNetV2-S @384": "models/effv2s384",
    "ConvNeXt-Tiny @384": "models/convnext384",
}


def training_curves(out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (name, d), c in zip(MODELS.items(), SERIES):
        h = json.loads((ROOT / d / "history.json").read_text())
        e = [r["epoch"] for r in h]; f = [r["val_macro_f1"] for r in h]
        ax.plot(e, f, color=c, lw=2, label=name)
        b = int(np.argmax(f))
        ax.plot(e[b], f[b], "o", ms=8, color=c, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(f"{f[b]:.3f}", (e[b], f[b]), xytext=(4, 6), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation macro-F1")
    ax.set_title("Validation macro-F1 per epoch (dot = best epoch, the one kept)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=160); plt.close(fig)


def confusion(y_true, y_pred, labels, title, path, annotate=True, size=(6.5, 5.5)):
    cm = pd.crosstab(pd.Categorical(y_true, labels), pd.Categorical(y_pred, labels), dropna=False).values.astype(float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=size)
    ax.grid(False)
    im = ax.imshow(cmn, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90 if len(labels) > 8 else 30, ha="center" if len(labels) > 8 else "right",
                  fontsize=7 if len(labels) > 8 else 9)
    ax.set_yticks(range(len(labels)), labels, fontsize=7 if len(labels) > 8 else 9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    if annotate:
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                            color="white" if cmn[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised share")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def classifier_confusions(pred_csv: Path, out: Path):
    from src.severity.grade import symptom_group

    p = pd.read_csv(pred_csv)
    tg, pg = p.label.map(symptom_group), p.pred_label.map(symptom_group)
    groups = ["necrotic", "chlorosis", "rust", "powdery", "generic", "healthy"]
    acc = (tg == pg).mean()
    confusion(tg, pg, groups, f"Symptom group of the top-1 prediction, test (acc. {acc:.1%})", out / "confusion_symptom_group.png")
    tp, pp = p.label.str.split("___").str[0], p.pred_label.str.split("___").str[0]
    plants = sorted(tp.unique())
    confusion(tp, pp, plants, f"Plant confusion, test (accuracy {(tp == pp).mean():.1%})", out / "confusion_plant.png",
              annotate=False, size=(9, 8))


def barh(ax, names, values, fmt, color, highlight=None, invert=True):
    """Horizontal bars; values may contain None (no bar, no label). With shared y-axes, invert only once."""
    y = np.arange(len(names))
    cols = [color if (highlight is None or n == highlight) else "#b9b8b2" for n in names]
    ax.barh(y, [0 if v is None else v for v in values], color=cols, height=0.6)
    ax.set_yticks(y, names); ax.grid(axis="y", visible=False)
    if invert:
        ax.invert_yaxis()
    for yi, v in zip(y, values):
        if v is not None:
            ax.text(v, yi, " " + fmt.format(v), va="center", color=INK2, fontsize=9)


def learned_predictions(eval_dir: Path, split: str = "test"):
    """Learned severity estimate (models/learned/xgb_regressor.json, fitted on validation) for every
    test image, plus the constant baseline (validation median). Returns a DataFrame or None."""
    feat_csv = eval_dir / split / "per_image.csv"
    model_dir = ROOT / "models" / "learned"
    if not (model_dir / "xgb_regressor.json").exists():
        return None
    from xgboost import XGBRegressor

    r = pd.read_csv(feat_csv)
    f = r[r.experiment == "features"].copy()
    if f.empty:
        return None
    for g in ["necrotic", "chlorosis", "rust", "powdery", "generic", "healthy"]:
        f[f"g_{g}"] = (f.pred_group == g).astype(int)
    meta = json.loads((model_dir / "meta.json").read_text())
    reg = XGBRegressor(); reg.load_model(model_dir / "xgb_regressor.json")
    f["learned_pct"] = np.clip(reg.predict(f[meta["features"]].to_numpy(float)), 0, 100)
    val = pd.read_csv(eval_dir / "val" / "per_image.csv")
    vf = val[(val.experiment == "features") & (val.source == "plantseg")]
    f["constant_pct"] = float(vf.gt_pct.median())        # same trivial baseline as learned_correction.py
    return f


def eval_charts(eval_dir: Path, out: Path, split: str = "test"):
    res = pd.read_csv(eval_dir / split / "per_image.csv")
    les = res[~res.experiment.str.startswith("leaf_") & (res.experiment != "features")]
    ps = les[les.source == "plantseg"]
    names = {"method_hsv": "HSV only", "method_lab": "Lab a*/b* only", "method_kmeans": "k-means only",
             "method_delta_e": "ΔE + hysteresis only", "generic": "generic profile (no CNN)",
             "aware_noclahe": "disease-aware, no CLAHE", "aware": "disease-aware (true disease)",
             "predicted_top1rule": "CNN-driven, top-1 fallback rule", "predicted_grouprule": "CNN-driven, group fallback rule",
             "predicted_nofallback": "CNN-driven, measured lesions"}
    exps = [e for e in names if e in set(ps.experiment)]
    rows: list[tuple[str, float, float | None]] = [(names[e], float((ps[ps.experiment == e].pred_pct - ps[ps.experiment == e].gt_pct).abs().mean()),
             float(ps[ps.experiment == e].dice.mean())) for e in exps]
    lp = learned_predictions(eval_dir, split)
    if lp is not None:
        d = lp[lp.source == "plantseg"]
        rows.insert(0, (f"constant {d.constant_pct.iloc[0]:.0f} % (trivial baseline)", float((d.constant_pct - d.gt_pct).abs().mean()), None))
        rows.insert(0, ("CNN-driven + learned correction (final)", float((d.learned_pct - d.gt_pct).abs().mean()), None))
    final = rows[0][0]
    rows = sorted(rows, key=lambda r: r[1])                       # best (lowest MAE) first
    labels = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.42 * len(rows) + 1.8), sharey=True)
    barh(axes[0], labels, [r[1] for r in rows], "{:.1f}", SERIES[0], final)
    axes[0].set_title("Severity MAE, pp (lower = better)"); axes[0].set_xlabel("percentage points")
    barh(axes[1], labels, [r[2] for r in rows], "{:.3f}", SERIES[0], "CNN-driven, measured lesions", invert=False)
    for i, r in enumerate(rows):
        if r[2] is None:
            axes[1].text(0.005, i, "n/a (outputs a %, not a mask)", va="center", color=INK2, fontsize=9)
    axes[1].set_title("Lesion Dice vs PlantSeg (higher = better)")
    fig.suptitle(f"Test split, round 2 (n = {ps[ps.experiment == exps[0]].shape[0]} diseased images)", x=0.01, ha="left",
                 fontsize=10, color=INK2)
    fig.tight_layout(); fig.savefig(out / "severity_ablation.png", dpi=160); plt.close(fig)

    m = ps[ps.experiment == ("predicted_nofallback" if "predicted_nofallback" in exps else "aware")]
    panels = [("measured lesion %", m.gt_pct.to_numpy(), m.pred_pct.to_numpy())]
    if lp is not None:
        d = lp[lp.source == "plantseg"]
        panels.append(("learned correction (final)", d.gt_pct.to_numpy(), d.learned_pct.to_numpy()))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.4))
    axes = np.atleast_1d(axes)
    for ax, (title, gt, pred) in zip(axes, panels):
        ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls="--", label="perfect agreement")
        ax.scatter(gt, pred, s=12, alpha=0.4, color=SERIES[0], edgecolors="none", label=f"test image (n={len(gt)})")
        r = np.corrcoef(gt, pred)[0, 1]
        mae = np.abs(pred - gt).mean()
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
        ax.set_xlabel("ground-truth infected % (PlantSeg mask / leaf)"); ax.set_ylabel("predicted infected %")
        ax.set_title(f"{title}: MAE {mae:.1f} pp, r = {r:.2f}")
        ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out / "severity_scatter.png", dpi=160); plt.close(fig)


def pipeline_strips(sample_dir: Path, out: Path, per_image_csv: Path | None = None):
    """Step strips (original -> ... -> overlay), one TYPICAL image per symptom group: the image whose
    Dice is closest to its group's median (healthy: median false-positive %), taken from the dev-set
    error analysis - not hand-picked. Contains dataset photos -> .jpg, kept out of git."""
    import cv2

    from src.imageproc import viz
    from src.imageproc.pipeline import run_pipeline, strip_for
    from src.severity.grade import symptom_group

    m = pd.read_csv(sample_dir / "manifest.csv")
    m["group"] = m.label.map(symptom_group)
    if per_image_csv is not None and per_image_csv.exists():
        pi = pd.read_csv(per_image_csv)
        pi["group"] = pi.label.map(symptom_group)
        picks = []
        for g, x in pi.groupby("group"):
            col = "pred_pct" if g == "healthy" else "dice"
            x = x.dropna(subset=[col])
            if len(x):
                picks.append(x.iloc[(x[col] - x[col].median()).abs().argsort()[:1]])
        m = pd.concat(picks)
    rows = []
    for g in ["necrotic", "chlorosis", "rust", "powdery", "healthy"]:
        for r in m[m.group == g].head(1).itertuples():
            res = run_pipeline(cv2.imread(str(sample_dir / str(r.image_path))), group=g)
            rows.append(strip_for(res, f"  [{g}: {str(r.label).split('___')[0]} {r.disease}]"[:60]))
    W = max(x.shape[1] for x in rows)
    rows = [np.hstack([x, np.full((x.shape[0], W - x.shape[1], 3), 255, np.uint8)]) for x in rows]
    viz.save(out / "pipeline_strips.jpg", np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None, help="dir with <split>/per_image.csv from run_eval")
    ap.add_argument("--pred", default="outputs/ensemble/test_predictions.csv")
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    training_curves(out)
    classifier_confusions(ROOT / a.pred, out)
    if a.eval:
        eval_charts(ROOT / a.eval, out)
    if (ROOT / "data" / "dev_val" / "manifest.csv").exists():
        pipeline_strips(ROOT / "data" / "dev_val", out, ROOT / "outputs" / "error_analysis_v2" / "per_image.csv")
    print("figures in", out)


if __name__ == "__main__":
    main()
