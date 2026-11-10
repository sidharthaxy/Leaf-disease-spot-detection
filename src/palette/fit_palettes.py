"""
Fit the per-species palettes on TRAIN, choose k / thresholds / label cut-offs on VAL.

    python -m src.palette.fit_palettes --data data/router --plantseg data/dev_val

Per species:
  1. TRAIN images -> Stage B preprocessing (no CLAHE) -> rough leaf prior (GrabCut, centre rectangle) -> Lab
     pixels (very dark L* < 8 and blown-out L* > 97 pixels are skipped: they carry no colour; dull grey/brown
     pixels are skipped too: soil, mulch, bark, dead tissue, see dull_or_brown)
  2. robust GMM with k = 6 components (an earlier full run searched k in {4, 6}: 5 species chose 4, 5 chose 6;
     k is fixed here to keep the larger search affordable): fit, drop the 10 % least likely pixels, refit (x2)
  3. thresholds = log-likelihood quantiles t_strong = Q(q_s), t_weak = Q(q_w) of either ALL fit pixels or only
     the CORE pixels kept by the robust trimming (stricter: "less likely than the plant's own core colours")
  4. VAL selection of (dull/brown exclusion on/off, trim 10 % / 30 %, threshold basis, q_s, q_w,
     brown seeds on/off): healthy val images must keep a median abnormal area <= 2 %
     (primary); among those, maximise Dice + image-level ROC-AUC on SYNTHETIC damage (real PlantSeg val lesions pasted
     onto the same val leaves, see synthetic.py). If no setting meets 2 %, the one with the lowest median wins.
Then, pooled over species on VAL: the "Normal" cut-off c1 = the abnormal % that best separates healthy
from synthetic-damaged val images (Youden's J, at least 2 %); c2 (Significant) = max(10 %, 3 x c1) is a
configurable assumption (there is no ground truth for "significant" on ornamentals).
Writes models/palettes/<species>.json and reports/palette_val.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from src.imageproc.preprocess import preprocess
from src.palette.palette import INFER, PALETTE_DIR, W_L, Palette, abnormal_mask, dull_or_brown, lab_features, leaf_mask
from src.palette.synthetic import build_lesion_bank, paste_lesions, rough_leaf_prior

ROOT = Path(__file__).resolve().parents[2]
KS = (6,)
QGRID = [(0.0005, 0.002), (0.001, 0.005), (0.002, 0.01), (0.005, 0.02), (0.01, 0.05), (0.02, 0.08), (0.05, 0.15), (0.05, 0.25), (0.1, 0.3)]
DARK_L = (20.0,)   # 0 vs 20 was searched in earlier runs: 20 won for 9 of 10 species
TRIMS = (0.10, 0.30)
MAX_HEALTHY_MEDIAN = 2.0


def load_pre(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return preprocess(img, {"max_side": INFER["max_side"], "clahe": INFER["clahe"]})[0]


def prior_pixels(pre: np.ndarray, prior: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """(all usable prior pixels, the same without dull/brown pixels), each subsampled to <= n."""
    X = lab_features(pre)[prior > 0]
    L = X[:, 0] / W_L
    X = X[(L >= 8) & (L <= 97)]
    keep = ~dull_or_brown(X[:, 0] / W_L, X[:, 1], X[:, 2])
    sub = lambda Z: Z[rng.choice(len(Z), n, replace=False)] if len(Z) > n else Z
    return sub(X), sub(X[keep])


def robust_gmm(X: np.ndarray, k: int, seed: int, rounds: int = 2, trim: float = 0.10,
               return_core: bool = False):
    """Fit, drop the `trim` least likely pixels, refit (`rounds` times). trim 0.10 keeps 81 % of the pixels,
    0.30 keeps 49 %: the denser "core" colours of the plant, so scene colours (soil, wall, sky) drop out."""
    g = GaussianMixture(k, covariance_type="full", reg_covar=1e-2, random_state=seed, max_iter=200).fit(X)
    for _ in range(rounds):
        ll = g.score_samples(X)
        X = X[ll >= np.quantile(ll, trim)]
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-2, random_state=seed, max_iter=200,
                            weights_init=g.weights_, means_init=g.means_).fit(X)
    return (g, X) if return_core else g


def dice(a, b):
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def to_dict(species, g, k, train_ll, qs, qw):
    return {"species": species, "k": k, "w_l": W_L, "weights": g.weights_.tolist(), "means": g.means_.tolist(),
            "covariances": g.covariances_.tolist(), "q_strong": qs, "q_weak": qw,
            "t_strong": float(np.quantile(train_ll, qs)), "t_weak": float(np.quantile(train_ll, qw))}


def fit_species(args) -> tuple[dict, list, list]:
    sp, data, plantseg, px_per_image, seed = args
    data = Path(data)
    man = pd.read_csv(data / "manifest.csv")
    rng = np.random.default_rng(seed + hash(sp) % 1000)
    bank = build_lesion_bank(plantseg, seed=seed)
    tr = man[(man.label == sp) & (man.split == "train")]
    va = man[(man.label == sp) & (man.split == "val")]
    XA, XE = [], []
    for p in tr.image_path:
        pre = load_pre(data / p)
        if pre is not None:
            xa, xe = prior_pixels(pre, rough_leaf_prior(pre), px_per_image, rng)
            XA.append(xa)
            XE.append(xe)
    pix = {False: np.concatenate(XA), True: np.concatenate(XE)}
    # val: healthy + a synthetic-damaged copy of each image (same leaf, known lesion mask); images where no
    # lesion could be placed are kept for the healthy check but skipped for Dice
    vals = []
    for p in va.image_path:
        pre = load_pre(data / p)
        if pre is None:
            continue
        dmg, gt = paste_lesions(pre, rough_leaf_prior(pre), bank, rng)
        vals.append((lab_features(pre), lab_features(dmg), gt))
    best = None
    for k, excl, trim in itertools.product(KS, (True, False), TRIMS):
        g, core = robust_gmm(pix[excl], k, seed, trim=trim, return_core=True)
        pal0 = Palette(to_dict(sp, g, k, g.score_samples(core), 0.01, 0.05))
        basis_ll = {"core": pal0.loglik(core), "all": pal0.loglik(pix[excl])}
        lls = [(pal0.loglik(fh), pal0.loglik(fd), gt, fh, fd) for fh, fd, gt in vals]
        for (qs, qw), dark_L, basis, brown in itertools.product(QGRID, DARK_L, ("core", "all"), (False, True)):
            cfg = {**INFER, "dark_L": dark_L, "brown_seeds": brown}
            ref = basis_ll[basis]
            ts, tw = float(np.quantile(ref, qs)), float(np.quantile(ref, qw))
            h_pct, d_pct, dices = [], [], []
            for llh, lld, gt, Fh, Fd in lls:
                leaf, _ = leaf_mask(llh, tw, cfg, Fh[..., 0] / W_L)
                ab, _ = abnormal_mask(llh, leaf, ts, tw, cfg, Fh[..., 0] / W_L, Fh)
                h_pct.append(100 * (ab > 0).sum() / max(1, (leaf > 0).sum()))
                if (gt > 0).any():
                    leaf, _ = leaf_mask(lld, tw, cfg, Fd[..., 0] / W_L)
                    ab, _ = abnormal_mask(lld, leaf, ts, tw, cfg, Fd[..., 0] / W_L, Fd)
                    d_pct.append(100 * (ab > 0).sum() / max(1, (leaf > 0).sum()))
                    dices.append(dice(ab, gt))
            med = float(np.median(h_pct))
            ok = med <= MAX_HEALTHY_MEDIAN
            auc = float(roc_auc_score(np.r_[np.zeros(len(h_pct)), np.ones(len(d_pct))], np.r_[h_pct, d_pct]))
            key = (ok, float(np.mean(dices)) + auc if ok else -med)
            row = {"species": sp, "k": k, "exclude_dull": excl, "trim": trim, "basis": basis, "q_strong": qs, "q_weak": qw,
                   "dark_L": dark_L, "val_n": len(lls), "val_n_synthetic": len(dices),
                   "healthy_median_pct": round(med, 2), "healthy_mean_pct": round(float(np.mean(h_pct)), 2),
                   "synthetic_dice": round(float(np.mean(dices)), 3),
                   "image_auc": round(auc, 3), "brown_seeds": brown,
                   "meets_2pct": ok}
            if best is None or key > best[0]:
                best = (key, row, {**to_dict(sp, g, k, ref, qs, qw), "infer": {"dark_L": dark_L, "brown_seeds": brown},
                                   "threshold_basis": basis,
                                   "exclude_dull_in_fit": excl, "trim": trim, "n_fit_pixels": int(len(pix[excl])),
                                   "n_core_pixels": int(len(core))}, h_pct, d_pct)
    _, row, d, h_pct, d_pct = best
    d.update({"n_train_images": int(len(tr)), "val": row})
    (PALETTE_DIR / f"{sp}.json").write_text(json.dumps(d, indent=1))
    print(json.dumps(row), flush=True)
    return row, h_pct, d_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "router"))
    ap.add_argument("--plantseg", default=str(ROOT / "data" / "dev_val"), help="PlantSeg VAL images+masks for the lesion bank")
    ap.add_argument("--px-per-image", type=int, default=1500)
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cutoffs-only", action="store_true", help="recompute the label cut-offs from the saved palettes")
    a = ap.parse_args()
    data = Path(a.data)
    man = pd.read_csv(data / "manifest.csv")
    species = a.species or sorted(set(man.label) - {"crop", "other_houseplant", "non_plant"})
    if a.cutoffs_only:
        healthy = [x for sp in species for x in val_healthy_pcts(sp, data)]
        write_cutoffs(species, healthy, None)
        return
    print(f"lesion bank: {len(build_lesion_bank(a.plantseg, seed=a.seed))} patches from PlantSeg val", flush=True)
    PALETTE_DIR.mkdir(parents=True, exist_ok=True)
    report, pooled = [], {"healthy": [], "damaged": []}

    from multiprocessing import Pool

    with Pool(a.workers) as pool:
        res = pool.map(fit_species, [(sp, str(data), a.plantseg, a.px_per_image, a.seed) for sp in species])
    for row, h_pct, d_pct in res:
        pooled["healthy"] += h_pct
        pooled["damaged"] += d_pct
        report.append(row)

    write_cutoffs(species, pooled["healthy"], report)


def val_healthy_pcts(sp: str, data: Path) -> list:
    """Abnormal % of every healthy VAL photo with the saved palette (same code path as the search)."""
    man = pd.read_csv(data / "manifest.csv")
    pal = Palette.load(sp)
    cfg = {**INFER, **pal.d.get("infer", {})}
    out = []
    for p in man[(man.label == sp) & (man.split == "val")].image_path:
        pre = load_pre(data / p)
        if pre is None:
            continue
        F = lab_features(pre)
        ll = pal.loglik(F)
        leaf, _ = leaf_mask(ll, pal.t_weak, cfg, F[..., 0] / W_L)
        ab, _ = abnormal_mask(ll, leaf, pal.t_strong, pal.t_weak, cfg, F[..., 0] / W_L, F)
        out.append(100 * (ab > 0).sum() / max(1, (leaf > 0).sum()))
    return out


def write_cutoffs(species: list, healthy: list, report: list | None) -> None:
    """Label cut-offs, pooled over species on VAL. Synthetic damage barely separates from healthy photos at
    image level (val AUC ~0.5-0.6, reports/palette_val.md), so a "best separating" cut-off would be arbitrary.
    Instead: c1 = 90th percentile of the healthy val photos' abnormal % (rounded up to 0.5 %), i.e. 90 % of
    healthy plants are labelled "Normal"; c2 = max(10 %, 3 x c1) for "Significant" (a configurable assumption)."""
    h = np.array(healthy)
    c1 = float(np.ceil(np.percentile(h, 90) * 2) / 2)
    c2 = float(max(10.0, 3 * c1))
    for sp in species:
        f = PALETTE_DIR / f"{sp}.json"
        d = json.loads(f.read_text())
        d["label_cutoffs_pct"] = [c1, c2]
        f.write_text(json.dumps(d, indent=1))
    if report is None:
        print(f"cut-offs: {c1}, {c2}")
        return
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    df = pd.DataFrame(report)
    (rep / "palette_val.md").write_text(
        "# Palette branch, VALIDATION (k, thresholds and cut-offs chosen here)\n\n"
        "Constraint: median abnormal area on healthy val leaves <= 2 %; objective: Dice + image-level ROC-AUC on synthetic damage "
        "(real PlantSeg val lesions pasted onto the same val leaves).\n\n" + df.to_markdown(index=False) +
        f"\n\nLabel cut-offs (pooled val): Normal < {c1:.1f} % (90 % of healthy val photos), "
        f"Significant >= {c2:.1f} % (configurable assumption).\n")
    print(df.to_string(index=False))
    print(f"cut-offs: {c1}, {c2}")


if __name__ == "__main__":
    main()
