"""
Palette branch: per-species "normal colour" model -> abnormal-tissue mask (no disease names).

Why: the crop pipeline assumes healthy tissue is GREEN. For a red croton, a purple Tradescantia or a
white-and-green pothos that assumption is wrong: normal red/white tissue looks like "lesion". Instead we
learn, per species, which colours are normal (from healthy photos) and flag what does not fit.

Colour model (fitted offline by src/palette/fit_palettes.py, saved to models/palettes/<species>.json):
  - pixel feature x = [w_L * L*, a*, b*] in CIE Lab (OpenCV float Lab: L 0..100, a/b about -127..127).
    a* (green-red) and b* (blue-yellow) carry the pigment colour; L* (lightness) mostly carries
    lighting, so it is down-weighted (w_L = 0.3): a shaded and a sunlit part of the same leaf fit the
    same component.
  - Gaussian mixture model (GMM): p(x) = sum_k w_k N(x; mu_k, Sigma_k), k = 3..6 components (chosen on
    val). One component per "normal colour" of the species, e.g. croton: green, yellow, orange, red.
  - robust fit: fit, drop the 10 % least likely pixels, refit (twice), so a few damaged leaves or
    background pixels in the web photos do not become "normal".
  - thresholds are log-likelihood quantiles of the TRAIN leaf pixels (q_strong < q_weak), chosen on
    VAL healthy images so that healthy leaves get a low abnormal area (median <= 2 %).

Inference (every step returns an image for the step strip):
  1. preprocess: Stage B step 1 (resize, median denoise) WITHOUT CLAHE: CLAHE re-maps L* depending on the
     neighbourhood, so healthy tissue next to a dark spot gets a different L* than the same tissue elsewhere
     and shows up as an off-palette halo (seen on the synthetic test leaf); the colour model needs each
     pixel's own colour
  2. log-likelihood map: log p(x) per pixel under the species GMM
  3. leaf mask ("ExG+ generalised"): normal tissue = log p >= t_weak, cleaned; then add low-likelihood
     pixels that are (i) near normal tissue AND (ii) inside its convex hull AND (iii) mostly bordered by
     normal tissue (the ExG+ symptom-expansion rules, with "green" replaced by "normal for this species");
     closing, hole filling, largest components
  4. abnormal tissue: hysteresis inside the leaf. Strong = log p < t_strong (clearly off-palette), weak =
     log p < t_weak; a weak region is kept only if it contains a strong pixel (so faint, noisy
     off-palette pixels alone are ignored, but a real spot keeps its full extent); then opening/closing
     and a minimum area. Optional per species (chosen on validation): dull brown / grey pixels (the colour
     of browning and necrosis, see dull_or_brown) are strong seeds as well
  5. measure: abnormal % = abnormal px / leaf px, spot count = connected components
  6. label: Normal / Some abnormal tissue / Significant abnormal tissue (cut-offs: configurable
     assumptions, chosen on val). Wording is "abnormal / damaged tissue", never a disease name: on
     ornamentals, browning is often sun scorch, over/under-watering or cold, not a pathogen.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.imageproc.leaf_seg import convex_hulls, filter_by_green_border
from src.imageproc.morphology import ellipse_kernel, fill_holes, keep_largest, open_close, remove_small, to_u8
from src.imageproc.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[2]
PALETTE_DIR = ROOT / "models" / "palettes"
W_L = 0.3
INFER = {"max_side": 640, "clahe": False, "near_frac": 0.04, "hull_min_area_frac": 0.002, "min_normal_border": 0.5,
         "border_px": 3, "close_k": 7, "open_k": 3, "min_spot_frac": 0.0005, "min_spot_px": 20,
         "keep_top_k": 10, "keep_min_frac": 0.05, "dark_L": 20.0, "max_hole_frac": 0.02, "max_low_frac": 0.08,
         "brown_seeds": False}
LABELS = ("Normal", "Some abnormal tissue", "Significant abnormal tissue")


# ----------------------------------------------------------------------------- features + GMM maths
def lab_features(img_bgr: np.ndarray, w_l: float = W_L) -> np.ndarray:
    """(H, W, 3) float32 features [w_L * L, a, b]."""
    lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lab[..., 0] *= w_l
    return lab


class Palette:
    """A fitted species palette (GMM + thresholds). Pure numpy at inference (no sklearn needed)."""

    def __init__(self, d: dict):
        self.d = d
        self.species = d["species"]
        self.w_l = d.get("w_l", W_L)
        self.weights = np.asarray(d["weights"], np.float64)
        self.means = np.asarray(d["means"], np.float64)
        self.covs = np.asarray(d["covariances"], np.float64)
        self.prec = np.linalg.inv(self.covs)
        _, logdet = np.linalg.slogdet(self.covs)
        self.log_norm = np.log(self.weights) - 0.5 * (3 * np.log(2 * np.pi) + logdet)
        self.t_strong = float(d["t_strong"])
        self.t_weak = float(d["t_weak"])
        self.cutoffs = d.get("label_cutoffs_pct", [3.0, 10.0])

    @classmethod
    def load(cls, species: str, folder: str | Path = PALETTE_DIR) -> "Palette":
        return cls(json.loads((Path(folder) / f"{species}.json").read_text()))

    def loglik(self, X: np.ndarray) -> np.ndarray:
        """log p(x) for features X (..., 3)."""
        sh = X.shape[:-1]
        X = X.reshape(-1, 3).astype(np.float64)
        comp = np.empty((X.shape[0], len(self.weights)))
        for k in range(len(self.weights)):
            dx = X - self.means[k]
            comp[:, k] = self.log_norm[k] - 0.5 * np.einsum("ni,ij,nj->n", dx, self.prec[k], dx)
        m = comp.max(1, keepdims=True)
        return (m[:, 0] + np.log(np.exp(comp - m).sum(1))).reshape(sh)

    def swatches_bgr(self) -> list[dict]:
        """Normal colours (GMM means) as BGR + hex, heaviest component first: for the demo."""
        out = []
        for k in np.argsort(self.weights)[::-1]:
            lab = self.means[k].copy()
            lab[0] /= self.w_l
            bgr = cv2.cvtColor(lab.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_Lab2BGR)[0, 0]
            b, g, r = (np.clip(bgr, 0, 1) * 255).round().astype(int)
            out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "weight": round(float(self.weights[k]), 3)})
        return out

    def label(self, abnormal_pct: float) -> str:
        if abnormal_pct < self.cutoffs[0]:
            return LABELS[0]
        return LABELS[1] if abnormal_pct < self.cutoffs[1] else LABELS[2]


def dull_or_brown(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pixels that look like soil, mulch, bark, concrete or dead/necrotic tissue: warm/neutral grey (chroma C* < 12,
    not bright white, not purple/red-tinted) or dull brown/tan (hue angle 55-85 deg, C* < 35, L* < 60). They are left OUT of the palette
    fit: the healthy palettes of the 10 species are green / yellow / red / pink / purple / orange / cream, and
    web photos show a lot of soil and mulch around the plant. Assumption (README): brown is never "normal"
    for these species, so brown-leaved cultivars are a known limitation."""
    C = np.hypot(a, b)
    h = (np.degrees(np.arctan2(b, a)) + 360) % 360
    tinted = (h >= 270) | (h < 20)          # purple / red-tinted greys are leaf colour (e.g. Tradescantia)
    grey = (C < 12) & (L < 75) & ~tinted
    brown = (h >= 55) & (h <= 85) & (C < 35) & (L < 60)
    return grey | brown


# ----------------------------------------------------------------------------- segmentation
def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Keep the connected components of `weak` that contain at least one `strong` pixel."""
    n, lab = cv2.connectedComponents(to_u8(weak), connectivity=8)
    if n <= 1:
        return np.zeros_like(to_u8(weak))
    hit = np.zeros(n, bool)
    hit[np.unique(lab[(strong > 0) & (weak > 0)])] = True
    hit[0] = False
    return to_u8(hit[lab])


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Fill enclosed holes up to `max_area` px. Big enclosed regions in whole-plant photos are usually
    background seen between leaves (wall, soil, pot), not damage, so they stay outside the leaf."""
    m = to_u8(mask)
    holes = cv2.bitwise_and(fill_holes(m), cv2.bitwise_not(m))
    small = holes & ~remove_small(holes, max_area + 1)
    return cv2.bitwise_or(m, to_u8(small))


def leaf_mask(ll: np.ndarray, t_weak: float, cfg: dict, L: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """L (CIE L*, 0..100) is optional: pixels darker than cfg["dark_L"] are treated as shadow/gaps, never
    as leaf expansion or abnormal tissue (deep shadow between leaves has no measurable colour)."""
    h, w = ll.shape
    dark = (L < cfg["dark_L"]) if (L is not None and cfg.get("dark_L", 0) > 0) else np.zeros_like(ll, bool)
    normal = open_close(to_u8(ll >= t_weak), cfg["open_k"], 5)
    normal = remove_small(normal, int(0.002 * h * w))
    normal = keep_largest(normal, cfg["keep_top_k"], cfg["keep_min_frac"])
    r = max(3, int(cfg["near_frac"] * max(h, w)))
    near = cv2.dilate(normal, ellipse_kernel(2 * r + 1))
    hull = convex_hulls(normal, int(cfg["hull_min_area_frac"] * h * w))
    low = to_u8((ll < t_weak) & ~dark) & near & hull
    low = filter_by_green_border(low, normal, cfg["min_normal_border"], cfg["border_px"])
    low = low & ~remove_small(low, int(cfg["max_low_frac"] * h * w) + 1)  # very large off-palette areas = background
    leaf = cv2.morphologyEx(cv2.bitwise_or(normal, low), cv2.MORPH_CLOSE, ellipse_kernel(cfg["close_k"]))
    leaf = fill_small_holes(leaf, int(cfg["max_hole_frac"] * h * w)) & to_u8(~dark | (normal > 0))
    leaf = keep_largest(leaf, cfg["keep_top_k"], cfg["keep_min_frac"])
    return leaf, {"normal": normal, "hull": hull, "low_near_normal": low}


def abnormal_mask(ll: np.ndarray, leaf: np.ndarray, t_strong: float, t_weak: float, cfg: dict,
                  L: np.ndarray | None = None, F: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Hysteresis: strong = clearly off-palette; weak = off-palette. With cfg["brown_seeds"] (chosen per species
    on validation), dull brown / grey pixels (dull_or_brown: the colour of browning and necrosis) inside the
    leaf are strong seeds too, even when the palette has absorbed some brown from soil in the web photos."""
    inside = leaf > 0
    if L is not None and cfg.get("dark_L", 0) > 0:
        inside = inside & (L >= cfg["dark_L"])
    strong_px = ll < t_strong
    weak_px = ll < t_weak
    if cfg.get("brown_seeds") and F is not None:
        brown = dull_or_brown(F[..., 0] / W_L, F[..., 1], F[..., 2])
        strong_px = strong_px | brown
        weak_px = weak_px | brown
    strong = to_u8(strong_px & inside)
    weak = to_u8(weak_px & inside)
    cand = hysteresis(strong, weak)
    ab = open_close(cand, cfg["open_k"], 5) & to_u8(inside)
    min_area = max(cfg["min_spot_px"], int(cfg["min_spot_frac"] * max(1, int(inside.sum()))))
    return remove_small(ab, min_area), cand


def loglik_vis(ll: np.ndarray, t_strong: float, t_weak: float) -> np.ndarray:
    """Colour map of the log-likelihood: bright = normal colour for this species, dark = off-palette."""
    lo, hi = t_strong - 4.0, float(np.percentile(ll, 99))
    v = np.clip((ll - lo) / max(1e-6, hi - lo), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def run_palette(img_bgr: np.ndarray, pal: Palette, cfg: dict | None = None, t_scale: tuple | None = None) -> dict:
    """Run the palette branch on one BGR image. `t_scale` overrides (t_strong, t_weak) (used for tuning)."""
    from src.imageproc import viz

    cfg = {**INFER, **pal.d.get("infer", {}), **(cfg or {})}
    t_strong, t_weak = t_scale if t_scale is not None else (pal.t_strong, pal.t_weak)
    pre, pre_steps = preprocess(img_bgr, {"max_side": cfg["max_side"], "clahe": cfg["clahe"]})
    F = lab_features(pre, pal.w_l)
    L = F[..., 0] / pal.w_l
    ll = pal.loglik(F)
    leaf, dbg = leaf_mask(ll, t_weak, cfg, L)
    ab, cand = abnormal_mask(ll, leaf, t_strong, t_weak, cfg, L, F)
    leaf_px = int((leaf > 0).sum())
    ab_px = int((ab > 0).sum())
    pct = 100.0 * ab_px / leaf_px if leaf_px else 0.0
    n_spots = max(0, cv2.connectedComponents(ab, connectivity=8)[0] - 1)
    m = {"species": pal.species, "abnormal_pct": round(pct, 2), "leaf_px": leaf_px, "abnormal_px": ab_px,
         "n_spots": int(n_spots), "label": pal.label(pct), "leaf_frac_of_image": round(leaf_px / leaf.size, 3)}
    steps = {"original": pre_steps["resized"], "preprocessed": pre,
             "palette likelihood": loglik_vis(ll, t_strong, t_weak),
             "leaf mask": viz.mask_on_image(pre, leaf),
             "off-palette candidates": viz.mask_on_image(pre, cand),
             "abnormal tissue": viz.mask_on_image(pre, ab),
             "overlay": viz.overlay(pre_steps["resized"], leaf, ab)}
    return {"measurements": m, "leaf_mask": leaf, "abnormal_mask": ab, "loglik": ll, "steps": steps, "normal": dbg["normal"]}


STRIP_ORDER = ["original", "preprocessed", "palette likelihood", "leaf mask", "off-palette candidates", "abnormal tissue", "overlay"]
