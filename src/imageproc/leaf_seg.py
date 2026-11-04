"""
Stage B.2: leaf segmentation (find the leaf pixels = the denominator of infected %).

Three methods, compared in the report:

1. ExG + Otsu
   Excess Green index on chromatic coordinates:  r = R/(R+G+B), g = G/(R+G+B), b = B/(R+G+B)
       ExG = 2g - r - b
   Normalising by R+G+B removes most of the brightness (shadow vs sun) effect. Vegetation has
   g > r, b, so ExG is high on leaves and low on soil / sky / walls.
   Otsu then picks the threshold automatically: it tries every threshold and chooses the one
   that maximises the between-class variance of the two resulting groups (leaf vs background).

2. HSV thresholding
   In HSV, Hue is the colour angle. Leaves (green to yellow-green) occupy a hue band; we also
   require some saturation (S) and brightness (V) to reject grey/white/black background.
   OpenCV hue range is 0..179 (degrees / 2): yellow ~ 30, green ~ 60.

3. GrabCut initialised from the ExG mask
   GrabCut models foreground and background colours with Gaussian Mixture Models and solves a
   graph cut (min-cut / max-flow) that balances colour fit against smooth boundaries. We seed it
   from the ExG mask: eroded ExG = "sure leaf", ExG = "probably leaf", near the leaf = "probably
   background", far away = "sure background". It recovers leaf pixels that ExG misses
   (e.g. large brown lesions on the leaf margin), at a higher compute cost.

4. ExG+ (our improved default, added after evaluating 1-3 on real PlantSeg images)
   Two failure modes of plain ExG + Otsu on in-the-wild photos:
     a) when the leaf fills the whole frame there is no background, but Otsu ALWAYS splits the
        histogram into two classes, so it splits the leaf into "greener" vs "less green" halves.
        Fix: a pixel that is clearly green (ExG above an absolute cap) is always leaf:
             threshold = min(Otsu threshold, cap)
     b) heavily diseased tissue (yellow / brown / white) is not green, so it is cut off, and hole
        filling only rescues lesions that are completely surrounded by green.
        Fix: "symptom expansion" - add symptom-coloured pixels (yellow/orange/brown, whitish,
        dark) that lie (i) within `expand_frac` x image size of green tissue (a dilation of the
        green mask) AND (ii) inside the CONVEX HULL of a green leaf region. The convex hull is the
        smallest convex polygon containing the region: a diseased leaf margin lies inside its
        leaf's hull, while brown soil next to the leaf lies outside it.
        (iii) "green border ratio": each connected symptom region is kept only if at least
        `min_green_border` of the ring of pixels just outside it is green tissue. A lesion (even on
        the leaf margin) is mostly surrounded by green; leaf litter / white backdrop / soil touching
        the leaf only shares one edge with it.

IMPORTANT: diseased tissue is NOT green, so pure colour methods punch holes where lesions are.
Post-processing fills enclosed holes so lesions inside the leaf count as leaf area. Lesions that
touch the leaf border can still be lost; GrabCut helps there. This is discussed in the report.
"""

from __future__ import annotations

import cv2
import numpy as np

from .morphology import ellipse_kernel, fill_holes, keep_largest, open_close, remove_small, to_u8


def exg_index(img: np.ndarray) -> np.ndarray:
    """Return ExG as float32 in [-1, 2]."""
    f = img.astype(np.float32)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    s = R + G + B + 1e-6
    r, g, b = R / s, G / s, B / s
    return 2 * g - r - b


def exg_otsu(img: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    cfg = cfg or {}
    exg = exg_index(img)
    # map ExG to 0..255 so OpenCV's Otsu can run on it
    lo, hi = float(exg.min()), float(exg.max())
    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_exg = lo + thr / 255.0 * (hi - lo)              # Otsu threshold back in ExG units
    if cfg.get("cap") is not None:                       # ExG+ fix (a): clearly green is always leaf
        thr_exg = min(thr_exg, float(cfg["cap"]))
        mask = ((exg > thr_exg) * 255).astype(np.uint8)
    return mask, {"exg": exg_u8, "otsu_threshold": float(thr), "exg_threshold": thr_exg}


def symptom_colours(img: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Pixels whose colour looks like diseased plant tissue (not green, not sky/blue)."""
    cfg = {**SYMPTOM_DEFAULTS, **(cfg or {})}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = ((H <= cfg["warm_h_max"]) | (H >= 170)) & (S >= cfg["warm_s_min"]) & (V >= cfg["warm_v_min"])  # yellow/orange/brown/red
    white = (S <= cfg["white_s_max"]) & (V >= cfg["white_v_min"])                                           # powdery / bleached
    dark = V <= cfg["dark_v_max"]                                                                            # black necrosis
    return to_u8(warm | white | dark)


def convex_hulls(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Filled convex hull of every connected region larger than `min_area`."""
    out = np.zeros_like(mask)
    contours, _ = cv2.findContours(to_u8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.fillPoly(out, [cv2.convexHull(c)], 255)
    return out


def filter_by_green_border(sym: np.ndarray, green: np.ndarray, min_frac: float, border_px: int = 3) -> np.ndarray:
    """Keep symptom regions whose outer ring (border_px wide) is at least `min_frac` green."""
    n, labels = cv2.connectedComponents(to_u8(sym), connectivity=8)
    if n <= 1:
        return to_u8(sym)
    # grey dilation of the label image spreads each region's id into its surrounding ring
    grown = cv2.dilate(labels.astype(np.float32), ellipse_kernel(2 * border_px + 1)).astype(np.int32)
    ring = (labels == 0) & (grown > 0)
    total = np.bincount(grown[ring], minlength=n)
    greenc = np.bincount(grown[ring & (green > 0)], minlength=n)
    frac = greenc / np.maximum(total, 1)
    keep = frac >= min_frac
    keep[0] = False
    return to_u8(keep[labels])


def exg_plus(img: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    cfg = {**EXGPLUS_DEFAULTS, **(cfg or {})}
    green, d = exg_otsu(img, {"cap": cfg["cap"]})
    green = open_close(green, 3, 5)
    r = max(3, int(cfg["expand_frac"] * max(img.shape[:2])))
    near = cv2.dilate(green, ellipse_kernel(2 * r + 1))      # "close to green tissue"
    hull = convex_hulls(green, min_area=int(cfg["hull_min_area_frac"] * green.size))
    sym = symptom_colours(img, cfg.get("symptom")) & near & hull
    if cfg.get("min_green_border", 0) > 0:
        sym = filter_by_green_border(sym, green, cfg["min_green_border"], cfg.get("border_px", 3))
    d.update({"green": green, "hull": hull, "symptom_near_green": sym})
    return cv2.bitwise_or(green, sym), d


def hsv_threshold(img: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    cfg = {**HSV_DEFAULTS, **(cfg or {})}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg["h_min"], cfg["s_min"], cfg["v_min"]], np.uint8)
    upper = np.array([cfg["h_max"], 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask, {"hsv": hsv}


def grabcut_refine(img: np.ndarray, init_mask: np.ndarray, iters: int = 4,
                   erode_k: int = 15, dilate_k: int = 41) -> np.ndarray:
    init = to_u8(init_mask)
    if init.sum() == 0:  # nothing to seed from
        return init
    gc = np.full(init.shape, cv2.GC_BGD, np.uint8)                        # far away: sure background
    near = cv2.dilate(init, np.ones((dilate_k, dilate_k), np.uint8))
    gc[near > 0] = cv2.GC_PR_BGD                                          # around the leaf: probably bg
    gc[init > 0] = cv2.GC_PR_FGD                                          # ExG leaf: probably leaf
    sure = cv2.erode(init, np.ones((erode_k, erode_k), np.uint8))
    gc[sure > 0] = cv2.GC_FGD                                             # leaf core: sure leaf
    if not (gc == cv2.GC_FGD).any():
        gc[init > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return init  # e.g. degenerate GMM when (almost) the whole image is one class
    return to_u8((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))


def postprocess_leaf(mask: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    cfg = {**POST_DEFAULTS, **(cfg or {})}
    h, w = mask.shape[:2]
    m = open_close(mask, cfg["open_k"], cfg["close_k"])
    m = fill_holes(m)                                   # lesions inside the leaf become leaf area
    m = remove_small(m, int(cfg["min_area_frac"] * h * w))
    m = keep_largest(m, top_k=cfg["top_k"], min_frac_of_largest=cfg["min_frac_of_largest"])
    return m


def segment_leaf(img: np.ndarray, method: str = "exg", cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """Return (final leaf mask {0,255}, debug dict with raw masks)."""
    cfg = cfg or {}
    debug = {}
    if method == "exg":
        raw, d = exg_otsu(img, cfg.get("exg"))
    elif method == "exg_plus":
        raw, d = exg_plus(img, cfg.get("exg_plus"))
    elif method == "hsv":
        raw, d = hsv_threshold(img, cfg.get("hsv"))
    elif method == "grabcut":
        seed, d = exg_otsu(img, cfg.get("exg"))
        seed = postprocess_leaf(seed, cfg.get("post"))
        debug["grabcut_seed"] = seed
        raw = grabcut_refine(img, seed, **cfg.get("grabcut", {}))
    else:
        raise ValueError(f"unknown leaf method: {method}")
    debug.update(d)
    debug["leaf_raw"] = raw
    final = postprocess_leaf(raw, cfg.get("post"))
    return final, debug


HSV_DEFAULTS = {"h_min": 18, "h_max": 95, "s_min": 40, "v_min": 30}
POST_DEFAULTS = {
    "open_k": 5,
    "close_k": 15,
    "min_area_frac": 0.002,       # drop blobs smaller than 0.2 % of the image
    "top_k": 5,                   # several leaves may be in the frame
    "min_frac_of_largest": 0.10,  # ...but only keep ones at least 10 % the size of the main leaf
}
EXGPLUS_DEFAULTS = {"cap": 0.06, "expand_frac": 0.04, "hull_min_area_frac": 0.005, "min_green_border": 0.3}
SYMPTOM_DEFAULTS = {"warm_h_max": 33, "warm_s_min": 45, "warm_v_min": 40,
                    "white_s_max": 45, "white_v_min": 170, "dark_v_max": 45}
