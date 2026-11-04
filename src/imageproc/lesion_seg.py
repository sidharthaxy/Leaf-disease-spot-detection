"""
Stage B.3 + B.4: lesion (diseased spot) detection INSIDE the leaf mask, then clean-up.

Three methods, compared in the report:

(a) HSV thresholding
    "nongreen" mode: define healthy tissue as a green hue band with enough saturation;
    every leaf pixel that is NOT healthy green is a lesion candidate.
    "powdery" mode: powdery mildew is a white/grey coating, i.e. LOW saturation and HIGH value.

(b) Lab channel thresholding
    In CIE Lab, a* runs green(-) -> red(+), b* runs blue(-) -> yellow(+).
    OpenCV stores them as uint8 with 128 = neutral, so green tissue has a* < 128 while brown,
    red and orange necrosis has a* > 128. We run Otsu on the a* values of LEAF pixels only
    (background would bias the histogram), and never let the threshold drop below a floor,
    otherwise a perfectly healthy leaf would still be split into two "classes" by Otsu.
    Chlorosis (yellowing) is better separated on b* (yellowness) or on L (it is lighter).

(c) k-means clustering in Lab space
    Cluster the Lab colours of leaf pixels into k groups (k = 3..4). k-means alternates between
    assigning each pixel to its nearest centre and moving each centre to the mean of its pixels.
    We then auto-pick the lesion clusters with a colour "score" per symptom group:
        hue    : how far the cluster's Lab hue angle is from green (necrotic, rust, chlorosis)
        a      : redness a*
        white  : L - chroma (whitish, for powdery mildew)
    The greenest cluster is taken as the healthy reference; clusters whose score exceeds it by
    `min_delta` are lesions.

(d) ΔE from the leaf's OWN healthy colour + hysteresis   (our adaptive method, added after validation)
    Fixed colour thresholds fail because "healthy green" differs per plant and per photo (dark,
    light, bluish cabbage, yellow-green wheat). So we estimate a healthy reference colour PER LEAF:
        ref = median Lab colour of the greenest `ref_frac` of leaf pixels (highest ExG)
    and score every leaf pixel by its colour difference to it (CIE76 ΔE, with lightness down-
    weighted by `w_L` so shadows / glare count less than a real colour change):
        ΔE = sqrt( (w_L·ΔL)² + Δa² + Δb² )
    Hysteresis thresholding (as in the Canny edge detector) then keeps
        strong pixels  ΔE > T_high   (lesion cores)   +
        weak pixels    ΔE > T_low    that are CONNECTED to a strong pixel (halos, faint margins)
    with T_high = max(Otsu(ΔE on leaf), floor) and T_low = ratio · T_high. The floor stops a healthy
    leaf (unimodal, small ΔE) from being split in two by Otsu.

Post-processing (B.4): restrict to leaf, opening (remove speckle), closing (merge fragments),
hole filling (spots with a lighter centre), minimum-area filter (drop noise blobs).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .morphology import fill_holes, open_close, remove_small, to_u8


# ----------------------------------------------------------------------------- (a) HSV
def hsv_lesions(img: np.ndarray, leaf: np.ndarray, p: dict) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inside = leaf > 0
    mode = p.get("mode", "nongreen")
    if mode == "nongreen":
        healthy = (H >= p.get("green_h_min", 33)) & (H <= p.get("green_h_max", 90)) & (S >= p.get("green_s_min", 40))
        cand = inside & ~healthy
    elif mode == "powdery":
        cand = inside & (S <= p.get("s_max", 60)) & (V >= p.get("v_min", 160))
    elif mode == "band":  # explicit hue band, e.g. rust orange or chlorosis yellow
        cand = inside & (H >= p["h_min"]) & (H <= p["h_max"]) & (S >= p.get("s_min", 50)) & (V >= p.get("v_min", 50))
    else:
        raise ValueError(f"unknown hsv mode {mode}")
    return to_u8(cand)


# ----------------------------------------------------------------------------- (b) Lab channel
LAB_INDEX = {"L": 0, "a": 1, "b": 2}


def lab_lesions(img: np.ndarray, leaf: np.ndarray, p: dict) -> tuple[np.ndarray, float]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ch = lab[..., LAB_INDEX[p.get("channel", "a")]]
    inside = leaf > 0
    vals = ch[inside]
    if vals.size < 10:
        return np.zeros_like(leaf), float("nan")
    if p.get("fixed") is not None:
        thr = float(p["fixed"])
    else:
        otsu, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(otsu) + p.get("offset", 0)
        if p.get("floor") is not None:
            thr = max(thr, float(p["floor"]))
    if p.get("direction", "high") == "high":
        cand = inside & (ch > thr)
    else:
        cand = inside & (ch < thr)
    return to_u8(cand), thr


# ----------------------------------------------------------------------------- (c) k-means
def _lab_float(img: np.ndarray) -> np.ndarray:
    """Real CIE Lab units: L in 0..100, a*, b* roughly -128..127."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def cluster_scores(centres: np.ndarray, score: str) -> np.ndarray:
    L, a, b = centres[:, 0], centres[:, 1], centres[:, 2]
    if score == "a":
        return a
    if score == "white":
        return L - np.hypot(a, b)
    if score == "hue":
        # Lab hue angle in degrees (0 = red/magenta, ~90 = yellow, ~135-180 = green).
        # "distance from green" = 150 - hue: necrosis/brown/orange score high, green scores ~0.
        hue = np.degrees(np.arctan2(b, a)) % 360
        return 150.0 - np.clip(hue, 0, 180)
    raise ValueError(f"unknown kmeans score {score}")


def kmeans_lesions(img: np.ndarray, leaf: np.ndarray, p: dict, seed: int = 0) -> tuple[np.ndarray, dict]:
    k = int(p.get("k", 4))
    inside = leaf > 0
    lab = _lab_float(img)
    X = lab[inside]
    if len(X) < k * 10:
        return np.zeros_like(leaf), {}
    rng = np.random.default_rng(seed)
    fit_X = X[rng.choice(len(X), size=min(len(X), p.get("max_fit_pixels", 20000)), replace=False)]
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(fit_X)  # fit on a subsample (fast)
    labels = km.predict(X)                                               # then label every leaf pixel
    scores = cluster_scores(km.cluster_centers_, p.get("score", "hue"))
    healthy = int(np.argmin(scores))
    lesion_clusters = [i for i in range(k) if scores[i] - scores[healthy] >= p.get("min_delta", 25)]
    lesion_flat = np.isin(labels, lesion_clusters)
    out = np.zeros(leaf.shape, bool)
    out[inside] = lesion_flat
    # a label image for visualisation (cluster id + 1, 0 = background)
    label_img = np.zeros(leaf.shape, np.uint8)
    label_img[inside] = labels + 1
    info = {"centres_Lab": km.cluster_centers_.round(1).tolist(), "scores": scores.round(1).tolist(),
            "healthy_cluster": healthy, "lesion_clusters": lesion_clusters, "label_img": label_img}
    return to_u8(out), info


# ----------------------------------------------------------------------------- (d) ΔE + hysteresis
def delta_e_map(img: np.ndarray, leaf: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray]:
    from .leaf_seg import exg_index

    inside = leaf > 0
    lab = _lab_float(img)
    exg = exg_index(img)[inside]
    X = lab[inside]
    k = max(10, int(p.get("ref_frac", 0.4) * len(X)))
    ref = np.median(X[np.argsort(-exg)[:k]], axis=0)          # healthy reference colour of THIS leaf
    w = np.array([p.get("w_L", 0.5), 1.0, 1.0], np.float32)
    de = np.zeros(leaf.shape, np.float32)
    de[inside] = np.sqrt((((X - ref) * w) ** 2).sum(1))
    return de, ref


def delta_e_lesions(img: np.ndarray, leaf: np.ndarray, p: dict) -> tuple[np.ndarray, dict]:
    from skimage.filters import apply_hysteresis_threshold

    inside = leaf > 0
    if inside.sum() < 50:
        return np.zeros_like(leaf), {}
    de, ref = delta_e_map(img, leaf, p)
    v = np.clip(de[inside], 0, 100)
    otsu, _ = cv2.threshold((v * 2.55).astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_high = max(otsu / 2.55, p.get("floor", 18.0))
    t_low = p.get("ratio", 0.6) * t_high
    cand = apply_hysteresis_threshold(np.where(inside, de, 0), t_low, t_high)
    return to_u8(cand), {"threshold": float(t_high), "t_low": float(t_low), "ref_Lab": ref.round(1).tolist(),
                         "delta_e": np.clip(de * 2.55, 0, 255).astype(np.uint8)}


# ----------------------------------------------------------------------------- post-processing
def postprocess_lesions(cand: np.ndarray, leaf: np.ndarray, p: dict | None = None) -> np.ndarray:
    p = {**POST_DEFAULTS, **(p or {})}
    m = cv2.bitwise_and(to_u8(cand), to_u8(leaf))
    m = open_close(m, p["open_k"], p["close_k"])
    if p["fill_holes"]:
        m = fill_holes(m)
    m = cv2.bitwise_and(m, to_u8(leaf))     # closing / filling must not leak outside the leaf
    m = remove_small(m, p["min_area_px"])
    return m


def affected_zone(lesion: np.ndarray, leaf: np.ndarray, frac: float) -> np.ndarray:
    """Merge dense lesion clusters into an 'affected zone': closing with a disk whose diameter is
    `frac` x sqrt(leaf area), then hole filling, restricted to the leaf. Used for rust, where tiny
    dense pustules are rated (and annotated in PlantSeg) as one infected area."""
    k = max(3, int(frac * np.sqrt((leaf > 0).sum())))
    z = cv2.morphologyEx(to_u8(lesion), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.bitwise_and(fill_holes(z), to_u8(leaf))


def segment_lesions(img: np.ndarray, leaf: np.ndarray, method: str, profile: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (final lesion/area mask, raw candidate mask, debug info).
    If the profile sets `zone_frac`, the final mask is the affected zone and info["spots"] keeps the
    individual spots (used for the spot count)."""
    info: dict[str, Any] = {}
    if "&" in method:
        # VOTING: a pixel is a lesion candidate only if every listed detector agrees (intersection).
        # Each detector makes different mistakes (Lab: yellow-green veins; HSV: dull green shade;
        # ΔE: lighting changes), so agreement removes many false positives.
        raw = {**profile, "zone_frac": None, "post": {"open_k": 0, "close_k": 0, "fill_holes": False, "min_area_px": 0}}
        parts = method.split("&")
        cand = segment_lesions(img, leaf, parts[0], raw)[0]
        for part in parts[1:]:
            cand = cv2.bitwise_and(cand, segment_lesions(img, leaf, part, raw)[0])
        clean = postprocess_lesions(cand, leaf, profile.get("post"))
        info["spots"] = clean
        if profile.get("zone_frac"):
            clean = affected_zone(clean, leaf, float(profile["zone_frac"]))
        return clean, cand, info
    if method == "hsv":
        cand = hsv_lesions(img, leaf, profile.get("hsv", {}))
    elif method == "lab":
        cand, thr = lab_lesions(img, leaf, profile.get("lab", {}))
        info["threshold"] = thr
    elif method == "kmeans":
        cand, info = kmeans_lesions(img, leaf, profile.get("kmeans", {}))
    elif method == "delta_e":
        cand, info = delta_e_lesions(img, leaf, profile.get("delta_e", {}))
    else:
        raise ValueError(f"unknown lesion method {method}")
    clean = postprocess_lesions(cand, leaf, profile.get("post"))
    info["spots"] = clean
    if profile.get("zone_frac"):
        clean = affected_zone(clean, leaf, float(profile["zone_frac"]))
    return clean, cand, info


POST_DEFAULTS = {"open_k": 3, "close_k": 5, "fill_holes": True, "min_area_px": 20}
