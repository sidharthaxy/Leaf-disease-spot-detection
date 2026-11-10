"""
Demo backend (FastAPI, CPU only, light enough for an 8 GB Mac).

    .venv/bin/uvicorn demo.backend.app:app --port 8000        (or: bash demo/run_demo.sh)

GET  /api/health
POST /api/analyze                  multipart "file" -> our system + the standard pipeline side by side
GET  /api/examples                 the example gallery (unseen TEST images with expert masks)
GET  /api/examples/{id}/compare    standard vs ours vs expert ground truth, with Dice and infected %

Router (models/router, optional): decides crop pipeline / palette branch / other houseplant / out of scope
before anything else (src/router/router.py). The crop route runs exactly the code it ran before.

Classifier: configs/inference.yaml (ensemble, temperature-calibrated), unless MODEL_DIR is set.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.evaluation.metrics import mask_metrics
from src.imageproc import viz
from src.imageproc.pipeline import run_pipeline
from src.imageproc.preprocess import resize_mask_like
from src.imageproc.plant_check import check_plant
from src.imageproc.preprocess import preprocess
from src.imageproc.standard import run_standard
from src.severity.grade import load_yaml, resolve_from_prediction, severity_label

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "outputs" / "model"))
EXAMPLES = ROOT / "demo" / "examples"
MAX_UPLOAD_SIDE = 1600   # downscale huge phone photos right away (RAM)
GROUPS = ("necrotic", "chlorosis", "rust", "powdery", "generic", "healthy")

app = FastAPI(title="Leaf disease severity demo")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

classifier = None
gradcam = None
learned = None   # XGBoost severity correction (src/evaluation/learned_correction.py), if trained
router = None    # plant router (src/router/router.py), if models/router exists
ROUTER_DIR = ROOT / "models" / "router"


@app.on_event("startup")
def load_models():
    """Load ONE classifier instance (ONNX), Grad-CAM weights and the learned correction, if present."""
    global classifier, gradcam, learned, router, MODEL_DIR
    inference_cfg = ROOT / "configs" / "inference.yaml"
    if "MODEL_DIR" not in os.environ and inference_cfg.exists():
        from src.classifier.infer import EnsembleClassifier

        classifier = EnsembleClassifier(inference_cfg)
        MODEL_DIR = ROOT / classifier.model_dirs[0]   # Grad-CAM explains the first (strongest) model
    elif (MODEL_DIR / "model.onnx").exists():
        from src.classifier.infer import OnnxClassifier

        classifier = OnnxClassifier(MODEL_DIR)
    if (MODEL_DIR / "best.pt").exists() and os.environ.get("GRADCAM", "1") == "1":
        import torch

        torch.set_num_threads(2)
        from src.classifier.gradcam import GradCAM

        gradcam = GradCAM(MODEL_DIR / "best.pt")
    lc = ROOT / "models" / "learned"
    if (lc / "xgb_regressor.json").exists() and (lc / "meta.json").exists():
        from xgboost import XGBRegressor

        reg = XGBRegressor()
        reg.load_model(lc / "xgb_regressor.json")
        learned = {"model": reg, "features": json.loads((lc / "meta.json").read_text())["features"]}
    if (ROUTER_DIR / "router_config.json").exists() and os.environ.get("ROUTER", "1") == "1":
        from src.router.router import Router

        router = Router(ROUTER_DIR)
    print(f"classifier={'yes' if classifier else 'NO'}  gradcam={'yes' if gradcam else 'no'}  "
          f"learned severity={'yes' if learned else 'no'}  router={'yes' if router else 'no'}  examples={EXAMPLES.exists()}")


# ----------------------------------------------------------------------------- helpers
def b64(img: np.ndarray, max_side: int = 900) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def diff_image(base: np.ndarray, ours: np.ndarray, standard: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
    """Pixel-level difference: blue = only ours, red = only standard, yellow = both; GT outline in white."""
    out = (base * 0.45).astype(np.uint8)
    o, s = ours > 0, standard > 0
    out[o & ~s] = (230, 120, 30)      # BGR blue-ish  : only our method
    out[s & ~o] = (40, 40, 220)       # BGR red       : only the standard method
    out[o & s] = (0, 210, 240)        # BGR yellow    : both
    if gt is not None:
        cs, _ = cv2.findContours((gt > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (255, 255, 255), 2)
    return out


def ours_full(img: np.ndarray) -> tuple[dict, dict]:
    """Our full system on one image. Returns (json-ready result, internal masks)."""
    pred = classifier.predict(img, k=len(classifier.maps["labels"])) if classifier else None
    res = resolve_from_prediction(pred["topk"] if pred else None)
    out = run_pipeline(img, group=res["group"])
    m = out["measurements"]

    estimated = None
    if learned is not None:
        from src.evaluation.features import image_features
        from src.imageproc.preprocess import preprocess

        pre, _ = preprocess(img)
        feats = {f"f_{k}": v for k, v in image_features(pre, out["leaf_mask"], res["group"]).items()}
        feats.update({f"g_{g}": int(res["group"] == g) for g in GROUPS})
        x = np.array([[feats.get(c, 0.0) for c in learned["features"]]], float)
        estimated = float(np.clip(learned["model"].predict(x)[0], 0, 100))

    images = {k: b64(v) for k, v in out["steps"].items()}
    cam_label = None
    if gradcam is not None:
        cam_img, cam_label = gradcam.overlay(img)
        images["gradcam"] = b64(cam_img)
    label = res["label"]
    non_leaf = label in set(load_yaml("disease_profiles.yaml").get("non_leaf_labels") or [])
    result = {
        "classifier": {
            "available": classifier is not None,
            "plant": pred["plant"][0] if pred else None,
            "plant_confidence": pred["plant"][1] if pred else None,
            "label": label,
            "confidence": res["confidence"],
            "uncertain": res["uncertain"],
            "min_confidence": load_yaml("severity.yaml")["confidence"]["min_confidence"],
            "top3": [{"label": l, "confidence": c} for l, c in res["topk"]],
            "gradcam_label": cam_label,
        },
        "severity": {
            "group": m["group"],
            "infected_pct": round(m["infected_pct"], 2),
            "severity": m["severity"],
            "n_spots": m["n_spots"],
            "mean_spot_area_px": round(m["mean_spot_area_px"], 1),
            "leaf_pct_of_image": round(m["leaf_pct_of_image"], 1),
            "leaf_method": m["leaf_method"],
            "lesion_method": m["lesion_method"],
            "non_leaf_warning": non_leaf,
            "estimated_affected_pct": None if estimated is None else round(estimated, 1),
            "estimated_severity": None if estimated is None else severity_label(estimated, res["group"]),
        },
        "images": images,
    }
    masks = {"leaf": out["leaf_mask"], "lesion": out["lesion_mask"], "base": out["steps"]["original"]}
    return result, masks


def router_json(route: dict) -> dict:
    from src.palette.species import display

    return {"route": route["route"], "reason": route["reason"],
            "species": display(route["species"]) if route.get("species") else None,
            "top3": [{**t, "name": display(t["class"])["common"] if t["class"] not in ("crop", "other_houseplant", "non_plant")
                      else {"crop": "Crop plant (PlantSeg/PlantDoc classes)", "other_houseplant": "Other houseplant",
                            "non_plant": "Not a plant"}[t["class"]]} for t in route["top3"]]}


def palette_result(img: np.ndarray, species: str) -> dict:
    """Palette branch for one image (src/palette/palette.py): abnormal tissue vs the species' normal colours."""
    from src.palette.palette import Palette, run_palette
    from src.palette.species import display

    pal = Palette.load(species)
    out = run_palette(img, pal)
    m = out["measurements"]
    return {"species": display(species), "abnormal_pct": m["abnormal_pct"], "label": m["label"], "n_spots": m["n_spots"],
            "leaf_pct_of_image": round(100 * m["leaf_frac_of_image"], 1), "swatches": pal.swatches_bgr(),
            "cutoffs_pct": pal.cutoffs, "images": {k: b64(v) for k, v in out["steps"].items()}}


def routed(img: np.ndarray, gate: dict, t0: float) -> dict | None:
    """Router decision. Returns a finished response for non-crop routes, or None -> run the crop pipeline."""
    if router is None:
        if gate["status"] == "out_of_scope":
            return {"rejected": True, "plant_check": gate, "images": {"original": b64(img)}, "timing_s": round(time.time() - t0, 2)}
        return None
    route = router.route(img, gate["green_pct"])
    rj = router_json(route)
    if route["route"] == "out_of_scope":
        return {"rejected": True, "plant_check": gate, "router": rj, "images": {"original": b64(img)},
                "timing_s": round(time.time() - t0, 2)}
    if route["route"] == "palette" and not (ROOT / "models" / "palettes" / f"{route['species']}.json").exists():
        rj.update({"route": "crop_uncertain", "reason": rj["reason"] + " (no palette model installed; using the crop pipeline)"})
        return {"crop_router": rj}
    if route["route"] == "palette":
        return {"route": "palette", "router": rj, "plant_check": gate, "palette": palette_result(img, route["species"]),
                "images": {"original": b64(img)}, "timing_s": round(time.time() - t0, 2)}
    if route["route"] == "other_houseplant":
        return {"route": "other_houseplant", "router": rj, "plant_check": gate, "images": {"original": b64(img)},
                "timing_s": round(time.time() - t0, 2)}
    return {"crop_router": rj}  # crop / crop_uncertain -> unchanged crop pipeline


def standard_json(std: dict) -> dict:
    m = std["measurements"]
    return {"infected_pct": round(m["infected_pct"], 2), "severity": severity_label(m["infected_pct"]),
            "n_spots": m["n_spots"], "leaf_pct_of_image": round(m["leaf_pct_of_image"], 1),
            "images": {"overlay": b64(std["overlay"]), "leaf mask": b64(std["leaf_view"])}}


def read_upload(data: bytes) -> np.ndarray:
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "image too large (max 15 MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    h, w = img.shape[:2]
    s = min(1.0, MAX_UPLOAD_SIDE / max(h, w))
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


# ----------------------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"ok": True, "classifier": classifier is not None, "gradcam": gradcam is not None,
            "learned_severity": learned is not None, "examples": EXAMPLES.exists()}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    t0 = time.time()
    img = read_upload(await file.read())
    # 1) scope gate: is there green leaf tissue? The CNN is closed-set and the leaf mask is relative, so without
    #    this a hand or a wall would be analysed as a leaf; non-green leaves are also out of scope
    #    (src/imageproc/plant_check.py)
    gate = check_plant(preprocess(img)[0], load_yaml("disease_profiles.yaml").get("plant_check"))
    # 2) router: crop pipeline / palette branch / other houseplant / out of scope (src/router/router.py)
    early = routed(img, gate, t0)
    if early is not None and "crop_router" not in early:
        return early
    result, masks = ours_full(img)
    result["plant_check"] = gate
    if early is not None:
        result["router"] = early["crop_router"]
    std = run_standard(img)
    result["standard"] = standard_json(std)
    result["comparison"] = {"diff": b64(diff_image(masks["base"], masks["lesion"], std["lesion_mask"]))}
    result["timing_s"] = round(time.time() - t0, 2)
    return result


def _examples() -> pd.DataFrame:
    if not (EXAMPLES / "manifest.csv").exists():
        raise HTTPException(404, "no examples found in demo/examples")
    return pd.read_csv(EXAMPLES / "manifest.csv").fillna("")


@app.get("/api/examples")
def examples():
    df = _examples()
    out = []
    for r in df.itertuples():
        img = cv2.imread(str(EXAMPLES / str(r.image_path)))
        out.append({"id": int(str(r.example_id)), "label": str(r.label), "plant": str(r.plant), "disease": str(r.disease),
                    "has_mask": bool(r.mask_path), "kind": str(getattr(r, "kind", "") or "crop"), "thumb": b64(img, max_side=220)})
    return {"note": "Official TEST-split images: never used to train the classifier, tune the pipeline or "
                    "fit the learned correction. Ornamental examples (palette branch) are held-out test photos "
                    "from iNaturalist.", "examples": out}


@app.get("/api/examples/{example_id}/compare")
def compare(example_id: int):
    t0 = time.time()
    df = _examples()
    row = df[df.example_id == example_id]
    if row.empty:
        raise HTTPException(404, "unknown example")
    r = row.iloc[0]
    img = cv2.imread(str(EXAMPLES / r.image_path))
    if str(r.get("kind", "")) == "palette":   # ornamental example: router -> palette branch (no expert mask)
        gate = check_plant(preprocess(img)[0], load_yaml("disease_profiles.yaml").get("plant_check"))
        res = routed(img, gate, t0) or {}
        if "crop_router" in res:   # router sent it to the crop pipeline: show that honestly
            res, _ = ours_full(img)
            res["plant_check"] = gate
        res["example"] = {"label": r.label, "licence": r.get("licence", ""), "attribution": r.get("attribution", ""),
                          "url": r.get("url", ""), "note": r.get("note", "")}
        res["images"] = {**res.get("images", {}), "original": b64(img)}
        res["timing_s"] = round(time.time() - t0, 2)
        return res
    result, masks = ours_full(img)
    result["plant_check"] = check_plant(preprocess(img)[0], load_yaml("disease_profiles.yaml").get("plant_check"))
    std = run_standard(img)
    base = masks["base"]
    gt = None
    if r.mask_path:
        gt = resize_mask_like(cv2.imread(str(EXAMPLES / r.mask_path), cv2.IMREAD_GRAYSCALE), base)
    else:
        gt = np.zeros(base.shape[:2], np.uint8)     # healthy leaf: expert truth = no lesions
    # expert infected %: same shared denominator as the evaluation, |our leaf ∪ GT| (README §9)
    denom = max(1, int(((masks["leaf"] > 0) | (gt > 0)).sum()))
    expert_pct = 100 * (gt > 0).sum() / denom

    def scores(lesion):
        mm = mask_metrics(lesion, gt)
        clean = lambda v: None if v != v else round(float(v), 3)       # NaN (healthy) -> null
        return {"dice": clean(mm["dice"]), "iou": clean(mm["iou"]), "precision": clean(mm["precision"]),
                "recall": clean(mm["recall"])}

    result["standard"] = standard_json(std)
    result["standard"]["scores"] = scores(std["lesion_mask"])
    result["severity"]["scores"] = scores(masks["lesion"])
    result["expert"] = {
        "label": r.label, "infected_pct": round(float(expert_pct), 2),
        "severity": severity_label(expert_pct), "healthy": not bool(r.mask_path),
        "overlay": b64(viz.overlay(base, None, gt)),
    }
    result["comparison"] = {"diff": b64(diff_image(base, masks["lesion"], std["lesion_mask"], gt))}
    result["timing_s"] = round(time.time() - t0, 2)
    return result
