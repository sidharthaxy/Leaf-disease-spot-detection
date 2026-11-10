"""
Collect the ROUTER dataset (runs on AWS as a CPU Processing job; small smoke tests run locally).

    python -m src.router.collect_router_data --out data/router --work data/_router_dl \
        [--skip-crop] [--skip-kaggle] [--skip-coco] [--per-species 300] [--limit-smoke N]

Router classes (what kind of plant / image is this?):
  crop              PlantSeg + PlantDoc-healthy TRAIN images (stratified over the 125 crop classes);
                    PlantSeg/PlantDoc VAL and TEST images are kept separately for the misrouting check,
                    they never enter router training.
  <palette species> 10 ornamental species from iNaturalist (CC-licensed photos, provenance recorded):
                    croton, coleus, caladium, purple_heart, copperleaf, polka_dot, bloodleaf, red_cordyline
                    (non-green / pink) and golden_pothos, dieffenbachia (variegated green).
  other_houseplant  Kaggle "House Plant Species" (CC BY-NC-SA 4.0), minus classes that ARE palette species
                    (Dumb Cane, Pothos, Polka Dot Plant) or ambiguous ones (Tradescantia = whole genus).
  non_plant         COCO 2017 val images with no plant/fruit/vegetable objects and (almost) no green
                    vegetation, plus GREEN non-plant objects from Wikimedia Commons categories.

Every image: resized so the long side <= 384 px (JPEG q90), dHash computed; near-duplicates (Hamming <= 4)
removed within and across classes and against the crop images; split 70/15/15 per class (crop keeps the
official PlantSeg / PlantDoc splits). Output: <out>/images/<class>/<split>/*.jpg, <out>/manifest.csv
(image_path, label, split, source, licence, attribution, url, dhash) and contact sheets in <out>/sheets/.
Images are NOT committed to git (dataset licences); the manifest + this script are.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import random
import subprocess
import sys
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "leaf-disease-student-project/1.0 (NIT Rourkela academic research)"}
MAX_SIDE = 384

PALETTE_SPECIES = {  # router label -> iNaturalist taxon id (checked via the iNat API, 2026-11-10)
    "croton": 127387, "coleus": 291284, "caladium": 283377, "purple_heart": 169872, "copperleaf": 281562,
    "polka_dot": 164011, "bloodleaf": 914138, "red_cordyline": 123572, "golden_pothos": 69223,
    "dieffenbachia": 206266,
}
KAGGLE_DATASET = "kacpergregorowicz/house-plant-species"
KAGGLE_EXCLUDE = ("Dumb Cane", "Pothos", "Polka Dot Plant", "Tradescantia")  # palette species / ambiguous genus
WIKI_GREEN_CATEGORIES = ["Green doors", "Green T-shirts", "Green shirts", "Green clothing", "Green bottles",
                         "Green chairs", "Green balloons", "Green buses", "Green toys", "Green tables", "Green bags"]
COCO_EXCLUDE = {"potted plant", "broccoli", "banana", "apple", "orange", "carrot"}
ICC = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"


# ----------------------------------------------------------------------------- helpers
def http_json(url: str, tries: int = 6) -> dict:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001  (429 / transient network errors: back off and retry)
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    return {}


def http_bytes(url: str, tries: int = 4) -> bytes | None:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            time.sleep(2 ** k)
    return None


def save_resized(data_or_path, dst: Path) -> int | None:
    """Decode, resize (long side <= MAX_SIDE), save JPEG. Returns dHash, or None if unreadable."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data_or_path) if isinstance(data_or_path, bytes) else data_or_path)
        img.draft("RGB", (MAX_SIDE * 2, MAX_SIDE * 2))
        img = img.convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    s = min(1.0, MAX_SIDE / max(img.size))
    if s < 1:
        img = img.resize((max(1, round(img.size[0] * s)), max(1, round(img.size[1] * s))), Image.Resampling.LANCZOS)
    if min(img.size) < 96:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=90)
    return dhash(img)


def dhash(img, size: int = 8) -> int:
    import numpy as np

    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


# ----------------------------------------------------------------------------- sources
def collect_palette(out: Path, per_species: int, rows: list) -> None:
    """iNaturalist observations with CC-licensed photos (first photo of each observation)."""
    for label, taxon in PALETTE_SPECIES.items():
        got, page = 0, 1
        jobs = []
        while got < per_species and page <= 10:
            # exclude observations annotated as flowering (13) or fruiting (14): we want LEAVES, and the most
            # popular iNat photos of e.g. pothos or cordyline are disproportionately rare flowers / berries
            q = {"taxon_id": taxon, "photo_license": ICC, "photos": "true", "per_page": 200, "page": page,
                 "without_term_value_id": "13,14"}
            res = http_json("https://api.inaturalist.org/v1/observations?" + urllib.parse.urlencode(q))
            time.sleep(1.1)  # iNaturalist asks for <= 1 request / second
            obs = res.get("results", [])
            if not obs:
                break
            for o in obs:
                ph = next((p for p in o.get("photos", []) if p.get("license_code")), None)
                if not ph or got >= per_species:
                    continue
                url = ph["url"].replace("/square.", "/medium.")
                jobs.append((url, out / "images" / label / "all" / f"inat_{o['id']}_{ph['id']}.jpg",
                             {"label": label, "source": "inaturalist", "licence": ph["license_code"],
                              "attribution": ph.get("attribution", ""), "url": f"https://www.inaturalist.org/observations/{o['id']}"}))
                got += 1
            page += 1
        _download_all(jobs, rows)
        print(f"[palette] {label}: {sum(1 for r in rows if r['label'] == label)} images", flush=True)


def collect_wikimedia_green(out: Path, per_category: int, rows: list) -> None:
    jobs = []
    for cat in WIKI_GREEN_CATEGORIES:
        q = {"action": "query", "generator": "categorymembers", "gcmtitle": "Category:" + cat, "gcmtype": "file",
             "gcmlimit": per_category, "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": 512,
             "format": "json"}
        res = http_json("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q))
        time.sleep(2.0)  # Commons rate-limits bursts
        for page in (res.get("query", {}).get("pages", {}) or {}).values():
            ii = (page.get("imageinfo") or [{}])[0]
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            artist = meta.get("Artist", {}).get("value", "")
            jobs.append((ii.get("thumburl") or ii.get("url"),
                         out / "images" / "non_plant" / "all" / f"wiki_{page['pageid']}.jpg",
                         {"label": "non_plant", "source": f"wikimedia:{cat}", "licence": lic,
                          "attribution": artist[:200], "url": ii.get("descriptionurl", "")}))
    _download_all(jobs, rows, workers=2, pause=0.5)
    print(f"[wikimedia] green non-plant: {sum(1 for r in rows if r['source'].startswith('wikimedia'))} images", flush=True)


def _download_all(jobs, rows, workers: int = 8, pause: float = 0.0):
    def one(j):
        url, dst, meta = j
        data = http_bytes(url)
        if pause:
            time.sleep(pause)
        h = save_resized(data, dst) if data else None
        return (dst, meta, h)

    with cf.ThreadPoolExecutor(workers) as ex:
        for dst, meta, h in ex.map(one, jobs):
            if h is not None:
                rows.append({**meta, "image_path": str(dst), "dhash": h})


def kaggle_download(dataset: str, work: Path) -> None:
    """Download + unzip a Kaggle dataset over the public HTTP API with the new-style access token
    (~/.kaggle/access_token, sent as a Bearer header). Independent of the kaggle CLI version (the CLI that
    pip picks for Python 3.9 only understands the old kaggle.json username/key format)."""
    token = (Path.home() / ".kaggle" / "access_token").read_text().strip()

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    req = urllib.request.Request(f"https://www.kaggle.com/api/v1/datasets/download/{dataset}",
                                 headers={**UA, "Authorization": f"Bearer {token}"})
    try:
        urllib.request.build_opener(NoRedirect).open(req, timeout=60)
        raise RuntimeError("expected a redirect to the dataset archive")
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307):
            raise
        url = e.headers["Location"]          # signed storage URL: fetched WITHOUT the token
    zpath = work / "kaggle.zip"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    print(f"[kaggle] downloaded {zpath.stat().st_size / 1e9:.2f} GB", flush=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(work)
    zpath.unlink()


def collect_kaggle(out: Path, work: Path, per_class: int, rows: list, seed: int) -> None:
    work.mkdir(parents=True, exist_ok=True)
    kaggle_download(KAGGLE_DATASET, work)
    root = next(p for p in work.rglob("*") if p.is_dir() and p.name.lower() == "house_plant_species")
    rng = random.Random(seed)
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(cls.name.startswith(x) for x in KAGGLE_EXCLUDE):
            print(f"[kaggle] skip {cls.name}", flush=True)
            continue
        files = sorted(f for f in cls.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        rng.shuffle(files)
        for i, f in enumerate(files[:per_class]):
            dst = out / "images" / "other_houseplant" / "all" / f"kaggle_{cls.name[:20].replace(' ', '_')}_{i:03d}.jpg"
            h = save_resized(f, dst)
            if h is not None:
                rows.append({"label": "other_houseplant", "source": f"kaggle:{cls.name}", "licence": "CC BY-NC-SA 4.0 (dataset)",
                             "attribution": "kacpergregorowicz/house-plant-species", "url": "", "image_path": str(dst), "dhash": h})
    print(f"[kaggle] other_houseplant: {sum(1 for r in rows if r['label'] == 'other_houseplant')} images", flush=True)


def collect_coco(out: Path, work: Path, n: int, rows: list, seed: int) -> None:
    """COCO 2017 val images without plant/fruit/vegetable objects and with (almost) no green vegetation."""
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.imageproc.plant_check import green_fraction

    work.mkdir(parents=True, exist_ok=True)
    for name in ("val2017.zip", "annotations_trainval2017.zip"):
        url = ("http://images.cocodataset.org/zips/" if name.startswith("val") else
               "http://images.cocodataset.org/annotations/") + name
        dst = work / name
        if not dst.exists():
            subprocess.check_call(["curl", "-sSfL", "-o", str(dst), url])
    with zipfile.ZipFile(work / "annotations_trainval2017.zip") as z:
        ann = json.load(z.open("annotations/instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    bad = {a["image_id"] for a in ann["annotations"] if cats[a["category_id"]] in COCO_EXCLUDE}
    ids = [im for im in ann["images"] if im["id"] not in bad]
    random.Random(seed).shuffle(ids)
    taken = 0
    with zipfile.ZipFile(work / "val2017.zip") as z:
        for im in ids:
            if taken >= n:
                break
            data = z.read("val2017/" + im["file_name"])
            arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))[:, :, ::-1].copy()
            if green_fraction(arr) > 0.02:  # visible vegetation (grass, trees) -> not a clean non-plant example
                continue
            dst = out / "images" / "non_plant" / "all" / f"coco_{im['id']}.jpg"
            h = save_resized(data, dst)
            if h is not None:
                rows.append({"label": "non_plant", "source": "coco2017-val", "licence": f"flickr licence id {im['license']}",
                             "attribution": "COCO 2017", "url": im.get("flickr_url", ""), "image_path": str(dst), "dhash": h})
                taken += 1
    print(f"[coco] non_plant: {taken} images", flush=True)


def collect_crop(out: Path, work: Path, n_train: int, rows: list, seed: int) -> None:
    """PlantSeg + PlantDoc via the existing data scripts; TRAIN sample for router training, all VAL/TEST for eval."""
    import pandas as pd

    root = Path(__file__).resolve().parents[2]
    raw, proc = work / "raw", work / "processed"
    subprocess.check_call([sys.executable, "-m", "src.data.prepare_raw_data", "--out", str(raw), "--work", str(work / "dl")], cwd=root)
    subprocess.check_call([sys.executable, "-m", "src.data.build_manifest", "--plantseg", str(raw / "plantseg" / "plantseg"),
                           "--plantdoc", str(raw / "plantdoc" / "PlantDoc-Dataset-master"), "--out", str(proc)], cwd=root)
    m = pd.read_csv(proc / "manifest.csv")
    m = m[~(m.dup_of.notna() & (m.split != "test"))]
    tr = m[m.split == "train"]
    # stratified: every crop class contributes, proportionally, at least 8 images
    per = (tr.label.value_counts() / len(tr) * n_train).clip(lower=8).round().astype(int)
    picks = pd.concat([g.sample(min(len(g), per[lab]), random_state=seed) for lab, g in tr.groupby("label")])
    sel = pd.concat([picks, m[m.split.isin(["val", "test"])]])
    for r in sel.itertuples():
        dst = out / "images" / "crop" / str(r.split) / Path(str(r.image_path)).name
        h = save_resized(proc / str(r.image_path), dst)
        if h is not None:
            rows.append({"label": "crop", "split": str(r.split), "source": f"{r.source}:{r.label}",
                         "licence": "CC BY-NC 4.0 (PlantSeg)" if r.source == "plantseg" else "CC BY 4.0 (PlantDoc)",
                         "attribution": str(r.source), "url": "", "image_path": str(dst), "dhash": h})
    print(f"[crop] train {len(picks)}, val {int((sel.split == 'val').sum())}, test {int((sel.split == 'test').sum())}", flush=True)


# ----------------------------------------------------------------------------- cleaning, splits, sheets
def dedupe_and_split(rows: list, out: Path, seed: int) -> list:
    """Drop near-duplicates (dHash Hamming <= 4): within a class, across classes, and against crop images.
    Crop images keep their official split; every other class is split 70/15/15."""
    crop = [r for r in rows if r["label"] == "crop"]
    other = [r for r in rows if r["label"] != "crop"]
    kept_hashes = [int(r["dhash"]) for r in crop]
    kept, dropped, seen_paths = [], 0, set()
    for r in other:
        if r["image_path"] in seen_paths:   # same file listed twice (e.g. one Commons file in two categories)
            dropped += 1
            continue
        seen_paths.add(r["image_path"])
        h = int(r["dhash"])
        if any(bin(h ^ k).count("1") <= 4 for k in kept_hashes):
            Path(r["image_path"]).unlink(missing_ok=True)
            dropped += 1
            continue
        kept_hashes.append(h)
        kept.append(r)
    print(f"[dedupe] removed {dropped} near-duplicates", flush=True)
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for r in kept:
        by_label.setdefault(r["label"], []).append(r)
    final = list(crop)
    for label, rs in by_label.items():
        rng.shuffle(rs)
        n = len(rs)
        for i, r in enumerate(rs):
            split = "train" if i < 0.70 * n else "val" if i < 0.85 * n else "test"
            src = Path(r["image_path"])
            dst = out / "images" / label / split / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
            final.append({**r, "split": split, "image_path": str(dst)})
    for d in out.glob("images/*/all"):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return final


def contact_sheets(rows: list, out: Path, per_page: int = 120, cols: int = 12, tile: int = 120) -> None:
    """Numbered thumbnail grids per class, for a human label check before training."""
    from PIL import Image, ImageDraw

    sheets = out / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    by_label: dict[str, list] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    for label, rs in sorted(by_label.items()):
        if label == "crop":  # crop data is the curated PlantSeg/PlantDoc set: one sample page is enough
            rs = random.Random(0).sample(rs, min(per_page, len(rs)))
        if label == "other_houseplant":
            rs = random.Random(0).sample(rs, min(2 * per_page, len(rs)))
        rs = sorted(rs, key=lambda r: r["image_path"])
        for p0 in range(0, len(rs), per_page):
            page = rs[p0:p0 + per_page]
            rows_n = (len(page) + cols - 1) // cols
            canvas = Image.new("RGB", (cols * tile, rows_n * (tile + 14)), "white")
            draw = ImageDraw.Draw(canvas)
            for i, r in enumerate(page):
                try:
                    im = Image.open(r["image_path"]).convert("RGB")
                except Exception:  # noqa: BLE001
                    continue
                im.thumbnail((tile, tile))
                x, y = (i % cols) * tile, (i // cols) * (tile + 14)
                canvas.paste(im, (x + (tile - im.size[0]) // 2, y))
                draw.text((x + 2, y + tile), f"{Path(r['image_path']).stem[-14:]}", fill=(0, 0, 0))
            canvas.save(sheets / f"{label}_{p0 // per_page + 1:02d}.jpg", quality=70)
    print(f"[sheets] written to {sheets}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/router")
    ap.add_argument("--work", default="data/_router_dl")
    ap.add_argument("--per-species", type=int, default=300)
    ap.add_argument("--per-kaggle-class", type=int, default=50)
    ap.add_argument("--n-coco", type=int, default=450)
    ap.add_argument("--per-wiki-category", type=int, default=30)
    ap.add_argument("--n-crop-train", type=int, default=2800)
    ap.add_argument("--skip-crop", action="store_true")
    ap.add_argument("--skip-kaggle", action="store_true")
    ap.add_argument("--skip-coco", action="store_true")
    ap.add_argument("--skip-wiki", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    steps = [("crop", not a.skip_crop, lambda: collect_crop(out, work, a.n_crop_train, rows, a.seed)),
             ("palette", True, lambda: collect_palette(out, a.per_species, rows)),
             ("wiki", not a.skip_wiki, lambda: collect_wikimedia_green(out, a.per_wiki_category, rows)),
             ("coco", not a.skip_coco, lambda: collect_coco(out, work, a.n_coco, rows, a.seed)),
             ("kaggle", not a.skip_kaggle, lambda: collect_kaggle(out, work, a.per_kaggle_class, rows, a.seed))]
    failed = []
    for name, on, fn in steps:  # one failing source must not lose the others (job output is uploaded at the end)
        if not on:
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failed.append(name)
            print(f"[{name}] FAILED, continuing with the other sources", flush=True)
    final = dedupe_and_split(rows, out, a.seed)
    rel = lambda p: str(Path(p).relative_to(out))
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "split", "source", "licence", "attribution", "url", "dhash"])
        w.writeheader()
        for r in final:
            w.writerow({**{k: r.get(k, "") for k in w.fieldnames}, "image_path": rel(r["image_path"])})
    contact_sheets(final, out)
    counts: dict = {}
    for r in final:
        counts.setdefault(r["label"], {}).setdefault(r["split"], 0)
        counts[r["label"]][r["split"]] += 1
    print(json.dumps(counts, indent=1), flush=True)
    if failed:
        print(f"[warn] failed sources: {failed}", flush=True)
    print(f"[done] {len(final)} images in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
