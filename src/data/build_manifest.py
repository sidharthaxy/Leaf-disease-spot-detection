"""
Data step 2: build the manifest + a training-ready (resized) copy.   python -m src.data.build_manifest

Verified facts this relies on (from the step-1 inspection report, PlantSeg Zenodo v7):
  - plantseg/{images,annotations}/{train,val,test}/<name>.{jpg,png}; official split in Metadata.csv
  - mask = grayscale PNG; 0 = background, lesion pixels = Metadata.Index + 1 (one disease per image)
  - 7,774 images, 34 plants, 115 diseases, NO healthy images
  - PlantDoc healthy classes are folders named "<Plant> leaf" (e.g. "Apple leaf", "grape leaf")

Outputs (-> data/processed/, sample -> data/sample/):
  manifest.csv          image_path, mask_path, plant, disease, label, split, source, mask_value,
                        mask_ratio, orig_h, orig_w, dhash, dup_of
  images/<split>/...    JPEGs resized so the long side <= 768 (quality 95)
  masks/<split>/...     binary lesion masks (0/255) resized with nearest neighbour
  class_counts.csv
  sample/               ~30 stratified images + masks for local development on the Mac
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

MAX_SIDE = 768
SPLIT_MAP = {"Training": "train", "Validation": "val", "Test": "test"}
PLANTDOC_HEALTHY = {  # PlantDoc folder -> PlantSeg plant name
    "Apple leaf": "Apple", "Bell_pepper leaf": "Bell pepper", "Blueberry leaf": "Blueberry",
    "Cherry leaf": "Cherry", "Peach leaf": "Peach", "Raspberry leaf": "Raspberry",
    "Soyabean leaf": "Soybean", "Strawberry leaf": "Strawberry", "Tomato leaf": "Tomato", "grape leaf": "Grape",
}


def ensure_deps():
    try:
        import numpy, pandas, PIL  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pillow", "numpy", "pandas"])


def label_of(plant: str, disease: str) -> str:
    """('Tomato', 'tomato early blight') -> 'Tomato___early_blight'"""
    d = disease.strip().lower()
    p = plant.strip().lower()
    if d.startswith(p + " "):
        d = d[len(p) + 1:]
    return f"{plant.strip().replace(' ', '_')}___{d.replace(' ', '_')}"


def dhash(img, size=8) -> int:
    """Difference hash: shrink to (size+1) x size grey, compare neighbours -> 64-bit fingerprint.
    Near-identical images (re-encoded / resized copies) have hashes within a few bits."""
    import numpy as np

    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def process_one(job):
    """Resize image (+ binarise and resize mask). Returns metadata dict."""
    from PIL import Image
    import numpy as np

    src_img, src_mask, dst_img, dst_mask, mask_value = job
    img = Image.open(src_img)
    img.draft("RGB", (MAX_SIDE * 2, MAX_SIDE * 2))
    img = img.convert("RGB")
    w, h = img.size  # NOTE: no EXIF rotation - PlantSeg masks were drawn on the raw pixel grid
    s = min(1.0, MAX_SIDE / max(img.size))
    new = (max(1, round(img.size[0] * s)), max(1, round(img.size[1] * s)))
    small = img.resize(new, Image.LANCZOS) if s < 1 else img
    Path(dst_img).parent.mkdir(parents=True, exist_ok=True)
    small.save(dst_img, quality=95)
    out = {"orig_w": w, "orig_h": h, "dhash": dhash(small)}
    if src_mask:
        m = np.asarray(Image.open(src_mask))
        out["mask_size_matches"] = bool(m.shape[:2] == (h, w))
        out["mask_values_found"] = ",".join(str(v) for v in np.unique(m) if v)
        # any non-zero pixel is lesion: one disease per image, and 12 PlantSeg masks carry a
        # neighbouring class value (annotation slip), which an exact match would turn into empty masks
        binm = Image.fromarray(((m > 0) * 255).astype("uint8"))
        binm = binm.resize(new, Image.NEAREST)
        Path(dst_mask).parent.mkdir(parents=True, exist_ok=True)
        binm.save(dst_mask)
        out["mask_ratio_resized"] = float((np.asarray(binm) > 0).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantseg", default="data/raw/plantseg/plantseg")
    ap.add_argument("--plantdoc", default="data/raw/plantdoc/PlantDoc-Dataset-master")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    ensure_deps()
    import pandas as pd

    rng = random.Random(args.seed)
    seg, doc, out = Path(args.plantseg), Path(args.plantdoc), Path(args.out)
    rows, jobs = [], []

    # ---- PlantSeg (official split) ----
    meta = pd.read_csv(seg / "Metadata.csv", encoding="utf-8-sig")
    meta.columns = [c.strip().replace(" ", "_") for c in meta.columns]  # 'Label file' -> Label_file
    meta = meta.rename(columns={"Index": "class_index"})  # 'Index' clashes with itertuples' row index
    for r in meta.itertuples():
        split = SPLIT_MAP[r.Split]
        stem = Path(r.Name).stem
        src_i = seg / "images" / split / r.Name
        src_m = seg / "annotations" / split / r.Label_file
        di, dm = f"images/{split}/{stem}.jpg", f"masks/{split}/{stem}.png"
        rows.append({"image_path": di, "mask_path": dm, "plant": r.Plant, "disease": r.Disease.lower(),
                     "label": label_of(r.Plant, r.Disease), "split": split, "source": "plantseg",
                     "mask_value": int(r.class_index) + 1, "mask_ratio": float(r.Mask_ratio)})
        jobs.append((str(src_i), str(src_m), str(out / di), str(out / dm), int(r.class_index) + 1))

    # ---- PlantDoc healthy (train -> train/val stratified, test -> test) ----
    for folder, plant in PLANTDOC_HEALTHY.items():
        for pd_split in ("train", "test"):
            files = sorted(p for p in (doc / pd_split / folder).glob("*") if p.is_file())
            if pd_split == "train":
                rng.shuffle(files)
                n_val = max(1, round(len(files) * args.val_frac))
                assign = ["val"] * n_val + ["train"] * (len(files) - n_val)
            else:
                assign = ["test"] * len(files)
            for i, (p, split) in enumerate(zip(files, assign)):
                stem = f"plantdoc_{plant.lower().replace(' ', '_')}_healthy_{pd_split}_{i:04d}"
                di = f"images/{split}/{stem}.jpg"
                rows.append({"image_path": di, "mask_path": "", "plant": plant, "disease": "healthy",
                             "label": f"{plant.replace(' ', '_')}___healthy", "split": split, "source": "plantdoc",
                             "mask_value": 0, "mask_ratio": 0.0})
                jobs.append((str(p), None, str(out / di), None, 0))

    print(f"processing {len(jobs)} images with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap(process_one, jobs, chunksize=16)):
            results.append(res)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(results)], axis=1)

    # sanity: does every mask contain exactly the expected class value?
    ps = df[df.source == "plantseg"]
    bad = ps[ps.mask_values_found.astype(str) != ps.mask_value.astype(str)]
    print(f"mask value check: {len(ps) - len(bad)}/{len(ps)} masks contain exactly Index+1; "
          f"{(ps.mask_values_found.fillna('') == '').sum()} empty masks; "
          f"{(~ps.mask_size_matches.astype(bool)).sum()} masks with a different size than their image", flush=True)

    # ---- near-duplicate check across splits (leakage) ----
    hashes = df.dhash.tolist()
    dup_of = [""] * len(df)
    order = {"train": 0, "val": 1, "test": 2}
    by_bucket = {}
    for i, h in enumerate(hashes):  # exact-hash buckets first (fast), then hamming <= 4 within 16-bit prefix buckets
        by_bucket.setdefault(h >> 48, []).append(i)
    n_pairs = 0
    for idxs in by_bucket.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if bin(hashes[i] ^ hashes[j]).count("1") <= 4 and df.split[i] != df.split[j]:
                    n_pairs += 1
                    # mark the one in the "earlier" split (train) as the duplicate to drop
                    k, other = (i, j) if order[df.split[i]] < order[df.split[j]] else (j, i)
                    dup_of[k] = df.image_path[other]
    df["dup_of"] = dup_of
    print(f"cross-split near-duplicate pairs (dHash hamming<=4, same 16-bit prefix): {n_pairs}; "
          f"{sum(1 for d in dup_of if d)} train/val rows flagged", flush=True)

    df.to_csv(out / "manifest.csv", index=False)
    counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(1)
    counts.sort_values("total").to_csv(out / "class_counts.csv")
    print(df.groupby(["source", "split"]).size().to_string(), flush=True)
    print(f"labels={df.label.nunique()} plants={df.plant.nunique()}", flush=True)

    # ---- ~30-image stratified dev sample: one per disease from a spread of classes + 4 healthy ----
    import shutil
    samp_dir = out.parent / "sample"
    test = df[df.split == "test"]
    labels = sorted(test[test.source == "plantseg"].label.unique())
    rng.shuffle(labels)
    pick = [test[test.label == l].sample(1, random_state=args.seed).iloc[0] for l in labels[:26]]
    pick += [r for _, r in test[test.source == "plantdoc"].groupby("plant").head(1).head(4).iterrows()]
    srows = []
    for r in pick:
        for col in ("image_path", "mask_path"):
            if r[col]:
                dst = samp_dir / r[col]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(out / r[col], dst)
        srows.append(r)
    pd.DataFrame(srows).to_csv(samp_dir / "manifest.csv", index=False)
    print(f"sample: {len(srows)} images", flush=True)


if __name__ == "__main__":
    main()
