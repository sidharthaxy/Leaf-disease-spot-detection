"""
Data step 1: download the raw datasets and inspect them.     python -m src.data.prepare_raw_data

  1. downloads PlantSeg (Zenodo v7, plantseg.zip, 1.1 GB) and verifies the MD5,
  2. downloads PlantDoc (GitHub, CC BY 4.0), used only for its healthy-leaf classes,
  3. extracts both into data/raw/{plantseg,plantdoc}/,
  4. writes an inspection report to data/raw/report/ (folder tree, file types, mask encoding, image sizes,
     split/class files), so the dataset format is VERIFIED instead of assumed.
Needs ~3 GB of free disk. Then run src/data/build_manifest.py.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

PLANTSEG_URL = "https://zenodo.org/records/17719108/files/plantseg.zip?download=1"
PLANTSEG_MD5 = "9358a66dff88cdd15c4fe009763c40a3"
PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_pillow():
    """Install Pillow / NumPy if this Python lacks them (checked without importing them)."""
    import importlib.util

    if any(importlib.util.find_spec(m) is None for m in ("PIL", "numpy")):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pillow", "numpy"])


def download(url, dest):
    print(f"[download] {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download)"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    print(f"[download] done, {os.path.getsize(dest) / 1e9:.2f} GB", flush=True)


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(zip_path, dest):
    print(f"[extract] {zip_path} -> {dest}", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    # drop macOS junk folders if present
    for junk in Path(dest).rglob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)


def tree_summary(root, max_depth=4, max_children=25):
    """Folder tree with file counts, collapsed so 100+ class folders stay readable."""
    lines = []
    root = Path(root)

    def walk(d, depth):
        entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        exts = collections.Counter(f.suffix.lower() for f in files)
        ind = "  " * depth
        lines.append(f"{ind}{d.name}/  ({len(dirs)} dirs, {len(files)} files {dict(exts)})")
        for f in files[:5]:
            lines.append(f"{ind}  - {f.name}  [{f.stat().st_size} B]")
        if len(files) > 5:
            lines.append(f"{ind}  - ... {len(files) - 5} more files")
        if depth < max_depth:
            for sub in dirs[:max_children]:
                walk(sub, depth + 1)
            if len(dirs) > max_children:
                lines.append(f"{ind}  ... {len(dirs) - max_children} more dirs: "
                             f"{[x.name for x in dirs[max_children:max_children + 40]]}")

    walk(root, 0)
    return lines


def small_text_files(root, limit=40):
    """Show heads of small text/json/csv files: split lists, class maps, metadata."""
    out = {}
    cands = [p for p in Path(root).rglob("*")
             if p.is_file() and p.suffix.lower() in {".txt", ".csv", ".json", ".md", ".yaml", ".yml", ".xlsx"}]
    # prefer files near the top of the tree, they are usually metadata (not per-image annotations)
    cands.sort(key=lambda p: (len(p.relative_to(root).parts), str(p)))
    for p in cands[:limit]:
        rel = str(p.relative_to(root))
        if p.suffix.lower() == ".xlsx":
            out[rel] = f"<xlsx, {p.stat().st_size} B>"
            continue
        try:
            with open(p, "r", errors="replace") as f:
                out[rel] = f.read(1500)
        except Exception as e:  # noqa: BLE001
            out[rel] = f"<unreadable: {e}>"
    return out


def inspect_images_and_masks(root, n_sample=300, seed=0):
    """Sample image files; classify PNGs that look like masks (single channel / few unique values)."""
    import numpy as np
    from PIL import Image

    files = [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
    by_dir = collections.Counter(str(p.parent.relative_to(root)) for p in files)
    rng = random.Random(seed)
    sample = rng.sample(files, min(n_sample, len(files)))

    mask_like, image_like = [], []
    union_values = collections.Counter()
    for p in sample:
        try:
            im = Image.open(p)
            arr = np.asarray(im)
        except Exception as e:  # noqa: BLE001
            image_like.append({"path": str(p.relative_to(root)), "error": str(e)})
            continue
        uniq = np.unique(arr)
        rec: dict[str, Any] = {
            "path": str(p.relative_to(root)),
            "mode": im.mode,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "n_unique": int(len(uniq)),
        }
        if len(uniq) <= 64 and (arr.ndim == 2 or im.mode in ("P", "L", "1", "I", "I;16")):
            rec["unique_values"] = [int(v) for v in uniq[:64]]
            rec["frac_nonzero"] = float((arr > 0).mean())
            union_values.update(int(v) for v in uniq)
            if im.mode == "P":
                rec["has_palette"] = True
            mask_like.append(rec)
        else:
            image_like.append(rec)

    sizes = collections.Counter(tuple(r["shape"][:2]) for r in image_like if "shape" in r)
    return {
        "n_image_files_total": len(files),
        "image_files_per_dir_top50": by_dir.most_common(50),
        "n_dirs_with_images": len(by_dir),
        "sampled": len(sample),
        "mask_like_count": len(mask_like),
        "mask_like_examples": mask_like[:25],
        "mask_union_unique_values_in_sample": sorted(union_values.items())[:200],
        "photo_like_count": len(image_like),
        "photo_like_examples": image_like[:10],
        "photo_size_hw_top15": [[list(k), v] for k, v in sizes.most_common(15)],
    }


def pairing_check(root):
    """How many photos have a same-stem file elsewhere (likely the mask/annotation)?"""
    stems = collections.defaultdict(list)
    for p in Path(root).rglob("*"):
        if p.is_file():
            stems[p.stem].append(str(p.relative_to(root)))
    multi = {k: v for k, v in stems.items() if len(v) > 1}
    ex = list(multi.items())[:10]
    return {"n_stems": len(stems), "n_stems_with_multiple_files": len(multi), "examples": ex}


def healthy_hits(root):
    hits = [str(p.relative_to(root)) for p in Path(root).rglob("*")
            if "healthy" in p.name.lower() and p.is_dir()]
    files = sum(1 for p in Path(root).rglob("*") if p.is_file() and "healthy" in str(p).lower())
    return {"healthy_dirs": hits[:50], "files_with_healthy_in_path": files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--work", default="data/_download")
    ap.add_argument("--skip-plantdoc", action="store_true")
    args = ap.parse_args()

    ensure_pillow()
    out = Path(args.out)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    seg_dir, doc_dir, rep_dir = out / "plantseg", out / "plantdoc", out / "report"
    for d in (seg_dir, doc_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {}

    # ---- PlantSeg ----
    zp = work / "plantseg.zip"
    download(PLANTSEG_URL, zp)
    got = md5sum(zp)
    report["plantseg_md5_ok"] = got == PLANTSEG_MD5
    report["plantseg_md5"] = got
    print(f"[plantseg] md5 {got} ok={report['plantseg_md5_ok']}", flush=True)
    extract(zp, seg_dir)
    zp.unlink()
    report["plantseg_tree"] = tree_summary(seg_dir)
    report["plantseg_text_files"] = small_text_files(seg_dir)
    report["plantseg_images"] = inspect_images_and_masks(seg_dir)
    report["plantseg_pairing"] = pairing_check(seg_dir)
    report["plantseg_healthy"] = healthy_hits(seg_dir)

    # ---- PlantDoc (healthy classes, if needed later) ----
    if not args.skip_plantdoc:
        try:
            zd = work / "plantdoc.zip"
            download(PLANTDOC_URL, zd)
            extract(zd, doc_dir)
            zd.unlink()
            report["plantdoc_tree"] = tree_summary(doc_dir, max_depth=3, max_children=40)
            report["plantdoc_healthy"] = healthy_hits(doc_dir)
        except Exception as e:  # noqa: BLE001  (PlantDoc is optional; never fail the job for it)
            report["plantdoc_error"] = repr(e)

    with open(rep_dir / "inspect_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(rep_dir / "plantseg_tree.txt", "w") as f:
        f.write("\n".join(report["plantseg_tree"]))
    print("[done] report written", flush=True)
    print("\n".join(report["plantseg_tree"][:80]), flush=True)


if __name__ == "__main__":
    main()
