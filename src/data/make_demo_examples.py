"""
Build the demo's example gallery: TEST-split images with expert masks, never seen in training or tuning.

    python -m src.data.make_demo_examples            # -> demo/examples/{images,masks,manifest.csv}

Why these are fair examples:
  - they are from the official PlantSeg TEST split (+ PlantDoc healthy test images);
  - the classifier was trained on train (validation for early stopping), the pipeline was tuned on
    validation, the learned severity correction was fitted on validation: none of them saw these images;
  - cross-split duplicates were removed from train/val with perceptual hashing, so no copy leaked in.
Source: data/sample/ (30 stratified test images written by src/data/build_manifest.py).
Non-photographs (document / poster scans) are skipped. Licence / attribution: demo/examples/ATTRIBUTION.md.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "sample"
DST = ROOT / "demo" / "examples"
SKIP = {"Raspberry___fire_blight"}   # the sample's copy is a scanned extension leaflet, not a photo


def main():
    m = pd.read_csv(SRC / "manifest.csv")
    m = m[(m.split == "test") & ~m.label.isin(SKIP)].reset_index(drop=True)
    for col in ("image_path", "mask_path"):
        for rel in m[col].dropna():
            if isinstance(rel, str) and rel:
                (DST / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(SRC / rel, DST / rel)
    m.insert(0, "example_id", range(len(m)))
    m.to_csv(DST / "manifest.csv", index=False)
    print(f"{len(m)} examples -> {DST}  ({(m.source == 'plantseg').sum()} diseased with masks, "
          f"{(m.source != 'plantseg').sum()} healthy)")


if __name__ == "__main__":
    main()
