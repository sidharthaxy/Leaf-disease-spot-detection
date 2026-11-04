"""
Manifest-driven dataset for Stage A.

Manifest CSV columns (built by src/data/build_manifest.py):
    image_path, mask_path, plant, disease, label, split, source
  - paths are RELATIVE to the data root (so the same CSV works on any machine)
  - label = "Plant___Disease" (e.g. "Tomato___Early_blight", "Tomato___healthy")
  - split in {train, val, test}

Augmentations are deliberately "realistic only": geometry (crop / flip / rotate) and mild
brightness / contrast / saturation changes. NO hue jitter: shifting hue can turn brown lesions
green or healthy tissue brown, which would literally change the disease the label describes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_label_maps(df: pd.DataFrame) -> dict:
    """Index maps built from the TRAIN split only (sorted -> deterministic)."""
    tr = df[df.split == "train"]
    labels = sorted(tr.label.unique())
    plants = sorted(tr.plant.unique())
    return {
        "labels": labels,
        "plants": plants,
        "label_to_plant": {l: tr[tr.label == l].plant.iloc[0] for l in labels},
    }


def train_transform(size: int):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.0),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int):
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LeafDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str | Path, maps: dict, transform):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.label_idx = {l: i for i, l in enumerate(maps["labels"])}
        self.plant_idx = {p: i for i, p in enumerate(maps["plants"])}
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        r = self.df.iloc[index]
        img = Image.open(self.root / r.image_path)
        img.draft("RGB", (1024, 1024))  # fast JPEG downscale-on-decode for huge photos (before loading)
        x = self.transform(img.convert("RGB"))
        return x, torch.tensor(self.label_idx[r.label]), torch.tensor(self.plant_idx[r.plant])
