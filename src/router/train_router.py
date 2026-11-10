"""
Router training: one small classifier that decides WHICH pipeline an image goes to.

    classes: crop | <10 palette species> | other_houseplant | non_plant
    (the exact list is read from data/router/manifest.csv, sorted, train split only)

Runs unchanged on a cloud T4, Kaggle/Colab, or the Mac (smoke test):
    python -m src.router.train_router --data-root data/router --out outputs/router
    python -m src.router.train_router --data-root data/router --limit 8 --epochs 1 --max-steps 2 --num-workers 0

Recipe: ImageNet-pretrained EfficientNet-B0 at 224 px, heads-only for 1 epoch, then full fine-tuning;
AdamW + warm-up/cosine; label smoothing; class-balanced sampling (1/count^0.5). Augmentation is
geometry + mild brightness/contrast only, NO hue/saturation jitter: colour is exactly what separates a
red croton from a diseased crop leaf, so it must not be scrambled.
Saves best.pt (by val macro-F1), model.onnx (softmax probabilities), label_map.json and the val/test
probabilities, so temperature and routing thresholds are fitted locally on VAL (src/router/calibrate_router.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from src.utils.common import get_device, seed_everything

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_tf(size):
    return T.Compose([T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.75, 1.33)), T.RandomHorizontalFlip(),
                      T.RandomVerticalFlip(), T.RandomRotation(20), T.ColorJitter(brightness=0.2, contrast=0.2),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_tf(size):
    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


class RouterDataset(Dataset):
    def __init__(self, df, root, classes, tf):
        self.df, self.root, self.tf = df.reset_index(drop=True), Path(root), tf
        self.idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r.image_path).convert("RGB")
        return self.tf(img), self.idx[r.label]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("SM_CHANNEL_TRAINING", "data/router"))
    ap.add_argument("--out", default=os.environ.get("SM_MODEL_DIR", "outputs/router"))
    ap.add_argument("--ckpt-dir", default="/opt/ml/checkpoints" if os.path.isdir("/opt/ml") else "outputs/router_ckpt")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--frozen-epochs", type=int, default=1)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--sampler-power", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help=">0: N images per class per split (smoke test)")
    ap.add_argument("--max-steps", type=int, default=0)
    return ap.parse_args()


@torch.no_grad()
def predict(model, loader, device, use_amp, max_steps=0):
    model.eval()
    ps, ys = [], []
    for i, (x, y) in enumerate(loader):
        if max_steps and i >= max_steps:
            break
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x.to(device, non_blocking=True))
        ps.append(logits.float().cpu())
        ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def main():
    a = parse()
    seed_everything(a.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    root, out, ck = Path(a.data_root), Path(a.out), Path(a.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "manifest.csv")
    if a.limit:
        df = df.groupby(["label", "split"], group_keys=False).apply(lambda g: g.head(a.limit))
    classes = sorted(df[df.split == "train"].label.unique())
    (out / "label_map.json").write_text(json.dumps({"classes": classes, "img_size": a.img_size, "backbone": a.backbone,
                                                     "mean": MEAN, "std": STD}, indent=1))
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    print(f"device={device} classes={len(classes)} train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    print(tr.label.value_counts().to_string(), flush=True)

    cnt = tr.label.value_counts()
    w = tr.label.map(lambda l: 1.0 / cnt[l] ** a.sampler_power).to_numpy().astype(float)
    sampler = WeightedRandomSampler(w.tolist(), num_samples=len(w), replacement=True)
    kw = dict(num_workers=a.num_workers, pin_memory=use_amp, persistent_workers=a.num_workers > 0)
    dl_tr = DataLoader(RouterDataset(tr, root, classes, train_tf(a.img_size)), batch_size=a.batch_size, sampler=sampler, drop_last=True, **kw)
    dl_va = DataLoader(RouterDataset(va, root, classes, eval_tf(a.img_size)), batch_size=a.batch_size * 2, **kw)
    dl_te = DataLoader(RouterDataset(te, root, classes, eval_tf(a.img_size)), batch_size=a.batch_size * 2, **kw)

    model = timm.create_model(a.backbone, pretrained=True, num_classes=len(classes), drop_rate=0.2).to(device)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": head, "lr": a.lr_head}, {"params": body, "lr": a.lr_backbone}], weight_decay=a.weight_decay)
    base_lrs = [a.lr_head, a.lr_backbone]
    steps_per_epoch = a.max_steps or len(dl_tr)
    total = steps_per_epoch * a.epochs
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    start, best, bad, hist, step = 0, -1.0, 0, [], 0
    if (ck / "last.pt").exists():  # spot interruption -> resume
        s = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        start, best, bad, hist = s["epoch"] + 1, s["best"], s["bad"], s["hist"]
        step = start * steps_per_epoch
        print(f"resumed from epoch {start}", flush=True)

    for ep in range(start, a.epochs):
        frozen = ep < a.frozen_epochs
        for p in body:
            p.requires_grad = not frozen
        model.train()
        t0, losses = time.time(), []
        for i, (x, y) in enumerate(dl_tr):
            if a.max_steps and i >= a.max_steps:
                break
            warm = steps_per_epoch
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = ce(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
        lv, yv = predict(model, dl_va, device, use_amp, a.max_steps)
        f1 = float(f1_score(yv, lv.argmax(1), average="macro"))
        acc = float((lv.argmax(1) == yv).mean())
        ci = classes.index("crop") if "crop" in classes else -1
        crop_mis = float((lv[yv == ci].argmax(1) != ci).mean()) if ci >= 0 and (yv == ci).any() else float("nan")
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_acc": acc, "val_f1": f1, "val_crop_misroute": crop_mis,
                     "sec": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if f1 > best:
            best, bad = f1, 0
            torch.save({"model": model.state_dict(), "classes": classes, "args": vars(a)}, out / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best, "bad": bad, "hist": hist}, ck / "last.pt")
        if bad >= a.patience and not frozen:
            print("early stop", flush=True)
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=False)["model"])
    model.to(device)
    for split, dl, d in (("val", dl_va, va), ("test", dl_te, te)):
        lg, y = predict(model, dl, device, use_amp, a.max_steps)
        np.savez_compressed(out / f"logits_{split}.npz", logits=lg, y=y, paths=d.image_path.to_numpy()[: len(y)])
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    model.eval().cpu().float()
    dummy = torch.randn(1, 3, a.img_size, a.img_size)
    torch.onnx.export(model, dummy, str(out / "model.onnx"), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    print(f"best val macro-F1 {best:.4f}; exported {out / 'model.onnx'}", flush=True)


if __name__ == "__main__":
    main()
