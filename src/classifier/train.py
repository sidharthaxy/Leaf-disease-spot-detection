"""
Stage A training: plant + disease classifier.

Runs unchanged on:
  - any CUDA machine (e.g. an RTX 3050 laptop or a cloud T4): python -m src.classifier.train --data-root data/processed
  - Kaggle / Colab GPU:  python -m src.classifier.train --data-root /kaggle/input/... --out outputs
  - Mac smoke test:       python -m src.classifier.train --data-root sample_data --limit 16 \
                              --epochs 1 --max-steps 2 --batch-size 4 --img-size 128 --num-workers 0

Recipe (see configs/train.yaml):
  - ImageNet-pretrained backbone, heads-only training for `frozen_epochs`, then full fine-tuning
  - AdamW, linear warm-up + cosine decay (per step), label smoothing
  - class imbalance: WeightedRandomSampler with weight = 1 / count^0.5
      (full inverse frequency would repeat a 10-image class ~100x per epoch and overfit it;
       the square root is a common middle ground between "natural" and "fully balanced")
  - AMP fp16 ONLY on CUDA (T4 tensor cores); fp32 on MPS / CPU
  - early stopping on validation macro-F1 (accuracy hides the rare classes)
  - checkpoint every epoch to the checkpoint dir -> an interrupted run resumes from it (re-run the same command)
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.classifier.model import ExportWrapper, TwoHeadClassifier
from src.data.dataset import LeafDataset, build_label_maps, eval_transform, train_transform
from src.utils.common import get_device, save_json, seed_everything

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "train.yaml"


def parse_args():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    ap = argparse.ArgumentParser()
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: str(s).lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    ap.add_argument("--data-root", default=os.environ.get("SM_CHANNEL_TRAINING", os.environ.get("SM_CHANNEL_TRAIN", "data/processed")))
    ap.add_argument("--manifest", default="manifest.csv", help="relative to data-root")
    ap.add_argument("--out", default=os.environ.get("SM_MODEL_DIR", "outputs/model"))
    ap.add_argument("--ckpt-dir", default="/opt/ml/checkpoints" if os.path.isdir("/opt/ml") else "outputs/checkpoints")
    ap.add_argument("--metrics-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "outputs/metrics"))
    return ap.parse_args()


def make_sampler(train_df: pd.DataFrame, power: float):
    counts = train_df.label.value_counts()
    w = train_df.label.map(lambda l: 1.0 / counts[l] ** power).to_numpy()
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True)


def cosine_with_warmup(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, loader, device, use_amp, max_steps=0):
    model.eval()
    ys, ps, pls, plys, top5_hits, losses = [], [], [], [], 0, []
    ce = nn.CrossEntropyLoss()
    probs_all = []
    for i, (x, y, yp) in enumerate(loader):
        if max_steps and i >= max_steps:
            break
        x, y, yp = x.to(device), y.to(device), yp.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            lo, po = model(x)
        lo, po = lo.float(), po.float()
        losses.append(ce(lo, y).item())
        k = min(5, lo.shape[1])
        top5_hits += (lo.topk(k, dim=1).indices == y[:, None]).any(1).sum().item()
        ys.append(y.cpu()); ps.append(lo.argmax(1).cpu())
        plys.append(yp.cpu()); pls.append(po.argmax(1).cpu())
        probs_all.append(lo.softmax(1).cpu())
    if not ys:  # empty split (only happens in tiny smoke tests)
        nan = float("nan")
        return {"loss": nan, "top1": nan, "top5": nan, "macro_f1": 0.0, "plant_acc": nan, "n": 0}, None
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
    yp = torch.cat(plys).numpy(); pp = torch.cat(pls).numpy()
    return {
        "loss": float(np.mean(losses)),
        "top1": float((y == p).mean()),
        "top5": top5_hits / len(y),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "plant_acc": float((yp == pp).mean()),
        "n": int(len(y)),
    }, (y, p, yp, pp, torch.cat(probs_all).numpy())


@torch.no_grad()
def predict_probs(model, loader, device, use_amp, tta=False):
    """Softmax probabilities for both heads. tta=True averages the image and its mirror image."""
    model.eval()
    L, P, Y, YP = [], [], [], []
    for x, y, yp in loader:
        x = x.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            lo, po = model(x)
            lp, pp = lo.float().softmax(1), po.float().softmax(1)
            if tta:
                lo2, po2 = model(torch.flip(x, dims=[3]))
                lp = (lp + lo2.float().softmax(1)) / 2
                pp = (pp + po2.float().softmax(1)) / 2
        L.append(lp.cpu()); P.append(pp.cpu()); Y.append(y); YP.append(yp)
    return (torch.cat(L).numpy(), torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(YP).numpy())


def save_probs(model, df, root, maps, a, device, use_amp, out_dir, split):
    """Save val/test probabilities (+TTA) so ensembles / plant-restriction can be computed offline."""
    if not len(df):
        return {}
    dl = DataLoader(LeafDataset(df, root, maps, eval_transform(a.img_size)), batch_size=a.batch_size,
                    num_workers=a.num_workers)
    res = {}
    arrays = {}
    for tta in (False, True):
        lp, pp, y, yp = predict_probs(model, dl, device, use_amp, tta)
        key = "tta" if tta else "plain"
        arrays[f"label_probs_{key}"], arrays[f"plant_probs_{key}"] = lp.astype(np.float16), pp.astype(np.float16)
        top5 = np.argsort(-lp, 1)[:, :5]
        res[key] = {"top1": float((lp.argmax(1) == y).mean()), "top5": float((top5 == y[:, None]).any(1).mean()),
                    "macro_f1": float(f1_score(y, lp.argmax(1), average="macro", zero_division=0)),
                    "plant_acc": float((pp.argmax(1) == yp).mean())}
    np.savez_compressed(out_dir / f"probs_{split}.npz", y=y, yp=yp, image_path=df.image_path.to_numpy(), **arrays)
    print(f"{split} probs saved: plain={res['plain']} tta={res['tta']}", flush=True)
    return res


def main():
    a = parse_args()
    seed_everything(a.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    print(f"device={device} amp={use_amp} args={vars(a)}", flush=True)

    root = Path(a.data_root)
    df = pd.read_csv(root / a.manifest)
    if "dup_of" in df.columns:  # drop train/val near-duplicates of images in another split (leakage)
        leak = df.dup_of.notna() & (df.split != "test")
        print(f"dropping {int(leak.sum())} train/val near-duplicates (dHash)", flush=True)
        df = df[~leak]
    if a.limit:
        df = df.groupby("split", group_keys=False).apply(lambda g: g.sample(min(len(g), a.limit), random_state=a.seed))
    maps = build_label_maps(df)
    df = df[df.label.isin(maps["labels"])]          # val/test labels never seen in train are dropped
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    print(f"labels={len(maps['labels'])} plants={len(maps['plants'])} train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    pin = device.type == "cuda"
    dl_tr = DataLoader(LeafDataset(tr, root, maps, train_transform(a.img_size)), batch_size=a.batch_size,
                       sampler=make_sampler(tr, a.sampler_power) if a.sampler_power > 0 else None,
                       shuffle=a.sampler_power <= 0, num_workers=a.num_workers, pin_memory=pin, drop_last=len(tr) > a.batch_size,
                       persistent_workers=a.num_workers > 0)
    dl_va = DataLoader(LeafDataset(va, root, maps, eval_transform(a.img_size)), batch_size=a.batch_size,
                       num_workers=a.num_workers, pin_memory=pin)

    model = TwoHeadClassifier(a.backbone, len(maps["plants"]), len(maps["labels"]), a.pretrained).to(device)
    head_params = list(model.plant_head.parameters()) + list(model.label_head.parameters())
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": a.lr_backbone},
        {"params": head_params, "lr": a.lr_head},
    ], weight_decay=a.weight_decay)
    steps_per_epoch = a.max_steps or len(dl_tr)
    total, warmup = a.epochs * steps_per_epoch, a.warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, total, warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce_label = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    ce_plant = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    ckpt_dir, out_dir = Path(a.ckpt_dir), Path(a.out)
    ckpt_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_f1, bad_epochs, history = 0, -1.0, 0, []
    last = ckpt_dir / "last.pt"
    if last.exists():  # spot interruption -> resume
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1, bad_epochs, history = ck["epoch"] + 1, ck["best_f1"], ck["bad_epochs"], ck["history"]
        print(f"resumed from epoch {ck['epoch']} best_f1={best_f1:.4f}", flush=True)

    for epoch in range(start_epoch, a.epochs):
        frozen = epoch < a.frozen_epochs
        model.set_backbone_trainable(not frozen)
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for i, (x, y, yp) in enumerate(dl_tr):
            if a.max_steps and i >= a.max_steps:
                break
            x, y, yp = x.to(device, non_blocking=True), y.to(device), yp.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                lo, po = model(x)
                loss = ce_label(lo, y) + a.plant_loss_weight * ce_plant(po, yp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(x); n += len(x)
        val, _ = evaluate(model, dl_va, device, use_amp, a.max_steps)
        rec = {"epoch": epoch, "frozen": frozen, "train_loss": run_loss / max(1, n), "lr": sched.get_last_lr()[0],
               "time_s": round(time.time() - t0, 1), **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        print(f"epoch={epoch} train_loss={rec['train_loss']:.4f} val_loss={val['loss']:.4f} "
              f"val_top1={val['top1']:.4f} val_top5={val['top5']:.4f} val_macro_f1={val['macro_f1']:.4f} "
              f"val_plant_acc={val['plant_acc']:.4f} frozen={frozen} time={rec['time_s']}s", flush=True)

        if val["macro_f1"] > best_f1:
            best_f1, bad_epochs = val["macro_f1"], 0
            torch.save({"model": model.state_dict(), "maps": maps, "args": vars(a), "epoch": epoch, "val": val},
                       out_dir / "best.pt")
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1,
                    "bad_epochs": bad_epochs, "history": history}, last)
        save_json(history, Path(a.metrics_dir) / "history.json")
        if bad_epochs >= a.patience and not frozen:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    # ---- final: best model on the test split, predictions for offline analysis ----
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    save_json(maps, out_dir / "label_maps.json")
    if len(te):
        dl_te = DataLoader(LeafDataset(te, root, maps, eval_transform(a.img_size)), batch_size=a.batch_size,
                           num_workers=a.num_workers)
        test, (y, p, yp, pp, probs) = evaluate(model, dl_te, device, use_amp, a.max_steps)
        print("test " + " ".join(f"test_{k}={v:.4f}" if isinstance(v, float) else f"test_{k}={v}" for k, v in test.items()), flush=True)
        save_json({"best_epoch": best["epoch"], "val": best["val"], "test": test}, out_dir / "summary.json")
        pred = te.iloc[: len(y)].reset_index(drop=True)[["image_path", "plant", "label"]].copy()
        pred["pred_label"] = [maps["labels"][i] for i in p]
        pred["pred_plant"] = [maps["plants"][i] for i in pp]
        pred["confidence"] = probs.max(1)
        top5 = np.argsort(-probs, 1)[:, :5]
        pred["top5"] = ["|".join(f"{maps['labels'][j]}:{probs[r, j]:.3f}" for j in row) for r, row in enumerate(top5)]
        pred.to_csv(out_dir / "test_predictions.csv", index=False)
    save_json(history, out_dir / "history.json")
    extra = {sp: save_probs(model, d, root, maps, a, device, use_amp, out_dir, sp) for sp, d in (("val", va), ("test", te))}
    save_json(extra, out_dir / "tta_summary.json")

    if a.export_onnx:
        export_onnx(model, a.img_size, out_dir / "model.onnx")


def export_onnx(model, img_size, path):
    m = ExportWrapper(model).cpu().eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(m, dummy, str(path), input_names=["image"], output_names=["label_probs", "plant_probs"],
                      dynamic_axes={"image": {0: "batch"}, "label_probs": {0: "batch"}, "plant_probs": {0: "batch"}},
                      opset_version=17)
    print(f"exported {path}", flush=True)


if __name__ == "__main__":
    main()
