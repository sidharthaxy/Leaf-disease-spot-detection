"""
Grad-CAM (Selvaraju et al., 2017) for the label head: where did the CNN look?

  1. forward the image, keep the last convolutional feature maps A_k (C x h x w)
  2. back-propagate the score y_c of the predicted class to those maps
  3. weight of each channel  alpha_k = mean over pixels of d y_c / d A_k   (global-average-pooled gradient)
  4. CAM = ReLU( sum_k alpha_k * A_k )  -> upsample to the image, normalise to 0..1, colour-map

It links the deep-learning stage back to image processing: a good model's heat-map should sit on the
lesions that our classical pipeline segments.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.classifier.infer import prepare
from src.classifier.model import TwoHeadClassifier


class GradCAM:
    def __init__(self, ckpt_path: str | Path, device: str = "cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        args = ck["args"]
        self.maps = ck["maps"]
        self.size = int(args["img_size"])
        self.model = TwoHeadClassifier(args["backbone"], len(self.maps["plants"]), len(self.maps["labels"]),
                                       pretrained=False).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.device = device

    def __call__(self, img_bgr: np.ndarray, class_idx: int | None = None) -> tuple[np.ndarray, str]:
        x = torch.from_numpy(prepare(img_bgr, self.size)).to(self.device)
        bb = self.model.backbone
        feats = bb.forward_features(x)                    # 1 x C x h x w   (last conv block)
        feats.retain_grad()
        pooled = bb.forward_head(feats, pre_logits=True)  # global average pool
        logits = self.model.label_head(pooled)
        c = int(logits.argmax()) if class_idx is None else class_idx
        self.model.zero_grad()
        logits[0, c].backward()
        w = feats.grad.mean(dim=(2, 3), keepdim=True)     # alpha_k
        cam = torch.relu((w * feats).sum(1))[0].detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam, self.maps["labels"][c]

    def overlay(self, img_bgr: np.ndarray, class_idx: int | None = None, alpha: float = 0.45) -> tuple[np.ndarray, str]:
        cam, label = self(img_bgr, class_idx)
        # the classifier saw a centre crop (short side resized to 1.14*size, then size x size):
        # map the CAM back onto that same crop of the original image
        h, w = img_bgr.shape[:2]
        side = int(min(h, w) / 1.14)
        y, x = (h - side) // 2, (w - side) // 2
        crop = img_bgr[y:y + side, x:x + side]
        heat = cv2.applyColorMap((cv2.resize(cam, (side, side)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(heat, alpha, crop, 1 - alpha, 0), label
