"""
Stage A model: ImageNet-pretrained timm backbone + two linear heads.

  image -> backbone (EfficientNet-B0 / ConvNeXt-Tiny) -> pooled feature vector
                                                       ├─> plant head   (which host plant)
                                                       └─> label head   (Plant___Disease, incl. Plant___healthy)

The label head is the main output (it already implies the plant). The plant head is an auxiliary
task: it gives an extra, easier training signal and a plant-level prediction we can show even when
the exact disease is uncertain.
"""

from __future__ import annotations

import timm
import torch.nn as nn


class TwoHeadClassifier(nn.Module):
    def __init__(self, backbone: str, n_plants: int, n_labels: int, pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        # num_classes=0 -> timm returns the pooled features instead of ImageNet logits
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.dropout = nn.Dropout(drop)
        self.plant_head = nn.Linear(d, n_plants)
        self.label_head = nn.Linear(d, n_labels)

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.label_head(f), self.plant_head(f)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable


class ExportWrapper(nn.Module):
    """For ONNX export: output softmax probabilities for both heads."""

    def __init__(self, model: TwoHeadClassifier):
        super().__init__()
        self.model = model

    def forward(self, x):
        label_logits, plant_logits = self.model(x)
        return label_logits.softmax(-1), plant_logits.softmax(-1)
