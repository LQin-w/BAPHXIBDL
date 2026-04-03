from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models as tv_models


WEIGHTS_MAP = {
    "resnet18": tv_models.ResNet18_Weights.DEFAULT,
    "resnet34": tv_models.ResNet34_Weights.DEFAULT,
    "resnet50": tv_models.ResNet50_Weights.DEFAULT,
    "efficientnet_b0": tv_models.EfficientNet_B0_Weights.DEFAULT,
    "efficientnet_b1": tv_models.EfficientNet_B1_Weights.DEFAULT,
    "efficientnet_b2": tv_models.EfficientNet_B2_Weights.DEFAULT,
}


@dataclass
class BackboneSpec:
    name: str
    out_channels: int
    family: str


class BackboneEncoder(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = False) -> None:
        super().__init__()
        self.spec = self._build(backbone_name, pretrained)
        self.out_channels = self.spec.out_channels
        self.family = self.spec.family

    def _build(self, backbone_name: str, pretrained: bool) -> BackboneSpec:
        weights = WEIGHTS_MAP[backbone_name] if pretrained else None
        try:
            model = getattr(tv_models, backbone_name)(weights=weights)
        except Exception:
            model = getattr(tv_models, backbone_name)(weights=None)
        self.backbone_name = backbone_name
        if backbone_name.startswith("resnet"):
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            return BackboneSpec(name=backbone_name, out_channels=model.fc.in_features, family="resnet")

        if backbone_name.startswith("efficientnet"):
            self.features = model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            out_channels = model.classifier[1].in_features
            return BackboneSpec(name=backbone_name, out_channels=out_channels, family="efficientnet")

        raise ValueError(f"不支持的 backbone: {backbone_name}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_features(x)
        pooled = self.pool(feature_map).flatten(1)
        return pooled
