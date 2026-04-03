from __future__ import annotations

import torch
import torch.nn as nn

from .cbam import CBAMBlock


class PatchEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, use_cbam: bool) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            layers.append(CBAMBlock(64))
        layers.extend(
            [
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            ]
        )
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x).flatten(1)
        return self.proj(features)


class AttentionPool(nn.Module):
    def __init__(self, feature_dim: int, metadata_dim: int) -> None:
        super().__init__()
        input_dim = feature_dim + metadata_dim if metadata_dim > 0 else feature_dim
        self.score = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_mask: torch.Tensor,
        metadata_context: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_patches, feature_dim = patch_features.shape
        if metadata_context is not None:
            repeated_meta = metadata_context.unsqueeze(1).expand(batch_size, num_patches, -1)
            score_input = torch.cat([patch_features, repeated_meta], dim=-1)
        else:
            score_input = patch_features

        scores = self.score(score_input).squeeze(-1)
        scores = scores.masked_fill(patch_mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(weights * patch_features, dim=1)
        return pooled


class ROIGeometryEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, roi_vector: torch.Tensor) -> torch.Tensor:
        return self.encoder(roi_vector)


class LocalBranch(nn.Module):
    def __init__(self, config: dict, metadata_dim: int) -> None:
        super().__init__()
        local_cfg = config["model"]["local_branch"]
        data_cfg = config["data"]
        self.mode = local_cfg["mode"]
        in_channels = {"patch": 1, "heatmap": 1, "patch_heatmap": 2}[self.mode]
        feature_dim = int(local_cfg["feature_dim"])
        geometry_dim = int(local_cfg["geometry_dim"])
        roi_dim = 4 + int(data_cfg["max_keypoints"]) * 3

        self.patch_encoder = PatchEncoder(
            in_channels=in_channels,
            out_dim=feature_dim,
            use_cbam=config["model"]["cbam"]["enabled"] and config["model"]["cbam"]["local_branch"],
        )
        self.attention_pool = AttentionPool(feature_dim=feature_dim, metadata_dim=metadata_dim)
        self.geometry_encoder = ROIGeometryEncoder(input_dim=roi_dim, output_dim=geometry_dim)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + geometry_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(local_cfg["dropout"]),
        )
        self.output_dim = feature_dim

    def forward(
        self,
        local_images: torch.Tensor,
        local_heatmaps: torch.Tensor,
        patch_mask: torch.Tensor,
        roi_vector: torch.Tensor,
        metadata_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mode == "patch":
            local_input = local_images
        elif self.mode == "heatmap":
            local_input = local_heatmaps
        else:
            local_input = torch.cat([local_images, local_heatmaps], dim=2)

        batch_size, num_patches, channels, height, width = local_input.shape
        flattened = local_input.reshape(batch_size * num_patches, channels, height, width)
        patch_features = self.patch_encoder(flattened).reshape(batch_size, num_patches, -1)
        pooled = self.attention_pool(patch_features, patch_mask, metadata_context)
        geometry_features = self.geometry_encoder(roi_vector)
        return self.fusion(torch.cat([pooled, geometry_features], dim=-1))
