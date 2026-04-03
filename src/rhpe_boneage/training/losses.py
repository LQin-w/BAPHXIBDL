from __future__ import annotations

import torch.nn as nn


def build_loss(name: str, smooth_l1_beta: float = 1.0):
    key = name.lower()
    if key == "smoothl1":
        return nn.SmoothL1Loss(beta=smooth_l1_beta)
    if key == "l1":
        return nn.L1Loss()
    if key == "mse":
        return nn.MSELoss()
    raise ValueError(f"不支持的损失函数: {name}")
