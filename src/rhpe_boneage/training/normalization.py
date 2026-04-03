from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ScalarNormalizer:
    mean: float = 0.0
    std: float = 1.0
    eps: float = 1e-6

    @classmethod
    def fit(cls, values) -> "ScalarNormalizer":
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return cls()
        std = float(array.std())
        if std < 1e-6:
            std = 1.0
        return cls(mean=float(array.mean()), std=std)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.eps)

    def inverse_transform(self, values):
        return values * (self.std + self.eps) + self.mean

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / (self.std + self.eps)

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.std + self.eps) + self.mean

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict[str, float] | None) -> "ScalarNormalizer":
        if not state:
            return cls()
        return cls(**state)
