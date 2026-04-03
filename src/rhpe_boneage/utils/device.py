from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import sys


@dataclass
class RuntimeInfo:
    python: str
    torch_version: str
    torchvision_version: str
    cuda_build: str | None
    cuda_available: bool
    device_count: int
    device_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_runtime() -> tuple[torch.device, RuntimeInfo]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names: list[str] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            names.append(torch.cuda.get_device_name(index))
    runtime = RuntimeInfo(
        python=sys.version.replace("\n", " "),
        torch_version=torch.__version__,
        torchvision_version=__import__("torchvision").__version__,
        cuda_build=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        device_names=names,
    )
    return device, runtime


def suggest_dataloader_kwargs(
    batch_size: int,
    use_cuda: bool,
    cpu_count: int | None = None,
) -> dict[str, Any]:
    if cpu_count is None:
        cpu_count = 0
        try:
            import os

            cpu_count = os.cpu_count() or 0
        except Exception:
            cpu_count = 0

    if cpu_count <= 2:
        workers = 0
    elif cpu_count <= 8:
        workers = min(2, max(1, cpu_count - 1))
    else:
        workers = min(8, max(2, cpu_count // 2))
        workers = min(workers, max(2, batch_size * 2))

    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(use_cuda),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def maybe_compile_model(model: torch.nn.Module, enabled: bool, logger) -> torch.nn.Module:
    if not enabled:
        logger.info("torch.compile: 配置关闭，跳过。")
        return model
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile: 当前 torch 不支持，自动降级。")
        return model
    try:
        compiled = torch.compile(model)
        logger.info("torch.compile: 已启用。")
        return compiled
    except Exception as exc:  # pragma: no cover - 编译失败时的保护逻辑
        logger.warning("torch.compile: 启用失败，自动降级。原因: %s", exc)
        return model
