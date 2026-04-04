from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
_SCIENTIFIC_NOTATION_PATTERN = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))[eE][+-]?\d+$")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"配置文件顶层必须是字典: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _assign_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _parse_scalar_override(raw_value: str) -> Any:
    text = raw_value.strip()
    if text == "":
        return None
    value = yaml.safe_load(text)
    if isinstance(value, str) and _SCIENTIFIC_NOTATION_PATTERN.fullmatch(text):
        return float(text)
    return value


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if not overrides:
        return parsed
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖参数格式必须是 key=value，收到: {item}")
        key, raw_value = item.split("=", 1)
        value = _parse_scalar_override(raw_value)
        _assign_nested(parsed, key.strip(), value)
    return parsed


def load_config(
    config_path: str | Path | None,
    overrides: list[str] | None = None,
    checkpoint_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        config = deep_merge(config, load_yaml(DEFAULT_CONFIG_PATH))
    if checkpoint_config:
        config = deep_merge(config, checkpoint_config)
    if config_path:
        config_file = Path(config_path)
        default_resolved = DEFAULT_CONFIG_PATH.resolve()
        config_resolved = config_file.resolve()
        if config_resolved != default_resolved:
            config = deep_merge(config, load_yaml(config_file))
    config = deep_merge(config, parse_overrides(overrides))
    return config


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
