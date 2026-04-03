from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .metrics import compute_regression_metrics


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def build_training_target(
    batch: dict[str, torch.Tensor],
    target_mode: str,
    target_normalizer,
    relative_direction: str = "boneage_minus_chronological",
) -> torch.Tensor:
    if target_mode == "relative":
        if relative_direction == "chronological_minus_boneage":
            raw_target = batch["chronological"] - batch["boneage"]
        else:
            raw_target = batch["boneage"] - batch["chronological"]
    else:
        raw_target = batch["boneage"]
    return target_normalizer.transform_tensor(raw_target)


def decode_boneage_prediction(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    target_mode: str,
    target_normalizer,
    relative_direction: str = "boneage_minus_chronological",
) -> torch.Tensor:
    target_pred = target_normalizer.inverse_transform_tensor(prediction)
    if target_mode == "relative":
        if relative_direction == "chronological_minus_boneage":
            return batch["chronological"] - target_pred
        return target_pred + batch["chronological"]
    return target_pred


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    target_mode: str,
    target_normalizer,
    train: bool,
    relative_direction: str = "boneage_minus_chronological",
    optimizer=None,
    scaler=None,
    gradient_clip: float | None = None,
    epoch: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.train(train)
    amp_enabled = train and device.type == "cuda" and scaler is not None
    total_loss = 0.0
    total_count = 0
    y_true: list[float] = []
    y_pred: list[float] = []
    rows: list[dict[str, Any]] = []

    progress = tqdm(loader, desc=f"{'train' if train else 'eval'}:{epoch}", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        has_target_mask = batch["has_target"].view(-1).bool()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp_enabled):
            outputs = model(batch)
            prediction = outputs["prediction"]
            loss = None
            if has_target_mask.any():
                normalized_target = build_training_target(
                    batch,
                    target_mode,
                    target_normalizer,
                    relative_direction=relative_direction,
                )
                loss = criterion(prediction[has_target_mask], normalized_target[has_target_mask])

        if train and loss is not None:
            if scaler is not None:
                scaler.scale(loss).backward()
                if gradient_clip and gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip and gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

        pred_boneage = decode_boneage_prediction(
            prediction,
            batch,
            target_mode,
            target_normalizer,
            relative_direction=relative_direction,
        )

        batch_size = prediction.shape[0]
        if loss is not None:
            total_loss += loss.item() * int(has_target_mask.sum().item())
            total_count += int(has_target_mask.sum().item())

        for index in range(batch_size):
            gt_value = float(batch["boneage"][index].item()) if bool(batch["has_target"][index].item()) else np.nan
            pred_value = float(pred_boneage[index].item())
            abs_error = abs(pred_value - gt_value) if not np.isnan(gt_value) else np.nan
            if not np.isnan(gt_value):
                if relative_direction == "chronological_minus_boneage":
                    relative_gt = float(batch["chronological"][index].item()) - gt_value
                    relative_pred = float(batch["chronological"][index].item()) - pred_value
                else:
                    relative_gt = gt_value - float(batch["chronological"][index].item())
                    relative_pred = pred_value - float(batch["chronological"][index].item())
            else:
                relative_gt = np.nan
                relative_pred = np.nan
            rows.append(
                {
                    "ID": batch["id"][index],
                    "gt_boneage": gt_value,
                    "pred_boneage": pred_value,
                    "abs_error": abs_error,
                    "sex": int(batch["male_index"][index].item()),
                    "chronological": float(batch["chronological"][index].item()),
                    "gt_relative_boneage": relative_gt,
                    "pred_relative_boneage": relative_pred,
                }
            )
            if not np.isnan(gt_value):
                y_true.append(gt_value)
                y_pred.append(pred_value)

        avg_loss = total_loss / max(total_count, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}")

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_count, 1) if total_count > 0 else None
    prediction_df = pd.DataFrame(rows)
    valid_df = prediction_df[prediction_df["gt_boneage"].notna()].copy()
    if len(valid_df) >= 2:
        relative_values = valid_df["gt_relative_boneage"].to_numpy(dtype=np.float32)
        abs_errors = valid_df["abs_error"].to_numpy(dtype=np.float32)
        if (
            np.std(relative_values) < 1e-8
            or np.std(abs_errors) < 1e-8
            or len(np.unique(relative_values)) < 2
        ):
            corr = None
            slope = None
        else:
            corr = float(np.corrcoef(relative_values, abs_errors)[0, 1])
            slope = float(np.polyfit(relative_values, abs_errors, deg=1)[0])
        metrics["relative_age_error_corr"] = corr
        metrics["relative_age_error_slope"] = slope
    else:
        metrics["relative_age_error_corr"] = None
        metrics["relative_age_error_slope"] = None
    return metrics, prediction_df
