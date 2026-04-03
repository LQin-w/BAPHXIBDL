from __future__ import annotations

from typing import Any

import numpy as np


def compute_regression_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y_true_arr.size == 0:
        return {"mae": None, "mad": None}
    abs_error = np.abs(y_true_arr - y_pred_arr)
    return {
        "mae": float(abs_error.mean()),
        "mad": float(np.median(abs_error)),
    }
