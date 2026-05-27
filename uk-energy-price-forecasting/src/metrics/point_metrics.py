import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(y_pred - y_true)[mask] / denom[mask]) * 100.0)


def skill_vs_baseline(model_error: float, baseline_error: float) -> float:
    """1 - model/baseline. Positive => model beats baseline. Higher is better."""
    if baseline_error == 0:
        return 0.0
    return 1.0 - model_error / baseline_error
