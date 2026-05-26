import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss for a single quantile q in (0,1)."""
    diff = y_true - y_pred
    loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
    return float(np.mean(loss))


def mean_pinball(y_true: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """Average pinball loss across a {quantile: prediction-array} mapping."""
    return float(np.mean([pinball_loss(y_true, p, q) for q, p in preds.items()]))
