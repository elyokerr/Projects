import numpy as np
from src.metrics.pinball import pinball_loss


def crps_from_quantiles(y_true: np.ndarray, preds: dict[float, np.ndarray]) -> float:
    """CRPS approximated as 2 * mean pinball over an evenly-spaced quantile grid."""
    qs = sorted(preds)
    return float(2.0 * np.mean([pinball_loss(y_true, preds[q], q) for q in qs]))
