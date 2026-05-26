import numpy as np


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical fraction of observations within [lower, upper] (inclusive)."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))
