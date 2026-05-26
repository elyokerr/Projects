import numpy as np
from src.metrics.crps import crps_from_quantiles


def test_crps_zero_when_perfect():
    y = np.array([5.0])
    preds = {q: np.array([5.0]) for q in [0.1, 0.5, 0.9]}
    assert abs(crps_from_quantiles(y, preds)) < 1e-9


def test_crps_positive_when_off():
    y = np.array([5.0])
    preds = {0.1: np.array([1.0]), 0.5: np.array([2.0]), 0.9: np.array([3.0])}
    assert crps_from_quantiles(y, preds) > 0
