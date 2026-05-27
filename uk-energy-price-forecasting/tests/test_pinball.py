import numpy as np
from src.metrics.pinball import pinball_loss


def test_pinball_underforecast_penalizes_by_quantile():
    assert pinball_loss(np.array([10.0]), np.array([8.0]), q=0.9) == 1.8


def test_pinball_overforecast():
    assert abs(pinball_loss(np.array([10.0]), np.array([12.0]), q=0.9) - 0.2) < 1e-9


def test_pinball_median_is_half_abs_error():
    assert pinball_loss(np.array([10.0]), np.array([7.0]), q=0.5) == 1.5


def test_mean_pinball_over_quantiles():
    from src.metrics.pinball import mean_pinball

    y = np.array([10.0])
    preds = {0.1: np.array([8.0]), 0.5: np.array([9.0]), 0.9: np.array([12.0])}
    expected = (0.2 + 0.5 + 0.2) / 3
    assert abs(mean_pinball(y, preds) - expected) < 1e-9
