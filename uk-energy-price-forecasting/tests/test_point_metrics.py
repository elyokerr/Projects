import numpy as np
from src.metrics.point_metrics import mae, rmse, smape, skill_vs_baseline


def test_mae():
    assert mae(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == 1.5


def test_rmse():
    assert abs(rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) - 3.5355339) < 1e-5


def test_smape_identical_zero():
    assert smape(np.array([5.0, 5.0]), np.array([5.0, 5.0])) == 0.0


def test_skill_positive_when_model_better():
    assert skill_vs_baseline(model_error=1.0, baseline_error=2.0) == 0.5
