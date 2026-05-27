import numpy as np
from src.metrics.coverage import interval_coverage


def test_coverage_all_inside():
    y = np.array([5.0, 6.0, 7.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([10.0, 10.0, 10.0])
    assert interval_coverage(y, lo, hi) == 1.0


def test_coverage_half_inside():
    y = np.array([5.0, 50.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([10.0, 10.0])
    assert interval_coverage(y, lo, hi) == 0.5


def test_coverage_boundary_inclusive():
    y = np.array([10.0])
    assert interval_coverage(y, np.array([0.0]), np.array([10.0])) == 1.0
