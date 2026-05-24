import numpy as np

from src.abtest.data import ExperimentData
from src.abtest.simulate import simulate_conversion


def test_simulate_returns_experimentdata_with_arms():
    ed = simulate_conversion(n_per_arm=5000, base_rate=0.20,
                             absolute_lift=0.0, seed=1)
    assert isinstance(ed, ExperimentData)
    assert ed.n_control == 5000 and ed.n_treatment == 5000


def test_simulate_lift_is_recoverable_in_expectation():
    ed = simulate_conversion(n_per_arm=200_000, base_rate=0.20,
                             absolute_lift=0.03, seed=2)
    diff = ed.treatment["converted"].mean() - ed.control["converted"].mean()
    assert abs(diff - 0.03) < 0.005


def test_covariate_is_correlated_with_outcome():
    ed = simulate_conversion(n_per_arm=50_000, base_rate=0.30,
                             absolute_lift=0.0, covariate_corr=0.6, seed=3)
    sub = ed.control
    r = np.corrcoef(sub["pre_covariate"], sub["converted"])[0, 1]
    assert r > 0.2
