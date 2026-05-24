import numpy as np
import pandas as pd

from src.abtest.data import ExperimentData
from src.abtest.frequentist import two_proportion_z
from src.abtest.simulate import simulate_conversion


def test_z_test_hand_computed():
    rows = (["control"] * 1000 + ["treatment"] * 1000)
    conv = ([1] * 100 + [0] * 900 + [1] * 140 + [0] * 860)
    df = pd.DataFrame({"variant": rows, "converted": conv})
    df["unit_id"] = range(len(df))
    ed = ExperimentData(df, metric_cols=["converted"])
    r = two_proportion_z(ed, "converted")
    assert abs(r.absolute_effect - 0.04) < 1e-9
    assert r.p_value < 0.01 and r.significant is True


def test_z_test_recovers_zero_effect():
    ed = simulate_conversion(n_per_arm=50000, base_rate=0.2,
                             absolute_lift=0.0, seed=11)
    r = two_proportion_z(ed, "converted")
    assert r.ci_low < 0 < r.ci_high


def _continuous(shift, seed, n=2000):
    rng = np.random.default_rng(seed)
    vals = np.concatenate([rng.normal(0, 1, n), rng.normal(shift, 1, n)])
    df = pd.DataFrame({"variant": ["control"] * n + ["treatment"] * n,
                       "value": vals})
    df["unit_id"] = range(len(df))
    return ExperimentData(df, metric_cols=["value"])


def test_welch_detects_mean_shift():
    from src.abtest.frequentist import welch_t
    r = welch_t(_continuous(0.3, 0), "value")
    assert r.absolute_effect > 0 and r.significant is True


def test_welch_not_significant_on_identical():
    from src.abtest.frequentist import welch_t
    r = welch_t(_continuous(0.0, 1), "value")
    assert r.significant is False


def test_mann_whitney_pvalue_in_range():
    from src.abtest.frequentist import mann_whitney
    r = mann_whitney(_continuous(0.5, 2, n=500), "value")
    assert 0.0 <= r.p_value <= 1.0
