import pandas as pd

from src.abtest.data import ExperimentData
from src.abtest.health import aa_test, srm_check
from src.abtest.simulate import simulate_conversion


def test_srm_passes_on_balanced_split():
    ed = simulate_conversion(n_per_arm=10000, base_rate=0.2,
                             absolute_lift=0.0, seed=4)
    r = srm_check(ed)
    assert r.passed is True and r.p_value > 0.01


def test_srm_fails_on_imbalanced_split():
    df = pd.DataFrame({"unit_id": range(11000),
                       "variant": ["control"] * 6000 + ["treatment"] * 5000,
                       "converted": [0] * 11000})
    ed = ExperimentData(df, metric_cols=["converted"])
    r = srm_check(ed, expected_ratio=0.5)
    assert r.passed is False and r.p_value < 0.01


def test_aa_false_positive_rate_near_alpha():
    fpr = aa_test(base_rate=0.2, n_per_arm=4000, n_simulations=400,
                  alpha=0.05, seed=7)
    assert 0.02 <= fpr <= 0.09
