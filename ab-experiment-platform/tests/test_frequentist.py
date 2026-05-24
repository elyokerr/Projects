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
