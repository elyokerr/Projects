from src.abtest.cuped import apply_cuped
from src.abtest.simulate import simulate_conversion


def test_cuped_reduces_variance_when_covariate_correlated():
    ed = simulate_conversion(n_per_arm=40000, base_rate=0.30,
                             absolute_lift=0.0, covariate_corr=0.7, seed=31)
    r = apply_cuped(ed, "converted", covariate="pre_covariate")
    assert r.variance_reduction > 0.05
    assert r.adjusted_ci_low < 0 < r.adjusted_ci_high


def test_cuped_unbiased_recovers_known_lift():
    ed = simulate_conversion(n_per_arm=80000, base_rate=0.30,
                             absolute_lift=0.03, covariate_corr=0.6, seed=32)
    r = apply_cuped(ed, "converted", covariate="pre_covariate")
    assert abs(r.adjusted_absolute_effect - 0.03) < 0.006
