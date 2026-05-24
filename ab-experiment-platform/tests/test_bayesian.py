from src.abtest.bayesian import beta_binomial
from src.abtest.simulate import simulate_conversion


def test_strong_positive_lift_gives_high_prob_better():
    ed = simulate_conversion(n_per_arm=20000, base_rate=0.20,
                             absolute_lift=0.05, seed=21)
    r = beta_binomial(ed, "converted", seed=0)
    assert r.prob_treatment_better > 0.99
    assert r.expected_loss_treatment < r.expected_loss_control


def test_zero_lift_gives_prob_near_half():
    ed = simulate_conversion(n_per_arm=20000, base_rate=0.20,
                             absolute_lift=0.0, seed=22)
    r = beta_binomial(ed, "converted", seed=0)
    assert 0.3 < r.prob_treatment_better < 0.7
