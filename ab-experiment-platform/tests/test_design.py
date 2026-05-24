from src.abtest.design import power_for_sample_size, required_sample_size


def test_sample_size_matches_statsmodels_reference():
    res = required_sample_size(baseline_rate=0.20, mde_absolute=0.02,
                               alpha=0.05, power=0.80)
    assert 6100 <= res.sample_size_per_arm <= 6600


def test_power_increases_with_sample_size():
    p_small = power_for_sample_size(baseline_rate=0.20, mde_absolute=0.02,
                                    n_per_arm=2000, alpha=0.05)
    p_big = power_for_sample_size(baseline_rate=0.20, mde_absolute=0.02,
                                  n_per_arm=8000, alpha=0.05)
    assert p_big > p_small


def test_duration_from_daily_traffic():
    res = required_sample_size(baseline_rate=0.20, mde_absolute=0.02,
                               alpha=0.05, power=0.80,
                               daily_traffic_per_arm=1000)
    assert res.duration_days is not None and res.duration_days > 0
