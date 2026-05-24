from src.abtest.results import FrequentistResult


def test_frequentist_result_fields():
    r = FrequentistResult(
        metric="retention_7", control_mean=0.19, treatment_mean=0.18,
        absolute_effect=-0.01, relative_effect=-0.05, ci_low=-0.03, ci_high=0.01,
        p_value=0.07, significant=False, test="two_proportion_z",
        verdict="no significant difference",
    )
    assert r.significant is False
    assert r.test == "two_proportion_z"
