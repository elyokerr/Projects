import src.abtest as ab


def test_public_api_exports():
    expected = [
        "ExperimentData", "simulate_conversion", "required_sample_size",
        "power_for_sample_size", "srm_check", "aa_test", "two_proportion_z",
        "welch_t", "mann_whitney", "correct", "beta_binomial", "apply_cuped",
        "msprt_stream", "naive_peeking_fpr", "obrien_fleming_bounds", "decide",
        "PowerResult", "HealthCheckResult", "FrequentistResult", "BayesianResult",
        "CupedResult", "SequentialResult", "Decision",
    ]
    for name in expected:
        assert hasattr(ab, name), f"missing export: {name}"
