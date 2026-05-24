import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="set RUN_SLOW=1 to run the end-to-end smoke",
)


def test_full_pipeline_produces_coherent_decision():
    from src.abtest import (
        apply_cuped,
        beta_binomial,
        decide,
        msprt_stream,
        required_sample_size,
        simulate_conversion,
        srm_check,
        two_proportion_z,
    )

    res = required_sample_size(baseline_rate=0.20, mde_absolute=0.03)
    assert res.sample_size_per_arm > 0

    ed = simulate_conversion(n_per_arm=res.sample_size_per_arm, base_rate=0.20,
                             absolute_lift=0.03, covariate_corr=0.5, seed=1)
    assert srm_check(ed).passed

    fr = two_proportion_z(ed, "converted")
    bayes = beta_binomial(ed, "converted", seed=0)
    cuped = apply_cuped(ed, "converted", covariate="pre_covariate")
    seq = msprt_stream(base_rate=0.20, lift=0.03, n_max=20000, seed=2)
    d = decide(fr, mde_absolute=0.03, guardrails_ok=True)

    assert d.recommendation in {"ship", "no_ship", "inconclusive"}
    assert bayes.prob_treatment_better >= 0.0
    assert cuped.variance_reduction == cuped.variance_reduction  # not NaN
    assert isinstance(seq.crossed, bool)
