from __future__ import annotations

import numpy as np
import pandas as pd

from src.abtest.data import ExperimentData


def simulate_conversion(*, n_per_arm: int, base_rate: float, absolute_lift: float,
                        covariate_corr: float = 0.0,
                        seed: int | None = None) -> ExperimentData:
    rng = np.random.default_rng(seed)
    n = n_per_arm
    variant = np.array(["control"] * n + ["treatment"] * n)
    latent = rng.normal(size=2 * n)
    # Baseline conversion probability varies with the latent factor, additively on
    # the probability scale. This lets a pre-period covariate correlate with the
    # outcome WITHOUT distorting the additive treatment effect (a logit shift would
    # not preserve an additive lift, biasing CUPED's recovered effect downward).
    base_p = base_rate + covariate_corr * 0.25 * latent
    lift = np.where(variant == "treatment", absolute_lift, 0.0)
    p = np.clip(base_p + lift, 1e-6, 1 - 1e-6)
    converted = rng.binomial(1, p)
    pre_cov = latent + rng.normal(scale=0.5, size=2 * n)
    df = pd.DataFrame({
        "unit_id": np.arange(2 * n),
        "variant": variant,
        "converted": converted,
        "pre_covariate": pre_cov,
    })
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return ExperimentData(df, variant_col="variant", metric_cols=["converted"],
                          covariate_col="pre_covariate")
