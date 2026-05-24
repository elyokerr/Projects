from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.abtest.data import ExperimentData


def simulate_conversion(*, n_per_arm: int, base_rate: float, absolute_lift: float,
                        covariate_corr: float = 0.0,
                        seed: int | None = None) -> ExperimentData:
    rng = np.random.default_rng(seed)
    n = n_per_arm
    variant = np.array(["control"] * n + ["treatment"] * n)
    latent = rng.normal(size=2 * n)
    p = np.where(variant == "control", base_rate, base_rate + absolute_lift)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if covariate_corr > 0:
        shift = covariate_corr * 2.5 * latent
        p = expit(logit(p) + shift - shift.mean())
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
