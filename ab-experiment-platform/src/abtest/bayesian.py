from __future__ import annotations

import numpy as np

from src.abtest.data import ExperimentData
from src.abtest.results import BayesianResult


def beta_binomial(data: ExperimentData, metric: str, *, prior_a: float = 1.0,
                  prior_b: float = 1.0, n_samples: int = 200_000,
                  seed: int | None = None) -> BayesianResult:
    rng = np.random.default_rng(seed)
    c = data.control[metric].dropna().to_numpy()
    t = data.treatment[metric].dropna().to_numpy()
    post_c = rng.beta(prior_a + c.sum(), prior_b + (len(c) - c.sum()), n_samples)
    post_t = rng.beta(prior_a + t.sum(), prior_b + (len(t) - t.sum()), n_samples)
    prob_better = float((post_t > post_c).mean())
    loss_t = float(np.maximum(post_c - post_t, 0).mean())
    loss_c = float(np.maximum(post_t - post_c, 0).mean())
    diff = post_t - post_c
    cred_low, cred_high = np.percentile(diff, [2.5, 97.5])
    verdict = ("treatment likely better" if prob_better > 0.95 else
               "control likely better" if prob_better < 0.05 else "inconclusive")
    return BayesianResult(metric=metric, prob_treatment_better=prob_better,
                          expected_loss_treatment=loss_t,
                          expected_loss_control=loss_c,
                          cred_low=float(cred_low), cred_high=float(cred_high),
                          verdict=verdict)
