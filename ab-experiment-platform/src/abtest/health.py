from __future__ import annotations

import numpy as np
from scipy import stats

from src.abtest.data import ExperimentData
from src.abtest.results import HealthCheckResult


def srm_check(data: ExperimentData, expected_ratio: float = 0.5) -> HealthCheckResult:
    n_c, n_t = data.n_control, data.n_treatment
    total = n_c + n_t
    exp_c = total * expected_ratio
    exp_t = total * (1 - expected_ratio)
    chi2, p = stats.chisquare([n_c, n_t], [exp_c, exp_t])
    passed = bool(p >= 0.001)
    detail = (f"control={n_c}, treatment={n_t}; expected {exp_c:.0f}/{exp_t:.0f}. "
              f"{'OK' if passed else 'SAMPLE RATIO MISMATCH - results untrustworthy'}")
    return HealthCheckResult(check="srm", passed=passed, statistic=float(chi2),
                             p_value=float(p), detail=detail)


def aa_test(*, base_rate: float, n_per_arm: int, n_simulations: int = 500,
            alpha: float = 0.05, seed: int | None = None) -> float:
    rng = np.random.default_rng(seed)
    sig = 0
    for _ in range(n_simulations):
        a = rng.binomial(1, base_rate, n_per_arm)
        b = rng.binomial(1, base_rate, n_per_arm)
        p1, p2 = a.mean(), b.mean()
        pooled = (a.sum() + b.sum()) / (2 * n_per_arm)
        se = np.sqrt(pooled * (1 - pooled) * (2 / n_per_arm))
        if se == 0:
            continue
        z = (p2 - p1) / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        if p < alpha:
            sig += 1
    return sig / n_simulations
