from __future__ import annotations

import numpy as np
from scipy import stats

from src.abtest.data import ExperimentData
from src.abtest.results import CupedResult


def apply_cuped(data: ExperimentData, metric: str, *, covariate: str,
                alpha: float = 0.05) -> CupedResult:
    df = data.df
    y = df[metric].to_numpy(dtype=float)
    x = df[covariate].to_numpy(dtype=float)
    theta = np.cov(y, x, bias=True)[0, 1] / np.var(x)
    y_adj = y - theta * (x - x.mean())
    is_t = (df[data.variant_col] == data.treatment_label).to_numpy()
    yc, yt = y_adj[~is_t], y_adj[is_t]
    effect = yt.mean() - yc.mean()
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    zcrit = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = effect - zcrit * se, effect + zcrit * se
    p_value = 2 * (1 - stats.norm.cdf(abs(effect / se)))
    var_reduction = 1.0 - (np.var(y_adj) / np.var(y))
    return CupedResult(metric=metric, theta=float(theta),
                       variance_reduction=float(var_reduction),
                       adjusted_absolute_effect=float(effect),
                       adjusted_ci_low=float(ci_low), adjusted_ci_high=float(ci_high),
                       adjusted_p_value=float(p_value))
