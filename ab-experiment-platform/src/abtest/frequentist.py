from __future__ import annotations

import numpy as np
from scipy import stats

from src.abtest.data import ExperimentData
from src.abtest.results import FrequentistResult


def two_proportion_z(data: ExperimentData, metric: str,
                     alpha: float = 0.05) -> FrequentistResult:
    c = data.control[metric].dropna().to_numpy()
    t = data.treatment[metric].dropna().to_numpy()
    p_c, p_t = c.mean(), t.mean()
    n_c, n_t = len(c), len(t)
    pooled = (c.sum() + t.sum()) / (n_c + n_t)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    ci_low, ci_high = diff - zcrit * se_unpooled, diff + zcrit * se_unpooled
    sig = bool(p_value < alpha)
    return FrequentistResult(
        metric=metric, control_mean=float(p_c), treatment_mean=float(p_t),
        absolute_effect=float(diff),
        relative_effect=float(diff / p_c) if p_c else float("nan"),
        ci_low=float(ci_low), ci_high=float(ci_high), p_value=float(p_value),
        significant=sig, test="two_proportion_z",
        verdict=("significant difference" if sig else "no significant difference"))


def welch_t(data: ExperimentData, metric: str,
            alpha: float = 0.05) -> FrequentistResult:
    c = data.control[metric].dropna().to_numpy(dtype=float)
    t = data.treatment[metric].dropna().to_numpy(dtype=float)
    mean_c, mean_t = c.mean(), t.mean()
    n_c, n_t = len(c), len(t)
    var_c, var_t = c.var(ddof=1), t.var(ddof=1)
    se = np.sqrt(var_t / n_t + var_c / n_c)
    _, p_value = stats.ttest_ind(t, c, equal_var=False)
    df = (var_t / n_t + var_c / n_c) ** 2 / (
        (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    diff = mean_t - mean_c
    ci_low, ci_high = diff - tcrit * se, diff + tcrit * se
    sig = bool(p_value < alpha)
    return FrequentistResult(
        metric=metric, control_mean=float(mean_c), treatment_mean=float(mean_t),
        absolute_effect=float(diff),
        relative_effect=float(diff / mean_c) if mean_c else float("nan"),
        ci_low=float(ci_low), ci_high=float(ci_high), p_value=float(p_value),
        significant=sig, test="welch_t",
        verdict=("significant difference" if sig else "no significant difference"))


def mann_whitney(data: ExperimentData, metric: str,
                 alpha: float = 0.05) -> FrequentistResult:
    c = data.control[metric].dropna().to_numpy(dtype=float)
    t = data.treatment[metric].dropna().to_numpy(dtype=float)
    _, p_value = stats.mannwhitneyu(t, c, alternative="two-sided")
    mean_c, mean_t = c.mean(), t.mean()
    diff = mean_t - mean_c
    sig = bool(p_value < alpha)
    return FrequentistResult(
        metric=metric, control_mean=float(mean_c), treatment_mean=float(mean_t),
        absolute_effect=float(diff),
        relative_effect=float(diff / mean_c) if mean_c else float("nan"),
        ci_low=float("nan"), ci_high=float("nan"), p_value=float(p_value),
        significant=sig, test="mann_whitney",
        verdict=("significant difference" if sig else "no significant difference"))
