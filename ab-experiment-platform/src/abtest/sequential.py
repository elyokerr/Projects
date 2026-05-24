from __future__ import annotations

import numpy as np
from scipy import stats

from src.abtest.results import SequentialResult


def msprt_stream(*, base_rate: float, lift: float, n_max: int, alpha: float = 0.05,
                 tau: float = 0.05, seed: int | None = None) -> SequentialResult:
    rng = np.random.default_rng(seed)
    c = rng.binomial(1, base_rate, n_max).astype(float)
    t = rng.binomial(1, base_rate + lift, n_max).astype(float)
    threshold = 1.0 / alpha
    cum_c = np.cumsum(c)
    cum_t = np.cumsum(t)
    ns = np.arange(1, n_max + 1)
    d = cum_t / ns - cum_c / ns
    var = base_rate * (1 - base_rate) * 2
    lr = np.sqrt(var / (var + ns * tau**2)) * np.exp(
        (ns**2 * tau**2 * d**2) / (2 * var * (var + ns * tau**2)))
    crossed_mask = lr >= threshold
    crossed = bool(np.any(crossed_mask))
    stop = int(np.argmax(crossed_mask)) if crossed else None
    final = float(lr[stop] if crossed else lr[-1])
    return SequentialResult(method="msprt", crossed=crossed, stop_index=stop,
                            final_statistic=final, threshold=float(threshold),
                            detail=f"tau={tau}, n_max={n_max}")


def naive_peeking_fpr(*, base_rate: float, n_max: int, look_every: int = 500,
                      n_sims: int = 200, alpha: float = 0.05,
                      seed: int | None = None) -> float:
    rng = np.random.default_rng(seed)
    looks = np.arange(look_every, n_max + 1, look_every)
    sig_count = 0
    for s in range(n_sims):
        c = rng.binomial(1, base_rate, n_max)
        t = rng.binomial(1, base_rate, n_max)
        cum_c = np.cumsum(c)
        cum_t = np.cumsum(t)
        ever = False
        for nlook in looks:
            sc, st = cum_c[nlook - 1], cum_t[nlook - 1]
            pooled = (sc + st) / (2 * nlook)
            se = np.sqrt(pooled * (1 - pooled) * (2 / nlook))
            if se == 0:
                continue
            z = (st / nlook - sc / nlook) / se
            if 2 * (1 - stats.norm.cdf(abs(z))) < alpha:
                ever = True
                break
        if ever:
            sig_count += 1
        if (s + 1) % 50 == 0:
            print(f"  naive-peeking sim {s + 1}/{n_sims}", flush=True)
    return sig_count / n_sims


def obrien_fleming_bounds(*, n_looks: int, alpha: float = 0.05) -> list[float]:
    ks = np.arange(1, n_looks + 1)
    t = ks / n_looks
    z_a = stats.norm.ppf(1 - alpha / 2)
    spend = 2 - 2 * stats.norm.cdf(z_a / np.sqrt(t))
    inc = np.diff(np.concatenate([[0.0], spend]))
    bounds = stats.norm.ppf(1 - inc / 2)
    return [float(b) for b in bounds]
