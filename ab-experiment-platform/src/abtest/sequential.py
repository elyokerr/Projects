from __future__ import annotations

import numpy as np

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
