from __future__ import annotations

import math

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src.abtest.results import PowerResult

_power_solver = NormalIndPower()


def required_sample_size(*, baseline_rate: float, mde_absolute: float,
                         alpha: float = 0.05, power: float = 0.80,
                         daily_traffic_per_arm: float | None = None) -> PowerResult:
    p1, p2 = baseline_rate, baseline_rate + mde_absolute
    effect = proportion_effectsize(p2, p1)
    n = _power_solver.solve_power(effect_size=effect, alpha=alpha, power=power,
                                  ratio=1.0, alternative="two-sided")
    n_per_arm = int(math.ceil(abs(n)))
    duration = None
    if daily_traffic_per_arm:
        duration = n_per_arm / daily_traffic_per_arm
    return PowerResult(baseline_rate=baseline_rate, mde_absolute=mde_absolute,
                       alpha=alpha, power=power, sample_size_per_arm=n_per_arm,
                       total_sample_size=2 * n_per_arm, duration_days=duration)


def power_for_sample_size(*, baseline_rate: float, mde_absolute: float,
                          n_per_arm: int, alpha: float = 0.05) -> float:
    effect = proportion_effectsize(baseline_rate + mde_absolute, baseline_rate)
    return float(_power_solver.solve_power(effect_size=effect, nobs1=n_per_arm,
                                           alpha=alpha, ratio=1.0,
                                           alternative="two-sided"))
