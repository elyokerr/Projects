"""Rolling-origin backtest harness (Phase 5).

Public API
----------
generate_origins(time_index, start, step="1D", end=None) -> list[pd.Timestamp]
    Enumerate forecast origins at successive midnight boundaries.

BacktestResult
    Dataclass holding actuals/forecasts for a completed backtest run.
    Exposes a `.metrics()` method.

run_backtest(model, bundle, origins, horizon=48, quantiles=(0.1,0.5,0.9),
             refit=False, model_name=None) -> BacktestResult
    Execute a rolling-origin evaluation loop.

build_ablation_table(results, baseline="seasonal_naive") -> pd.DataFrame
    Summarise multiple BacktestResult objects into a skill-scored DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.metrics.coverage import interval_coverage
from src.metrics.crps import crps_from_quantiles
from src.metrics.pinball import mean_pinball
from src.metrics.point_metrics import mae, rmse, skill_vs_baseline, smape


# ---------------------------------------------------------------------------
# Module 1 — generate_origins
# ---------------------------------------------------------------------------


def generate_origins(
    time_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    step: str = "1D",
    end: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """Return forecast origins at successive midnight boundaries.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        The full time index of the panel (tz-naive or tz-aware).
    start : pd.Timestamp
        Earliest desired origin.  The first returned origin is the first
        midnight in *time_index* that is >= start.
    step : str
        Pandas frequency string for the spacing between consecutive origins.
        Default ``"1D"`` (daily).
    end : pd.Timestamp or None
        Latest desired origin (inclusive).  If None, uses the last timestamp
        in *time_index*.

    Returns
    -------
    list[pd.Timestamp]
        Timestamps that are midnight (00:00) entries in *time_index*, spaced
        by *step*, within [start, end].
    """
    if end is None:
        end = time_index[-1]

    # All midnight timestamps present in the index
    midnights = time_index[(time_index.hour == 0) & (time_index.minute == 0)]

    # Keep those within [start, end]
    midnights = midnights[(midnights >= start) & (midnights <= end)]

    if len(midnights) == 0:
        return []

    # Build the desired grid starting from the first valid midnight
    grid = pd.date_range(start=midnights[0], end=end, freq=step)

    # Keep only grid entries that are actually in the midnight set
    midnight_set = set(midnights)
    return [ts for ts in grid if ts in midnight_set]
