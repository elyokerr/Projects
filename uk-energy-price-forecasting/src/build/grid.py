"""
Grid alignment utilities for UK settlement-period data.

Provides functions to:
- Reindex a DataFrame onto a continuous 30-minute grid (forward-filling short gaps).
- Count settlement periods per Europe/London calendar day.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def to_settlement_period_grid(
    df: pd.DataFrame,
    value_cols: list[str],
    freq: str = "30min",
) -> pd.DataFrame:
    """Reindex *df* onto a continuous half-hourly grid and forward-fill short gaps.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a tz-aware DatetimeIndex.
    value_cols : list[str]
        Columns to retain (others are dropped).
    freq : str
        Grid frequency; default ``"30min"``.

    Returns
    -------
    pd.DataFrame
        Reindexed DataFrame with gaps of ≤ 2 settlement periods forward-filled;
        longer gaps remain NaN.
    """
    # Work only on the requested columns.
    df = df[value_cols].copy()

    # Build a continuous grid spanning the data's min/max timestamps.
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq,
        tz=df.index.tz,
    )

    # Reindex onto the full grid, then forward-fill gaps of at most 2 periods.
    df = df.reindex(full_idx)
    df = df.ffill(limit=2)

    return df


def assert_settlement_periods_per_day(df: pd.DataFrame) -> dict[date, int]:
    """Return the number of settlement periods per Europe/London calendar day.

    The DataFrame index must be a tz-aware UTC DatetimeIndex.  Rows are
    grouped by their local (Europe/London) calendar date; the returned dict
    maps each ``datetime.date`` to the row count for that day.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a tz-aware DatetimeIndex (UTC recommended).

    Returns
    -------
    dict[datetime.date, int]
        Mapping from local calendar date to settlement-period count.
    """
    # GB settlement periods are defined in local (Europe/London) time: a normal
    # local day has 48, a spring clock-change day 46, an autumn clock-change day
    # 50. We therefore group by the LOCAL calendar date. A continuous UTC grid
    # always has 48 entries per UTC date, so grouping by UTC would never surface
    # the 46/50 days this function exists to detect.
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local_dates = idx.tz_convert("Europe/London").normalize().date
    counts: dict[date, int] = {}
    for d in local_dates:
        counts[d] = counts.get(d, 0) + 1
    return counts
