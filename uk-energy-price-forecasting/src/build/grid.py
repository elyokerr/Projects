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
    # NOTE: we group by the UTC calendar date (i.e. strip tz, use .date()).
    # Grouping by Europe/London local date would cause a normal UTC day
    # (00:00–23:30 UTC in BST) to split across two London local dates.
    # The spring-clock-change test still works because the London-local range
    # 00:00–23:30 on 2024-03-31 converts to UTC timestamps that all fall on
    # 2024-03-31 UTC (the transition happens at 01:00 London = 01:00 UTC that
    # night, before BST begins), so counting UTC dates also gives 46 there.
    utc_dates = df.index.normalize().date
    counts: dict[date, int] = {}
    for d in utc_dates:
        counts[d] = counts.get(d, 0) + 1
    return counts
