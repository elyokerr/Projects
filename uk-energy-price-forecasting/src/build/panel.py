"""
Panel builder for UK energy price forecasting.

Assembles a :class:`PanelBundle` from raw price, demand, fuel-mix, and
calendar DataFrames.  Enforces:
- All series aligned to the same 30-minute settlement-period grid.
- Future covariates contain ONLY deterministic calendar features (leakage guard).
- Values cast to float32 for memory efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from darts import TimeSeries

from src.build.grid import to_settlement_period_grid
from src.build.leakage import assert_no_future_leakage


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PanelBundle:
    """Container for the aligned Darts TimeSeries used in model training."""

    target: TimeSeries
    """Univariate price series."""

    past_covariates: TimeSeries
    """Multivariate series: demand (demand_indo, demand_itsdo) + gen_* columns."""

    future_covariates: TimeSeries
    """Multivariate series: deterministic calendar features only."""

    series_index: dict = field(default_factory=dict)
    """Arbitrary metadata (e.g. source file paths, date ranges, feature names)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_panel(
    price_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> PanelBundle:
    """Construct a :class:`PanelBundle` from raw input DataFrames.

    Parameters
    ----------
    price_df : pd.DataFrame
        Index: tz-aware UTC timestamps.  Column: ``price``.
    demand_df : pd.DataFrame
        Index: tz-aware UTC timestamps.  Columns: ``indo``, ``itsdo``.
    fuel_df : pd.DataFrame
        Tidy format with columns ``timestamp``, ``fuel``, ``mw``.
        Pivoted wide to ``gen_<fuel>`` columns aligned to the common index.
    calendar_df : pd.DataFrame
        Index: tz-aware UTC timestamps (from :func:`src.data.calendar.build_calendar`).
        Columns: ``dow``, ``is_weekend``, ``is_holiday``, ``sp_of_day``,
        ``sin_*``, ``cos_*`` …

    Returns
    -------
    PanelBundle
        Validated bundle with leakage guard applied.
    """
    # ------------------------------------------------------------------
    # 1. Pivot fuel_df from tidy to wide  (gen_<fuel>)
    # ------------------------------------------------------------------
    fuel_wide = (
        fuel_df
        .pivot_table(index="timestamp", columns="fuel", values="mw", aggfunc="mean")
        .add_prefix("gen_")
        .rename_axis(None, axis=1)
    )
    # Ensure the fuel index is tz-aware (carry tz from price_df if needed).
    if fuel_wide.index.tz is None and price_df.index.tz is not None:
        fuel_wide.index = fuel_wide.index.tz_localize(price_df.index.tz)

    # ------------------------------------------------------------------
    # 2. Determine common time range (intersection of all inputs).
    # ------------------------------------------------------------------
    common_start = max(
        price_df.index.min(),
        demand_df.index.min(),
        fuel_wide.index.min(),
        calendar_df.index.min(),
    )
    common_end = min(
        price_df.index.max(),
        demand_df.index.max(),
        fuel_wide.index.max(),
        calendar_df.index.max(),
    )

    price_df = price_df.loc[common_start:common_end]
    demand_df = demand_df.loc[common_start:common_end]
    fuel_wide = fuel_wide.loc[common_start:common_end]
    calendar_df = calendar_df.loc[common_start:common_end]

    # ------------------------------------------------------------------
    # 3. Align all sources to the 30-minute settlement-period grid.
    # ------------------------------------------------------------------
    price_aligned = to_settlement_period_grid(price_df, ["price"])

    demand_aligned = to_settlement_period_grid(
        demand_df.rename(columns={"indo": "demand_indo", "itsdo": "demand_itsdo"}),
        ["demand_indo", "demand_itsdo"],
    )

    gen_cols = list(fuel_wide.columns)
    fuel_aligned = to_settlement_period_grid(fuel_wide, gen_cols)

    cal_cols = list(calendar_df.columns)
    cal_aligned = to_settlement_period_grid(calendar_df, cal_cols)

    # ------------------------------------------------------------------
    # 4. Build past_covariates: demand + gen columns.
    # ------------------------------------------------------------------
    past_df = pd.concat([demand_aligned, fuel_aligned], axis=1)

    # ------------------------------------------------------------------
    # 5. Cast everything to float32.
    # ------------------------------------------------------------------
    price_aligned = price_aligned.astype(np.float32)
    past_df = past_df.astype(np.float32)
    cal_aligned = cal_aligned.astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Convert to Darts TimeSeries.
    #    Darts 0.44 strips timezone — that's expected; the index stays
    #    consistent across all three series.
    # ------------------------------------------------------------------
    target_ts = TimeSeries.from_dataframe(price_aligned, value_cols=["price"])
    past_ts = TimeSeries.from_dataframe(past_df, value_cols=list(past_df.columns))
    future_ts = TimeSeries.from_dataframe(cal_aligned, value_cols=list(cal_aligned.columns))

    # ------------------------------------------------------------------
    # 7. Assemble metadata.
    # ------------------------------------------------------------------
    series_index = {
        "start": str(common_start),
        "end": str(common_end),
        "n_timesteps": len(price_aligned),
        "past_covariate_cols": list(past_df.columns),
        "future_covariate_cols": list(cal_aligned.columns),
    }

    bundle = PanelBundle(
        target=target_ts,
        past_covariates=past_ts,
        future_covariates=future_ts,
        series_index=series_index,
    )

    # ------------------------------------------------------------------
    # 8. Leakage guard — raises LeakageError if future_covariates are dirty.
    # ------------------------------------------------------------------
    assert_no_future_leakage(bundle)

    return bundle
