import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

from src.build.panel import PanelBundle, build_panel
from src.data.calendar import build_calendar


def _make_tiny_inputs(n: int = 96):
    """Return synthetic (price_df, demand_df, fuel_df, calendar_df) over *n* half-hours."""
    idx = pd.date_range("2024-06-01", periods=n, freq="30min", tz="UTC")

    price_df = pd.DataFrame({"price": np.linspace(50.0, 100.0, n)}, index=idx)

    demand_df = pd.DataFrame(
        {"indo": np.linspace(20000.0, 30000.0, n),
         "itsdo": np.linspace(19000.0, 29000.0, n)},
        index=idx,
    )

    fuels = ["gas", "wind", "nuclear"]
    records = [
        {"timestamp": ts, "fuel": fuel, "mw": float(i * 100 + j * 500)}
        for i, ts in enumerate(idx)
        for j, fuel in enumerate(fuels)
    ]
    fuel_df = pd.DataFrame(records)

    calendar_df = build_calendar(idx)

    return price_df, demand_df, fuel_df, calendar_df


@pytest.fixture
def tiny_bundle() -> PanelBundle:
    """A valid PanelBundle built from ~2 days (96 SPs) of synthetic data."""
    price_df, demand_df, fuel_df, calendar_df = _make_tiny_inputs()
    return build_panel(price_df, demand_df, fuel_df, calendar_df)


@pytest.fixture
def tiny_bundle_with_leak(tiny_bundle: PanelBundle) -> PanelBundle:
    """A PanelBundle with a forbidden 'gen_gas' column injected into future_covariates.

    Constructed by directly wrapping a DataFrame that contains a forbidden
    out-turn column — bypassing build_panel's leakage guard.
    """
    n = tiny_bundle.target.n_timesteps
    idx = pd.date_range("2024-06-01", periods=n, freq="30min")

    # Start from the clean future_covariates and add the forbidden column.
    fut_df = tiny_bundle.future_covariates.to_dataframe()
    fut_df.index = idx  # align to tz-naive index Darts uses internally
    fut_df["gen_gas"] = np.random.rand(n).astype(np.float32)

    dirty_future_ts = TimeSeries.from_dataframe(fut_df, value_cols=list(fut_df.columns))

    return PanelBundle(
        target=tiny_bundle.target,
        past_covariates=tiny_bundle.past_covariates,
        future_covariates=dirty_future_ts,
        series_index=tiny_bundle.series_index,
    )
