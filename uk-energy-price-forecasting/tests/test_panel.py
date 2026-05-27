"""Tests for src/build/panel.py — PanelBundle construction."""
import pandas as pd
import numpy as np
import pytest

from src.build.panel import build_panel, PanelBundle


# ---------------------------------------------------------------------------
# Helpers to build synthetic 96-SP (2-day) input frames
# ---------------------------------------------------------------------------

def _make_inputs(n: int = 96):
    """Return (price_df, demand_df, fuel_df, calendar_df) over *n* half-hours."""
    idx = pd.date_range("2024-06-01", periods=n, freq="30min", tz="UTC")

    price_df = pd.DataFrame({"price": np.random.rand(n) * 100 + 50}, index=idx)

    demand_df = pd.DataFrame(
        {"indo": np.random.rand(n) * 30000 + 20000,
         "itsdo": np.random.rand(n) * 30000 + 20000},
        index=idx,
    )

    # Tidy fuel DataFrame
    fuels = ["gas", "wind", "nuclear"]
    records = []
    for ts in idx:
        for fuel in fuels:
            records.append({"timestamp": ts, "fuel": fuel, "mw": np.random.rand() * 5000})
    fuel_df = pd.DataFrame(records)

    from src.data.calendar import build_calendar
    calendar_df = build_calendar(idx)

    return price_df, demand_df, fuel_df, calendar_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_panel_time_indices_aligned():
    price_df, demand_df, fuel_df, calendar_df = _make_inputs()
    bundle = build_panel(price_df, demand_df, fuel_df, calendar_df)
    # All three TimeSeries must share the same time index length
    assert len(bundle.target.time_index) == len(bundle.past_covariates.time_index)
    assert len(bundle.target.time_index) == len(bundle.future_covariates.time_index)


def test_panel_past_covariate_components():
    price_df, demand_df, fuel_df, calendar_df = _make_inputs()
    bundle = build_panel(price_df, demand_df, fuel_df, calendar_df)
    past_cols = list(bundle.past_covariates.components)
    assert "demand_indo" in past_cols
    assert any(c.startswith("gen_") for c in past_cols)


def test_panel_future_covariate_components():
    price_df, demand_df, fuel_df, calendar_df = _make_inputs()
    bundle = build_panel(price_df, demand_df, fuel_df, calendar_df)
    fut_cols = list(bundle.future_covariates.components)
    assert "dow" in fut_cols
    assert "sin_daily_1" in fut_cols


def test_panel_price_not_in_future_covariates():
    price_df, demand_df, fuel_df, calendar_df = _make_inputs()
    bundle = build_panel(price_df, demand_df, fuel_df, calendar_df)
    assert "price" not in bundle.future_covariates.components


def test_panel_target_is_univariate():
    price_df, demand_df, fuel_df, calendar_df = _make_inputs()
    bundle = build_panel(price_df, demand_df, fuel_df, calendar_df)
    assert bundle.target.n_components == 1
    assert "price" in bundle.target.components


def test_panel_values_float32():
    price_df, demand_df, fuel_df, calendar_df = _make_inputs()
    bundle = build_panel(price_df, demand_df, fuel_df, calendar_df)
    import numpy as np
    assert bundle.target.dtype == np.float32
    assert bundle.past_covariates.dtype == np.float32
    assert bundle.future_covariates.dtype == np.float32
