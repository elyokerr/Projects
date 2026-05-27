import pandas as pd
import numpy as np
from src.build.grid import to_settlement_period_grid, assert_settlement_periods_per_day


def test_grid_fills_short_gap():
    idx = pd.date_range("2024-01-01", periods=6, freq="30min", tz="UTC")
    df = pd.DataFrame({"v": [1., 2., np.nan, np.nan, 5., 6.]}, index=idx)
    out = to_settlement_period_grid(df, ["v"])
    assert out["v"].isna().sum() == 0
    assert out["v"].iloc[2] == 2.0


def test_grid_leaves_long_gap_nan():
    idx = pd.date_range("2024-01-01", periods=6, freq="30min", tz="UTC")
    df = pd.DataFrame({"v": [1., np.nan, np.nan, np.nan, np.nan, 6.]}, index=idx)
    out = to_settlement_period_grid(df, ["v"])
    assert out["v"].isna().sum() >= 1


def test_normal_day_has_48_sps():
    idx = pd.date_range("2024-06-01 00:00", "2024-06-01 23:30", freq="30min", tz="UTC")
    df = pd.DataFrame({"v": range(48)}, index=idx)
    counts = assert_settlement_periods_per_day(df)
    assert counts[pd.Timestamp("2024-06-01").date()] == 48


def test_spring_clock_change_has_46_sps():
    idx = pd.date_range("2024-03-31 00:00", "2024-03-31 23:30", freq="30min", tz="Europe/London")
    df = pd.DataFrame({"v": range(len(idx))}, index=idx.tz_convert("UTC"))
    counts = assert_settlement_periods_per_day(df)
    assert counts[pd.Timestamp("2024-03-31").date()] == 46
