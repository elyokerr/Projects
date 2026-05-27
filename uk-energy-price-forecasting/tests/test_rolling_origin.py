"""Tests for src/backtest/rolling_origin.py (Phase 5 — rolling-origin harness)."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.rolling_origin import generate_origins


# ---------------------------------------------------------------------------
# Module 1: generate_origins
# ---------------------------------------------------------------------------


def test_generate_origins_daily_boundaries():
    idx = pd.date_range("2024-01-01", periods=5 * 48, freq="30min")  # 5 days
    origins = generate_origins(idx, start=pd.Timestamp("2024-01-02"))
    # midnights for days 2,3,4,5 -> 4 origins
    assert len(origins) == 4
    assert all(o.hour == 0 and o.minute == 0 for o in origins)
    assert origins[0] == pd.Timestamp("2024-01-02")


def test_generate_origins_respects_end():
    idx = pd.date_range("2024-01-01", periods=5 * 48, freq="30min")
    origins = generate_origins(
        idx,
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-03"),
    )
    # Only 2024-01-02 and 2024-01-03 midnights
    assert len(origins) == 2
    assert origins[-1] == pd.Timestamp("2024-01-03")


def test_generate_origins_step_two_days():
    idx = pd.date_range("2024-01-01", periods=7 * 48, freq="30min")
    origins = generate_origins(
        idx,
        start=pd.Timestamp("2024-01-01"),
        step="2D",
    )
    # Day 1, Day 3, Day 5, Day 7 — all must be midnights in the index
    assert len(origins) == 4
    assert origins[0] == pd.Timestamp("2024-01-01")
    assert origins[1] == pd.Timestamp("2024-01-03")
