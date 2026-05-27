"""
Synthetic fixture panel for UK energy price forecasting.

Provides:
- _generate_fixture_frame(): produces and saves a reproducible 30-day wide parquet.
- load_fixture_panel(): loads the parquet and returns a PanelBundle.

The parquet at tests/fixtures/fixture_panel.parquet is committed to the repo so
the Streamlit demo and smoke tests can run without any real API credentials.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.build.panel import PanelBundle, build_panel
from src.data.calendar import build_calendar

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
_PARQUET_PATH = _FIXTURES_DIR / "fixture_panel.parquet"


# ---------------------------------------------------------------------------
# Generator (run once to produce the committed parquet)
# ---------------------------------------------------------------------------

def _generate_fixture_frame(seed: int = 42) -> pd.DataFrame:
    """Create a reproducible 30-day half-hourly wide DataFrame and save to parquet.

    Returns
    -------
    pd.DataFrame
        The generated DataFrame (also written to ``tests/fixtures/fixture_panel.parquet``).
    """
    rng = np.random.default_rng(seed)

    n = 30 * 48  # 1440 rows
    index = pd.date_range("2024-01-01 00:00", periods=n, freq="30min", tz="UTC")
    t = np.arange(n, dtype=float)

    # -- Settlement period of day (1-48) for solar profile ----------------
    sp_of_day = (t % 48).astype(int)  # 0-indexed here; midday = ~24

    # -- Price: daily sinusoid + weekly component + noise + spikes ---------
    daily_component = 15.0 * np.sin(2 * np.pi * t / 48)
    weekly_component = 5.0 * np.sin(2 * np.pi * t / (48 * 7))
    noise = rng.normal(0.0, 3.0, n)
    price_base = 60.0 + daily_component + weekly_component + noise

    # Add a handful of random positive spikes
    spike_idx = rng.choice(n, size=12, replace=False)
    price_base[spike_idx] += rng.uniform(30.0, 100.0, size=12)
    price = np.clip(price_base, 5.0, None)  # keep mostly positive

    # -- Demand: correlated with daily shape + noise -----------------------
    demand_base = 28000.0 + 6000.0 * np.sin(2 * np.pi * t / 48 - 0.5)
    indo = demand_base + rng.normal(0.0, 500.0, n)
    itsdo = indo * rng.uniform(0.97, 1.00, n) + rng.normal(0.0, 200.0, n)

    # -- Fuels -------------------------------------------------------------
    # Nuclear: flat baseload
    gen_nuclear = np.full(n, 6500.0) + rng.normal(0.0, 50.0, n)

    # Wind: random-walk-ish, clipped positive
    wind_increments = rng.normal(0.0, 150.0, n)
    gen_wind = np.maximum(0.0, 4000.0 + np.cumsum(wind_increments) * 0.05)
    gen_wind = np.clip(gen_wind, 0.0, 12000.0)

    # Solar: daytime bump using settlement-period position
    # sp_of_day is 0-47; midday UTC ≈ sp 24 (noon UTC ≈ 12:00, sp index 24)
    solar_profile = np.maximum(
        0.0,
        np.sin(np.pi * (sp_of_day - 8) / 30.0)
    )
    solar_profile[sp_of_day < 8] = 0.0
    solar_profile[sp_of_day > 38] = 0.0
    gen_solar = solar_profile * 5000.0 + rng.normal(0.0, 100.0, n)
    gen_solar = np.clip(gen_solar, 0.0, None)

    # Biomass: small constant + noise
    gen_biomass = np.full(n, 1200.0) + rng.normal(0.0, 50.0, n)
    gen_biomass = np.clip(gen_biomass, 0.0, None)

    # Gas: fills residual demand (demand - nuclear - wind - solar - biomass)
    gen_gas = indo - gen_nuclear - gen_wind - gen_solar - gen_biomass
    gen_gas = np.clip(gen_gas, 500.0, None)  # always some gas running

    # -- Assemble and save -------------------------------------------------
    df = pd.DataFrame(
        {
            "price": price.astype(np.float32),
            "indo": indo.astype(np.float32),
            "itsdo": itsdo.astype(np.float32),
            "gen_gas": gen_gas.astype(np.float32),
            "gen_wind": gen_wind.astype(np.float32),
            "gen_nuclear": gen_nuclear.astype(np.float32),
            "gen_biomass": gen_biomass.astype(np.float32),
            "gen_solar": gen_solar.astype(np.float32),
        },
        index=index,
    )
    df.index.name = "timestamp"

    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PARQUET_PATH)
    return df


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_fixture_panel() -> PanelBundle:
    """Load the committed synthetic fixture parquet and return a PanelBundle.

    Reads ``tests/fixtures/fixture_panel.parquet`` (resolved relative to this
    module, not the working directory) and reconstructs the four inputs that
    :func:`src.build.panel.build_panel` expects.

    Returns
    -------
    PanelBundle
        Ready-to-use bundle with target, past_covariates, future_covariates.
    """
    df = pd.read_parquet(_PARQUET_PATH)

    # Ensure the index is UTC-aware (parquet may drop tz on round-trip).
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    index = df.index

    # -- price_df ----------------------------------------------------------
    price_df = df[["price"]].copy()

    # -- demand_df ---------------------------------------------------------
    demand_df = df[["indo", "itsdo"]].copy()

    # -- fuel_df: melt gen_* wide columns back to tidy format -------------
    # build_panel pivots on "fuel" column and adds the "gen_" prefix itself,
    # so we strip the prefix here before melting.
    fuel_wide = df[["gen_gas", "gen_wind", "gen_nuclear", "gen_biomass", "gen_solar"]].copy()
    fuel_wide.index.name = "timestamp"
    # Strip gen_ prefix from column names
    fuel_wide.columns = [c.replace("gen_", "", 1) for c in fuel_wide.columns]
    # Reset index so timestamp becomes a column for melt
    fuel_wide_reset = fuel_wide.reset_index()
    fuel_df = fuel_wide_reset.melt(
        id_vars="timestamp",
        var_name="fuel",
        value_name="mw",
    )

    # -- calendar_df -------------------------------------------------------
    calendar_df = build_calendar(index)

    return build_panel(price_df, demand_df, fuel_df, calendar_df)


# ---------------------------------------------------------------------------
# CLI entry point — run this module directly to regenerate the parquet
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    frame = _generate_fixture_frame()
    print(f"Generated fixture parquet: {_PARQUET_PATH}")
    print(f"Shape: {frame.shape}")
    print(frame.head())
