"""
Calendar / covariate feature builder for UK energy price forecasting.

Converts a pd.DatetimeIndex of half-hourly UTC timestamps into a DataFrame
of calendar features suitable for use as model covariates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of settlement periods per day (GB half-hourly market).
_SPS_PER_DAY = 48
_SPS_PER_WEEK = _SPS_PER_DAY * 7
_SPS_PER_YEAR = _SPS_PER_DAY * 365.25  # approximate

# Fourier harmonics per seasonality component.
_N_HARMONICS = 2

# Seasonality periods (in number of SPs) for Fourier terms.
_PERIODS = {
    "daily": _SPS_PER_DAY,
    "weekly": _SPS_PER_WEEK,
    "annual": _SPS_PER_YEAR,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a calendar feature DataFrame aligned to ``index``.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Typically a half-hourly UTC index.  May be tz-aware or tz-naive; tz-aware
        UTC is recommended.

    Returns
    -------
    pd.DataFrame
        Indexed by ``index`` with columns:

        - ``dow``        : int 0-6 (Monday=0) computed in Europe/London local time.
        - ``is_weekend`` : int 0/1.
        - ``is_holiday`` : int 0/1 — England & Wales bank holidays.
        - ``sp_of_day``  : int 1-48 — settlement period within the local calendar day.
        - Fourier columns: ``sin_daily_1``, ``cos_daily_1``, …,
          ``sin_annual_2``, ``cos_annual_2`` (2 harmonics each for daily,
          weekly, annual seasonality).
    """
    # Convert to Europe/London for local-time computations.
    if index.tz is None:
        local_index = index.tz_localize("UTC").tz_convert("Europe/London")
    else:
        local_index = index.tz_convert("Europe/London")

    # -- Day of week (0=Monday, 6=Sunday) in local time -------------------
    dow = local_index.dayofweek
    is_weekend = (dow >= 5).astype(int)

    # -- Settlement period of day -----------------------------------------
    # sp_of_day = hour * 2 + minute // 30 + 1  (1-indexed)
    sp_of_day = local_index.hour * 2 + local_index.minute // 30 + 1

    # -- Bank holiday flag (England) --------------------------------------
    # Use country_holidays('GB', subdiv='ENG') if available in this version,
    # falling back to holidays.UK() for older releases.
    try:
        eng_holidays = holidays.country_holidays("GB", subdiv="ENG")
    except (AttributeError, KeyError, TypeError):
        eng_holidays = holidays.UK()  # type: ignore[attr-defined]

    local_dates = local_index.normalize().date
    is_holiday = np.array(
        [1 if d in eng_holidays else 0 for d in local_dates], dtype=int
    )

    # -- Fourier terms ----------------------------------------------------
    # Build a cumulative ordinal SP counter (0-based) anchored to the first
    # row of the index — Fourier terms are relative, not absolute.
    # Using an ordinal counter rather than absolute epoch allows the features
    # to generalise across different training windows.
    ordinal_sp = np.arange(len(index), dtype=float)

    fourier_cols: dict[str, np.ndarray] = {}
    for season_name, period in _PERIODS.items():
        for k in range(1, _N_HARMONICS + 1):
            angle = 2.0 * np.pi * k * ordinal_sp / period
            fourier_cols[f"sin_{season_name}_{k}"] = np.sin(angle)
            fourier_cols[f"cos_{season_name}_{k}"] = np.cos(angle)

    # -- Assemble DataFrame -----------------------------------------------
    data: dict[str, object] = {
        "dow": dow.to_numpy(),
        "is_weekend": is_weekend,  # already ndarray from (dow >= 5).astype(int)
        "is_holiday": is_holiday,
        "sp_of_day": sp_of_day.to_numpy(),
    }
    data.update(fourier_cols)

    return pd.DataFrame(data, index=index)
