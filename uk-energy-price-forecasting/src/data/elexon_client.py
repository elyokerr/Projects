"""
Elexon BMRS Insights API client.

Base URL: https://data.elexon.co.uk/bmrs/api/v1
No API key required.

Design: each public function accepts an optional ``transport`` callable.
  - When ``transport`` is None, a real HTTP request is made via ``requests``.
  - When provided, ``transport(url, params) -> dict`` is called instead
    (used in tests with fixture data — no live network calls in tests).

Caching: successful responses are written to
  ``data/raw/elexon/<endpoint>_<date_from>_<date_to>.parquet`` and re-used on
  subsequent calls with the same parameters.

# TODO: confirm field names against live Elexon Insights docs at
#       build-verification time.  The field names used here
#       (startTime, fuelType, generation, demand) are best-effort based on
#       public Elexon Insights API documentation.
"""
from __future__ import annotations

import time
import logging
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
_CACHE_ROOT = Path("data/raw/elexon")
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# TODO: confirm field names against live Elexon Insights docs at build-verification time.
_FUELHH_TIMESTAMP_FIELD = "startTime"
_FUELHH_FUEL_FIELD = "fuelType"
_FUELHH_MW_FIELD = "generation"
_DEMAND_TIMESTAMP_FIELD = "startTime"
_DEMAND_VALUE_FIELD = "demand"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _request(
    url: str,
    params: dict,
    transport: Optional[Callable] = None,
) -> dict:
    """Fetch ``url`` with ``params``, retrying up to 3 times on errors/429/5xx.

    If ``transport`` is provided it is called instead of ``requests.get``.
    """
    if transport is not None:
        return transport(url, params)

    headers = {"User-Agent": _USER_AGENT}
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp
                )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            log.warning("Attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"All 3 attempts failed for {url}") from last_exc


def _cache_path(endpoint: str, date_from: date, date_to: date) -> Path:
    return (
        _CACHE_ROOT
        / f"{endpoint}_{date_from.isoformat()}_{date_to.isoformat()}.parquet"
    )


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        log.info("Loading cached data from %s", path)
        return pd.read_parquet(path)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Cached data to %s", path)


def _parse_utc_timestamp(ts_str: str) -> pd.Timestamp:
    """Parse an ISO-8601 timestamp string to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_generation_by_fuel(
    date_from: date,
    date_to: date,
    transport: Optional[Callable] = None,
) -> pd.DataFrame:
    """Fetch half-hourly generation by fuel type from Elexon BMRS /datasets/FUELHH.

    Parameters
    ----------
    date_from : date  Start date (inclusive).
    date_to   : date  End date (inclusive).
    transport : callable, optional
        Injected transport ``(url, params) -> dict``.  If None, uses
        ``requests.get`` (makes a live network call).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (UTC datetime), ``fuel`` (str), ``mw`` (float).
    """
    cache_file = _cache_path("FUELHH", date_from, date_to)
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached

    url = f"{_BASE_URL}/datasets/FUELHH"
    params = {
        "settlementDateFrom": date_from.isoformat(),
        "settlementDateTo": date_to.isoformat(),
    }
    raw = _request(url, params, transport)

    records = raw.get("data", [])
    rows = []
    for rec in records:
        rows.append({
            "timestamp": _parse_utc_timestamp(rec[_FUELHH_TIMESTAMP_FIELD]),
            "fuel": rec[_FUELHH_FUEL_FIELD],
            "mw": float(rec[_FUELHH_MW_FIELD]),
        })

    df = pd.DataFrame(rows, columns=["timestamp", "fuel", "mw"])

    if transport is None:
        _save_cache(df, cache_file)

    return df


def fetch_demand(
    date_from: date,
    date_to: date,
    transport: Optional[Callable] = None,
) -> pd.DataFrame:
    """Fetch half-hourly demand from Elexon BMRS /datasets/INDO and /datasets/ITSDO.

    Both datasets are fetched and merged on ``timestamp``.

    Parameters
    ----------
    date_from : date  Start date (inclusive).
    date_to   : date  End date (inclusive).
    transport : callable, optional
        Injected transport ``(url, params) -> dict``.  If None, uses
        ``requests.get`` (makes a live network call).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (UTC datetime), ``indo`` (float), ``itsdo`` (float).
    """
    cache_file = _cache_path("DEMAND", date_from, date_to)
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached

    params = {
        "settlementDateFrom": date_from.isoformat(),
        "settlementDateTo": date_to.isoformat(),
    }

    def _fetch_one(endpoint: str) -> pd.DataFrame:
        url = f"{_BASE_URL}/datasets/{endpoint}"
        raw = _request(url, params, transport)
        records = raw.get("data", [])
        rows = []
        for rec in records:
            rows.append({
                "timestamp": _parse_utc_timestamp(rec[_DEMAND_TIMESTAMP_FIELD]),
                "value": float(rec[_DEMAND_VALUE_FIELD]),
            })
        return pd.DataFrame(rows, columns=["timestamp", "value"])

    indo_df = _fetch_one("INDO").rename(columns={"value": "indo"})
    itsdo_df = _fetch_one("ITSDO").rename(columns={"value": "itsdo"})

    df = pd.merge(indo_df, itsdo_df, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if transport is None:
        _save_cache(df, cache_file)

    return df
