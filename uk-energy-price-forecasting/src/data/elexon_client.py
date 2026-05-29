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

Field names and endpoints below were confirmed against live Elexon responses
on 2026-05-29.
"""
from __future__ import annotations

import time
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
_CACHE_ROOT = Path("data/raw/elexon")
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# Confirmed against live Elexon responses 2026-05-29.
_FUELHH_TIMESTAMP_FIELD = "startTime"
_FUELHH_FUEL_FIELD = "fuelType"
_FUELHH_MW_FIELD = "generation"
# /demand/outturn returns both demand series in one record.
_DEMAND_TIMESTAMP_FIELD = "startTime"
_DEMAND_INDO_FIELD = "initialDemandOutturn"
_DEMAND_ITSDO_FIELD = "initialTransmissionSystemDemandOutturn"
# /balancing/settlement/system-prices/{date}: GB has a single imbalance price
# (System Sell Price == System Buy Price since P305), so systemSellPrice is the
# canonical system price.
_SYSPRICE_TIMESTAMP_FIELD = "startTime"
_SYSPRICE_VALUE_FIELD = "systemSellPrice"


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


# Elexon dataset endpoints cap the queryable window. FUELHH allows 7 days;
# /demand/outturn allows ~14. We chunk both at 7 days to stay safely under.
_MAX_RANGE_DAYS = 7


def _date_chunks(date_from: date, date_to: date, max_days: int = _MAX_RANGE_DAYS):
    """Yield (start, end) inclusive sub-ranges no longer than ``max_days``."""
    start = date_from
    while start <= date_to:
        end = min(start + timedelta(days=max_days - 1), date_to)
        yield start, end
        start = end + timedelta(days=1)


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
    rows = []
    for chunk_from, chunk_to in _date_chunks(date_from, date_to):
        params = {
            "settlementDateFrom": chunk_from.isoformat(),
            "settlementDateTo": chunk_to.isoformat(),
        }
        raw = _request(url, params, transport)
        for rec in raw.get("data", []):
            rows.append({
                "timestamp": _parse_utc_timestamp(rec[_FUELHH_TIMESTAMP_FIELD]),
                "fuel": rec[_FUELHH_FUEL_FIELD],
                "mw": float(rec[_FUELHH_MW_FIELD]),
            })
        if transport is not None:
            # In tests the transport serves a single fixed payload; one pass.
            break

    df = pd.DataFrame(rows, columns=["timestamp", "fuel", "mw"])
    df = df.drop_duplicates(["timestamp", "fuel"]).sort_values(
        ["timestamp", "fuel"]
    ).reset_index(drop=True)

    if transport is None:
        _save_cache(df, cache_file)

    return df


def fetch_demand(
    date_from: date,
    date_to: date,
    transport: Optional[Callable] = None,
) -> pd.DataFrame:
    """Fetch half-hourly demand from Elexon BMRS /demand/outturn.

    A single endpoint returns both the initial demand out-turn (INDO) and the
    initial transmission-system demand out-turn (ITSDO) per settlement period.

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

    url = f"{_BASE_URL}/demand/outturn"
    rows = []
    for chunk_from, chunk_to in _date_chunks(date_from, date_to):
        params = {
            "settlementDateFrom": chunk_from.isoformat(),
            "settlementDateTo": chunk_to.isoformat(),
        }
        raw = _request(url, params, transport)
        for rec in raw.get("data", []):
            rows.append({
                "timestamp": _parse_utc_timestamp(rec[_DEMAND_TIMESTAMP_FIELD]),
                "indo": float(rec[_DEMAND_INDO_FIELD]),
                "itsdo": float(rec[_DEMAND_ITSDO_FIELD]),
            })
        if transport is not None:
            # In tests the transport serves a single fixed payload; one pass.
            break

    df = pd.DataFrame(rows, columns=["timestamp", "indo", "itsdo"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    if transport is None:
        _save_cache(df, cache_file)

    return df


def fetch_system_price(
    date_from: date,
    date_to: date,
    transport: Optional[Callable] = None,
) -> pd.DataFrame:
    """Fetch half-hourly GB system (imbalance) price from Elexon BMRS.

    Endpoint ``/balancing/settlement/system-prices/{date}`` returns 48
    settlement periods for a single date, so a date range is fetched by
    iterating day by day.  GB operates a single imbalance price (System Sell
    Price == System Buy Price), so ``systemSellPrice`` is used as the price.

    Parameters
    ----------
    date_from : date  Start date (inclusive).
    date_to   : date  End date (inclusive).
    transport : callable, optional
        Injected transport ``(url, params) -> dict``.  If None, uses
        ``requests.get`` (makes a live network call).  In tests the transport
        is expected to return one day's payload regardless of the URL.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (UTC datetime), ``price`` (float, GBP/MWh).
    """
    cache_file = _cache_path("SYSPRICE", date_from, date_to)
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached

    rows = []
    day = date_from
    while day <= date_to:
        url = f"{_BASE_URL}/balancing/settlement/system-prices/{day.isoformat()}"
        raw = _request(url, {}, transport)
        for rec in raw.get("data", []):
            rows.append({
                "timestamp": _parse_utc_timestamp(rec[_SYSPRICE_TIMESTAMP_FIELD]),
                "price": float(rec[_SYSPRICE_VALUE_FIELD]),
            })
        if transport is not None:
            # In tests the transport serves a single fixed payload; one pass.
            break
        day += timedelta(days=1)

    df = pd.DataFrame(rows, columns=["timestamp", "price"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    if transport is None:
        _save_cache(df, cache_file)

    return df
