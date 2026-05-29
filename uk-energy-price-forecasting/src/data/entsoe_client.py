"""
ENTSO-E Transparency Platform REST API client.

Base URL: https://web-api.tp.entsoe.eu/api
Authentication: API token (env var ``ENTSOE_TOKEN``); obtain a free token by
emailing transparency@entsoe.eu.

Design: the public function accepts an optional ``transport`` callable.
  - When ``transport`` is None and a token is available, a real HTTP request is
    made via ``requests``.
  - When provided, ``transport(url, params) -> str`` is called instead
    (used in tests with fixture XML — no live network calls in tests).

Caching: successful responses are written to
  ``data/raw/entsoe/dayahead_<date_from>_<date_to>.parquet`` and re-used on
  subsequent calls with the same parameters.

Prices are returned in the native published currency (EUR/MWh as published by
ENTSO-E). No FX conversion is applied here; callers should note that GB
day-ahead prices may occasionally be published in GBP after post-Brexit
arrangements — verify against live data at integration time.

# TODO: confirm namespace + element names against a real ENTSO-E A44 response
#       at build-verification time.  The namespace URI and element structure
#       used here are based on the documented ENTSO-E XML schema but have not
#       been verified against a live API response.
"""
from __future__ import annotations

import os
import logging
from datetime import date
from pathlib import Path
from typing import Callable, Optional
import xml.etree.ElementTree as ET

import pandas as pd
import requests

log = logging.getLogger(__name__)

_BASE_URL = "https://web-api.tp.entsoe.eu/api"
_CACHE_ROOT = Path("data/raw/entsoe")
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# GB bidding zone EIC code.
_GB_DOMAIN = "10YGB----------A"

# TODO: confirm namespace + element names against a real ENTSO-E A44 response
#       at build-verification time.
_NS_URI = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"
_NS = {"ns": _NS_URI}

# Resolution string → pd.Timedelta mapping (extend as needed).
_RESOLUTION_MAP = {
    "PT15M": pd.Timedelta(minutes=15),
    "PT30M": pd.Timedelta(minutes=30),
    "PT60M": pd.Timedelta(hours=1),
    "P1D": pd.Timedelta(days=1),
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cache_path(date_from: date, date_to: date) -> Path:
    # strftime("%Y-%m-%d") avoids colons (illegal in Windows filenames) that
    # appear when callers pass a datetime instead of a date.
    return (
        _CACHE_ROOT
        / f"dayahead_{date_from.strftime('%Y-%m-%d')}_{date_to.strftime('%Y-%m-%d')}.parquet"
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


def _parse_xml(xml_text: str) -> pd.DataFrame:
    """Parse ENTSO-E Publication_MarketDocument XML into a tidy DataFrame.

    Returns columns: ``timestamp`` (UTC), ``price`` (float).
    """
    root = ET.fromstring(xml_text)

    # Handle documents with and without the default namespace prefix.
    # TODO: confirm namespace URI against a real ENTSO-E A44 response at
    #       build-verification time.
    ns = _NS

    rows = []
    for ts_elem in root.findall("ns:TimeSeries", ns):
        for period_elem in ts_elem.findall("ns:Period", ns):
            interval_elem = period_elem.find("ns:timeInterval", ns)
            resolution_str = period_elem.findtext("ns:resolution", namespaces=ns)

            if interval_elem is None or resolution_str is None:
                log.warning("Skipping Period with missing timeInterval or resolution")
                continue

            start_str = interval_elem.findtext("ns:start", namespaces=ns)
            if start_str is None:
                log.warning("Skipping Period with missing timeInterval/start")
                continue

            period_start = pd.Timestamp(start_str).tz_convert("UTC")
            resolution = _RESOLUTION_MAP.get(resolution_str)
            if resolution is None:
                log.warning("Unknown resolution %s — skipping Period", resolution_str)
                continue

            for point_elem in period_elem.findall("ns:Point", ns):
                pos_text = point_elem.findtext("ns:position", namespaces=ns)
                price_text = point_elem.findtext("ns:price.amount", namespaces=ns)
                if pos_text is None or price_text is None:
                    continue
                position = int(pos_text)
                price = float(price_text)
                # Timestamp = period_start + (position - 1) * resolution
                ts = period_start + (position - 1) * resolution
                rows.append({"timestamp": ts, "price": price})

    df = pd.DataFrame(rows, columns=["timestamp", "price"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_day_ahead_price(
    date_from: date,
    date_to: date,
    token: Optional[str] = None,
    transport: Optional[Callable] = None,
) -> pd.DataFrame:
    """Fetch GB day-ahead electricity prices from ENTSO-E Transparency Platform.

    Prices are in EUR/MWh as published (native currency; no FX conversion).

    Parameters
    ----------
    date_from : date  Start date (inclusive).
    date_to   : date  End date (inclusive).
    token : str, optional
        ENTSO-E API token.  Defaults to ``os.environ.get("ENTSOE_TOKEN")``.
    transport : callable, optional
        Injected transport ``(url, params) -> str`` (returns raw XML text).
        If None, a live HTTP request is made.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (UTC datetime), ``price`` (float, EUR/MWh).

    Raises
    ------
    ValueError
        If no token is available and no transport is injected.
    """
    if token is None:
        token = os.environ.get("ENTSOE_TOKEN")

    if token is None and transport is None:
        raise ValueError(
            "ENTSOE_TOKEN not set; obtain a free token by emailing "
            "transparency@entsoe.eu"
        )

    cache_file = _cache_path(date_from, date_to)
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached

    # ENTSO-E expects UTC datetime strings in the format YYYYMMDDHHmm.
    # The day-ahead market covers CET/CEST midnight → midnight, but using
    # UTC dates with 23:00Z boundaries is the standard approach.
    period_start = f"{date_from.strftime('%Y%m%d')}2300"
    period_end = f"{date_to.strftime('%Y%m%d')}2300"

    params = {
        "documentType": "A44",
        "in_Domain": _GB_DOMAIN,
        "out_Domain": _GB_DOMAIN,
        "periodStart": period_start,
        "periodEnd": period_end,
        "securityToken": token,
    }

    url = _BASE_URL

    if transport is not None:
        xml_text = transport(url, params)
    else:
        headers = {"User-Agent": _USER_AGENT}
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        xml_text = resp.text

    df = _parse_xml(xml_text)

    if transport is None:
        _save_cache(df, cache_file)

    return df
