"""Tests for src/data/elexon_client.py — uses injected transport; NO live network calls."""
import json
from pathlib import Path
from datetime import date

import pandas as pd

from src.data.elexon_client import (
    fetch_demand,
    fetch_generation_by_fuel,
    fetch_system_price,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _make_fuelhh_transport():
    """Return a transport callable that serves the FUELHH fixture."""
    fixture_path = FIXTURES / "elexon_fuelhh_sample.json"
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))

    def transport(url: str, params: dict):
        return raw

    return transport


def _make_fixture_transport(filename: str):
    """Return a transport callable that serves a single JSON fixture."""
    raw = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))

    def transport(url: str, params: dict):
        return raw

    return transport


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------

class TestFetchGenerationByFuel:
    def test_columns_present(self):
        df = fetch_generation_by_fuel(
            date(2024, 1, 15), date(2024, 1, 15),
            transport=_make_fuelhh_transport(),
        )
        assert set(df.columns) >= {"timestamp", "fuel", "mw"}, f"Columns: {df.columns.tolist()}"

    def test_timestamps_are_utc(self):
        df = fetch_generation_by_fuel(
            date(2024, 1, 15), date(2024, 1, 15),
            transport=_make_fuelhh_transport(),
        )
        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_multiple_fuels(self):
        df = fetch_generation_by_fuel(
            date(2024, 1, 15), date(2024, 1, 15),
            transport=_make_fuelhh_transport(),
        )
        assert df["fuel"].nunique() > 1, "Expected more than one distinct fuel type"

    def test_mw_is_numeric(self):
        df = fetch_generation_by_fuel(
            date(2024, 1, 15), date(2024, 1, 15),
            transport=_make_fuelhh_transport(),
        )
        assert pd.api.types.is_numeric_dtype(df["mw"]), "mw column should be numeric"

    def test_non_empty(self):
        df = fetch_generation_by_fuel(
            date(2024, 1, 15), date(2024, 1, 15),
            transport=_make_fuelhh_transport(),
        )
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Demand tests — single /demand/outturn endpoint returns both series
# ---------------------------------------------------------------------------

class TestFetchDemand:
    def _get_df(self):
        return fetch_demand(
            date(2024, 1, 15), date(2024, 1, 15),
            transport=_make_fixture_transport("elexon_demand_sample.json"),
        )

    def test_columns_present(self):
        df = self._get_df()
        assert set(df.columns) >= {"timestamp", "indo", "itsdo"}, f"Columns: {df.columns.tolist()}"

    def test_timestamps_are_utc(self):
        df = self._get_df()
        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_demand_columns_numeric(self):
        df = self._get_df()
        assert pd.api.types.is_numeric_dtype(df["indo"])
        assert pd.api.types.is_numeric_dtype(df["itsdo"])

    def test_non_empty(self):
        df = self._get_df()
        assert len(df) > 0

    def test_rows_match_fixture_count(self):
        df = self._get_df()
        assert len(df) == 4


# ---------------------------------------------------------------------------
# System price tests
# ---------------------------------------------------------------------------

class TestFetchSystemPrice:
    def _get_df(self):
        return fetch_system_price(
            date(2024, 1, 15), date(2024, 1, 15),
            transport=_make_fixture_transport("elexon_sysprice_sample.json"),
        )

    def test_columns_present(self):
        df = self._get_df()
        assert set(df.columns) >= {"timestamp", "price"}, f"Columns: {df.columns.tolist()}"

    def test_timestamps_are_utc(self):
        df = self._get_df()
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_price_is_numeric(self):
        df = self._get_df()
        assert pd.api.types.is_numeric_dtype(df["price"])

    def test_negative_prices_preserved(self):
        # GB system price can go negative; the client must not clip it.
        df = self._get_df()
        assert (df["price"] < 0).any()

    def test_rows_match_fixture_count(self):
        df = self._get_df()
        assert len(df) == 4
