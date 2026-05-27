"""Tests for src/data/entsoe_client.py — uses injected transport; NO live network calls."""
from pathlib import Path
from datetime import date

import pandas as pd
import pytest

from src.data.entsoe_client import fetch_day_ahead_price

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _xml_transport():
    """Return a transport callable that serves the ENTSO-E day-ahead fixture XML."""
    fixture_path = FIXTURES / "entsoe_dayahead_sample.xml"
    xml_text = fixture_path.read_text(encoding="utf-8")

    def transport(url: str, params: dict) -> str:
        return xml_text

    return transport


class TestFetchDayAheadPrice:
    def _get_df(self):
        return fetch_day_ahead_price(
            date(2024, 1, 15), date(2024, 1, 16),
            token="DUMMY_TOKEN",
            transport=_xml_transport(),
        )

    def test_columns_present(self):
        df = self._get_df()
        assert set(df.columns) >= {"timestamp", "price"}, f"Columns: {df.columns.tolist()}"

    def test_timestamps_are_utc(self):
        df = self._get_df()
        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_price_is_numeric(self):
        df = self._get_df()
        assert pd.api.types.is_numeric_dtype(df["price"])

    def test_correct_row_count(self):
        df = self._get_df()
        # Fixture has 4 Points in one Period with PT60M resolution.
        assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    def test_timestamps_monotonic_increasing(self):
        df = self._get_df()
        assert df["timestamp"].is_monotonic_increasing

    def test_first_timestamp(self):
        df = self._get_df()
        # Period start is 2024-01-15T23:00Z, position 1 → offset 0.
        expected = pd.Timestamp("2024-01-15T23:00:00", tz="UTC")
        assert df["timestamp"].iloc[0] == expected

    def test_prices_match_fixture(self):
        df = self._get_df()
        expected_prices = [95.50, 88.25, 82.10, 79.80]
        for i, price in enumerate(expected_prices):
            assert abs(df["price"].iloc[i] - price) < 1e-6


class TestFetchDayAheadPriceNoToken:
    def test_raises_without_token_and_transport(self, monkeypatch):
        """Should raise ValueError when ENTSOE_TOKEN is absent and no transport injected."""
        monkeypatch.delenv("ENTSOE_TOKEN", raising=False)
        with pytest.raises(ValueError, match="ENTSOE_TOKEN"):
            fetch_day_ahead_price(date(2024, 1, 15), date(2024, 1, 16))
