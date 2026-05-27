"""Tests for src/data/calendar.py"""
import pandas as pd
import pytest

from src.data.calendar import build_calendar

FOURIER_COLS = [
    "sin_daily_1", "cos_daily_1", "sin_daily_2", "cos_daily_2",
    "sin_weekly_1", "cos_weekly_1", "sin_weekly_2", "cos_weekly_2",
    "sin_annual_1", "cos_annual_1", "sin_annual_2", "cos_annual_2",
]


@pytest.fixture
def two_day_index():
    """Half-hourly UTC index over 2024-06-01 (Saturday) and 2024-06-02 (Sunday)."""
    return pd.date_range(
        start="2024-06-01 00:00:00",
        end="2024-06-02 23:30:00",
        freq="30min",
        tz="UTC",
    )


@pytest.fixture
def cal(two_day_index):
    return build_calendar(two_day_index)


class TestBuildCalendarStructure:
    def test_required_columns_present(self, cal):
        required = {"dow", "is_weekend", "is_holiday", "sp_of_day"} | set(FOURIER_COLS)
        missing = required - set(cal.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_index_preserved(self, cal, two_day_index):
        assert len(cal) == len(two_day_index)
        assert (cal.index == two_day_index).all()

    def test_sp_of_day_range(self, cal):
        assert cal["sp_of_day"].min() == 1
        assert cal["sp_of_day"].max() == 48

    def test_sp_of_day_complete_within_day(self, cal, two_day_index):
        # Each calendar day should have 48 distinct SPs (index is 2 full days).
        for day_str in ["2024-06-01", "2024-06-02"]:
            day_mask = cal.index.normalize() == pd.Timestamp(day_str, tz="UTC")
            sps = cal.loc[day_mask, "sp_of_day"]
            assert sorted(sps.tolist()) == list(range(1, 49)), (
                f"SPs for {day_str}: {sorted(sps.tolist())}"
            )


class TestWeekendFlag:
    def test_saturday_is_weekend(self, cal):
        sat_mask = cal.index.normalize() == pd.Timestamp("2024-06-01", tz="UTC")
        assert (cal.loc[sat_mask, "is_weekend"] == 1).all(), "Saturday rows should have is_weekend=1"

    def test_sunday_is_weekend(self, cal):
        # 2024-06-02 is Sunday.  The index runs through 23:30 UTC, but in
        # BST (UTC+1) the last two slots (23:00 UTC, 23:30 UTC) fall on Monday
        # local time, so we only check rows up to 22:30 UTC which are
        # unambiguously Sunday in Europe/London.
        sun_mask = (
            (cal.index >= pd.Timestamp("2024-06-02 00:00:00", tz="UTC"))
            & (cal.index <= pd.Timestamp("2024-06-02 22:30:00", tz="UTC"))
        )
        assert (cal.loc[sun_mask, "is_weekend"] == 1).all(), "Sunday rows should have is_weekend=1"


class TestHolidayFlag:
    def test_christmas_day_is_holiday(self):
        """2024-12-25 is Christmas Day — a bank holiday in England."""
        idx = pd.date_range(
            start="2024-12-25 00:00:00",
            periods=48,
            freq="30min",
            tz="UTC",
        )
        cal = build_calendar(idx)
        assert (cal["is_holiday"] == 1).all(), (
            "All rows on 2024-12-25 should have is_holiday=1"
        )

    def test_non_holiday_not_flagged(self, cal):
        # 2024-06-01 is a Saturday but NOT a bank holiday in England.
        sat_mask = cal.index.normalize() == pd.Timestamp("2024-06-01", tz="UTC")
        assert (cal.loc[sat_mask, "is_holiday"] == 0).all(), (
            "2024-06-01 should not be a bank holiday"
        )


class TestFourierTerms:
    def test_fourier_bounds(self, cal):
        for col in FOURIER_COLS:
            assert col in cal.columns, f"Missing Fourier column: {col}"
            assert cal[col].between(-1.0, 1.0).all(), (
                f"Column {col} has values outside [-1, 1]: "
                f"min={cal[col].min():.4f}, max={cal[col].max():.4f}"
            )

    def test_fourier_columns_not_constant(self, cal):
        for col in FOURIER_COLS:
            assert cal[col].nunique() > 1, f"Fourier column {col} is constant — likely a bug"
