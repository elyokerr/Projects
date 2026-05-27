"""Tests for src/build/leakage.py — the critical leakage guard."""
import pytest
from src.build.leakage import assert_no_future_leakage, LeakageError


def test_outturn_never_in_future_covariates(tiny_bundle):
    fut_cols = set(tiny_bundle.future_covariates.components)
    assert fut_cols.isdisjoint({"demand_indo", "gen_gas", "price"})


def test_leakage_guard_raises_when_outturn_injected(tiny_bundle_with_leak):
    with pytest.raises(LeakageError):
        assert_no_future_leakage(tiny_bundle_with_leak)


def test_clean_bundle_passes(tiny_bundle):
    assert assert_no_future_leakage(tiny_bundle) is True
