"""
Tests for the synthetic fixture panel.

Validates that the committed parquet loads correctly into a PanelBundle and
passes the leakage guard.
"""
from src.build.fixtures import load_fixture_panel
from src.build.leakage import assert_no_future_leakage


def test_fixture_panel_loads_and_passes_guard():
    bundle = load_fixture_panel()
    assert assert_no_future_leakage(bundle) is True


def test_fixture_panel_has_expected_length():
    bundle = load_fixture_panel()
    # ~30 days * 48 SPs (allow a small tolerance for grid edge alignment)
    assert 1430 <= len(bundle.target) <= 1440


def test_fixture_panel_has_fuel_and_demand_in_past_covariates():
    bundle = load_fixture_panel()
    comps = set(bundle.past_covariates.components)
    assert "demand_indo" in comps
    assert any(c.startswith("gen_") for c in comps)
