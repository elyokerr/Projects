"""Tests for src/models/baselines.py — SeasonalNaive and ThetaBaseline."""
import numpy as np
import pandas as pd
from darts import TimeSeries

from src.build.panel import PanelBundle
from src.models.baselines import SeasonalNaive, ThetaBaseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _periodic_series(n_days=5, season=48):
    """Values repeat every `season` steps exactly."""
    base = np.sin(np.arange(season) / season * 2 * np.pi) * 10 + 50
    vals = np.tile(base, n_days)
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="30min")
    return TimeSeries.from_times_and_values(idx, vals)


def _make_bundle(ts: TimeSeries) -> PanelBundle:
    """Wrap a TimeSeries in a PanelBundle (all slots use the same series)."""
    return PanelBundle(target=ts, past_covariates=ts, future_covariates=ts, series_index={})


# ---------------------------------------------------------------------------
# SeasonalNaive — verbatim tests from spec
# ---------------------------------------------------------------------------

def test_seasonal_naive_zero_error_on_periodic_series():
    ts = _periodic_series()
    bundle = _make_bundle(ts)
    model = SeasonalNaive(season=48).fit(bundle)
    # origin at end of day 3 (position 4*48 - 1 = 191), forecast next 48
    origin = ts.time_index[4 * 48 - 1]
    preds = model.predict_quantiles(bundle, origin, horizon=48, quantiles=(0.1, 0.5, 0.9))
    truth = ts.values().flatten()[4 * 48: 5 * 48]
    assert np.allclose(preds[0.5], truth, atol=1e-9)


def test_seasonal_naive_quantiles_all_equal_point():
    ts = _periodic_series()
    bundle = _make_bundle(ts)
    model = SeasonalNaive(season=48).fit(bundle)
    origin = ts.time_index[4 * 48 - 1]
    preds = model.predict_quantiles(bundle, origin, horizon=48)
    assert np.allclose(preds[0.1], preds[0.9])


# ---------------------------------------------------------------------------
# SeasonalNaive — additional tests
# ---------------------------------------------------------------------------

def test_seasonal_naive_fit_returns_self():
    ts = _periodic_series()
    bundle = _make_bundle(ts)
    model = SeasonalNaive(season=48)
    result = model.fit(bundle)
    assert result is model


def test_seasonal_naive_returns_all_requested_quantiles():
    ts = _periodic_series()
    bundle = _make_bundle(ts)
    model = SeasonalNaive(season=48).fit(bundle)
    origin = ts.time_index[4 * 48 - 1]
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    preds = model.predict_quantiles(bundle, origin, horizon=10, quantiles=quantiles)
    assert set(preds.keys()) == set(quantiles)
    for q in quantiles:
        assert len(preds[q]) == 10


def test_seasonal_naive_forecast_length_matches_horizon():
    ts = _periodic_series()
    bundle = _make_bundle(ts)
    model = SeasonalNaive(season=48).fit(bundle)
    origin = ts.time_index[4 * 48 - 1]
    for h in [1, 10, 48]:
        preds = model.predict_quantiles(bundle, origin, horizon=h)
        assert all(len(v) == h for v in preds.values())


def test_seasonal_naive_default_quantiles():
    ts = _periodic_series()
    bundle = _make_bundle(ts)
    model = SeasonalNaive(season=48).fit(bundle)
    origin = ts.time_index[4 * 48 - 1]
    preds = model.predict_quantiles(bundle, origin, horizon=5)
    # default quantiles from spec: (0.1, 0.5, 0.9)
    assert set(preds.keys()) == {0.1, 0.5, 0.9}


# ---------------------------------------------------------------------------
# ThetaBaseline — smoke tests
# ---------------------------------------------------------------------------

def test_theta_baseline_smoke():
    """Theta fits on fixture target and returns dict of arrays of length 48."""
    from src.build.fixtures import load_fixture_panel
    bundle = load_fixture_panel()
    origin = bundle.target.time_index[-49]
    model = ThetaBaseline().fit(bundle, train_end=origin)
    preds = model.predict_quantiles(bundle, origin, horizon=48, quantiles=(0.1, 0.5, 0.9))
    assert set(preds.keys()) == {0.1, 0.5, 0.9}
    assert all(len(preds[q]) == 48 for q in preds)


def test_theta_baseline_quantiles_all_equal():
    """Theta is deterministic — all quantiles should be the same array."""
    from src.build.fixtures import load_fixture_panel
    bundle = load_fixture_panel()
    origin = bundle.target.time_index[-49]
    model = ThetaBaseline().fit(bundle, train_end=origin)
    preds = model.predict_quantiles(bundle, origin, horizon=48, quantiles=(0.1, 0.5, 0.9))
    assert np.allclose(preds[0.1], preds[0.9])


def test_theta_baseline_fit_returns_self():
    from src.build.fixtures import load_fixture_panel
    bundle = load_fixture_panel()
    model = ThetaBaseline()
    result = model.fit(bundle)
    assert result is model
