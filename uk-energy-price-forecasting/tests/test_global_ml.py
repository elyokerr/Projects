"""Tests for src/models/global_ml.py — GlobalLGBM quantile model."""
import numpy as np

from src.build.fixtures import load_fixture_panel
from src.models.global_ml import GlobalLGBM


# ---------------------------------------------------------------------------
# Verbatim test from spec
# ---------------------------------------------------------------------------

def test_global_lgbm_produces_monotone_quantiles():
    bundle = load_fixture_panel()
    origin = bundle.target.time_index[-49]   # leave 48 steps to forecast
    model = GlobalLGBM(quantiles=(0.1, 0.5, 0.9), lags=48).fit(bundle, train_end=origin)
    preds = model.predict_quantiles(bundle, origin, horizon=48, quantiles=(0.1, 0.5, 0.9))
    assert set(preds) == {0.1, 0.5, 0.9}
    assert all(len(preds[q]) == 48 for q in preds)
    assert np.all(preds[0.1] <= preds[0.5] + 1e-6)
    assert np.all(preds[0.5] <= preds[0.9] + 1e-6)


# ---------------------------------------------------------------------------
# Additional correctness tests
# ---------------------------------------------------------------------------

def test_global_lgbm_fit_returns_self():
    bundle = load_fixture_panel()
    origin = bundle.target.time_index[-49]
    model = GlobalLGBM(quantiles=(0.1, 0.5, 0.9), lags=48)
    result = model.fit(bundle, train_end=origin)
    assert result is model


def test_global_lgbm_output_shape():
    """predict_quantiles returns arrays of exactly `horizon` length."""
    bundle = load_fixture_panel()
    origin = bundle.target.time_index[-49]
    model = GlobalLGBM(quantiles=(0.1, 0.5, 0.9), lags=48).fit(bundle, train_end=origin)
    for horizon in [1, 24, 48]:
        preds = model.predict_quantiles(bundle, origin, horizon=horizon)
        assert all(len(v) == horizon for v in preds.values()), (
            f"Expected length {horizon}, got {[len(v) for v in preds.values()]}"
        )


def test_global_lgbm_custom_quantiles():
    """Model honours a different set of requested quantiles at predict time."""
    bundle = load_fixture_panel()
    origin = bundle.target.time_index[-49]
    model = GlobalLGBM(quantiles=(0.1, 0.5, 0.9), lags=48).fit(bundle, train_end=origin)
    preds = model.predict_quantiles(bundle, origin, horizon=10, quantiles=(0.1, 0.9))
    assert set(preds.keys()) == {0.1, 0.9}
