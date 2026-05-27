"""Tests for src/models/global_ml.py — GlobalLGBM quantile model.

The 144-sub-model fit (output_chunk_length=48 x 3 quantiles) is the slow part,
so the fitted model is shared across tests via a module-scoped fixture rather
than refitting in every test.
"""
import numpy as np
import pytest

from src.build.fixtures import load_fixture_panel
from src.models.global_ml import GlobalLGBM


@pytest.fixture(scope="module")
def bundle():
    return load_fixture_panel()


@pytest.fixture(scope="module")
def origin(bundle):
    return bundle.target.time_index[-49]  # leave 48 steps to forecast


@pytest.fixture(scope="module")
def fitted(bundle, origin):
    return GlobalLGBM(quantiles=(0.1, 0.5, 0.9)).fit(bundle, train_end=origin)


def test_global_lgbm_produces_monotone_quantiles(fitted, bundle, origin):
    preds = fitted.predict_quantiles(bundle, origin, horizon=48, quantiles=(0.1, 0.5, 0.9))
    assert set(preds) == {0.1, 0.5, 0.9}
    assert all(len(preds[q]) == 48 for q in preds)
    assert np.all(preds[0.1] <= preds[0.5] + 1e-6)
    assert np.all(preds[0.5] <= preds[0.9] + 1e-6)


def test_global_lgbm_output_shape(fitted, bundle, origin):
    """predict_quantiles returns arrays of exactly `horizon` length."""
    for horizon in [1, 24, 48]:
        preds = fitted.predict_quantiles(bundle, origin, horizon=horizon)
        assert all(len(v) == horizon for v in preds.values()), (
            f"Expected length {horizon}, got {[len(v) for v in preds.values()]}"
        )


def test_global_lgbm_custom_quantiles(fitted, bundle, origin):
    """Model honours a different set of requested quantiles at predict time."""
    preds = fitted.predict_quantiles(bundle, origin, horizon=10, quantiles=(0.1, 0.9))
    assert set(preds.keys()) == {0.1, 0.9}


def test_global_lgbm_fit_returns_self(bundle, origin):
    # Tiny config — only verifies the fit() return contract, kept fast.
    model = GlobalLGBM(quantiles=(0.5,), lags=[-1], lags_past_covariates=[-1], n_estimators=5)
    assert model.fit(bundle, train_end=origin) is model
