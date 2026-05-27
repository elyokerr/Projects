"""Tests for src/models/foundation.py — zero-shot foundation model interfaces.

Both chronos-forecasting and timesfm are NOT installed in the local venv
(they are Colab-only GPU dependencies).  These tests verify that the
guard-clause ImportError is raised with a clear, actionable message.
"""
import numpy as np
import pytest

from src.models.foundation import chronos_forecast, timesfm_forecast


def test_chronos_raises_clear_error_when_not_installed():
    with pytest.raises(ImportError, match="chronos"):
        chronos_forecast(np.arange(100, dtype=float), horizon=48)


def test_timesfm_raises_clear_error_when_not_installed():
    with pytest.raises(ImportError, match="timesfm"):
        timesfm_forecast(np.arange(100, dtype=float), horizon=48)
