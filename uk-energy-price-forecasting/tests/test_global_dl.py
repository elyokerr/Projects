"""Smoke tests for src/models/global_dl.py — GlobalTiDE and GlobalTFT.

Deep-model training is CPU-intensive (~1-3 min for both models on fixture data).
These tests are SKIPPED by default.  Set the environment variable RUN_SLOW=1
to enable them:

    PowerShell:
        $env:RUN_SLOW=1; .venv\\Scripts\\python -m pytest tests/test_global_dl.py -v; $env:RUN_SLOW=$null

    Bash:
        RUN_SLOW=1 python -m pytest tests/test_global_dl.py -v
"""
import os

import numpy as np
import pytest

from src.build.fixtures import load_fixture_panel
from src.models.global_dl import GlobalTFT, GlobalTiDE

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="deep-model training is slow; set RUN_SLOW=1 to run",
)


@pytest.mark.parametrize("ModelCls", [GlobalTiDE, GlobalTFT])
def test_deep_model_smoke(ModelCls):
    bundle = load_fixture_panel()
    origin = bundle.target.time_index[-49]
    model = ModelCls().fit(bundle, train_end=origin)
    preds = model.predict_quantiles(bundle, origin, horizon=48, quantiles=(0.1, 0.5, 0.9))
    assert all(len(preds[q]) == 48 for q in preds)
    assert np.all(preds[0.1] <= preds[0.5] + 1e-6)
    assert np.all(preds[0.5] <= preds[0.9] + 1e-6)
