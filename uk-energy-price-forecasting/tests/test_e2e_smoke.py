"""End-to-end smoke test, gated behind RUN_SLOW=1.

Exercises the full pipeline against the fixture panel: load -> fit a global
model -> rolling-origin backtest -> ablation table. This is where integration
bugs (interface mismatches across modules) surface; ordinary `pytest` skips it.

Run with:  RUN_SLOW=1 pytest tests/test_e2e_smoke.py -v
PowerShell: $env:RUN_SLOW=1; .venv\\Scripts\\python -m pytest tests/test_e2e_smoke.py -v; $env:RUN_SLOW=$null
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="end-to-end smoke is slow; set RUN_SLOW=1 to run",
)


def test_end_to_end_pipeline():
    from src.backtest.rolling_origin import (
        build_ablation_table,
        generate_origins,
        run_backtest,
    )
    from src.build.fixtures import load_fixture_panel
    from src.build.leakage import assert_no_future_leakage
    from src.models.baselines import SeasonalNaive
    from src.models.global_ml import GlobalLGBM

    # 1. Load fixture panel and confirm the leakage invariant holds.
    bundle = load_fixture_panel()
    assert assert_no_future_leakage(bundle) is True

    # 2. A couple of interior rolling origins.
    idx = bundle.target.time_index
    origins = generate_origins(idx, start=idx[48 * 22])[:2]
    assert len(origins) == 2

    # 3. Backtest the baseline and a (tiny) global model on identical origins.
    res_naive = run_backtest(
        SeasonalNaive(), bundle, origins, horizon=48, model_name="seasonal_naive"
    )
    res_lgbm = run_backtest(
        GlobalLGBM(n_estimators=30), bundle, origins, horizon=48, model_name="global_lgbm"
    )

    # 4. Metrics are well-formed.
    for res in (res_naive, res_lgbm):
        m = res.metrics()
        assert m["pinball"] >= 0
        assert 0.0 <= m["coverage_80"] <= 1.0
        assert res.actuals.shape == (2, 48)

    # 5. Ablation table builds with skill scores; baseline skill is zero.
    table = build_ablation_table(
        {"seasonal_naive": res_naive, "global_lgbm": res_lgbm},
        baseline="seasonal_naive",
    )
    assert table.loc["seasonal_naive", "skill_pinball"] == 0.0
    assert "global_lgbm" in table.index
