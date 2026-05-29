"""Run the full model ladder on the real panel and write the ablation table.

Local CPU alternative to the Colab notebook. Fits each global model once
(refit=False) and backtests over a fixed set of rolling origins.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.rolling_origin import (  # noqa: E402
    build_ablation_table,
    generate_origins,
    run_backtest,
)
from src.build.fixtures import load_panel_from_parquet  # noqa: E402
from src.models.baselines import SeasonalNaive  # noqa: E402
from src.models.global_dl import GlobalTFT, GlobalTiDE  # noqa: E402
from src.models.global_ml import GlobalLGBM  # noqa: E402

PANEL = "data/processed/real_panel.parquet"
N_ORIGINS = 7
TIDE_EPOCHS = 20
TFT_EPOCHS = 8


def main() -> None:
    bundle = load_panel_from_parquet(PANEL)
    idx = bundle.target.time_index
    origins = generate_origins(idx, start=idx[-(N_ORIGINS + 1) * 48])
    print(f"panel len {len(bundle.target)} | {len(origins)} origins", flush=True)

    results = {}
    print("seasonal_naive ...", flush=True)
    results["seasonal_naive"] = run_backtest(
        SeasonalNaive(), bundle, origins, horizon=48, model_name="seasonal_naive"
    )
    print("global_lgbm ...", flush=True)
    results["global_lgbm"] = run_backtest(
        GlobalLGBM(), bundle, origins, horizon=48, model_name="global_lgbm"
    )
    print("global_tide ...", flush=True)
    results["global_tide"] = run_backtest(
        GlobalTiDE(n_epochs=TIDE_EPOCHS), bundle, origins, horizon=48, model_name="global_tide"
    )
    print("global_tft ...", flush=True)
    results["global_tft"] = run_backtest(
        GlobalTFT(n_epochs=TFT_EPOCHS), bundle, origins, horizon=48, model_name="global_tft"
    )

    table = build_ablation_table(results, baseline="seasonal_naive").round(3)
    out = Path("reports/ablation_real.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(), flush=True)
    print(f"\nsaved {out}", flush=True)
    print("any NaN:", bool(table.drop(columns=[]).isna().to_numpy().any()), flush=True)


if __name__ == "__main__":
    main()
