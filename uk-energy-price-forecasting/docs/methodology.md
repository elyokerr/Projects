# Methodology

## Forecasting task

Forecast the GB system (imbalance) price for all 48 settlement periods of the next delivery day, a 24-hour half-hourly horizon. Forecasts are probabilistic: quantiles and prediction intervals, not a single point.

## Global, multi-series modelling

Instead of fitting one model per series, a single global model is trained jointly across a panel of related GB energy series: price, national demand, and generation by fuel type. The model shares parameters across series, and series identity and static metadata distinguish them. The evaluated target is price; the wider panel supplies training signal and covariates.

## Covariate typing and the leakage discipline

Covariates are split by what is known at the forecast origin:

- **Past covariates:** out-turn demand and generation by fuel. Known only up to the origin, and used with look-back (negative) lags only.
- **Future covariates:** deterministic calendar features (day-of-week, holiday flag, settlement-period index, Fourier seasonal terms) and any published wind/solar forecast. Known for the delivery day at forecast time.

A leakage guard (`src/build/leakage.py`) enforces, at panel-construction time, that no out-turn series (anything named `gen_*`, `demand_*`, or `price`) appears in the future-covariate set. Only an explicit allowlist of calendar features is permitted, and the invariant is covered by a regression test.

Because past covariates are not known over the horizon, the LightGBM model uses a direct multi-horizon configuration (`output_chunk_length = 48`) so that all 48 steps are produced from information available at the origin, without needing future values of the out-turn series.

## Settlement-period grid

All series are aligned to a continuous half-hourly UTC grid. Short gaps (2 periods or fewer) are forward-filled; longer gaps are left missing and logged. Clock-change days are detected by counting settlement periods per Europe/London local day: 46 on the spring transition, 50 on the autumn transition, 48 otherwise.

## Scaling and missing data for deep models

Tree models (LightGBM) are scale-invariant and tolerate missing values, but the neural models (TiDE, TFT) are not. Two steps make them work on real data:

- Inputs (target and covariates) are scaled with a min-max scaler fitted on the training window only, so no future information leaks into the transform.
- Missing values are filled before training. A `gen_*` column that is missing means that source was not running or not yet commissioned (for example a new interconnector), so its generation was zero; price and demand gaps are interpolated.

## Model ladder

| Model | Role |
|---|---|
| Seasonal-naive (lag-48) | Baseline and skill-score denominator |
| Theta / AutoARIMA | Classical statistical reference (per-series, local) |
| Global LightGBM (quantile) | Gradient-boosted global model with quantile loss |
| Global TFT / TiDE (quantile likelihood) | Deep global models |
| Chronos / TimesFM (zero-shot) | Pretrained foundation-model benchmark |

## Evaluation: rolling-origin backtesting

The forecast origin is fixed at successive midnight day boundaries. At each origin the model forecasts the next 48 settlement periods, and the origin then advances across the held-out window. Every model is scored on the same set of origins, which gives a like-for-like comparison and avoids single-split luck.

### Metrics

- **Pinball (quantile) loss:** averaged across forecast quantiles, the headline metric. Minimised by the true quantile.
- **Interval coverage:** the empirical fraction of out-turns inside the predicted interval, compared against the nominal level (an 80% interval from the 0.1 and 0.9 quantiles).
- **CRPS:** approximated as twice the mean pinball over the quantile grid, a combined sharpness-and-calibration summary.
- **MAE / RMSE / sMAPE:** point metrics on the predictive median.
- **Skill vs naive:** `1 - model_error / baseline_error`; positive means the model beats seasonal-naive.

### Calibration

A reliability diagram compares nominal to empirical coverage, so a model that scores well on point metrics but is miscalibrated on intervals is identified rather than hidden. Seasonal-naive, with zero interval width, is the extreme case (coverage near zero).

## Reproducibility

A committed synthetic fixture panel (`tests/fixtures/fixture_panel.parquet`, generated with a fixed seed) backs the test suite, the end-to-end smoke test, and the Streamlit demo, so the repository runs end to end with no API keys. The real headline numbers come from `scripts/build_real_panel.py` (the open Elexon pull) and `scripts/run_real_ablation.py`. Deep-model training also runs in Colab against a saved panel; the application and backtests load artifacts and never train.
