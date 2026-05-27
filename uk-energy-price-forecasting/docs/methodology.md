# Methodology

## Forecasting task

Forecast the GB day-ahead wholesale electricity price for all 48 settlement periods of the next delivery day (a 24-hour, half-hourly horizon), producing **probabilistic** forecasts — quantiles and prediction intervals, not a single point.

## Global, multi-series modelling

Rather than fitting one model per series, a single **global** model is trained jointly across a panel of related GB energy series — day-ahead price, national demand, and generation by fuel type. The model shares parameters across series; series identity and static metadata distinguish them. The headline evaluated target is price; the wider panel supplies training signal and covariates.

## Covariate typing and the leakage discipline

Covariates are split by what is known at the forecast origin:

- **Past covariates** — out-turn demand and generation by fuel. Known only up to the origin. Used with negative (look-back) lags only.
- **Future covariates** — deterministic calendar features (day-of-week, holiday flag, settlement-period index, Fourier seasonal terms) and any published wind/solar forecast. Known for the delivery day at forecast time.

A leakage guard (`src/build/leakage.py`) enforces, at panel-construction time, that no out-turn series (anything named `gen_*`, `demand_*`, or `price`) ever appears in the future-covariate set. Only an explicit allowlist of calendar features is permitted. This invariant is covered by a dedicated regression test.

Because past covariates are not known over the horizon, the LightGBM model uses a **direct multi-horizon** configuration (`output_chunk_length = 48`) so that all 48 steps are produced from information available at the origin, without needing future values of the out-turn series.

## Settlement-period grid

All series are aligned to a continuous half-hourly UTC grid. Short gaps (≤ 2 periods) are forward-filled; longer gaps are left missing and logged. Clock-change days are detected by counting settlement periods per **Europe/London local** day — 46 on the spring transition, 50 on the autumn transition, 48 otherwise.

## Model ladder

| Model | Role |
|---|---|
| Seasonal-naive (lag-48) | Baseline; the skill-score denominator |
| Theta / AutoARIMA | Classical statistical reference (per-series, local) |
| Global LightGBM (quantile) | Gradient-boosted global model with quantile loss |
| Global TFT / TiDE (quantile likelihood) | Deep global models |
| Chronos / TimesFM (zero-shot) | Pretrained foundation-model benchmark |

## Evaluation — rolling-origin backtesting

The forecast origin is fixed at successive midnight day boundaries; at each origin the model forecasts the next 48 settlement periods; the origin then advances across the held-out window. Every model is scored on the identical set of origins, giving a like-for-like comparison and avoiding single-split luck.

### Metrics

- **Pinball (quantile) loss** — averaged across forecast quantiles; the headline metric. Minimised by the true quantile.
- **Interval coverage** — empirical fraction of out-turns inside the predicted interval, compared against the nominal level (e.g. an 80% interval from the 0.1 and 0.9 quantiles).
- **CRPS** — approximated as twice the mean pinball over the quantile grid; a combined sharpness-and-calibration summary.
- **MAE / RMSE / sMAPE** — point metrics on the predictive median, for intuition.
- **Skill vs naive** — `1 − model_error / baseline_error`; positive means the model beats seasonal-naive.

### Calibration

A reliability diagram compares nominal to empirical coverage. A model that is accurate on point metrics but miscalibrated on intervals is identified rather than hidden — seasonal-naive, with zero interval width, is the extreme illustrative case (coverage ≈ 0).

## Reproducibility

A committed synthetic fixture panel (`tests/fixtures/fixture_panel.parquet`, generated with a fixed seed) backs the entire test suite, the end-to-end smoke test, and the Streamlit demo, so the repository runs end-to-end with no API keys. Heavy training (deep models, foundation models) runs in Colab against a saved panel; the application and backtests load artifacts and never train.
