# UK Energy Price Forecasting — Design Document

*Date: 2026-05-26*

## 1. Project overview

A probabilistic, day-ahead forecasting system for GB wholesale electricity prices. The system forecasts the half-hourly day-ahead price for all 48 settlement periods of the next delivery day and reports calibrated uncertainty intervals alongside the central estimate.

Forecasts are produced by a single **global** model trained jointly across a panel of related GB energy series — day-ahead price, national demand, and generation by fuel type — rather than by fitting an independent model per series. The headline evaluated target is day-ahead price; the wider panel supplies both training signal and covariates.

The project is built end-to-end: data ingestion from public APIs, a reproducible feature/panel build, a ladder of forecasting models from naive baselines through global deep-learning models, a rolling-origin backtesting harness, probabilistic evaluation, and an interactive Streamlit application.

## 2. Problem statement

GB wholesale electricity prices are set in a day-ahead auction and are highly volatile. Price is driven by national demand, the generation mix (in particular intermittent wind and solar), interconnector flows, and fuel costs. Periods of low renewable output coincident with high demand produce sharp price spikes; periods of high renewable output can drive prices toward or below zero.

For decisions that depend on the next day's prices, a single point forecast is insufficient because the cost of being wrong is asymmetric and concentrated in the tails. The system therefore produces **probabilistic** forecasts — quantiles and prediction intervals — so that the shape and uncertainty of the next-day price curve are explicit.

The system must defend a concrete claim: a global deep-learning model with appropriate covariates and a probabilistic output forecasts day-ahead price more accurately than (a) a seasonal-naive baseline and (b) a zero-shot time-series foundation model, measured by pinball loss and interval coverage under rolling-origin backtesting.

## 3. Users

- **Energy trader / analyst** — needs the next-day half-hourly price curve with uncertainty intervals to inform day-ahead bids and hedging.
- **Battery / flexibility operator** — needs the expected price *shape* across the 48 settlement periods to schedule charging and discharging against intraday spreads.
- **Grid / risk analyst** — needs calibrated intervals to quantify exposure to price spikes.

The interactive application targets the first two directly: a forecast view for the next-day curve and a backtest view for inspecting historical model behaviour.

## 4. Dataset

All sources are free and reproducible.

| Source | Data | Granularity |
|---|---|---|
| Elexon BMRS Insights API | Generation by fuel type (`FUELHH`); national demand out-turn and initial estimate (`INDO`/`ITSDO`); embedded wind and solar forecast | Half-hourly |
| ENTSO-E Transparency Platform (free API token) | GB day-ahead auction price | Hourly / half-hourly, resampled to settlement periods |
| Derived calendar | UK bank holidays, day-of-week, period-of-day, Fourier seasonal terms | Half-hourly |

- **Coverage:** approximately three to four years of half-hourly history. The most recent months are held out for rolling-origin backtesting.
- **Panel composition:** day-ahead price plus national demand plus generation-by-fuel series (roughly 8–12 series depending on fuel categories published).
- **Settlement-period grid:** all series are aligned to the 48-settlement-period day. Clock-change days (46 or 50 periods) are handled explicitly.
- **Covariate typing:**
  - *Past covariates* — generation out-turn and demand out-turn (known only up to the forecast origin).
  - *Future covariates* — calendar features and published wind/solar forecasts (known for the delivery day at forecast time).
- **No-secrets fixture:** a small committed fixture panel allows the test suite and the application demo to run without any API tokens.

## 5. Tech stack

| Tool | Role | Why |
|---|---|---|
| Python 3.11 | Language | `pyarrow` wheels and the ML stack are reliable on 3.11 across platforms |
| Darts | Forecasting framework | Unified API for statistical, ML, and deep models; native global-model training across many series; built-in probabilistic forecasts and backtesting |
| LightGBM | Global ML model | Strong tabular gradient-boosting baseline; quantile objective for probabilistic output |
| PyTorch (CPU local / GPU on Colab) | Deep-model backend | Backs the TFT / TiDE models in Darts |
| TFT / TiDE (via Darts) | Global deep models | Attention- and MLP-based architectures designed for multi-series forecasting with covariates and quantile likelihoods |
| Chronos / TimesFM | Zero-shot foundation baseline | Benchmarks trained models against a modern pretrained time-series model; runs on a free Colab T4 |
| pandas, pyarrow | Data handling | Half-hourly panel construction and parquet caching |
| Streamlit + Plotly | Application | Fan-chart forecast view and backtest explorer; free hosting on Community Cloud |
| pytest, ruff | Testing and linting | Unit tests, leakage guard, end-to-end smoke; consistent style |
| Docker | Reproducibility | Local parity with the cloud deployment |
| GitHub Actions | CI | Lint and test on every change, path-filtered to this project |
| Google Colab (T4) | Heavy training | Global deep-model and foundation-model training |

## 6. Architecture

The system is library-first: pure, testable functions and classes under `src/`, with notebooks and the application as thin consumers.

```
ingest (API clients) ──► raw parquet cache ──► build (clean / align / resample to 48-SP grid)
        │                                              │
        ▼                                              ▼
   Darts panel TimeSeries  ◄── covariates (past: demand/generation out-turn;
        │                       future: calendar, wind/solar forecast)
        ▼
   model ladder ── seasonal-naive baseline
               ├─ statistical (Theta / AutoARIMA)
               ├─ global LightGBM (Darts RegressionModel, quantile)
               ├─ global deep (TFT / TiDE, quantile likelihood)
               └─ zero-shot foundation (Chronos / TimesFM)  [benchmark only]
        │
        ▼
   rolling-origin backtesting harness ──► metrics (pinball, coverage, CRPS, MAE, RMSE, sMAPE)
        │
        ▼
   Streamlit app (fan-chart forecast view + backtest explorer)
```

Key design choices:

- **Single global model across the panel.** All series share model weights; series identity and static covariates distinguish them. This is the mechanism by which the model forecasts at scale across many related series.
- **Strict past- vs future-covariate separation.** Only information genuinely available at the forecast origin enters as a future covariate. Generation and demand out-turns are past covariates. This prevents target leakage.
- **Probabilistic output throughout.** ML and deep models emit quantiles via quantile objectives / likelihoods; pinball loss is both a training objective and an evaluation metric.
- **Training decoupled from inference.** Models are trained in Colab notebooks and serialized; the application and backtests load artifacts and never train.

## 7. Repository structure

```
uk-energy-price-forecasting/
├── README.md                 # 9-section overview
├── requirements.txt          # Python 3.11; darts, lightgbm, torch, chronos/timesfm, pandas, pyarrow
├── .gitignore
├── src/
│   ├── data/                 # elexon_client.py, entsoe_client.py, calendar.py
│   ├── build/                # alignment / resampling to the 48-SP grid; panel + covariate construction
│   ├── models/               # baselines.py, global_ml.py, global_dl.py, foundation.py
│   ├── backtest/             # rolling_origin.py (the backtesting harness)
│   └── metrics/              # pinball.py, coverage.py, crps.py, point_metrics.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_covariates.ipynb
│   ├── 03_colab_train_global.ipynb
│   ├── 04_backtest_ablation.ipynb
│   └── 05_probabilistic_eval.ipynb
├── app/
│   └── streamlit_app.py      # fan-chart forecast + backtest explorer
├── data/                     # raw/ interim/ processed/ (contents gitignored)
├── models/                   # serialized trained artifacts
├── reports/figures/
├── tests/
└── docs/                     # this design doc, methodology.md, data/README.md
```

## 8. Evaluation methodology

- **Rolling-origin (expanding-window) backtesting.** The forecast origin is fixed at successive day boundaries; at each origin the model forecasts the next 48 settlement periods; the origin then advances across the held-out period (approximately the final six months). Every model is scored on the identical set of origins, giving a like-for-like comparison and avoiding single-split luck.
- **Probabilistic metrics (primary):**
  - **Pinball / quantile loss** averaged across forecast quantiles — the headline metric.
  - **Interval coverage** at the 80% and 90% nominal levels — the empirical proportion of out-turns falling inside the predicted interval.
  - **CRPS** — overall sharpness-and-calibration summary.
- **Point metrics (secondary):** MAE, RMSE, sMAPE on the predictive median, for intuition.
- **Skill scores:** every model is reported as a percentage improvement over the seasonal-naive baseline, making the practical value of each model explicit.
- **Ablation table** (headline artifact): rows are the models in the ladder (seasonal-naive, Theta/AutoARIMA, global LightGBM, global TFT/TiDE, zero-shot Chronos/TimesFM); columns are pinball, coverage@90, CRPS, MAE, and skill-vs-naive.
- **Per-horizon analysis:** error as a function of forecast horizon (settlement period within the day), identifying where each model wins.
- **Calibration check:** a reliability diagram comparing nominal to empirical coverage, so an accurate-but-miscalibrated model is identified rather than hidden.

## 9. Error handling

- **API ingestion:** retry with backoff on Elexon and ENTSO-E rate limits; raw responses cached to parquet so re-runs do not re-hit the APIs; the expected settlement-period count per day is validated (48 normally; 46 or 50 on clock-change days handled explicitly).
- **Missing or irregular data:** gaps are flagged and imputed under a documented rule — short gaps forward-filled, long gaps dropped and logged. Values are never silently zero-filled.
- **Compute limits:** foundation-model and deep-model code guards against out-of-memory and quota exhaustion by falling back to CPU or a reduced context window. Notebooks guard on the absence of a saved panel and skip gracefully so the repository runs without secrets.
- **No-secrets operation:** a committed fixture panel backs the test suite and the application demo, so neither requires API tokens.

## 10. Testing strategy

- `tests/conftest.py` prepends the project root to `sys.path` so tests run from any working directory.
- **Metric unit tests:** pinball, coverage, and CRPS implementations checked against hand-computed values.
- **Alignment / resampling tests:** the 48-settlement-period grid builder, including clock-change days (46 and 50 periods).
- **Covariate builder tests:** calendar feature construction.
- **API client tests:** Elexon and ENTSO-E clients exercised against recorded fixture responses.
- **Leakage regression test:** asserts that future-covariate construction at origin *t* uses no information dated after *t*.
- **Backtest-harness test:** on a synthetic series with a known pattern, asserts the seasonal-naive model recovers the expected error and that the harness produces the correct number of origins and forecasts.
- **End-to-end smoke test** gated behind `RUN_SLOW=1`: ingest fixture → build panel → fit a small global model → backtest two origins → produce a forecast. This surfaces integration issues that unit tests do not.
- **Lint and CI:** ruff-clean codebase; GitHub Actions runs lint and tests on every change, path-filtered to this project.

## 11. Deployment

- **Streamlit Community Cloud** (free) hosts the public demo with two views:
  1. **Day-ahead forecast** — a fan chart of the 48-settlement-period price curve with 80% and 90% intervals, served from a cached trained model and the latest panel.
  2. **Backtest explorer** — selection of a model and a historical origin, with the forecast overlaid on the realised out-turn and the metrics for that window.
- **Inference is decoupled from training.** Trained artifacts produced in Colab are loaded by the application; the application never trains, keeping it within Community Cloud resource limits.
- **Docker** provides local and reproducible runs with parity to the cloud deployment.
- **Secrets are optional.** The committed fixture panel allows the demo to run with no tokens; a live mode reads API tokens from environment variables when present.

## 12. Scaling path

- **More series:** the global model already generalises across the panel, so adding regional (grid-supply-point) series or finer fuel categories is additional series under the same architecture.
- **Higher frequency and lower latency:** move from notebook retraining to a scheduled orchestration flow with incremental refits.
- **Larger models and GPU:** the deep models scale to larger hidden sizes on a dedicated GPU; the foundation model could be fine-tuned rather than used zero-shot.
- **Serving:** if multiple consumers emerge, inference can be wrapped in a service API behind the application.

## 13. Definition of Done

- Ingestion clients pull generation-by-fuel, demand, and wind/solar forecast from Elexon and day-ahead price from ENTSO-E; raw data cached to parquet; a fixture panel committed for no-secrets runs.
- The build step produces an aligned 48-settlement-period Darts panel with correctly typed past and future covariates; clock-change days handled.
- The full model ladder is implemented: seasonal-naive, Theta/AutoARIMA, global LightGBM, global TFT/TiDE, and zero-shot Chronos/TimesFM.
- The rolling-origin backtest harness scores all models on identical origins and produces the ablation table, per-horizon plot, and reliability diagram.
- Probabilistic metrics (pinball, coverage@80/90, CRPS), point metrics, and skill-vs-naive are reported; the deep model's pinball loss is compared against the seasonal-naive baseline, with the result reported honestly whether or not it wins.
- The leakage test, backtest-harness test, metric unit tests, and the `RUN_SLOW=1` end-to-end smoke test all pass; the codebase is ruff-clean; CI is green.
- The Streamlit application is live on Community Cloud with both views; the README follows the nine-section template with hero metrics; `methodology.md` and `data/README.md` are complete.
