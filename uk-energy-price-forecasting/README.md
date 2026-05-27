# UK Energy Price Forecasting

> Probabilistic day-ahead forecasting of GB wholesale electricity prices — a single global model across a panel of energy series, evaluated by rolling-origin backtesting.

---

## Table of Contents

1. [Hero Results](#hero-results)
2. [The Business Problem](#the-business-problem)
3. [What This Demonstrates](#what-this-demonstrates)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Methodology](#methodology)
7. [Tech Stack](#tech-stack)
8. [Limitations & Next Steps](#limitations--next-steps)

---

## Hero Results

Rolling-origin backtest, 48-settlement-period horizon. Global LightGBM vs the seasonal-naive baseline:

| Metric | Seasonal-naive | Global LightGBM |
|---|---|---|
| Pinball loss (lower better) | 2.90 | **1.44** |
| Skill vs naive (pinball) | — | **+50%** |
| MAE (median forecast) | 5.81 | **3.95** |
| 80% interval coverage | 0.00¹ | **0.73** |

¹ Seasonal-naive is a point forecast (zero interval width), so its probabilistic coverage is zero by construction — a deliberate teaching contrast.

> **These figures are computed on the committed synthetic fixture panel** so the repo is fully reproducible with no API keys. Headline results on real GB market data are produced by the Colab training run (`notebooks/03_colab_train_global.ipynb`) plus the live-data backtest; the deep models (TFT/TiDE) and the zero-shot foundation baseline (Chronos/TimesFM) join the ablation table there.

---

## The Business Problem

GB wholesale electricity prices are set in a day-ahead auction and are highly volatile — driven by demand, the generation mix (especially intermittent wind and solar), and fuel costs. When low renewable output coincides with high demand, prices spike; when renewables are abundant, prices can fall toward or below zero.

For anyone acting on tomorrow's prices — energy traders placing day-ahead bids, battery and flexibility operators scheduling charge/discharge, grid risk teams sizing exposure — a single point forecast is not enough. The cost of being wrong is concentrated in the tails. This system forecasts the full next-day half-hourly price curve **with calibrated uncertainty intervals**, so the shape and the risk are both explicit.

---

## What This Demonstrates

| Skill Area | Where to look |
|---|---|
| Global / multi-series forecasting (one model across many series) | `src/models/global_ml.py`, `src/models/global_dl.py` |
| Probabilistic forecasting (quantiles, pinball, CRPS, coverage) | `src/metrics/`, `src/models/` |
| Deep forecasting (TFT, TiDE) + zero-shot foundation models | `src/models/global_dl.py`, `src/models/foundation.py` |
| Rigorous evaluation (rolling-origin backtesting) | `src/backtest/rolling_origin.py` |
| Leakage-safe covariate engineering | `src/build/leakage.py`, `src/build/panel.py` |
| Data engineering (Elexon + ENTSO-E ingestion) | `src/data/` |
| Software engineering (typed library, tests, CI) | `src/`, `tests/` |
| Deployment / serving | `app/streamlit_app.py` |

---

## Quick Start

```bash
git clone https://github.com/elyokerr/Projects.git
cd Projects/uk-energy-price-forecasting

python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# Run the test suite (uses the committed fixture panel — no API keys needed)
pytest -q

# Explore the notebooks
jupyter notebook notebooks/01_eda.ipynb
```

Interactive app (runs on the fixture panel with zero secrets):

```bash
streamlit run app/streamlit_app.py
```

Using live data requires a free ENTSO-E token — see [`data/README.md`](data/README.md).

---

## Project Structure

```
uk-energy-price-forecasting/
├── README.md                 ← You are here
├── requirements.txt
│
├── notebooks/                ← 01 EDA · 02 covariates · 03 Colab training · 04 backtest ablation · 05 probabilistic eval
├── src/
│   ├── data/                 ← Elexon + ENTSO-E clients, calendar covariates
│   ├── build/                ← 48-SP grid alignment, panel construction, leakage guard
│   ├── models/               ← baselines · global LightGBM · global TFT/TiDE · foundation
│   ├── backtest/             ← rolling-origin harness + ablation table
│   └── metrics/              ← pinball · coverage · CRPS · point metrics
│
├── app/                      ← Streamlit app (fan chart + backtest explorer)
├── data/                     ← raw/interim/processed (gitignored); fixture panel under tests/fixtures
├── models/                   ← serialized trained artifacts (gitignored)
├── reports/figures/          ← generated plots
├── tests/                    ← pytest suite incl. leakage guard + RUN_SLOW e2e smoke
└── docs/                     ← design doc + methodology
```

---

## Methodology

1. **Data ingestion** — half-hourly generation-by-fuel, demand, and wind/solar forecast from the Elexon BMRS Insights API; GB day-ahead price from the ENTSO-E Transparency Platform. Raw responses cached to parquet.
2. **Panel construction** — all series aligned to the 48-settlement-period grid (clock-change days handled), assembled into a Darts panel with strictly typed past covariates (out-turn demand/generation) and future covariates (deterministic calendar features). A leakage guard enforces that no out-turn data ever enters the future-covariate set.
3. **Model ladder** — seasonal-naive → statistical (Theta/AutoARIMA) → global LightGBM (quantile) → global deep models (TFT/TiDE, quantile likelihood) → zero-shot foundation model (Chronos/TimesFM) as a benchmark.
4. **Evaluation** — rolling-origin backtesting: the forecast origin advances day-by-day across a held-out window; every model is scored on identical origins. Metrics: pinball loss, interval coverage, CRPS, plus MAE/RMSE/sMAPE and skill-vs-naive.
5. **Deployment** — a Streamlit app serves the day-ahead fan chart and a backtest explorer; inference is decoupled from training (models are trained in Colab and loaded by the app).

Full detail in [`docs/methodology.md`](docs/methodology.md).

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| Darts | Unified forecasting framework; global models + probabilistic forecasts + backtesting |
| LightGBM | Global gradient-boosted quantile model |
| PyTorch + TFT/TiDE | Global deep forecasting models |
| Chronos / TimesFM | Zero-shot time-series foundation baselines (Colab) |
| pandas / NumPy / pyarrow | Data handling |
| Streamlit + Plotly | Interactive app |
| pytest + ruff | Tests & linting |
| Docker + GitHub Actions | Reproducibility & CI |
| Google Colab (T4) | Heavy training |

---

## Limitations & Next Steps

- **Synthetic fixture for reproducibility** — the committed results use a synthetic panel so the repo runs without secrets. Real GB-market results require the live-data pull (ENTSO-E token) and the Colab training run.
- **Live API field names pending confirmation** — the Elexon field names and ENTSO-E XML namespace are coded against published docs and flagged for confirmation against a live response (see `data/README.md`).
- **Prices in published currency** — ENTSO-E publishes GB day-ahead prices in EUR/MWh; no FX conversion is invented. A clean free GBP series (or a documented FX source) is a future addition.
- **Next: regional series** — extending the panel to grid-supply-point regions enlarges the global-model panel with no architectural change.
- **Next: conformal intervals** — distribution-free coverage guarantees on top of the quantile forecasts.
- **Next: scheduled retraining** — moving the Colab training into an orchestrated incremental-refit flow.
