# UK Energy Price Forecasting

> Probabilistic forecasting of the GB system (imbalance) price using a single global model across a panel of energy series, evaluated by rolling-origin backtesting.

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

Real 2024 GB system-price data, rolling-origin backtest, 48-settlement-period horizon. Every model is scored on identical origins. Values are GBP/MWh; lower is better except coverage.

| Model | Pinball | Skill vs naive | CRPS | MAE | 80% coverage |
|---|---|---|---|---|---|
| Seasonal-naive | 16.94 | n/a | 33.88 | 33.88 | 0.01 |
| Global LightGBM | 10.11 | +40% | 20.23 | 33.08 | 0.63 |
| Global TiDE | 11.01 | +35% | 22.01 | 33.97 | 0.56 |
| **Global TFT** | **7.98** | **+53%** | **15.97** | **25.35** | **0.76** |

The Temporal Fusion Transformer is best on every metric, including calibration (empirical coverage 0.76 against the 0.80 nominal target). Seasonal-naive is a point forecast with zero interval width, so its probabilistic coverage is near zero by construction.

Numbers come from the real 2024 GB imbalance price (built with `scripts/build_real_panel.py`, no API key; reproduce with `scripts/run_real_ablation.py`). Chronos/TimesFM zero-shot is an optional extra row from the Colab notebook. The committed synthetic fixture panel runs the full pipeline with no data download.

---

## The Business Problem

The GB electricity system (imbalance) price settles the difference between what generators and suppliers contracted and what they actually delivered in each half-hour. It is volatile: it follows demand, the generation mix (especially intermittent wind and solar) and the system's real-time balancing position, and it routinely goes negative when renewables are abundant or spikes when the system is short.

A single point forecast is not enough for the people exposed to imbalance, such as battery and flexibility operators scheduling charge and discharge, suppliers and generators managing their balancing position, and trading desks. The cost of being wrong sits in the tails. This system forecasts the full next-day half-hourly price curve (48 settlement periods) with calibrated uncertainty intervals, so both the shape and the risk are explicit.

The system price is the currently published GB price series. The GB day-ahead auction price is no longer available from ENTSO-E after Brexit (see [`data/README.md`](data/README.md)).

---

## What This Demonstrates

| Skill Area | Where to look |
|---|---|
| Global / multi-series forecasting (one model across many series) | `src/models/global_ml.py`, `src/models/global_dl.py` |
| Probabilistic forecasting (quantiles, pinball, CRPS, coverage) | `src/metrics/`, `src/models/` |
| Deep forecasting (TFT, TiDE) and zero-shot foundation models | `src/models/global_dl.py`, `src/models/foundation.py` |
| Rolling-origin backtesting | `src/backtest/rolling_origin.py` |
| Leakage-safe covariate engineering | `src/build/leakage.py`, `src/build/panel.py` |
| Data engineering (Elexon, ENTSO-E ingestion) | `src/data/` |
| Software engineering (typed library, tests, CI) | `src/`, `tests/` |
| Deployment / serving | `app/streamlit_app.py` |

---

## Quick Start

```bash
git clone https://github.com/elyokerr/Projects.git
cd Projects/uk-energy-price-forecasting

python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# Run the test suite (uses the committed fixture panel, no API keys needed)
pytest -q

# Explore the notebooks
jupyter notebook notebooks/01_eda.ipynb
```

Interactive app (runs on the fixture panel with no secrets):

```bash
streamlit run app/streamlit_app.py
```

Build a real GB panel (no API key needed; Elexon is open):

```bash
python scripts/build_real_panel.py --start 2023-01-01 --end 2024-12-31
```

See [`data/README.md`](data/README.md) for sources.

---

## Project Structure

```
uk-energy-price-forecasting/
├── README.md                 (you are here)
├── requirements.txt
│
├── notebooks/                01 EDA, 02 covariates, 03 Colab training, 04 backtest ablation, 05 probabilistic eval
├── src/
│   ├── data/                 Elexon + ENTSO-E clients, calendar covariates
│   ├── build/                48-SP grid alignment, panel construction, leakage guard
│   ├── models/               baselines, global LightGBM, global TFT/TiDE, foundation
│   ├── backtest/             rolling-origin harness + ablation table
│   └── metrics/              pinball, coverage, CRPS, point metrics
│
├── app/                      Streamlit app (fan chart + backtest explorer)
├── data/                     raw/interim/processed (gitignored); fixture panel under tests/fixtures
├── models/                   serialized trained artifacts (gitignored)
├── reports/                  ablation tables and figures
├── tests/                    pytest suite including leakage guard and RUN_SLOW e2e smoke
└── docs/                     design doc + methodology
```

---

## Methodology

1. **Data ingestion.** Half-hourly generation-by-fuel, demand out-turn, and the GB system (imbalance) price from the Elexon BMRS API (no key required). Raw responses are cached to parquet.
2. **Panel construction.** All series are aligned to the 48-settlement-period grid, with clock-change days handled, and assembled into a Darts panel with typed past covariates (out-turn demand and generation) and future covariates (deterministic calendar features). A leakage guard enforces that no out-turn data enters the future-covariate set.
3. **Model ladder.** Seasonal-naive, then statistical (Theta/AutoARIMA), then global LightGBM (quantile), then global deep models (TFT/TiDE, quantile likelihood), then a zero-shot foundation model (Chronos/TimesFM) as a benchmark.
4. **Evaluation.** Rolling-origin backtesting: the forecast origin advances day by day across a held-out window, and every model is scored on identical origins. Metrics are pinball loss, interval coverage, CRPS, MAE/RMSE/sMAPE and skill-vs-naive.
5. **Deployment.** A Streamlit app serves the next-day forecast fan chart and a backtest explorer. Inference is decoupled from training; models are trained in Colab and loaded by the app.

Full detail in [`docs/methodology.md`](docs/methodology.md).

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| Darts | Forecasting framework: global models, probabilistic forecasts, backtesting |
| LightGBM | Global gradient-boosted quantile model |
| PyTorch + TFT/TiDE | Global deep forecasting models |
| Chronos / TimesFM | Zero-shot time-series foundation baselines (Colab) |
| pandas / NumPy / pyarrow | Data handling |
| Streamlit + Plotly | Interactive app |
| pytest + ruff | Tests and linting |
| Docker + GitHub Actions | Reproducibility and CI |
| Google Colab (T4) | Deep-model training |

---
