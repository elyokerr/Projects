# UK Energy Price Forecasting

> Probabilistic forecasting of the GB system (imbalance) price — a single global model across a panel of energy series, evaluated by rolling-origin backtesting.

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

Real 2024 GB system-price data, rolling-origin backtest, 48-settlement-period horizon. Every model scored on identical origins (GBP/MWh; lower is better except coverage):

| Model | Pinball | Skill vs naive | CRPS | MAE | 80% coverage |
|---|---|---|---|---|---|
| Seasonal-naive | 16.94 | — | 33.88 | 33.88 | 0.01¹ |
| Global LightGBM | 10.11 | +40% | 20.23 | 33.08 | 0.63 |
| Global TiDE | 11.01 | +35% | 22.01 | 33.97 | 0.56 |
| **Global TFT** | **7.98** | **+53%** | **15.97** | **25.35** | **0.76** |

The Temporal Fusion Transformer wins on every metric — pinball, CRPS, MAE, and calibration (empirical coverage 0.76 against the 0.80 nominal target). The ordering (naive → boosting/TiDE → TFT) is exactly what the model ladder is built to surface.

¹ Seasonal-naive is a point forecast (zero interval width), so its probabilistic coverage is ≈0 by construction — a deliberate teaching contrast.

> Numbers are GBP/MWh on the **real 2024 GB imbalance price** (built with `scripts/build_real_panel.py`, no API key; reproduce with `scripts/run_real_ablation.py`). The zero-shot foundation baseline (Chronos/TimesFM) is an optional extra row produced by the Colab notebook. The committed synthetic fixture panel reproduces the full pipeline with no data download.

---

## The Business Problem

The GB electricity **system (imbalance) price** settles the difference between what generators and suppliers contracted and what they actually delivered each half-hour. It is highly volatile — driven by demand, the generation mix (especially intermittent wind and solar), and the system's real-time balancing position — and routinely goes negative when renewables are abundant or spikes sharply when the system is short.

For anyone exposed to imbalance — battery and flexibility operators deciding when to charge/discharge, suppliers and generators managing their balancing position, trading desks — a single point forecast is not enough. The cost of being wrong is concentrated in the tails. This system forecasts the full next-day half-hourly price curve (48 settlement periods) **with calibrated uncertainty intervals**, so the shape and the risk are both explicit.

*(The system price is the currently-published GB price series. The GB day-ahead auction price is no longer available from ENTSO-E post-Brexit — see [`data/README.md`](data/README.md).)*

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

Build a real GB panel (no API key needed — Elexon is open):

```bash
python scripts/build_real_panel.py --start 2023-01-01 --end 2024-12-31
```

See [`data/README.md`](data/README.md) for sources.

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

1. **Data ingestion** — half-hourly generation-by-fuel, demand out-turn, and the GB system (imbalance) price from the Elexon BMRS API (no key required). Raw responses cached to parquet.
2. **Panel construction** — all series aligned to the 48-settlement-period grid (clock-change days handled), assembled into a Darts panel with strictly typed past covariates (out-turn demand/generation) and future covariates (deterministic calendar features). A leakage guard enforces that no out-turn data ever enters the future-covariate set.
3. **Model ladder** — seasonal-naive → statistical (Theta/AutoARIMA) → global LightGBM (quantile) → global deep models (TFT/TiDE, quantile likelihood) → zero-shot foundation model (Chronos/TimesFM) as a benchmark.
4. **Evaluation** — rolling-origin backtesting: the forecast origin advances day-by-day across a held-out window; every model is scored on identical origins. Metrics: pinball loss, interval coverage, CRPS, plus MAE/RMSE/sMAPE and skill-vs-naive.
5. **Deployment** — a Streamlit app serves the next-day forecast fan chart and a backtest explorer; inference is decoupled from training (models are trained in Colab and loaded by the app).

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

- **Synthetic fixture for the committed hero numbers** — the table above is computed on a synthetic panel so the repo runs without secrets. Real GB-market results come from the open Elexon pull (`scripts/build_real_panel.py`) plus the Colab training run.
- **Target is the imbalance price, not the day-ahead auction price** — the GB day-ahead price is no longer published by ENTSO-E post-Brexit (data ends 2020). The system/imbalance price is the current, free, GB-specific alternative; it is more volatile, so absolute errors look larger than a day-ahead series would.
- **Single imbalance price assumption** — uses `systemSellPrice` (= `systemBuyPrice` under the GB single-price regime).
- **Next: regional series** — extending the panel to grid-supply-point regions enlarges the global-model panel with no architectural change.
- **Next: conformal intervals** — distribution-free coverage guarantees on top of the quantile forecasts.
- **Next: scheduled retraining** — moving the Colab training into an orchestrated incremental-refit flow.
