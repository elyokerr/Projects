# Data Science & Machine Learning Portfolio

Data-science and machine-learning projects, each taken from raw data through to a running, reviewable deliverable. Every project follows the same internal structure (see [`CONTRIBUTING.md`](CONTRIBUTING.md)), so each one is easy to navigate, reproduce, and review.

---

## Projects

### [`saas-churn-prediction/`](saas-churn-prediction/) - SaaS Customer Churn Prediction Platform

Predicts which SaaS customers are about to cancel, quantifies the revenue at stake, and serves the results through a REST API and a Streamlit dashboard. The whole stack comes up with a single `docker compose up`.

| Highlight | Value |
|---|---|
| Annual revenue at risk identified | **£64,000+** |
| Recall on churners | **~82%** |
| Engineered features | **45** (via SQL + Python) |
| Interfaces | Streamlit dashboard, FastAPI service, MLflow tracking |

**Stack:** Python · pandas · scikit-learn · XGBoost · MLflow · SHAP · FastAPI · Streamlit · PostgreSQL · Docker Compose · GitHub Actions CI

--> [Open project](saas-churn-prediction/)

---

### [`filings-rag/`](filings-rag/) - Filings-RAG: Question-Answering over UK FTSE 100 Annual Reports

A retrieval-augmented question-answering system over the annual reports of UK FTSE 100 companies. Hybrid retrieval (BM25 + BGE dense vectors), cross-encoder re-ranking, **forced citations**, and a Ragas evaluation pipeline tracked in MLflow. Streamlit chat UI with click-through-to-source citations.

| Highlight | Value |
|---|---|
| Corpus | **23 FTSE 100 companies · 34 annual reports · 10,722 pages** |
| Chunks indexed | **19,399** (hybrid BM25 + BGE dense, RRF fusion, cross-encoder re-rank) |
| Citation discipline | every `[TICKER\|YEAR\|p.PAGE]` is post-validated; chain regenerates once if any citation is unverifiable |
| Compute | runs locally on CPU via fastembed (ONNX) - no GPU required |
| Eval | 40-question hand-labelled test set (incl. 10 adversarial refusal cases); Ragas metrics + MLflow tracking |

**Stack:** LangChain (LCEL) · ChromaDB · `bm25s` · `bge-small-en-v1.5` (fastembed/ONNX) · `bge-reranker-v2-m3` (CrossEncoder) · Groq Llama 3.3 70B + Gemini 2.0 Flash fallback · PyMuPDF · Ragas · MLflow · Streamlit · Docker · GitHub Actions CI

--> [Open project](filings-rag/)

---

### [`uk-housing-mds/`](uk-housing-mds/) - UK Housing Market Modern Data Stack

A scheduled batch ELT pipeline over UK Land Registry Price Paid Data, ONS NSPL, and UK HPI. Dual-warehouse (DuckDB local / BigQuery production) with profile-switched dbt models, Prefect orchestration on GitHub Actions cron, Great Expectations landing-zone validation, and an Evidence.dev analytics site auto-published to GitHub Pages.

| Highlight | Value |
|---|---|
| dbt project | **11 models** (5-layer) · **38 tests** passing |
| Data quality | **3 GE landing suites** + 3 singular dbt tests + 1 checkpoint |
| Orchestration | **Prefect** + **GitHub Actions** monthly cron |
| Warehouse | **DuckDB** dev/CI · **BigQuery free tier** prod (quota-aware fallback) |
| BI | **Evidence.dev** static site → GitHub Pages (4 routes) |
| Verified | **End-to-end fixture smoke** passes in ~4m30s |

**Stack:** Python · Prefect · dbt-core (`dbt-duckdb` + `dbt-bigquery`) · DuckDB · BigQuery · Great Expectations · Evidence.dev · GitHub Actions · `ruff` · `sqlfluff`

--> [Open project](uk-housing-mds/)

---

### [`ab-experiment-platform/`](ab-experiment-platform/) - A/B Experiment Platform

A library-first online-experimentation toolkit covering the full A/B test lifecycle: power/sample-size design, sample-ratio-mismatch and A/A health checks, frequentist (z-test, Welch, Mann-Whitney) and Bayesian (Beta-Binomial) analysis, CUPED variance reduction, **always-valid sequential testing (mSPRT)**, and a ship/no-ship decision engine. A built-in ground-truth simulator lets the test suite prove the statistics are correct. Streamlit app + 6 notebooks.

| Highlight | Value |
|---|---|
| Sequential monitoring false-positive rate | **~31% (naive peeking) → ~4% (mSPRT)** at nominal 5% |
| CUPED variance reduction | **~10%** with a correlated pre-period covariate |
| Estimator CI coverage (500 simulated experiments) | **~96%** for nominal 95% |
| Power calibration | **80.0% predicted vs ~79% empirical** at the designed n |
| Tests | **34 unit tests** + simulation-based correctness validation + e2e smoke |

**Stack:** Python 3.11 · NumPy · SciPy · statsmodels · pandas · Streamlit · Plotly · matplotlib · pytest · ruff · Docker · GitHub Actions · Streamlit Community Cloud

--> [Open project](ab-experiment-platform/)

---

### [`uk-energy-price-forecasting/`](uk-energy-price-forecasting/) - UK Energy Price Forecasting

Probabilistic forecasting of the GB system (imbalance) price. A single **global** model trained across a panel of energy series (price, national demand, generation-by-fuel) produces calibrated quantile forecasts for all 48 half-hourly settlement periods, benchmarked against a full model ladder under rolling-origin backtesting with strict past/future covariate separation (a leakage guard enforces it).

| Highlight | Value |
|---|---|
| Best model (TFT) pinball loss vs seasonal-naive | **7.98 vs 16.94 (+53% skill)** ¹ |
| Best model (TFT) MAE vs seasonal-naive | **25.35 vs 33.88** ¹ |
| Best model (TFT) 80% interval coverage | **0.76** (nominal 0.80) |
| Model ladder | seasonal-naive → AutoARIMA → global LightGBM → global TFT/TiDE → zero-shot Chronos/TimesFM |
| Tests | **17 test files** incl. a leakage-guard regression test + `RUN_SLOW` end-to-end smoke |

¹ Real 2024 GB imbalance-price data, GBP/MWh, every model scored on identical rolling origins. The Elexon pull needs no API key (`scripts/build_real_panel.py`); the committed synthetic fixture panel reproduces the pipeline with no data download.

**Stack:** Python 3.11 · Darts · LightGBM · PyTorch (TFT/TiDE) · Chronos/TimesFM · pandas · pyarrow · Streamlit · Plotly · pytest · ruff · Docker · GitHub Actions · Google Colab (T4)

--> [Open project](uk-energy-price-forecasting/)

---

### [`career-gap-agent/`](career-gap-agent/) - AI Career Gap Analyst

A tool-using AI agent that turns live UK job postings into a personalised, evidence-backed skills-gap plan. A single **LangGraph** agent searches Adzuna postings, extracts required skills with an LLM, normalises them to the **ESCO** skills taxonomy by embedding match, and returns a ranked "skills to learn" report where every gap cites how many postings demanded it. Served through a mobile-responsive FastAPI web app and traced with Langfuse.

| Highlight | Value |
|---|---|
| Architecture | Single **LangGraph** tool-calling agent with a deterministic groundedness check and a hard iteration cap |
| Skill matching | LLM extraction normalised to the **ESCO** taxonomy by embedding similarity (skill entity-linking) |
| Evaluation | Component eval (precision / recall / F1) over a hand-labelled gold set + agent groundedness checks |
| Observability | **Langfuse** tracing of every agent step, tool call, latency, and token cost |
| Reproducibility | Runs with **zero secrets** on a committed ESCO index + Adzuna snapshot (36 tests, lint clean) |

**Stack:** Python 3.11 · LangGraph · LangChain (Groq Llama 3.3 70B + Gemini fallback) · fastembed/BGE · ESCO · PyMuPDF · Langfuse · FastAPI · HTMX · Tailwind · Docker · GitHub Actions · Hugging Face Spaces

--> [Open project](career-gap-agent/)

---

## How this repo is organised

```
Projects/
├── README.md              ← You are here
├── CONTRIBUTING.md        ← The standard every project follows
├── LICENSE
├── .gitignore             ← Repo-wide ignores (data/, mlruns/, etc.)
│
├── _template/             ← Skeleton for new projects
│
├── saas-churn-prediction/
├── filings-rag/
├── uk-housing-mds/
├── ab-experiment-platform/
├── uk-energy-price-forecasting/
└── career-gap-agent/
```

Every project folder uses the same internal layout:

```
<project>/
├── README.md          ← 9-section overview (hero results → limitations)
├── requirements.txt
├── notebooks/         ← Numbered EDA & modelling notebooks
├── src/               ← Reusable Python modules (data, features, models, utils)
├── data/              ← raw/, interim/, processed/, external/ (data gitignored)
├── models/            ← Serialised trained artifacts (.joblib, .pkl)
├── reports/figures/   ← Generated plots & screenshots
├── app/               ← Optional Streamlit / FastAPI / dashboard
├── tests/             ← Pytest tests
└── docs/              ← Extended documentation (optional)
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full convention and the README template used by every project.

---

## About

These projects are part of an ongoing portfolio focused on practical data science: not just model training, but the surrounding work that makes a model usable, such as clean data pipelines, honest validation (group splits where it matters), explainability (SHAP), business-framed metrics, and interactive deliverables a stakeholder can actually open.

New projects are added as they're built. Each one is self-contained, documented end to end, and follows the conventions above.

---

## Licence

This repository is shared for portfolio and educational purposes. Individual projects may use third-party datasets under their respective terms - see each project's README for data attribution.
