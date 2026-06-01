# Data Science & Machine Learning Portfolio

Samuel Demelo - MSc Data Science, University of Bristol.

A portfolio of end-to-end data-science, machine-learning, and data-engineering projects. Each project is now its own repository, taken from raw data through to a running, reviewable deliverable with tests, documentation, and honest evaluation. This page is the index; follow the links to each repo.

---

## Projects

### [saas-churn-prediction](https://github.com/elyokerr/saas-churn-prediction) - SaaS Customer Churn Prediction Platform

Predicts which SaaS customers are about to cancel, quantifies the revenue at stake, and serves the results through a REST API and a Streamlit dashboard. The whole stack comes up with a single `docker compose up`.

| Highlight | Value |
|---|---|
| Annual revenue at risk identified | **£64,000+** |
| Recall on churners | **~82%** |
| Engineered features | **45** (via SQL + Python) |

**Stack:** Python · pandas · scikit-learn · XGBoost · MLflow · SHAP · FastAPI · Streamlit · PostgreSQL · Docker Compose · GitHub Actions CI

---

### [filings-rag](https://github.com/elyokerr/filings-rag) - Question-Answering over UK FTSE 100 Annual Reports

A retrieval-augmented question-answering system over UK FTSE 100 annual reports. Hybrid retrieval (BM25 + BGE dense vectors), cross-encoder re-ranking, forced citations, and a Ragas evaluation pipeline tracked in MLflow. Streamlit chat UI with click-through-to-source citations.

| Highlight | Value |
|---|---|
| Corpus | **23 companies · 34 reports · 10,722 pages · 19,399 chunks** |
| Retrieval | hybrid BM25 + BGE dense, RRF fusion, cross-encoder re-rank |
| Compute | runs locally on CPU via fastembed (ONNX), no GPU required |
| Eval | 40-question hand-labelled test set incl. 10 adversarial refusal cases |

**Stack:** LangChain (LCEL) · ChromaDB · `bm25s` · `bge-small-en-v1.5` · `bge-reranker-v2-m3` · Groq Llama 3.3 70B + Gemini fallback · PyMuPDF · Ragas · MLflow · Streamlit · Docker · GitHub Actions CI

---

### [uk-housing-mds](https://github.com/elyokerr/uk-housing-mds) - UK Housing Market Modern Data Stack

A scheduled batch ELT pipeline over UK Land Registry Price Paid Data, ONS NSPL, and UK HPI. Dual-warehouse (DuckDB local / BigQuery production) with profile-switched dbt models, Prefect orchestration on GitHub Actions cron, Great Expectations validation, and an Evidence.dev analytics site published to GitHub Pages.

| Highlight | Value |
|---|---|
| dbt project | **11 models** (5-layer) · **38 tests** passing |
| Data quality | **3 GE landing suites** + singular dbt tests + checkpoint |
| Orchestration | **Prefect** + **GitHub Actions** monthly cron |
| Warehouse | **DuckDB** dev/CI · **BigQuery free tier** prod (quota-aware fallback) |
| BI | **Evidence.dev** static site to GitHub Pages |

**Stack:** Python · Prefect · dbt-core (`dbt-duckdb` + `dbt-bigquery`) · DuckDB · BigQuery · Great Expectations · Evidence.dev · GitHub Actions · `ruff` · `sqlfluff`

---

### [ab-experiment-platform](https://github.com/elyokerr/ab-experiment-platform) - A/B Experiment Platform

A library-first online-experimentation toolkit covering the full A/B test lifecycle: power/sample-size design, sample-ratio-mismatch and A/A health checks, frequentist and Bayesian analysis, CUPED variance reduction, always-valid sequential testing (mSPRT), and a ship/no-ship decision engine. A built-in ground-truth simulator lets the test suite prove the statistics are correct.

| Highlight | Value |
|---|---|
| Sequential monitoring false-positive rate | **~31% (naive peeking) to ~4% (mSPRT)** at nominal 5% |
| CUPED variance reduction | **~10%** with a correlated pre-period covariate |
| Estimator CI coverage (500 simulated experiments) | **~96%** for nominal 95% |
| Tests | **34 unit tests** + simulation-based correctness validation + e2e smoke |

**Stack:** Python 3.11 · NumPy · SciPy · statsmodels · pandas · Streamlit · Plotly · pytest · ruff · Docker · GitHub Actions

---

### [uk-energy-price-forecasting](https://github.com/elyokerr/uk-energy-price-forecasting) - UK Energy Price Forecasting

Probabilistic forecasting of the GB system (imbalance) price. A single global model trained across a panel of energy series (price, national demand, generation-by-fuel) produces calibrated quantile forecasts for all 48 half-hourly settlement periods, benchmarked against a full model ladder under rolling-origin backtesting with a strict leakage guard.

| Highlight | Value |
|---|---|
| Best model (TFT) pinball loss vs seasonal-naive | **7.98 vs 16.94 (+53% skill)** ¹ |
| Best model (TFT) MAE vs seasonal-naive | **25.35 vs 33.88** ¹ |
| Best model (TFT) 80% interval coverage | **0.76** (nominal 0.80) |
| Model ladder | seasonal-naive to AutoARIMA to global LightGBM to global TFT/TiDE to zero-shot Chronos/TimesFM |

¹ Real 2024 GB imbalance-price data, GBP/MWh, every model scored on identical rolling origins.

**Stack:** Python 3.11 · Darts · LightGBM · PyTorch (TFT/TiDE) · Chronos/TimesFM · pandas · Streamlit · Plotly · pytest · ruff · Docker · GitHub Actions · Google Colab (T4)

---

### [career-gap-agent](https://github.com/elyokerr/career-gap-agent) - AI Career Gap Analyst

A tool-using AI agent that turns live UK job postings into a personalised, evidence-backed skills-gap plan. A single LangGraph agent searches Adzuna postings, extracts required skills with an LLM, normalises them to the ESCO skills taxonomy by embedding match, and returns a ranked report where every gap cites how many postings demanded it. Served through a mobile-responsive FastAPI web app and traced with Langfuse.

| Highlight | Value |
|---|---|
| Architecture | Single **LangGraph** tool-calling agent with a deterministic groundedness check and hard iteration cap |
| Skill matching | LLM extraction normalised to the **ESCO** taxonomy by embedding similarity |
| Evaluation | Component eval (precision / recall / F1) over a hand-labelled gold set |
| Observability | **Langfuse** tracing of every agent step, tool call, latency, and token cost |
| Reproducibility | Runs with **zero secrets** on a committed ESCO index + Adzuna snapshot (36 tests, lint clean) |

**Stack:** Python 3.11 · LangGraph · LangChain (Groq Llama 3.3 70B + Gemini fallback) · fastembed/BGE · ESCO · PyMuPDF · Langfuse · FastAPI · HTMX · Tailwind · Docker · GitHub Actions · Hugging Face Spaces

---

### [uk-retail-recommender](https://github.com/elyokerr/uk-retail-recommender) - UK Retail Recommender

A two-stage personalised product recommender on real UK e-commerce data (Online Retail II). Multi-source retrieval (popularity, item-to-item co-purchase, ALS matrix factorisation + FAISS, and a neural two-tower model) feeds a LightGBM LambdaMART ranker, evaluated on a temporal split and served through a mobile-friendly FastAPI demo.

| Highlight | Value |
|---|---|
| Recall@10 (model ladder) | popularity 0.08 · item-to-item 0.42 · ALS 0.22 · **two-stage 0.58** |
| Ranker quality | recovers **~96%** of the retrieval recall ceiling (0.60) |
| Evaluation | temporal split · recall@k / NDCG@k / MAP@k · retrieval-ceiling diagnostic |
| Reproducibility | full ladder runs with **zero download** on a committed sample (39 tests, lint clean) |

**Stack:** Python 3.11 · pandas · `implicit` (ALS) · FAISS · LightGBM (LambdaMART) · PyTorch (two-tower) · FastAPI · HTMX · Tailwind · Docker · GitHub Actions · Hugging Face Spaces

---

### [realtime-fraud-detection](https://github.com/elyokerr/realtime-fraud-detection) - Real-Time Card-Fraud Detection

A streaming pipeline that scores card transactions for fraud as they happen, with online features served the same way they were trained. Transactions flow through Redpanda and a Quix Streams processor that computes per-card velocity features; Feast serves them online from Redis; a model flags fraud in milliseconds; and a live web dashboard shows it all. The whole stack runs with `docker compose up`.

| Highlight | Value |
|---|---|
| Model PR-AUC (imbalanced fraud, held-out) | **0.84** |
| Recall at precision 0.90 | **0.74** |
| Train/serve feature parity | **guaranteed** (one shared module, asserted by a test) |
| Scoring latency | single-digit to low-tens of milliseconds per transaction |
| Tests | **44 passing**, lint clean, broker-free in CI |

**Stack:** Python 3.11 · Quix Streams · Redpanda · Feast + Redis · LightGBM · FastAPI · sse-starlette · HTMX · Tailwind · pandas · pytest · ruff · Docker Compose · GitHub Actions

---

## About

These projects focus on practical data science: not just model training, but the surrounding work that makes a model usable, such as clean data pipelines, honest validation, explainability, business-framed metrics, and interactive deliverables a stakeholder can actually open. Each repository is self-contained, documented end to end, and reproducible (free and CPU-friendly wherever possible, with Google Colab used only for heavy training).

New projects are added as they are built.

## Licence

Shared for portfolio and educational purposes. Individual projects may use third-party datasets under their respective terms; see each project's README for data attribution.
