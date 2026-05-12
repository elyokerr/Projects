# Technical Architecture

## Overview

The SaaS Churn Prediction Platform is built as a modular, layered system where each component can run independently or as part of the full stack via Docker Compose. The architecture follows patterns common in production ML systems: a data pipeline feeds engineered features into a trained model, which is served via a REST API and visualised through an interactive dashboard.

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI/CD                           │
│            Lint (Ruff) → Test (pytest) → Verify artifacts        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     docker compose up                             │
│                                                                   │
│  ┌──────────────┐   ┌───────────────┐   ┌───────────────────┐   │
│  │  PostgreSQL   │   │   FastAPI     │   │   Streamlit       │   │
│  │  Database     │◄──│   API         │   │   Dashboard       │   │
│  │  port 5432    │   │  port 8000    │   │   port 8501       │   │
│  │               │   │               │   │                   │   │
│  │  - customers  │   │  /health      │   │  KPI cards        │   │
│  │  - features   │   │  /predict     │   │  Risk charts      │   │
│  │  - scores     │   │  /predict/    │   │  Drilldown        │   │
│  │               │   │    batch      │   │  CSV export       │   │
│  └──────────────┘   └───────────────┘   └───────────────────┘   │
│                                                                   │
│  Health checks: PostgreSQL → API → Dashboard (ordered startup)   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

### Layer 1: Data Pipeline

The data pipeline transforms raw CSV data into 45 engineered features stored in a relational database. It runs as three sequential Python modules.

**`src/data/ingest.py`** — Loads the Kaggle Telco Customer Churn CSV, reshapes columns to SaaS terminology, cleans data types, and inserts records into SQLite (local/Colab) or PostgreSQL (Docker).

**`src/data/generate_synthetic.py`** — Creates realistic engagement metrics (login frequency, feature adoption, session duration) and support ticket data, correlated with actual churn outcomes. This simulates the behavioural data a real SaaS platform would have.

**`src/data/feature_engineering.py`** — Executes SQL queries using CTEs, window functions, and joins to produce the final feature set. The SQL-based approach demonstrates that feature engineering can happen in the database layer, not just in pandas.

### Layer 2: Model Training

**`src/models/train.py`** — Trains three models (Logistic Regression, Random Forest, XGBoost) with MLflow experiment tracking. Each run logs parameters, metrics, and model artifacts.

**`src/models/evaluate.py`** — Generates SHAP explanations, ROC/PR curves, confusion matrices, and performs threshold optimisation to maximise F1 score. Also calculates revenue impact at the optimised threshold.

**`src/models/predict.py`** — Scores all customers using the best model, assigns risk levels (critical/high/medium/low), and calculates monthly revenue at risk per customer.

### Layer 3: API Service

**`api/main.py`** — FastAPI application with three endpoints. Model artifacts load at startup via the lifespan context manager. Input is validated against Pydantic schemas. Predictions include probability, risk level, revenue at risk, and top risk factors.

**`api/schemas.py`** — Pydantic models defining the data contracts. CustomerFeatures validates 29 input fields with type checking and value constraints. PredictionResponse defines the output structure.

### Layer 4: Dashboard

**`dashboard/app.py`** — Streamlit application that reads scored customer data and model metadata. Provides KPI cards, interactive Plotly charts, customer drilldown, and CSV export. Sidebar filters update all views reactively.

### Layer 5: Infrastructure

**`docker-compose.yml`** — Orchestrates three services (PostgreSQL, API, Dashboard) with health checks ensuring correct startup order. The API waits for the database to be ready; the dashboard waits for the API.

**`.github/workflows/ci.yml`** — GitHub Actions pipeline that runs Ruff linting and pytest on every push to main. Verifies model artifacts exist to catch broken builds.

**`Makefile`** — Common development commands: `make stack-up`, `make test`, `make lint`, `make pipeline`.

---

## Key Architectural Decisions

### Why SQLite locally, PostgreSQL in Docker?

SQLite requires zero setup — essential for Google Colab and quick local development. PostgreSQL in Docker demonstrates awareness of production database practices. The codebase supports both via environment variables, showing the candidate can build for multiple deployment targets.

### Why FastAPI over Flask?

FastAPI provides automatic request validation (Pydantic), auto-generated Swagger documentation, async support, and is the current industry standard for ML model serving. Flask would work but requires more boilerplate for the same functionality.

### Why Streamlit over Plotly Dash or React?

Streamlit is Python-native and produces professional dashboards with minimal code. For a portfolio project, the goal is demonstrating data storytelling and stakeholder communication, not frontend engineering.

### Why Docker Compose with health checks?

Health checks ensure services start in the correct order (database before API, API before dashboard). Without them, the API would crash on startup if it tried to connect to a database that wasn't ready yet. This is a production pattern that demonstrates infrastructure awareness.

### Why GitHub Actions over Jenkins or CircleCI?

GitHub Actions is free for public repositories, tightly integrated with GitHub, and the most common CI/CD tool candidates will encounter. It requires no external setup.

### Why Ruff over flake8/pylint?

Ruff is 10-100x faster than traditional Python linters and combines the functionality of flake8, isort, and pyupgrade in a single tool. It's increasingly adopted in the Python ecosystem.

### Why a Makefile?

Makefiles are a standard way to document and automate common project commands. A hiring manager can open the Makefile and immediately understand all available operations without reading documentation.

---

## Environment Compatibility

| Environment | Database | API Testing | Dashboard |
|---|---|---|---|
| **Google Colab** | SQLite | TestClient (in-process) | Plotly charts inline |
| **Local (Windows/Mac/Linux)** | SQLite | `uvicorn api.main:app` | `streamlit run dashboard/app.py` |
| **Docker** | PostgreSQL | Container network | Container network |

No GPU is needed — all models are tabular and train in seconds.

---

## File Responsibilities

| File | Responsibility |
|---|---|
| `src/config.py` | Centralised paths, constants, and environment detection |
| `src/db.py` | Database engine factory (SQLite or PostgreSQL based on environment) |
| `src/data/ingest.py` | Raw CSV → cleaned tables in database |
| `src/data/generate_synthetic.py` | Create engagement and support data |
| `src/data/feature_engineering.py` | Execute SQL features and export CSV |
| `src/sql/features.sql` | CTE-based feature engineering queries |
| `src/models/train.py` | Train 3 models with MLflow tracking |
| `src/models/evaluate.py` | SHAP, ROC/PR, confusion matrix, threshold tuning |
| `src/models/predict.py` | Batch scoring with risk levels and revenue impact |
| `api/main.py` | FastAPI endpoints for real-time prediction |
| `api/schemas.py` | Pydantic request/response validation |
| `dashboard/app.py` | Streamlit interactive dashboard |
| `tests/test_api.py` | 18 automated API tests |
| `docker-compose.yml` | Full stack orchestration |
| `.github/workflows/ci.yml` | CI/CD pipeline |
