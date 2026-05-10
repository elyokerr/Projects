# Technical Architecture

> Why the system is built the way it is — the reasoning behind every major technical decision.

---

## Design Philosophy

This project prioritises three things in order:

1. **Reproducibility** — Anyone can clone the repo and run the pipeline with zero configuration
2. **Readability** — The code is structured so each file has a clear, single responsibility
3. **Realism** — The architecture mirrors how production ML systems are actually built in industry

The system is deliberately not over-engineered. It uses the simplest tool that gets the job done well, and only adds complexity where it provides genuine value.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        NOTEBOOK LAYER                        │
│  notebooks/01_full_pipeline.ipynb  (data + modelling)        │
│  notebooks/02_fastapi_serving.ipynb (API dev + testing)      │
└────────────────────────────┬─────────────────────────────────┘
                             │ imports from
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                         │
│                                                              │
│  src/data/              src/models/           src/sql/       │
│  ├── ingest.py          ├── train.py          ├── init.sql   │
│  ├── generate_synthetic ├── evaluate.py       └── features   │
│  └── feature_engineering└── predict.py            .sql       │
│                                                              │
│  src/config.py          src/db.py                            │
│  (paths, settings)      (database connections)               │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                       API LAYER (Phase 3)                    │
│                                                              │
│  api/main.py          api/schemas.py        api/Dockerfile   │
│  (FastAPI endpoints)  (Pydantic validation) (container)      │
│                                                              │
│  tests/test_api.py    (18 automated pytest tests)            │
└────────────────────────────┬─────────────────────────────────┘
                             │ reads/writes
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                       STORAGE LAYER                          │
│                                                              │
│  data/raw/         data/processed/     models/    mlruns/    │
│  (source CSV)      (features CSV,      (trained   (MLflow    │
│                     scored customers)   model)     logs)      │
│                                                              │
│  data/churn.db     (SQLite database with 6 tables)           │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Decisions

### Notebook as orchestrator, not as the codebase

**Decision:** The Jupyter notebook imports from `src/` modules rather than containing all logic inline.

**Why:** A common mistake in portfolio projects is putting everything in a single notebook. This makes the code hard to test, impossible to reuse, and signals to hiring managers that the candidate doesn't understand software engineering. By keeping the core logic in Python modules and using the notebook only for orchestration and visualisation, the repo demonstrates both data science skills (the notebook) and engineering maturity (the modular code).

**Trade-off:** Slightly more complex setup — the notebook needs to know where the `src/` directory is. This is handled by the Colab/Drive setup cell.

### SQLite for development, PostgreSQL for production

**Decision:** The notebook and local development use SQLite. The `docker-compose.yml` defines a PostgreSQL setup for production.

**Why:** SQLite requires zero installation — no Docker, no server, no credentials. This means anyone can clone the repo and run it immediately. The SQL queries in `features.sql` are written to work on both engines, so switching to PostgreSQL is a configuration change, not a rewrite.

**Trade-off:** Some PostgreSQL-specific features (e.g., `DATE_PART`, `EXTRACT`) needed to be written with SQLite-compatible alternatives.

### MLflow for experiment tracking

**Decision:** Every training run is logged to MLflow with parameters, metrics, and model artifacts.

**Why:** MLflow is the most widely adopted experiment tracking tool in industry. Including it demonstrates awareness of MLOps practices and shows that the candidate thinks about reproducibility and auditability. It also makes the project genuinely useful — if you want to try different hyperparameters or feature sets, you can compare runs directly.

**Trade-off:** Adds a dependency and creates the `mlruns/` directory. The directory is gitignored since experiment logs are environment-specific.

### SHAP over simpler explanation methods

**Decision:** SHAP values are used for model explainability rather than simpler alternatives like feature importance from tree models.

**Why:** Tree-based feature importance (from scikit-learn or XGBoost) only tells you which features are important overall — it doesn't explain individual predictions or show the direction of impact. SHAP provides both global and local explanations, is model-agnostic, and is theoretically grounded in Shapley values from cooperative game theory. It's also what most data science teams use in practice.

**Trade-off:** SHAP computation is slower, especially for large datasets. For this project size, it runs in under a minute.

### Modular configuration management

**Decision:** All paths, database settings, and model parameters are centralised in `src/config.py`.

**Why:** Hard-coded paths scattered across files is one of the most common sources of bugs in data projects. By centralising configuration, changing the project location (e.g., from a local machine to Google Drive) requires editing one file, not twenty.

### Synthetic data generation as a separate, explicit step

**Decision:** Rather than downloading a more complete dataset, the pipeline generates synthetic engagement data programmatically.

**Why:** This demonstrates an important real-world skill — understanding what data should exist, designing it with realistic distributions, and correlating it with the target variable. It also means the project doesn't depend on any specific external data source beyond the base CSV. The synthetic data is generated deterministically (fixed random seed) so results are reproducible.

---

## File Responsibilities

Each file in the `src/` directory has a single, clear purpose:

| File | Responsibility | Inputs | Outputs |
|---|---|---|---|
| `config.py` | Define all paths, database settings, model params | — | Configuration constants |
| `db.py` | Create and manage database connections | Config | SQLAlchemy engine |
| `data/ingest.py` | Load raw CSV, reshape, write to database | Raw CSV | Database tables |
| `data/generate_synthetic.py` | Create engagement and support data | Database tables | New database tables |
| `data/feature_engineering.py` | Run SQL + Python feature transforms | Database, SQL files | `features.csv` |
| `models/train.py` | Train 3 models, log to MLflow, save best | `features.csv` | Model artifact, metadata |
| `models/evaluate.py` | Generate plots, SHAP analysis, metrics | Model, features | Plots, reports |
| `models/predict.py` | Score all customers, assign risk levels | Model, features | `scored_customers.csv` |
| `sql/init.sql` | Define database schema | — | Table definitions |
| `sql/features.sql` | Feature engineering queries | Database tables | Feature set |
| `api/main.py` | FastAPI app with 3 endpoints | Model artifacts | JSON predictions |
| `api/schemas.py` | Pydantic request/response validation | — | Data contracts |
| `tests/test_api.py` | 18 automated API tests | API app | Pass/fail results |

---

## Dependencies and Constraints

**Hardware requirements:** The entire pipeline runs on a basic laptop or Google Colab free tier. No GPU is needed — all models are tabular and train in seconds.

**External dependencies:** Standard Python data science libraries (pandas, scikit-learn, XGBoost, SHAP, MLflow, matplotlib, seaborn, SQLAlchemy) plus FastAPI, Pydantic, uvicorn, and pytest for the API layer. All are installable via pip with no system-level dependencies.

**Data dependencies:** The only external input is the Telco Customer Churn CSV from Kaggle. Everything else is generated by the pipeline.

---

## Phase 3 Architecture Decisions

### Why FastAPI over Flask

**Decision:** FastAPI was chosen as the API framework instead of Flask.

**Why:** FastAPI provides three things Flask doesn't out of the box. First, Pydantic integration means request validation is automatic — if someone sends negative revenue or an invalid contract type, the API rejects it with a clear error message before the model ever sees it. Second, Swagger documentation is auto-generated at `/docs`, giving anyone a browser-based interface to test the API without writing any code. Third, FastAPI supports async natively, which matters for high-concurrency production deployments. Flask can do all of these things with extensions, but FastAPI includes them by default. For ML model serving specifically, FastAPI has become the industry standard.

### Why Pydantic schemas in a separate file

**Decision:** Request and response models live in `api/schemas.py`, not in `api/main.py`.

**Why:** Separating data contracts from endpoint logic follows the same single-responsibility principle used throughout the project. It makes the schemas reusable (a future frontend or SDK could import them directly), keeps `main.py` focused on business logic, and makes the API easier to maintain as it grows.

### Why the lifespan pattern for model loading

**Decision:** The model is loaded once at startup using FastAPI's `lifespan` context manager, not on every request.

**Why:** Loading a joblib model from disk takes time. If it happened on every request, each prediction would be significantly slower. Loading once at startup means the model lives in memory and predictions are near-instant. The lifespan pattern is how FastAPI recommends managing startup/shutdown resources — it replaces the older `@app.on_event("startup")` approach.

### Why TestClient for Colab testing

**Decision:** The notebook uses FastAPI's TestClient rather than starting a real HTTP server.

**Why:** Google Colab can't expose persistent ports to the browser without workarounds like ngrok. TestClient simulates HTTP requests in-process, which means all API functionality can be tested without a running server. This isn't a compromise — it's exactly how production API tests are written with pytest. The notebook demonstrates both approaches: TestClient for testing, and a background-thread server for real HTTP requests when running locally.

### Why 18 tests in 4 categories

**Decision:** Tests are organised into health checks, single predictions, batch predictions, and input validation.

**Why:** This mirrors how production APIs are tested. Health checks verify the service is alive. Prediction tests verify correct output shape, valid probability ranges, and logical consistency (e.g., revenue at risk is £0 when the customer isn't predicted to churn). Validation tests verify that bad input is rejected — not silently accepted. The 18 tests provide good coverage without being excessive for a portfolio project.

---

## Completed Architecture Components

**Phase 1 — Data Pipeline:** Ingestion, synthetic data generation, SQL feature engineering with SQLite.

**Phase 2 — Modelling:** Three-model comparison with MLflow tracking, SHAP explainability, threshold optimisation, revenue impact analysis, batch scoring.

**Phase 3 — API Service:** FastAPI with Pydantic validation, Swagger documentation, pytest test suite, Dockerfile for containerised deployment.

---

## Planned Architecture Extensions

The following components are planned for future phases:

**Streamlit dashboard** (`dashboard/app.py`) — An interactive web application for customer success teams to explore churn risk across the customer base, drill into individual customers, and understand the revenue impact. This will demonstrate data visualisation, stakeholder communication, and product thinking.

**Docker Compose** — The full stack (database + API + dashboard) will be containerised so the entire system can be started with a single `docker-compose up` command.

**GitHub Actions CI** — Automated linting (ruff/flake8) and testing on every push, demonstrating CI/CD awareness.
