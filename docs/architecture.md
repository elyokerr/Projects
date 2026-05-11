# Technical Architecture

## System Overview

The SaaS Churn Prediction Platform is a multi-layer system that processes raw customer data through a feature engineering pipeline, trains and evaluates ML models, serves predictions through a REST API, and presents insights through an interactive dashboard.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│                                                                     │
│  Kaggle CSV ──→ ingest.py ──→ SQLite ──→ synthetic.py               │
│                                  │                                  │
│                                  ▼                                  │
│                          features.sql (CTEs, window functions)      │
│                                  │                                  │
│                                  ▼                                  │
│                          features.csv (45 features)                 │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODELLING LAYER                                │
│                                                                     │
│  train.py ──→ 3 models compared (MLflow tracking)                   │
│      │                                                              │
│      ├──→ evaluate.py ──→ SHAP, ROC, confusion matrix               │
│      │                                                              │
│      └──→ predict.py ──→ scored_customers.csv (batch scoring)       │
│                                                                     │
│  Artifacts: best_model.joblib, scaler.joblib, model_metadata.joblib │
└──────────┬──────────────────────────┬───────────────────────────────┘
           │                          │
           ▼                          ▼
┌────────────────────────┐  ┌────────────────────────────────────────┐
│     API LAYER          │  │         DASHBOARD LAYER                │
│                        │  │                                        │
│  FastAPI (api/main.py) │  │  Streamlit (dashboard/app.py)          │
│                        │  │                                        │
│  GET  /health          │  │  KPI cards, risk charts                │
│  POST /predict         │  │  Revenue at risk analysis              │
│  POST /predict/batch   │  │  Customer drilldown                    │
│                        │  │  High-risk table + CSV export          │
│  Pydantic validation   │  │  Sidebar filters                      │
│  Swagger docs (/docs)  │  │  Plotly interactive charts             │
│  18 pytest tests       │  │                                        │
└────────────────────────┘  └────────────────────────────────────────┘
```

---

## File Responsibilities

| File | Layer | Purpose |
|---|---|---|
| `src/config.py` | Config | Centralised paths, database URL, model parameters |
| `src/db.py` | Data | SQLAlchemy engine creation, SQLite/PostgreSQL toggle |
| `src/data/ingest.py` | Data | Load Kaggle CSV, reshape for SaaS context, insert into database |
| `src/data/synthetic.py` | Data | Generate realistic engagement metrics (logins, sessions, support) |
| `src/data/feature_engineering.py` | Data | Execute SQL feature queries, export to CSV |
| `src/sql/init.sql` | Data | Database schema definition |
| `src/sql/features.sql` | Data | Feature engineering queries (CTEs, window functions, JOINs) |
| `src/models/train.py` | Model | Training pipeline: 3 models, MLflow tracking, threshold tuning |
| `src/models/evaluate.py` | Model | SHAP analysis, confusion matrix, ROC/PR curves |
| `src/models/predict.py` | Model | Batch scoring with risk levels and revenue at risk |
| `api/main.py` | API | FastAPI application with 3 endpoints |
| `api/schemas.py` | API | Pydantic request/response schemas with validation rules |
| `dashboard/app.py` | Dashboard | Streamlit interactive dashboard with Plotly charts |
| `tests/test_api.py` | Testing | 18 automated tests (health, predict, batch, validation) |

---

## Architectural Decisions

### Why SQLite over PostgreSQL?

The project supports both. SQLite is the default because it requires zero setup — no Docker, no database server, no credentials. This makes the project fully reproducible for anyone who clones the repo. PostgreSQL is available via Docker Compose for users who want a more production-realistic setup. The toggle is a single environment variable (`USE_POSTGRES=true`).

### Why a notebook-driven pipeline?

For a portfolio project, Colab notebooks are the most accessible execution environment. They allow reviewers to see inputs, outputs, and visualisations in one place. The actual logic lives in modular `src/` files that the notebooks import — so the code is production-structured while the execution is notebook-friendly.

### Why MLflow for experiment tracking?

MLflow is the most widely used experiment tracking tool in industry. Including it demonstrates awareness of ML lifecycle management — comparing runs, logging metrics, storing model artifacts. The file-based backend (`mlruns/`) requires no server setup.

### Why Logistic Regression can beat XGBoost here?

With only ~7,000 rows and 45 engineered features, the dataset is small enough that a well-tuned linear model can match or beat a tree ensemble. The threshold optimisation step matters more than model selection — moving from the default 0.5 threshold to the optimal ~0.76 dramatically improves business outcomes.

### Why FastAPI over Flask?

FastAPI provides three things Flask doesn't: automatic request validation via Pydantic (the API rejects bad data before it reaches the model), auto-generated Swagger documentation (interactive API docs at `/docs`), and native async support. It's the current industry standard for ML model serving in Python.

### Why separate Pydantic schemas?

Keeping `schemas.py` separate from `main.py` follows the single-responsibility principle. The schemas define the data contracts independently of the endpoint logic. This makes them testable, reusable, and self-documenting — the Swagger UI is generated directly from these schemas.

### Why TestClient instead of a running server in Colab?

Colab can't easily expose ports for a persistent HTTP server. FastAPI's `TestClient` (built on `httpx`) simulates real HTTP requests without needing a running server process. This means all API testing works seamlessly in Colab while still exercising the full request/response cycle.

### Why Streamlit over Plotly Dash or React?

Streamlit is Python-native, requires no frontend skills, and produces professional-looking dashboards in a fraction of the time. For a portfolio project, the goal is demonstrating data storytelling and stakeholder communication, not frontend engineering. Streamlit achieves this with minimal code.

### Why Plotly over matplotlib/seaborn for the dashboard?

matplotlib and seaborn produce static images. Plotly produces interactive charts that users can hover, zoom, and pan. For a dashboard aimed at business stakeholders, interactivity is essential — a customer success manager needs to hover over a data point to see which customer it represents.

### Why a CSV download button?

The most actionable output of a churn prediction system is a list of high-risk customers. The download button lets a customer success manager export this list and import it directly into their CRM or outreach tool. This small feature demonstrates product thinking — understanding how the output will actually be used.

---

## Data Flow

### Phase 1: Raw → Features

1. `ingest.py` loads the Kaggle Telco CSV and reshapes columns for a SaaS context
2. Data is inserted into SQLite tables (customers, subscriptions, usage, support)
3. `synthetic.py` generates engagement metrics correlated with actual churn outcomes
4. `features.sql` runs CTE-based queries that produce 45 features per customer
5. `feature_engineering.py` executes the SQL and exports `features.csv`

### Phase 2: Features → Model → Scores

1. `train.py` loads features, splits data, trains 3 models with MLflow tracking
2. Best model selected by ROC-AUC, threshold optimised by F1 score
3. `evaluate.py` generates SHAP explanations, ROC curves, confusion matrix
4. `predict.py` scores all customers with risk levels and revenue at risk
5. Results saved to `scored_customers.csv`

### Phase 3: Model → API

1. `api/main.py` loads model artifacts at startup via FastAPI's lifespan context
2. Incoming requests are validated against Pydantic schemas
3. Features are transformed to match training format (column order, scaling)
4. Predictions include probability, risk level, revenue at risk, and top risk factors

### Phase 4: Scores → Dashboard

1. `dashboard/app.py` reads `scored_customers.csv` and `model_metadata.joblib`
2. Data is cached with `@st.cache_data` for performance
3. Sidebar filters update all charts and tables reactively
4. Customer drilldown merges scored data with full features for detailed analysis

---

## Environment Compatibility

The project is designed to run in three environments:

| Environment | Database | API Testing | Dashboard |
|---|---|---|---|
| **Google Colab** | SQLite | TestClient (in-process) | Plotly charts inline |
| **Local (Windows/Mac/Linux)** | SQLite | `uvicorn api.main:app` | `streamlit run dashboard/app.py` |
| **Docker** | PostgreSQL | Container network | Container network |

No GPU is needed — all models are tabular and train in seconds.

**External dependencies:** Standard Python data science libraries (pandas, scikit-learn, XGBoost, SHAP, MLflow, FastAPI, Streamlit, Plotly). All are installable via pip with no system-level dependencies.

**Data dependencies:** The only external input is the Telco Customer Churn CSV from Kaggle. Everything else is generated by the pipeline.

---

## Planned Architecture Extensions

**Docker Compose** — The full stack (database + API + dashboard) will be containerised so the entire system can be started with a single `docker-compose up` command.

**GitHub Actions CI** — Automated linting (ruff/flake8) and testing on every push, demonstrating CI/CD awareness.
