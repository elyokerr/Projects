# SaaS Customer Churn Prediction Platform

Predicts which SaaS customers are about to cancel, puts a pound figure on the revenue at stake, and serves the results through a REST API and a Streamlit dashboard. `docker compose up` brings the whole stack up; CI runs the linter and tests on every push.

---

## Table of Contents

- [The Business Problem](#the-business-problem)
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Results](#results)
- [API Usage](#api-usage)
- [Dashboard](#dashboard)
- [Technology Stack](#technology-stack)
- [Documentation](#documentation)
- [License](#license)

---

## The Business Problem

Every cancellation in a subscription business is lost recurring revenue, and winning a new customer back costs far more than keeping one you already have. So catching at-risk accounts early is worth real money.

A model that labels churners is only half of it. A customer success team also needs predictions they can explain, scored against a threshold that reflects business cost rather than a default 0.5, ranked by how much revenue is on the line, and reachable from tools they already use. This project covers that whole path, from raw data through to the dashboard a non-engineer would actually open.

---

## What This Project Demonstrates

| Capability | Where to look |
|---|---|
| Relational data modeling and SQL feature engineering | `src/sql/`, `src/data/feature_engineering.py` |
| ETL pipeline with reproducible synthetic data generation | `src/data/ingest.py`, `src/data/generate_synthetic.py` |
| Multi-model comparison with experiment tracking | `src/models/train.py` (MLflow integration) |
| Threshold optimization based on business cost | `src/models/train.py`, `docs/modeling.md` |
| Model explainability with SHAP | `src/models/evaluate.py` |
| Revenue impact estimation in financial terms | `src/models/train.py` |
| REST API with request validation and testing | `api/`, `tests/test_api.py` |
| Interactive dashboard for non-technical stakeholders | `dashboard/app.py` |
| Multi-service containerization | `docker-compose.yml`, `api/Dockerfile`, `dashboard/Dockerfile` |
| Continuous integration with linting and testing | `.github/workflows/ci.yml` |
| Reproducible workflow automation | `Makefile`, `notebooks/` |

---

## Quick Start

### Option 1: Docker (recommended)

The fastest way to see everything running. Requires Docker Desktop.

```bash
git clone https://github.com/YOUR_USERNAME/saas-churn-prediction.git
cd saas-churn-prediction
docker compose up --build
```

When the containers report healthy:

- Dashboard: <http://localhost:8501>
- API documentation (Swagger UI): <http://localhost:8000/docs>
- PostgreSQL: `localhost:5432` (user `churn_user`, password `churn_pass`, database `saas_churn`)

Stop everything with `docker compose down`. Add `-v` to also wipe the database volume.

### Option 2: Local Python (no Docker)

```bash
git clone https://github.com/YOUR_USERNAME/saas-churn-prediction.git
cd saas-churn-prediction
pip install -r requirements.txt

# Run the data pipeline (writes to a local SQLite database)
python -m src.data.ingest
python -m src.data.generate_synthetic
python -m src.data.feature_engineering

# Train the model
python -m src.models.train
python -m src.models.evaluate
python -m src.models.predict

# Start the API (terminal 1)
uvicorn api.main:app --reload --port 8000

# Start the dashboard (terminal 2)
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

### Option 3: Google Colab

Every notebook in `notebooks/` is Colab-ready. Each one mounts Google Drive, navigates to the project directory, installs dependencies, and runs the relevant stage. Open them in order:

1. `01_full_pipeline.ipynb` -data pipeline plus model training and evaluation
2. `02_fastapi_serving.ipynb` -API walkthrough using FastAPI's TestClient
3. `03_dashboard_demo.ipynb` -inline preview of the dashboard's charts
4. `04_docker_cicd.ipynb` -Docker and CI configuration walkthrough

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  Raw CSV  →  Python ingestion  →  SQLite / PostgreSQL               │
│  Synthetic engagement + support data                                 │
│  SQL feature engineering (CTEs, window functions)  →  45 features    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODELING LAYER                                 │
│  Multi-model comparison:  Logistic Regression / Random Forest / XGB  │
│  MLflow experiment tracking · SHAP explainability                    │
│  F1-optimized threshold · Revenue impact estimation                  │
│  Batch scoring with critical / high / medium / low risk tiers        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────┴──────────────────────────────────────────┐
│  ┌─────────────────────┐         ┌──────────────────────────┐       │
│  │    FastAPI Service   │         │   Streamlit Dashboard    │       │
│  │  POST /predict       │         │  KPI cards & risk charts │       │
│  │  POST /predict/batch │         │  Customer drilldown      │       │
│  │  GET  /health        │         │  Revenue at risk view    │       │
│  │  /docs (Swagger)     │         │  CSV export for CRM use  │       │
│  └─────────────────────┘         └──────────────────────────┘       │
│                                                                      │
│         All services containerized · CI runs on every push           │
└─────────────────────────────────────────────────────────────────────┘
```

For a deeper architectural walkthrough see [`docs/architecture.md`](docs/architecture.md).

---

## Project Structure

```
saas-churn-prediction/
│
├── api/                              FastAPI prediction service
│   ├── main.py                       Application entry point and endpoints
│   ├── schemas.py                    Pydantic request and response models
│   ├── Dockerfile                    Container definition
│   └── requirements.txt              Service-specific dependencies
│
├── dashboard/                        Streamlit interactive dashboard
│   ├── app.py                        Dashboard application
│   ├── Dockerfile                    Container definition
│   └── requirements.txt              Service-specific dependencies
│
├── data/
│   ├── raw/                          Raw CSV (excluded from git, see Setup)
│   └── processed/                    Engineered features and scored customers
│
├── docs/                             Detailed component documentation
│   ├── architecture.md               System architecture and design choices
│   ├── data_pipeline.md              Data layer walkthrough
│   ├── modeling.md                   Model training and evaluation approach
│   ├── api_service.md                API endpoints, validation, and testing
│   ├── dashboard.md                  Dashboard features and usage
│   └── screenshots/                  Dashboard screenshots used in this README
│
├── models/                           Trained artifacts (excluded from git)
│   ├── best_model.joblib             Serialized winning model
│   ├── scaler.joblib                 Fitted feature scaler
│   ├── model_metadata.joblib         Threshold, metrics, revenue impact
│   └── plots/                        Generated evaluation visualizations
│
├── notebooks/                        Colab-compatible Jupyter notebooks
│   ├── 01_full_pipeline.ipynb        Data pipeline + training + evaluation
│   ├── 02_fastapi_serving.ipynb      API walkthrough
│   ├── 03_dashboard_demo.ipynb       Dashboard charts preview
│   └── 04_docker_cicd.ipynb          Containerization and CI walkthrough
│
├── src/                              Core library
│   ├── config.py                     Centralized paths and settings
│   ├── db.py                         Database engine factory
│   ├── data/
│   │   ├── ingest.py                 Raw data ingestion and schema reshaping
│   │   ├── generate_synthetic.py     Synthetic engagement and support data
│   │   └── feature_engineering.py    SQL + Python feature pipeline
│   ├── models/
│   │   ├── train.py                  Training pipeline with MLflow
│   │   ├── evaluate.py               Metrics, plots, SHAP explanations
│   │   └── predict.py                Batch scoring with risk levels
│   └── sql/
│       ├── init.sql                  PostgreSQL schema definition
│       └── features.sql              Feature engineering queries
│
├── tests/
│   └── test_api.py                   Eighteen automated API tests
│
├── .github/workflows/
│   └── ci.yml                        Lint and test on every push
│
├── docker-compose.yml                Full stack orchestration
├── .dockerignore                     Files excluded from container builds
├── .gitignore                        Files excluded from version control
├── Makefile                          Common development commands
├── ruff.toml                         Linter configuration
├── requirements.txt                  Top-level Python dependencies
├── LICENSE                           MIT License
├── CONTRIBUTING.md                   Notes for anyone forking the project
└── README.md                         This file
```

---

## Results

| Metric | Value |
|---|---|
| Customers in the dataset | 7,043 |
| Engineered features | 45 |
| Best-performing model | Logistic Regression (winner varies slightly between runs) |
| Recall on the churn class | ~82% |
| Precision on the churn class | ~65% |
| F1 score on the churn class | ~0.72 |
| ROC-AUC | ~0.85 |
| Monthly recurring revenue at risk (identified) | £5,300+ |
| Estimated annual savings (20% intervention success rate) | £64,000+ |
| API endpoints | 3 (`/health`, `/predict`, `/predict/batch`) |
| Automated API tests | 18 |

Exact figures will vary slightly between runs because synthetic engagement data is generated with a fixed seed but downstream cross-validation and threshold optimization can shift the winning model. The `model_metadata.joblib` file always reflects the most recent run.

---

## API Usage

The API exposes three endpoints. Full documentation, including the interactive Swagger UI, is auto-generated and available at `/docs` when the API is running.

### Health check

```bash
curl http://localhost:8000/health
```

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_revenue": 95.00,
    "tenure_months": 2,
    "contract_type": 0,
    "total_charges": 190.00
  }'
```

Response:

```json
{
  "churn_probability": 0.87,
  "churn_prediction": true,
  "risk_level": "critical",
  "monthly_revenue_at_risk": 95.00,
  "top_risk_factors": ["monthly_revenue", "tenure_months", "contract_type"]
}
```

### Batch prediction

`POST /predict/batch` accepts up to 1000 customers per request and returns both individual predictions and aggregate revenue exposure. See [`docs/api_service.md`](docs/api_service.md) for the full schema.

---

## Dashboard

The Streamlit dashboard puts the model output in front of a customer success team without anyone needing to touch code. It includes:

- Five KPI cards summarizing customer base health and revenue exposure
- Risk-level distribution and revenue breakdown
- Churn probability histogram with the model's decision threshold overlaid
- Revenue versus probability scatter plot for spotting high-value at-risk accounts
- Individual customer drilldown with a feature-level explanation
- Downloadable CSV of the high-risk customer list, ready for CRM import
- Sidebar filters for risk level, revenue range, and probability range

Screenshots live in `docs/screenshots/` and are referenced in [`docs/dashboard.md`](docs/dashboard.md).

---

## Technology Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Data storage | SQLite (local) / PostgreSQL 16 (Docker) |
| Data processing | pandas, NumPy, SQLAlchemy |
| Machine learning | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| API framework | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, FastAPI TestClient |
| Linting | Ruff |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Automation | Make |
| Version control | Git, GitHub |

---

## Documentation

Each major component has its own detailed write-up under `docs/`:

- [`docs/architecture.md`](docs/architecture.md) -system design, layer responsibilities, and the rationale behind key decisions
- [`docs/data_pipeline.md`](docs/data_pipeline.md) -how raw data becomes 45 features
- [`docs/modeling.md`](docs/modeling.md) -model selection, evaluation, threshold tuning, and SHAP analysis
- [`docs/api_service.md`](docs/api_service.md) -endpoints, validation, testing, and running locally
- [`docs/dashboard.md`](docs/dashboard.md) -dashboard sections, filters, and design choices

---


## License

Released under the [MIT License](LICENSE). The underlying Telco Customer Churn dataset is sourced from Kaggle under its own terms of use.
