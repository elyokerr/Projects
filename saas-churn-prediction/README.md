# SaaS Customer Churn Prediction Platform

![CI Pipeline](https://github.com/YOUR_USERNAME/saas-churn-prediction/actions/workflows/ci.yml/badge.svg)

> An end-to-end machine learning system that predicts which SaaS customers are at risk of cancelling their subscriptions, estimates the revenue at risk, and provides actionable insights — served as a production-ready REST API with an interactive dashboard, fully containerised with Docker and tested via CI/CD.

---

## The Business Problem

Subscription-based software companies lose significant revenue when customers cancel — a process known as **churn**. Acquiring new customers costs 5–25x more than retaining existing ones, making churn prediction one of the highest-ROI applications of machine learning in business.

This project builds a complete churn prediction system that a customer success team could actually use: from raw data ingestion and feature engineering, through model training and evaluation, to a REST API for real-time predictions and an interactive dashboard for business stakeholders.

---

## Quick Start

### Option 1: Docker (recommended — one command)

```bash
git clone https://github.com/YOUR_USERNAME/saas-churn-prediction.git
cd saas-churn-prediction
docker compose up --build
```

Then open:
- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

### Option 2: Local (no Docker)

```bash
git clone https://github.com/YOUR_USERNAME/saas-churn-prediction.git
cd saas-churn-prediction
pip install -r requirements.txt

# Run the data pipeline
python -m src.data.ingest
python -m src.data.generate_synthetic
python -m src.data.feature_engineering

# Start the API
uvicorn api.main:app --reload --port 8000

# Start the dashboard (separate terminal)
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

### Option 3: Google Colab

Open the notebooks in the `notebooks/` folder — each one mounts Google Drive and runs the full pipeline interactively.

---

## Dashboard Preview

![Dashboard Overview](docs/screenshots/dashboard_overview.png)
![Customer Drilldown](docs/screenshots/customer_drilldown.png)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  Kaggle CSV → Python ingestion → SQLite/PostgreSQL               │
│  Synthetic engagement data → SQL feature engineering (45 features)│
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                              │
│  3-model comparison (Logistic Regression, Random Forest, XGBoost)│
│  MLflow tracking · SHAP explainability · Threshold optimisation  │
│  Revenue impact analysis · Batch scoring                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────┴──────────────────────────────────────┐
│  ┌─────────────────────┐       ┌──────────────────────────┐     │
│  │    FastAPI Service   │       │   Streamlit Dashboard    │     │
│  │  POST /predict       │       │  KPI cards & risk charts │     │
│  │  POST /predict/batch │       │  Customer drilldown      │     │
│  │  GET /health         │       │  Revenue at risk         │     │
│  │  Swagger docs at     │       │  CSV export for CRM      │     │
│  │  /docs               │       │  Sidebar filters         │     │
│  └─────────────────────┘       └──────────────────────────┘     │
│                                                                  │
│         All containerised via Docker Compose + CI/CD             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Skills Demonstrated

| Skill Area | What This Project Shows |
|---|---|
| **Data Engineering** | SQL feature engineering (CTEs, window functions, joins), ETL pipeline, database design |
| **Machine Learning** | Multi-model comparison, hyperparameter awareness, threshold optimisation |
| **MLOps** | MLflow experiment tracking, model serialisation, reproducible training |
| **Explainability** | SHAP values (global + local), feature importance, risk factor extraction |
| **API Development** | FastAPI REST endpoints, Pydantic validation, Swagger docs, automated tests |
| **Data Visualisation** | Streamlit dashboard, Plotly interactive charts, KPI cards |
| **DevOps** | Docker Compose (3-service stack), GitHub Actions CI, Makefile automation |
| **Software Engineering** | Modular architecture, pytest, clean code, comprehensive documentation |

---

## Results Summary

| Metric | Value |
|---|---|
| Best Model | Logistic Regression (or XGBoost depending on run) |
| Recall (churners caught) | ~82% |
| Precision | ~65% |
| F1 Score | ~0.72 |
| Total MRR at Risk | £5,300+ monthly |
| Estimated Annual Savings | £64,000+ |
| Features Engineered | 45 |
| API Tests | 18 passing |
| Risk Levels | Critical / High / Medium / Low |

---

## API — Serving Predictions

The FastAPI service exposes three endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status and model info |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 1000 customers) |

**Example request:**
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

**Example response:**
```json
{
  "churn_probability": 0.87,
  "churn_prediction": true,
  "risk_level": "critical",
  "monthly_revenue_at_risk": 95.00,
  "top_risk_factors": ["monthly_revenue", "tenure_months", "contract_type"]
}
```

Interactive Swagger documentation available at http://localhost:8000/docs.

---

## Dashboard — Business Intelligence

The Streamlit dashboard provides an interactive interface for customer success teams:

- **KPI cards** — total customers, predicted churners, churn rate, MRR at risk
- **Risk distribution** — colour-coded bar chart by severity level
- **Revenue at risk** — breakdown by risk category
- **Probability distribution** — histogram with decision threshold line
- **Customer drilldown** — individual risk profile with feature explorer
- **High-risk table** — sortable, filterable, with CSV export for CRM integration
- **Sidebar filters** — risk level, revenue range, probability range

---

## Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **SQL (SQLite / PostgreSQL)** | Data storage and feature engineering |
| **pandas / NumPy** | Data manipulation and analysis |
| **scikit-learn** | Machine learning pipeline and preprocessing |
| **XGBoost** | Gradient boosted tree model |
| **MLflow** | Experiment tracking and model registry |
| **SHAP** | Model explainability |
| **FastAPI** | REST API for model serving |
| **Pydantic** | Request/response validation |
| **Streamlit** | Interactive business dashboard |
| **Plotly** | Interactive data visualisations |
| **pytest** | Automated testing (18 tests) |
| **Docker / Docker Compose** | Full stack containerisation |
| **GitHub Actions** | CI/CD pipeline (lint + test) |
| **Ruff** | Python linter |
| **Make** | Build automation |
| **Git / GitHub** | Version control |

---

## Project Structure

```
saas-churn-prediction/
│
├── api/                          # FastAPI prediction service
│   ├── __init__.py
│   ├── main.py                   # API endpoints (/health, /predict, /predict/batch)
│   ├── schemas.py                # Pydantic request/response models
│   ├── Dockerfile                # API container definition
│   └── requirements.txt          # API-specific dependencies
│
├── dashboard/                    # Streamlit interactive dashboard
│   ├── app.py                    # Main dashboard application
│   ├── Dockerfile                # Dashboard container definition
│   └── requirements.txt          # Dashboard dependencies
│
├── data/
│   ├── raw/                      # Original Kaggle CSV
│   └── processed/                # Engineered features and scored customers
│       ├── features.csv
│       └── scored_customers.csv
│
├── docs/                         # Project documentation
│   ├── screenshots/              # Dashboard screenshots
│   ├── architecture.md           # Technical architecture and decisions
│   ├── data_pipeline.md          # Data layer documentation
│   ├── modelling.md              # Model training and evaluation
│   ├── api_service.md            # API documentation
│   └── dashboard.md              # Dashboard documentation
│
├── models/                       # Trained model artifacts
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── model_metadata.joblib
│   └── plots/                    # Generated visualisations
│
├── notebooks/                    # Colab-compatible Jupyter notebooks
│   ├── 01_full_pipeline.ipynb    # Data + modelling pipeline
│   ├── 02_fastapi_serving.ipynb  # API demo and testing
│   ├── 03_dashboard_demo.ipynb   # Dashboard charts preview
│   └── 04_docker_cicd.ipynb      # Docker & CI/CD walkthrough
│
├── src/                          # Core Python source code
│   ├── config.py                 # Project paths and settings
│   ├── db.py                     # Database connection management
│   ├── data/
│   │   ├── ingest.py             # Raw data ingestion
│   │   ├── generate_synthetic.py # Synthetic engagement data
│   │   └── feature_engineering.py# SQL-based feature engineering
│   ├── models/
│   │   ├── train.py              # Model training with MLflow
│   │   ├── evaluate.py           # Evaluation, SHAP, threshold tuning
│   │   └── predict.py            # Batch scoring with risk levels
│   └── sql/
│       ├── init.sql              # Database schema
│       └── features.sql          # Feature engineering queries
│
├── tests/
│   └── test_api.py               # 18 automated API tests
│
├── .github/workflows/
│   └── ci.yml                    # GitHub Actions CI pipeline
│
├── docker-compose.yml            # Full stack orchestration
├── .dockerignore                 # Docker build exclusions
├── Makefile                      # Common development commands
├── ruff.toml                     # Linter configuration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git exclusions
└── README.md                     # This file
```

---

## Roadmap

- [x] Phase 1 — Data pipeline (ingestion, synthetic data, SQL feature engineering)
- [x] Phase 2 — Modelling (training, evaluation, SHAP, revenue impact, batch scoring)
- [x] Phase 3 — FastAPI REST API with tests, validation, and Swagger docs
- [x] Phase 4 — Streamlit dashboard for customer success teams
- [x] Phase 5 — Docker containerisation of the full stack
- [x] Phase 6 — CI/CD with GitHub Actions

---

## Documentation

Detailed documentation for each component:

- [Data Pipeline](docs/data_pipeline.md) — how raw data becomes 45 predictive features
- [Modelling Approach](docs/modelling.md) — model selection, evaluation, threshold tuning, SHAP
- [API Service](docs/api_service.md) — endpoints, validation, testing, running locally and in Colab
- [Dashboard](docs/dashboard.md) — dashboard sections, filters, design decisions
- [Technical Architecture](docs/architecture.md) — system design and architectural decisions

---

## Future Improvements

- Model retraining pipeline with data drift detection
- Add authentication and rate limiting to the API
- A/B testing framework for churn intervention strategies
- Real-time streaming predictions with Kafka
- Model monitoring and alerting

---

## Author

Built as part of a portfolio demonstrating end-to-end data science and ML engineering skills for UK graduate roles.

---

## Licence

This project is for portfolio and educational purposes. The underlying dataset is sourced from Kaggle under its terms of use.
