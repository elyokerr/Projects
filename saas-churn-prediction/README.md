# SaaS Customer Churn Prediction Platform

> An end-to-end machine learning system that predicts which SaaS customers are at risk of cancelling their subscriptions, estimates the revenue at risk, and surfaces actionable insights through a REST API and interactive dashboard.

---

## Dashboard Preview

![Dashboard Overview](docs/screenshots/dashboard_overview.png)

![Customer Drilldown](docs/screenshots/customer_drilldown.png)

---

## The Business Problem

Subscription-based software companies lose significant revenue when customers cancel — a process known as **churn**. Identifying at-risk customers *before* they leave allows customer success teams to intervene with targeted retention strategies: personalised outreach, discounted renewals, feature onboarding, or escalated support.

This project builds the complete system a data team would deliver to solve this problem — from raw data ingestion through feature engineering, model training with experiment tracking, a production-ready prediction API, and an interactive dashboard for non-technical stakeholders.

---

## What This Project Does

```
Raw Data (Kaggle CSV)
    │
    ▼
Data Ingestion & Reshaping ──→ SQLite Database
    │
    ▼
Synthetic Engagement Data ──→ Login patterns, feature adoption, support tickets
    │
    ▼
SQL Feature Engineering ──→ 45 features (CTEs, window functions, aggregations)
    │
    ▼
Model Training ──→ 3 models compared (Logistic Regression, Random Forest, XGBoost)
    │                  MLflow experiment tracking
    ▼
Threshold Optimisation ──→ Business-optimal precision/recall balance
    │
    ▼
SHAP Explainability ──→ Global + individual prediction explanations
    │
    ▼
Batch Scoring ──→ All customers scored with risk levels + revenue at risk
    │
    ├──→ FastAPI REST API ──→ /health, /predict, /predict/batch
    │         │
    │         ▼
    │    Swagger Documentation ──→ Auto-generated interactive docs
    │
    └──→ Streamlit Dashboard ──→ KPI cards, charts, customer drilldown, CSV export
```

---

## Results Summary

| Metric | Value |
|---|---|
| Best Model | Logistic Regression (optimised threshold) |
| Recall (churners caught) | ~82% |
| Precision | ~65% |
| Optimal Threshold | 0.7602 |
| Total MRR at Risk | £5,300+ monthly |
| Estimated Annual Savings | £64,000+ |
| Features Engineered | 45 |
| API Endpoints | 3 (`/health`, `/predict`, `/predict/batch`) |
| Automated Tests | 18 (all passing) |
| Risk Levels | Critical / High / Medium / Low |

*Note: exact numbers may vary slightly between runs due to random seeds in synthetic data generation.*

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
| **pytest** | Automated testing |
| **matplotlib / seaborn** | Static data visualisation |
| **SQLAlchemy** | Database ORM and connection management |
| **Docker** | Containerisation for API and database |
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
│   ├── Dockerfile                # Container definition
│   └── requirements.txt          # API-specific dependencies
│
├── dashboard/                    # Streamlit interactive dashboard
│   ├── app.py                    # Main dashboard application
│   └── requirements.txt          # Dashboard dependencies
│
├── data/
│   ├── raw/                      # Original Kaggle CSV
│   └── processed/                # Engineered features and scored customers
│       ├── features.csv
│       └── scored_customers.csv
│
├── docs/                         # Project documentation
│   ├── screenshots/              # Dashboard screenshots for README
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
│   └── 03_dashboard_demo.ipynb   # Dashboard chart previews
│
├── src/                          # Core Python modules
│   ├── config.py                 # Centralised paths and settings
│   ├── db.py                     # Database connection utilities
│   ├── data/
│   │   ├── ingest.py             # Raw data loading and reshaping
│   │   ├── synthetic.py          # Engagement data generation
│   │   └── feature_engineering.py # SQL-based feature construction
│   ├── models/
│   │   ├── train.py              # Training pipeline with MLflow
│   │   ├── evaluate.py           # SHAP, confusion matrix, ROC curves
│   │   └── predict.py            # Batch scoring
│   └── sql/
│       ├── init.sql              # Database schema
│       └── features.sql          # Feature engineering queries
│
├── tests/
│   └── test_api.py               # 18 automated API tests
│
├── docker-compose.yml            # Container orchestration
├── Makefile                      # Build automation commands
├── requirements.txt              # Project-wide dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

### Option 1: Google Colab (Recommended)

The entire project runs in Google Colab with no local setup required.

1. Clone the repo to Google Drive:
```bash
cd /content/drive/MyDrive/Github/Projects
git clone https://github.com/YOUR_USERNAME/saas-churn-prediction.git
```

2. Open notebooks in order:
   - `notebooks/01_full_pipeline.ipynb` — runs the data and modelling pipeline
   - `notebooks/02_fastapi_serving.ipynb` — tests the API endpoints
   - `notebooks/03_dashboard_demo.ipynb` — previews dashboard visualisations

Each notebook mounts Google Drive automatically and installs dependencies.

### Option 2: Local Setup

```bash
# Clone and enter the project
git clone https://github.com/YOUR_USERNAME/saas-churn-prediction.git
cd saas-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (trains model, generates features, scores customers)
python -m src.data.ingest
python -m src.data.synthetic
python -m src.data.feature_engineering
python -m src.models.train
python -m src.models.evaluate
python -m src.models.predict

# Start the API
uvicorn api.main:app --reload --port 8000
# → Interactive Swagger docs at http://localhost:8000/docs

# Start the dashboard
streamlit run dashboard/app.py
# → Dashboard at http://localhost:8501

# Run tests
python -m pytest tests/test_api.py -v
```

---

## API — Serving Predictions

The FastAPI service exposes three endpoints for real-time and batch churn predictions.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status, model info, threshold |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Score up to 1000 customers at once |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_revenue": 79.85,
    "tenure_months": 48,
    "contract_type": 2,
    "total_charges": 3832.80
  }'
```

**Example response:**

```json
{
  "churn_probability": 0.23,
  "churn_prediction": false,
  "risk_level": "low",
  "monthly_revenue_at_risk": 0.0,
  "top_risk_factors": ["monthly_revenue", "has_dependents", "has_partner"]
}
```

Full API documentation is auto-generated at `/docs` (Swagger UI) when the server is running.

---

## Dashboard — Business Intelligence

The Streamlit dashboard provides an interactive interface for customer success teams.

**Features:**
- 5 KPI cards — total customers, predicted churners, churn rate, MRR at risk, critical count
- Risk level distribution chart with colour-coded severity
- Revenue at risk breakdown by risk level
- Churn probability distribution with decision threshold line
- Revenue vs probability scatter plot for identifying high-value at-risk customers
- Customer drilldown with engagement metrics and feature explorer
- High-risk customer table with CSV export for CRM integration
- Sidebar filters for risk level, revenue range, and probability range

```bash
# Run locally
streamlit run dashboard/app.py
```

---

## Sample Outputs

The pipeline generates the following visualisations (saved to `models/plots/`):

- **Churn Distribution** — baseline class balance in the dataset
- **Feature Correlation Heatmap** — relationships between engineered features
- **Churn by Segment** — churn rates across contract types, tenure groups, payment methods
- **ROC Curve** — model discrimination ability across all thresholds
- **Precision-Recall Curve** — performance on the minority (churn) class
- **Confusion Matrix** — true/false positive/negative breakdown at the optimised threshold
- **SHAP Summary Plot** — global feature importance with direction of impact
- **SHAP Bar Plot** — simplified feature importance ranking
- **SHAP Waterfall Plot** — explanation of an individual high-risk prediction

---

## Documentation

Detailed documentation for each component is available in the `docs/` folder:

- [Data Pipeline](docs/data_pipeline.md) — how raw data becomes 45 predictive features
- [Modelling Approach](docs/modelling.md) — model selection, evaluation, threshold tuning, SHAP
- [API Service](docs/api_service.md) — endpoints, validation, testing, running locally and in Colab
- [Dashboard](docs/dashboard.md) — dashboard sections, filters, design decisions
- [Technical Architecture](docs/architecture.md) — system design and architectural decisions

---

## Roadmap

- [x] Phase 1 — Data pipeline (ingestion, synthetic data, SQL feature engineering)
- [x] Phase 2 — Modelling (training, evaluation, SHAP, revenue impact, batch scoring)
- [x] Phase 3 — FastAPI REST API with tests, validation, and Swagger docs
- [x] Phase 4 — Streamlit dashboard for customer success teams
- [ ] Phase 5 — Docker containerisation of the full stack
- [ ] Phase 6 — CI/CD with GitHub Actions

---

## Future Improvements

- Containerise the full stack (API + dashboard + database) with Docker Compose
- Add GitHub Actions CI pipeline for linting and testing
- Implement model retraining pipeline with data drift detection
- Add authentication and rate limiting to the API
- A/B testing framework for churn intervention strategies

---

## Author

Built as part of a portfolio demonstrating end-to-end data science and ML engineering skills for UK graduate roles.

---

## Licence

This project is for portfolio and educational purposes. The underlying dataset is sourced from Kaggle under its terms of use.
