# SaaS Customer Churn Prediction Platform

> An end-to-end machine learning system that predicts which SaaS customers are at risk of cancelling their subscriptions, estimates the revenue at risk, and provides actionable insights — served as a production-ready REST API.

---

## The Business Problem

Subscription-based software companies lose significant revenue when customers cancel — a process known as **churn**. The cost of acquiring a new customer is typically 5–7× higher than retaining an existing one, which makes early identification of at-risk customers one of the most impactful things a data team can do.

This project builds a complete churn prediction system — not just a model, but the full pipeline a real data team would deliver: a structured database, engineered features, trained and evaluated models, explainable predictions, revenue impact analysis, and a REST API for serving predictions to other systems.

**Key results from this system:**

- Identified **£64,000+** in annual recurring revenue at risk across the customer base
- Achieved **82% recall** — the model catches 4 out of every 5 customers who would actually churn
- Engineered **45 predictive features** from raw data using SQL and Python
- Every prediction is explainable — the system tells you *why* a customer is flagged, not just *that* they are
- Model is served via a **FastAPI REST API** with automatic input validation and interactive documentation

---

## What This Project Demonstrates

This is a portfolio project designed to showcase how a Data Scientist or ML Engineer approaches real-world problems. It goes well beyond a typical notebook exercise.

| Skill Area | What's Shown |
|---|---|
| **Data Engineering** | Raw data → structured database → engineered features via SQL pipelines |
| **SQL** | Complex feature queries using CTEs, window functions, JOINs, and aggregations |
| **Machine Learning** | 3-model comparison (Logistic Regression, Random Forest, XGBoost) with proper train/test methodology |
| **Experiment Tracking** | MLflow for logging metrics, parameters, and model artifacts across experiments |
| **Model Explainability** | SHAP values for both global feature importance and individual prediction explanations |
| **API Development** | FastAPI REST service with Pydantic validation, Swagger docs, and health monitoring |
| **Testing** | 18 automated pytest tests covering all API endpoints and input validation |
| **Business Thinking** | Threshold tuning based on business cost, revenue impact estimation, risk-level scoring |
| **Software Engineering** | Modular codebase with separate concerns, configuration management, reproducible pipeline |
| **Containerisation** | Docker setup for both database infrastructure and API deployment |
| **Automation** | Makefile for one-command pipeline execution |

---

## How It Works

The system follows a standard industry ML pipeline:

```
Raw CSV Data
     │
     ▼
┌─────────────────────────────────┐
│  1. DATA INGESTION              │  Load raw data, reshape to SaaS context,
│     src/data/ingest.py          │  store in relational database
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  2. SYNTHETIC DATA GENERATION   │  Generate realistic engagement metrics,
│     src/data/generate_synthetic │  support tickets, and usage patterns
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  3. FEATURE ENGINEERING         │  45 features via SQL (CTEs, window
│     src/sql/features.sql        │  functions) + Python transformations
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  4. MODEL TRAINING              │  3 algorithms compared, tracked with
│     src/models/train.py         │  MLflow, threshold optimised for business
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  5. EVALUATION & EXPLAINABILITY │  SHAP explanations, confusion matrices,
│     src/models/evaluate.py      │  ROC curves, revenue impact analysis
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  6. BATCH SCORING               │  Score all customers, assign risk levels
│     src/models/predict.py       │  (Critical / High / Medium / Low)
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  7. REST API                    │  FastAPI service with /health, /predict,
│     api/main.py                 │  /predict/batch — validated, documented,
│                                 │  tested, containerised
└─────────────────────────────────┘
```

---

## API — Serving Predictions

The trained model is served as a REST API using FastAPI. Any application can send customer data and receive a churn prediction.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status, model info, threshold |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Score up to 1000 customers at once |

### Example: Predict churn for a single customer

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_revenue": 95.0, "tenure_months": 2, "contract_type": 0, "total_charges": 190.0}'
```

**Response:**
```json
{
  "churn_probability": 1.0,
  "churn_prediction": true,
  "risk_level": "critical",
  "monthly_revenue_at_risk": 95.0,
  "top_risk_factors": ["monthly_revenue", "paperless_billing", "has_partner"]
}
```

### Interactive Documentation

When the API is running locally, Swagger UI is auto-generated at [http://localhost:8000/docs](http://localhost:8000/docs) — you can test all endpoints directly in the browser.

For full API details, see [docs/api_service.md](docs/api_service.md).

---

## Project Structure

```
saas-churn-prediction/
│
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── Makefile                           ← Automation commands
├── docker-compose.yml                 ← Database container setup
├── .gitignore                         ← Files excluded from version control
│
├── api/                               ← FastAPI REST service
│   ├── __init__.py
│   ├── main.py                        ← API application (3 endpoints)
│   ├── schemas.py                     ← Pydantic request/response validation
│   ├── Dockerfile                     ← Container definition for deployment
│   └── requirements.txt               ← API-specific dependencies
│
├── notebooks/
│   ├── 01_full_pipeline.ipynb         ← Phase 1+2: data pipeline + modelling
│   └── 02_fastapi_serving.ipynb       ← Phase 3: API development + testing
│
├── src/
│   ├── __init__.py
│   ├── config.py                      ← Centralised paths and settings
│   ├── db.py                          ← Database connection helper
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py                  ← Raw data loading and SaaS reshaping
│   │   ├── generate_synthetic.py      ← Engagement and support ticket generation
│   │   └── feature_engineering.py     ← Python-side feature transforms
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                   ← Training pipeline with MLflow
│   │   ├── evaluate.py                ← Metrics, plots, SHAP analysis
│   │   └── predict.py                 ← Batch scoring and risk classification
│   │
│   └── sql/
│       ├── init.sql                   ← Database schema creation
│       └── features.sql               ← SQL feature engineering queries
│
├── tests/
│   └── test_api.py                    ← 18 automated API tests (pytest)
│
├── data/
│   ├── raw/                           ← Original dataset (see Data section below)
│   └── processed/                     ← Engineered features and scored customers
│
├── models/
│   ├── best_model.joblib              ← Serialised best model
│   ├── model_metadata.joblib          ← Metrics, threshold, revenue impact
│   ├── scaler.joblib                  ← Feature scaler (if applicable)
│   └── plots/                         ← All generated visualisations
│
├── mlruns/                            ← MLflow experiment tracking logs
│
└── docs/
    ├── data_pipeline.md               ← Data layer documentation
    ├── modelling.md                   ← Modelling approach documentation
    ├── api_service.md                 ← API service documentation
    └── architecture.md                ← Technical architecture decisions
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Google Colab account (recommended) or Jupyter Notebook

### Option 1: Run on Google Colab (Recommended)

1. Clone or copy this repository to your Google Drive
2. Open `notebooks/01_full_pipeline.ipynb` in Google Colab for the data pipeline and modelling
3. Open `notebooks/02_fastapi_serving.ipynb` in Google Colab for the API
4. Update the `PROJECT_PATH` variable in the first cell to match your Drive location:
   ```python
   PROJECT_PATH = "/content/drive/MyDrive/path/to/saas-churn-prediction"
   ```
5. Run all cells from top to bottom — each notebook handles everything

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/saas-churn-prediction.git
cd saas-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the full data + modelling pipeline
make all

# Start the API server
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000

# Open http://localhost:8000/docs for interactive Swagger UI
```

### Option 3: Run the API with Docker

```bash
docker build -t churn-api -f api/Dockerfile .
docker run -p 8000:8000 churn-api
```

### Running Tests

```bash
python -m pytest tests/test_api.py -v
```

All 18 tests should pass.

### Data

This project uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle, which is reshaped to simulate a B2B SaaS context. The raw CSV should be placed at `data/raw/telco_churn_raw.csv`. Synthetic engagement data (usage events, support tickets) is generated automatically by the pipeline to create a richer, more realistic feature set.

---

## Key Technical Decisions

**Why SQLite instead of PostgreSQL for the notebook?** The Docker Compose file includes PostgreSQL for production use, but the notebook uses SQLite so everything runs without any infrastructure setup. The SQL queries (CTEs, window functions, aggregations) are written to be compatible with both.

**Why three models instead of just XGBoost?** Comparing a simple model (Logistic Regression), a moderate one (Random Forest), and a strong one (XGBoost) demonstrates understanding of the bias-variance tradeoff and shows that the candidate doesn't blindly reach for the most complex option.

**Why threshold tuning?** The default 0.5 probability threshold is rarely optimal for business problems. By tuning the threshold based on the relative cost of false positives vs. false negatives, the system maximises business value rather than just accuracy.

**Why SHAP?** Stakeholders need to understand *why* a model makes predictions. SHAP provides mathematically grounded explanations at both the global level and individual prediction level.

**Why FastAPI over Flask?** FastAPI provides automatic request validation (Pydantic), auto-generated interactive documentation (Swagger), async support, and type hints throughout. It's the modern industry standard for ML model serving in Python.

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
| **pytest** | Automated testing |
| **matplotlib / seaborn** | Data visualisation |
| **SQLAlchemy** | Database ORM and connection management |
| **Docker** | Containerisation for API and database |
| **Make** | Build automation |
| **Git / GitHub** | Version control |

---

## Roadmap

- [x] Phase 1 — Data pipeline (ingestion, synthetic data, SQL feature engineering)
- [x] Phase 2 — Modelling (training, evaluation, SHAP, revenue impact, batch scoring)
- [x] Phase 3 — FastAPI REST API with tests, validation, and Swagger docs
- [ ] Phase 4 — Streamlit dashboard for customer success teams
- [ ] Phase 5 — Docker containerisation of the full stack
- [ ] Phase 6 — CI/CD with GitHub Actions

---

## Future Improvements

- Build interactive Streamlit dashboard for non-technical stakeholders
- Containerise the full stack (API + dashboard + database) with Docker Compose
- Add GitHub Actions CI pipeline for linting and testing
- Implement model retraining pipeline with data drift detection
- A/B testing framework for churn intervention strategies
- Add authentication and rate limiting to the API

---

## Documentation

Detailed documentation for each component is available in the `docs/` folder:

- [Data Pipeline](docs/data_pipeline.md) — How raw data becomes 45 predictive features
- [Modelling Approach](docs/modelling.md) — Model selection, evaluation, threshold tuning, SHAP
- [API Service](docs/api_service.md) — Endpoints, running locally and in Colab, validation, testing
- [Technical Architecture](docs/architecture.md) — Why the system is built the way it is

---

## Author

Built as part of a portfolio demonstrating end-to-end data science and ML engineering skills for UK graduate roles.

---

## Licence

This project is for portfolio and educational purposes. The underlying dataset is sourced from Kaggle under its terms of use.
