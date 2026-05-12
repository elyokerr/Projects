# SaaS Customer Churn Prediction Platform

> An end-to-end machine learning system that predicts which SaaS customers are at risk of cancelling their subscriptions, estimates the revenue at risk, and provides actionable insights for customer success teams.

---

## The Business Problem

Subscription-based software companies lose significant revenue when customers cancel — a process known as **churn**. The cost of acquiring a new customer is typically 5–7× higher than retaining an existing one, which makes early identification of at-risk customers one of the most impactful things a data team can do.

This project builds a complete churn prediction system — not just a model, but the full pipeline a real data team would deliver: a structured database, engineered features, trained and evaluated models, explainable predictions, and revenue impact analysis.

**Key results from this system:**

- Identified **£64,000+** in annual recurring revenue at risk across the customer base
- Achieved **82% recall** — the model catches 4 out of every 5 customers who would actually churn
- Engineered **45 predictive features** from raw data using SQL and Python
- Every prediction is explainable — the system tells you *why* a customer is flagged, not just *that* they are

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
| **Business Thinking** | Threshold tuning based on business cost, revenue impact estimation, risk-level scoring |
| **Software Engineering** | Modular codebase with separate concerns, configuration management, reproducible pipeline |
| **Containerisation** | Docker Compose setup for database infrastructure |
| **Automation** | Makefile for one-command pipeline execution |

---

## How It Works

The system follows a standard industry ML pipeline with six stages:

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
└─────────────────────────────────┘
```

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
├── notebooks/
│   └── 01_full_pipeline.ipynb         ← Interactive notebook (runs the full pipeline)
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
├── data/
│   ├── raw/                           ← Original dataset (see Data section below)
│   └── processed/                     ← Engineered features and scored customers
│
├── models/
│   ├── best_model.joblib              ← Serialised best model
│   ├── model_metadata.joblib          ← Metrics, threshold, revenue impact
│   └── plots/                         ← All generated visualisations
│
├── mlruns/                            ← MLflow experiment tracking logs
│
└── docs/
    ├── data_pipeline.md               ← Data layer documentation
    ├── modelling.md                    ← Modelling approach documentation
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
2. Open `notebooks/01_full_pipeline.ipynb` in Google Colab
3. Update the `PROJECT_PATH` variable in the first cell to match your Drive location:
   ```python
   PROJECT_PATH = "/content/drive/MyDrive/path/to/saas-churn-prediction"
   ```
4. Run all cells from top to bottom — the notebook handles everything

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/saas-churn-prediction.git
cd saas-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
make all
# Or run each step individually:
# python -m src.data.ingest
# python -m src.data.generate_synthetic
# python -m src.data.feature_engineering
# python -m src.models.train
# python -m src.models.evaluate
# python -m src.models.predict
```

### Data

This project uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle, which is reshaped to simulate a B2B SaaS context. The raw CSV should be placed at `data/raw/telco_churn_raw.csv`. Synthetic engagement data (usage events, support tickets) is generated automatically by the pipeline to create a richer, more realistic feature set.

---

## Key Technical Decisions

**Why SQLite instead of PostgreSQL for the notebook?** The Docker Compose file includes PostgreSQL for production use, but the notebook uses SQLite so everything runs without any infrastructure setup. The SQL queries (CTEs, window functions, aggregations) are written to be compatible with both — the same `features.sql` works on either database engine.

**Why three models instead of just XGBoost?** Comparing a simple model (Logistic Regression), a moderate one (Random Forest), and a strong one (XGBoost) demonstrates understanding of the bias-variance tradeoff and shows that the candidate doesn't blindly reach for the most complex option. MLflow tracks every experiment so results are reproducible.

**Why threshold tuning?** The default 0.5 probability threshold is rarely optimal for business problems. By tuning the threshold based on the relative cost of false positives vs. false negatives, the system maximises business value rather than just accuracy.

**Why SHAP?** In industry, stakeholders need to understand *why* a model makes predictions. SHAP provides mathematically grounded explanations at both the global level (which features matter most overall) and the individual level (why this specific customer was flagged).

---

## Results Summary

| Metric | Value |
|---|---|
| Best Model | XGBoost (or Logistic Regression depending on run) |
| Recall (churners caught) | ~82% |
| Precision | ~65% |
| F1 Score | ~0.72 |
| Total MRR at Risk | £5,300+ monthly |
| Estimated Annual Savings | £64,000+ |
| Features Engineered | 45 |
| Risk Levels | Critical / High / Medium / Low |

*Note: exact numbers may vary slightly between runs due to random seeds in synthetic data generation.*

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
| **matplotlib / seaborn** | Data visualisation |
| **SQLAlchemy** | Database ORM and connection management |
| **Docker Compose** | Container orchestration for database |
| **Make** | Build automation |
| **Git / GitHub** | Version control |

---

---

## Licence

This project is for portfolio and educational purposes. The underlying dataset is sourced from Kaggle under its terms of use.
