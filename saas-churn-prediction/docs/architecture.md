# System Architecture

This document explains how the platform fits together, what each layer is responsible for, and why I made the design decisions I did.

## High-Level View

The project is organized as three loosely coupled layers stacked on top of a shared database:

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                      │
│   FastAPI service              Streamlit dashboard           │
│   (REST API, port 8000)        (interactive UI, port 8501)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                       Modeling Layer                         │
│   Training pipeline · MLflow tracking · SHAP explainability  │
│   Threshold optimization · Batch scoring                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                            │
│   Raw ingestion · Synthetic generation · SQL features        │
│   PostgreSQL (Docker) / SQLite (local development)           │
└─────────────────────────────────────────────────────────────┘
```

The layers communicate through artifacts written to disk and tables in the database, not through in-memory function calls. That separation matters because it means each layer can be developed, tested, and operated independently. The training pipeline can run once a week on a beefy machine while the API runs continuously on a small one, and neither has to know about the other.

## Layer Responsibilities

### Data Layer

The data layer owns everything from raw CSV through to the analytics-ready feature set:

- `src/data/ingest.py` reads the Telco Churn CSV and reshapes it into a relational schema (`customers`, `subscriptions`, `product_usage`) using SaaS-appropriate column names
- `src/data/generate_synthetic.py` populates `usage_events` and `support_tickets` with behavioral data that's correlated with the existing churn label
- `src/data/feature_engineering.py` runs `src/sql/features.sql` and adds a handful of Python-computed features, producing a 45-column `feature_store` table

The data layer's only output is the feature store. The modeling layer doesn't know or care where the features come from, only that they exist in a known location with a known schema.

### Modeling Layer

The modeling layer reads from the feature store and writes serialized model artifacts to `models/`:

- `src/models/train.py` trains three candidate models, tracks runs in MLflow, picks the winner, optimizes the decision threshold for F1, and writes `best_model.joblib`, `scaler.joblib`, and `model_metadata.joblib`
- `src/models/evaluate.py` produces evaluation plots and SHAP explanations
- `src/models/predict.py` provides both single-customer and batch scoring, writing `scored_customers.csv` for the dashboard

Model metadata is the contract between training and serving. It contains the feature names in the exact order the model expects, the optimal threshold, and the business impact estimates. Anything downstream that reads metadata gets a consistent view of how the model should be used.

### Presentation Layer

Two services that consume the model:

- The FastAPI service (`api/`) wraps the model in a REST API with input validation, business-friendly risk tiers, and revenue exposure calculations
- The Streamlit dashboard (`dashboard/`) reads `scored_customers.csv` and `model_metadata.joblib` directly and renders interactive charts, KPIs, and a customer drilldown

The two presentation surfaces are deliberately independent. The API is for system-to-system integration (CRM, marketing automation, internal tools). The dashboard is for humans. Neither calls the other.

## Key Design Decisions

### Two database backends

The same codebase runs against SQLite (local development) and PostgreSQL (Docker stack). The `USE_POSTGRES` environment variable switches between them in `src/config.py`. SQL is written to standard CTEs that work on both engines.

I chose this approach because asking a reviewer to install PostgreSQL just to look at a portfolio project would have been friction with no upside. Local development with SQLite is a `pip install` and three commands. The PostgreSQL path demonstrates that the code is production-aware.

### MLflow for experiment tracking

Even for a single-developer project, MLflow is worth the small overhead. It captures every training run with parameters, metrics, and the serialized model. When I wanted to compare a tweaked hyperparameter configuration against the previous one, the comparison was free.

For a production deployment, MLflow's model registry would become a more significant part of the workflow - promoting models from staging to production, rolling back when needed. This project doesn't exercise that machinery but is structured so it could.

### Threshold optimization, not default 0.5

The default sklearn threshold of 0.5 is rarely the right choice for business problems. Choosing 0.5 implicitly says false positives and false negatives have equal cost, which is almost never true in retention contexts.

I optimize for F1 on the test set as a sensible default. A real deployment would let the business specify the cost ratio (a customer success team member costs X per intervention, an average customer is worth Y if retained) and compute the threshold from those numbers.

### Risk tiers, not just probabilities

The API and dashboard both report a `risk_level` (critical / high / medium / low) in addition to the raw probability. This isn't just a UI nicety. Customer success workflows are typically organized by tier - critical accounts get a phone call within 24 hours, high-risk accounts get a templated email, medium-risk get included in monthly health-check newsletters. The model output should line up with the workflow, not force the team to translate from probabilities.

### Synthetic data, with documented assumptions

The original Kaggle dataset has no behavioral data - no logins, no support tickets. Rather than ship a model trained on a dataset that doesn't reflect how a real SaaS company would think about churn, I generate synthetic engagement data that's plausibly correlated with the existing churn label.

This is documented prominently in `docs/data_pipeline.md` because it's the most important caveat about the modeling results. The model is learning real patterns - tenure, contract type, monthly revenue - and synthetic patterns that I deliberately built in. The performance numbers reflect both.

## Service Communication

The three containers talk to each other only at the edges:

- `postgres` exposes port 5432; the API connects to it via SQLAlchemy
- `api` exposes port 8000; nothing in the stack consumes the API at runtime (the dashboard reads files directly)
- `dashboard` exposes port 8501

The dashboard's independence from the API is intentional. Streamlit can be deployed to environments where the API isn't reachable (corporate Streamlit Cloud, internal analytics portals) by including the latest `scored_customers.csv` as part of the build artifact.

## Reproducibility

Every random source in the project is seeded:

- `RANDOM_STATE = 42` in `src/config.py` is used for the train/test split, cross-validation, model initialization, and synthetic data generation
- The same seed plus the same input data produces the same outputs - useful for debugging, less useful in production where weekly retraining would override determinism

The CI pipeline runs the test suite on a clean checkout on every push, which catches any reproducibility regressions early.

## Where I'd Take It Next

If this were a real production system, the highest-leverage next steps would be:

- **Authentication and rate limiting on the API.** A simple API key check is a 30-minute task; a proper OAuth integration is a project on its own.
- **Per-prediction SHAP values.** The API currently returns a feature-magnitude approximation for the top risk factors. SHAP values would give a real per-prediction explanation, at the cost of about 50ms per request.
- **Drift monitoring.** Schedule weekly recomputation of the feature distribution and compare against the training distribution. Alert when KL divergence on any feature exceeds a threshold.
- **A/B testing framework for interventions.** The model identifies who to contact, but it doesn't know which intervention will work. An experiment framework would close the loop.
