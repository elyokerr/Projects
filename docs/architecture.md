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
│              notebooks/01_full_pipeline.ipynb                │
│         (Orchestration, EDA, visualisation, narrative)        │
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

---

## Dependencies and Constraints

**Hardware requirements:** The entire pipeline runs on a basic laptop or Google Colab free tier. No GPU is needed — all models are tabular and train in seconds.

**External dependencies:** Only standard Python data science libraries (pandas, scikit-learn, XGBoost, SHAP, MLflow, matplotlib, seaborn, SQLAlchemy). All are installable via pip with no system-level dependencies.

**Data dependencies:** The only external input is the Telco Customer Churn CSV from Kaggle. Everything else is generated by the pipeline.

---

## Planned Architecture Extensions

The following components are planned for Phases 3–6:

**FastAPI endpoint** (`api/main.py`) — A REST API that accepts customer features and returns churn predictions with SHAP explanations in real time. This will demonstrate API development, input validation with Pydantic, and model serving.

**Streamlit dashboard** (`dashboard/app.py`) — An interactive web application for customer success teams to explore churn risk across the customer base, drill into individual customers, and understand the revenue impact. This will demonstrate data visualisation, stakeholder communication, and product thinking.

**Docker Compose** — The full stack (database + API + dashboard) will be containerised so the entire system can be started with a single `docker-compose up` command.

**GitHub Actions CI** — Automated linting (ruff/flake8) and testing on every push, demonstrating CI/CD awareness.
