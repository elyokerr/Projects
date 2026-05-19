# Data Science & Machine Learning Portfolio

A curated collection of end-to-end data-science and machine-learning projects. Every project here follows the same internal structure (see [`CONTRIBUTING.md`](CONTRIBUTING.md)) so each one is easy to navigate, reproduce, and review.

---

## Projects

### [`saas-churn-prediction/`](saas-churn-prediction/) — SaaS Customer Churn Prediction Platform

An end-to-end ML system that predicts which SaaS customers are at risk of cancelling, quantifies the revenue exposure, and surfaces the findings through a REST API and an interactive dashboard. The full stack runs with a single `docker compose up`.

| Highlight | Value |
|---|---|
| Annual revenue at risk identified | **£64,000+** |
| Recall on churners | **~82%** |
| Engineered features | **45** (via SQL + Python) |
| Interfaces | Streamlit dashboard, FastAPI service, MLflow tracking |

**Stack:** Python · pandas · scikit-learn · XGBoost · MLflow · SHAP · FastAPI · Streamlit · PostgreSQL · Docker Compose · GitHub Actions CI

➡ [Open project](saas-churn-prediction/)

---

## How this repo is organised

```
Projects/
├── README.md              ← You are here
├── CONTRIBUTING.md        ← The standard every project follows
├── LICENSE
├── .gitignore             ← Repo-wide ignores (data/, mlruns/, etc.)
│
├── _template/             ← Skeleton for new projects
│
└── saas-churn-prediction/
```

Every project folder uses the same internal layout:

```
<project>/
├── README.md          ← 9-section overview (hero results → limitations)
├── requirements.txt
├── notebooks/         ← Numbered EDA & modelling notebooks
├── src/               ← Reusable Python modules (data, features, models, utils)
├── data/              ← raw/, interim/, processed/, external/ (data gitignored)
├── models/            ← Serialised trained artifacts (.joblib, .pkl)
├── reports/figures/   ← Generated plots & screenshots
├── app/               ← Optional Streamlit / FastAPI / dashboard
├── tests/             ← Pytest tests
└── docs/              ← Extended documentation (optional)
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full convention and the README template used by every project.

---

## About

These projects are part of an ongoing portfolio focused on **practical, production-style data science** — not just model training, but the surrounding work that makes a model usable: clean data pipelines, honest validation (group splits where it matters), explainability (SHAP), business-framed metrics, and interactive deliverables a stakeholder can actually open.

New projects are added regularly. Each one is self-contained, documented end-to-end, and follows the conventions above so the whole repo stays scalable.

---

## Licence

This repository is shared for portfolio and educational purposes. Individual projects may use third-party datasets under their respective terms — see each project's README for data attribution.
