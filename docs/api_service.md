# API Service Documentation

## Overview

The FastAPI prediction service exposes the trained churn model as a REST API. It accepts customer features, validates them against Pydantic schemas, runs the model, and returns a structured prediction with churn probability, risk level, revenue at risk, and the top contributing features.

## How It Works

```
Client Request (JSON)
    │
    ▼
FastAPI receives request
    │
    ▼
Pydantic validates input ──→ 422 error if invalid
    │
    ▼
Features transformed to match training format
    │
    ▼
Model predicts churn probability
    │
    ▼
Risk level classified (low/medium/high/critical)
    │
    ▼
Response returned (JSON)
```

---

## Endpoints

### `GET /health`

Checks whether the API is running and the model is loaded. Used by monitoring systems, load balancers, and container orchestrators.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "logistic_regression",
  "optimal_threshold": 0.7602,
  "version": "1.0.0"
}
```

### `POST /predict`

Accepts a single customer's features and returns a churn prediction.

**Request body (minimum required fields):**
```json
{
  "monthly_revenue": 79.85,
  "tenure_months": 48,
  "contract_type": 2,
  "total_charges": 3832.80
}
```

All other features have sensible defaults and are optional.

**Response:**
```json
{
  "churn_probability": 0.23,
  "churn_prediction": false,
  "risk_level": "low",
  "monthly_revenue_at_risk": 0.0,
  "top_risk_factors": ["monthly_revenue", "has_dependents", "has_partner"]
}
```

**Response fields:**

| Field | Description |
|---|---|
| `churn_probability` | Model confidence the customer will churn (0–1) |
| `churn_prediction` | Whether probability exceeds the optimised threshold |
| `risk_level` | Business category: `critical` (≥0.8), `high` (≥0.6), `medium` (≥0.4), `low` (<0.4) |
| `monthly_revenue_at_risk` | Revenue exposure if the customer churns (£0 if predicted to stay) |
| `top_risk_factors` | Top 5 features contributing to the prediction |

### `POST /predict/batch`

Scores up to 1000 customers in a single request and returns both individual predictions and aggregate statistics.

**Request body:**
```json
{
  "customers": [
    {"monthly_revenue": 79.85, "tenure_months": 48, "contract_type": 2, "total_charges": 3832.80},
    {"monthly_revenue": 95.00, "tenure_months": 2, "contract_type": 0, "total_charges": 190.00}
  ]
}
```

**Response includes:**
```json
{
  "predictions": [...],
  "summary": {
    "total_customers_scored": 2,
    "high_risk_customers": 1,
    "total_monthly_revenue_at_risk": 95.00,
    "estimated_annual_revenue_at_risk": 1140.00
  }
}
```

---

## Input Validation

Pydantic schemas automatically reject invalid requests with a 422 status code and a clear error message:

| Invalid Input | Result |
|---|---|
| Missing `monthly_revenue` (required field) | `422: Field required` |
| Negative revenue (`-50`) | `422: Input should be greater than or equal to 0` |
| Invalid contract type (`99`) | `422: Input should be 0, 1, or 2` |
| Empty batch (`{"customers": []}`) | `422: List should have at least 1 item` |

This prevents garbage-in-garbage-out predictions in production.

---

## Running the API

### Locally

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

Interactive Swagger docs at `http://localhost:8000/docs`. ReDoc at `http://localhost:8000/redoc`.

**Important:** Run `uvicorn` from the project root (`saas-churn-prediction/`), not from inside the `api/` folder.

### In Google Colab

The notebook `02_fastapi_serving.ipynb` uses FastAPI's `TestClient` to test all endpoints without running a server. This simulates real HTTP requests in-process.

```python
from fastapi.testclient import TestClient
from api.main import app

with TestClient(app) as client:
    response = client.post("/predict", json=customer_data)
    print(response.json())
```

---

## Automated Tests

18 tests in `tests/test_api.py` covering four categories:

| Category | Tests | What's Verified |
|---|---|---|
| Health Endpoint | 3 | Returns 200, correct structure, model is loaded |
| Predict Endpoint | 6 | Returns 200, correct fields, probability 0–1, valid risk level, revenue logic, risk factors |
| Batch Endpoint | 4 | Returns 200, correct structure, summary stats, count matches input |
| Input Validation | 5 | Missing fields rejected, negative values rejected, invalid enums rejected, empty batch rejected |

Run with:
```bash
python -m pytest tests/test_api.py -v
```

---

## Technologies

| Technology | Purpose |
|---|---|
| **FastAPI** | Modern async Python API framework — industry standard for ML serving |
| **Pydantic** | Request/response validation with automatic type checking |
| **Swagger / OpenAPI** | Auto-generated interactive API documentation |
| **pytest** | Automated testing with FastAPI's TestClient |
| **uvicorn** | ASGI server to run the API |
| **Docker** | Containerisation for deployment |
