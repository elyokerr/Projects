# API Service Documentation

> How the trained churn prediction model is served as a REST API, making it accessible to any application, dashboard, or service.

---

## What This Is

Phase 3 transforms the trained model from Phase 2 into a **REST API** — a web service that accepts customer data over HTTP and returns churn predictions. Instead of opening a notebook and running code manually, any application (a website, a CRM system, a mobile app, a dashboard) can now send a request and get a prediction back in milliseconds.

This is the standard way ML models are deployed in industry. The model sits behind an API, and other systems talk to it.

---

## How It Works

```
Client (browser, app, dashboard, script)
      │
      │  HTTP Request (customer data as JSON)
      ▼
┌─────────────────────────────────┐
│        FastAPI Service           │
│                                  │
│  1. Validate input (Pydantic)    │
│  2. Load features into model     │
│  3. Run prediction               │
│  4. Classify risk level          │
│  5. Return JSON response         │
└─────────────────────────────────┘
      │
      │  HTTP Response (prediction as JSON)
      ▼
Client receives:
  - churn probability (0-1)
  - binary prediction (churn / stay)
  - risk level (critical / high / medium / low)
  - revenue at risk (£)
  - top risk factors
```

---

## Endpoints

### `GET /health` — Is the service alive?

Every production API has a health check. Load balancers, monitoring tools, and container orchestrators (Docker, Kubernetes) use this to know whether the service is alive and the model is loaded.

**Example response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "logistic_regression",
  "optimal_threshold": 0.7602,
  "version": "1.0.0"
}
```

### `POST /predict` — Score a single customer

The core endpoint. Send one customer's features, get back a complete churn risk assessment.

**Example request:**
```json
{
  "monthly_revenue": 79.85,
  "tenure_months": 48,
  "contract_type": 2,
  "total_charges": 3832.80,
  "payment_method": 3,
  "login_frequency": 12.0,
  "feature_adoption_rate": 0.85,
  "engagement_score": 82.0,
  "support_tickets_total": 4,
  "support_tickets_last_90d": 0
}
```

**Example response:**
```json
{
  "churn_probability": 1.0,
  "churn_prediction": true,
  "risk_level": "critical",
  "monthly_revenue_at_risk": 79.85,
  "top_risk_factors": ["monthly_revenue", "has_dependents", "has_partner"]
}
```

**What each field means:**

- **churn_probability** — The model's confidence (0 to 1) that this customer will cancel. Higher = more likely to churn.
- **churn_prediction** — Whether the probability exceeds the optimised business threshold. `true` = the model recommends intervention.
- **risk_level** — A human-readable classification: critical (≥0.8), high (0.6–0.8), medium (0.4–0.6), or low (<0.4).
- **monthly_revenue_at_risk** — If the customer is predicted to churn, this shows how much monthly revenue the company stands to lose. If not predicted to churn, this is £0.
- **top_risk_factors** — The features contributing most to this prediction, giving the customer success team actionable insight into *why* the customer is at risk.

### `POST /predict/batch` — Score up to 1000 customers

For bulk scoring — used in nightly batch runs, CRM integrations, or feeding a dashboard. Returns individual predictions plus an aggregate summary.

**Example response (summary section):**
```json
{
  "summary": {
    "total_customers_scored": 20,
    "high_risk_customers": 18,
    "total_monthly_revenue_at_risk": 1186.09,
    "estimated_annual_revenue_at_risk": 14233.08
  }
}
```

---

## Running the API

There are two ways to run and test the API.

### Option 1: Run Locally (Windows/Mac/Linux)

This gives you the full experience including the interactive Swagger UI in your browser.

**Step 1 — Open a terminal in the project root folder:**
```bash
cd saas-churn-prediction
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r api/requirements.txt
```

**Step 3 — Start the server:**
```bash
uvicorn api.main:app --reload
```

You should see:
```
Uvicorn running on http://127.0.0.1:8000
```

**Step 4 — Open the interactive API documentation:**

Open your browser and go to:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) — Interactive testing interface where you can fill in fields and hit "Execute"
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) — Clean, readable documentation view
- **Health check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) — Quick JSON check

**Troubleshooting (common issues):**

If the page doesn't load (`ERR_CONNECTION_REFUSED`), the server isn't running. Check the terminal for errors.

If you see a `ModuleNotFoundError`, you're probably running `uvicorn` from the wrong directory. You **must** run it from the project root (`saas-churn-prediction/`), not from inside `api/` or `src/`.

To verify the import works:
```python
from api.main import app
print(app)
```

If this errors, your project structure is missing an `__init__.py` file or the folder layout is incorrect.

### Option 2: Run in Google Colab (via TestClient)

Since Colab can't expose a persistent web server to a browser, we use FastAPI's **TestClient** — it simulates HTTP requests without needing a running server. This is the same approach used for automated testing.

**Step 1 — Open `notebooks/02_fastapi_serving.ipynb` in Google Colab**

**Step 2 — Run all cells from top to bottom.** The notebook:
1. Mounts Google Drive and navigates to the project
2. Installs dependencies
3. Creates a TestClient
4. Calls all three endpoints and prints the results
5. Demonstrates input validation
6. Runs the full 18-test pytest suite

The TestClient approach is how professional teams write API tests — it's not a workaround, it's best practice.

---

## Input Validation

One of FastAPI's strongest features is automatic input validation via Pydantic. Every request is checked before the model ever sees it.

**Missing required fields** → 422 error with a clear message:
```json
{"detail": [{"loc": ["body", "monthly_revenue"], "msg": "Field required"}]}
```

**Invalid values** → 422 error explaining the constraint:
- Negative revenue → rejected
- Contract type outside 0-2 → rejected
- Empty batch → rejected

This prevents garbage-in-garbage-out predictions in production.

---

## Automated Tests

The `tests/test_api.py` file contains 18 structured tests across four categories:

- **Health check tests** (3) — Service responds, correct structure, model is loaded
- **Single prediction tests** (7) — Returns 200, correct response shape, probability in valid range, risk level valid, revenue logic correct, high-risk customer detected, risk factors returned
- **Batch prediction tests** (4) — Returns 200, correct structure, summary stats present, output count matches input
- **Validation tests** (4) — Missing fields rejected, negative values rejected, invalid enum values rejected, empty batch rejected

Run them with:
```bash
cd saas-churn-prediction
python -m pytest tests/test_api.py -v
```

All 18 tests pass.

---

## File Structure

```
api/
├── __init__.py          ← Makes api/ a Python package
├── main.py              ← FastAPI application (endpoints, model loading, prediction logic)
├── schemas.py           ← Pydantic models (request/response data contracts)
├── Dockerfile           ← Container definition for deployment
└── requirements.txt     ← API-specific dependencies

tests/
└── test_api.py          ← 18 automated tests for all endpoints

notebooks/
└── 02_fastapi_serving.ipynb  ← Interactive Colab notebook demonstrating the API
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| **FastAPI** | Modern async Python API framework — industry standard for ML serving |
| **Pydantic** | Request/response validation with automatic type checking |
| **Swagger / OpenAPI** | Auto-generated interactive API documentation |
| **pytest** | Automated testing with FastAPI's TestClient |
| **uvicorn** | ASGI server to run the API |
| **Docker** | Containerisation for deployment |
