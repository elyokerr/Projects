# API Service

A FastAPI service that exposes the trained churn model as a REST API. It validates incoming requests with Pydantic, runs predictions, and returns probabilities together with business-friendly risk levels and revenue exposure estimates.

## Why FastAPI

I reached for FastAPI for a few concrete reasons:

- Type hints become input validation automatically through Pydantic
- Swagger UI and ReDoc are generated for free from the same type hints
- Async support is built in, though this project doesn't need it
- Performance is competitive with Go and Node.js for IO-bound workloads

For a project where the model itself is the main attraction, the framework should get out of the way. FastAPI does.

## Endpoints

The service exposes three endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service status and model info |
| `POST` | `/predict` | Single-customer prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 1000 customers) |

Interactive documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc) when the service is running.

### Health check

```bash
curl http://localhost:8000/health
```

Returns:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "logistic_regression",
  "optimal_threshold": 0.4023,
  "version": "1.0.0"
}
```

Used by load balancers and orchestrators to decide whether the service is ready to handle traffic. The Docker Compose configuration uses it to sequence container startup - the dashboard waits for the API to report healthy before starting.

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

Returns:

```json
{
  "churn_probability": 0.8742,
  "churn_prediction": true,
  "risk_level": "critical",
  "monthly_revenue_at_risk": 95.00,
  "top_risk_factors": [
    "monthly_revenue",
    "total_charges",
    "tenure_months",
    "support_tickets_total",
    "contract_type"
  ]
}
```

The four required fields are `monthly_revenue`, `tenure_months`, `contract_type`, and `total_charges`. Everything else has a sensible default in the Pydantic schema, so the API can be called with minimal information when richer behavioral data isn't available.

### Batch prediction

`POST /predict/batch` accepts up to 1000 customers per request. The 1000 cap is enforced at the schema level with Pydantic's `max_length` validator.

The response includes individual predictions plus aggregate statistics:

```json
{
  "predictions": [...],
  "summary": {
    "total_customers_scored": 250,
    "high_risk_customers": 47,
    "total_monthly_revenue_at_risk": 4521.30,
    "estimated_annual_revenue_at_risk": 54255.60
  }
}
```

The aggregate is what most consumers actually care about. The individual predictions are there for when the consumer needs to act on specific customers (push to CRM, generate intervention tasks).

## Input Validation

Every request is validated by Pydantic before the prediction code runs. The schema enforces:

- All required fields must be present
- Numeric fields must be non-negative where that makes sense (revenue, ticket counts)
- Categorical fields are bounded (`contract_type` is 0-2, `payment_method` is 0-3)
- Probability-like fields are in [0, 1] (`feature_adoption_rate`)
- Batch requests must contain at least one customer and no more than 1000

Failed validation returns HTTP 422 with a structured error payload pointing at exactly which field failed and why. This is FastAPI's default behavior and it's better than anything I could write by hand.

## Risk Tiers

The API converts raw probabilities into four business-friendly tiers:

| Tier | Probability range |
|---|---|
| `critical` | ≥ 0.80 |
| `high` | 0.60 - 0.80 |
| `medium` | 0.40 - 0.60 |
| `low` | < 0.40 |

These tiers exist because customer success workflows are typically organized by tier - critical accounts get a phone call within 24 hours, high-risk accounts get a templated email, medium-risk get included in monthly health-check newsletters. The model output should match the workflow, not force the team to translate from probabilities.

The thresholds are deliberately different from the F1-optimized threshold the model uses for binary classification. The binary threshold is about whether to act at all; the tiers are about how urgently to act.

## Top Risk Factors

The API returns up to five "top risk factors" per prediction. The current implementation ranks features by absolute value - the features with the largest magnitudes after preprocessing. This is a quick approximation that produces sensible-looking output.

A better implementation would use per-prediction SHAP values, which would tell the consumer exactly which features pushed this specific prediction toward churn. That's noted as a planned improvement in `docs/architecture.md`. The cost would be about 50ms per request, which is acceptable for retention workflows but not for high-throughput inference.

## Lifespan and Model Loading

The model is loaded once at startup via FastAPI's lifespan handler in `api/main.py`. The artifacts (model, scaler, metadata) are bound to module-level globals and shared across every request. This pattern is appropriate for read-only state and keeps per-request latency low.

If the model fails to load - for example, because the training pipeline hasn't been run yet - the API still starts but `/health` will report `model_loaded: false` and the prediction endpoints will return HTTP 503. This is a deliberate choice: the service should come up so that operators can check `/health` and see what's wrong, rather than crashing on startup with no way to investigate.

## Testing

The test suite uses FastAPI's `TestClient`, which makes in-process HTTP requests against the application without spinning up a real server. There are 18 tests across four categories:

| Category | Tests |
|---|---|
| Health endpoint | 3 |
| Single prediction | 7 |
| Batch endpoint | 4 |
| Input validation | 4 |

Run them with:

```bash
make test
```

The `TestClient` fixture in `tests/test_api.py` uses the context-manager form, which triggers the lifespan handler so the model loads correctly. Without that the prediction endpoints would return 503 in tests.

The CI pipeline runs the same tests on every push, so anything that breaks the API contract gets caught before it lands on `main`.

## Running Locally

Without Docker:

```bash
make api
```

Or directly:

```bash
uvicorn api.main:app --reload --port 8000
```

The `--reload` flag watches the source files and restarts the server when anything changes, which speeds up the development loop.

With Docker:

```bash
docker compose up api
```

The Docker setup includes a health check that polls `/health` every 15 seconds. The dashboard container depends on the API health check, so the stack starts up in a sensible order.

## CORS

The API is configured with permissive CORS settings - any origin can call any endpoint. This is appropriate for a portfolio project where the friction of restrictive CORS would outweigh the benefit. A production deployment would narrow `allow_origins` to the specific origins that should be allowed (the dashboard's domain, internal tools, etc.).

## Future Work

A few things I'd add for production use:

- **Authentication.** A simple API key check via FastAPI's dependency injection would take 30 minutes. A proper OAuth integration would be a project on its own.
- **Rate limiting.** Per-key rate limits via `slowapi` or an upstream API gateway like Kong.
- **Request logging.** Structured logs of every request and response, ideally to a log aggregator like CloudWatch or Datadog.
- **Async batch processing.** For very large batches, an async endpoint that accepts the batch, returns a job ID, and writes results to S3 would scale better than holding 1000 customers in memory.
