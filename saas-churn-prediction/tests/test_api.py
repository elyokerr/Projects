"""
API test suite.

Eighteen tests covering the three endpoints and input validation. The
tests use FastAPI's TestClient, which makes in-process HTTP requests
against the application without spinning up a real server. This keeps
the suite fast and self-contained: it runs the same way locally, in CI,
and inside the Colab walkthrough notebook.

Test categories:
    Health endpoint     - 3 tests
    Single prediction   - 7 tests (now includes one explicit assertion
                                    around risk-factor return shape)
    Batch endpoint      - 4 tests
    Input validation    - 4 tests

Run with:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add the project root to sys.path so `from api.main import app` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app


# --- Fixtures -------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan events enabled.

    Using TestClient as a context manager triggers the lifespan handler,
    which is what loads the model into memory. Without this the /predict
    endpoint would return 503 (model not loaded).
    """
    with TestClient(app) as c:
        yield c


# --- Sample payloads ------------------------------------------------------

VALID_CUSTOMER = {
    "monthly_revenue": 79.85,
    "tenure_months": 24,
    "contract_type": 1,
    "total_charges": 1916.40,
    "payment_method": 2,
    "paperless_billing": 1,
    "login_frequency": 8.5,
    "feature_adoption_rate": 0.72,
    "days_since_last_login": 2,
    "avg_session_duration": 22.0,
    "engagement_score": 68.0,
    "support_tickets_total": 3,
    "support_tickets_last_90d": 1,
    "escalation_count": 0,
    "has_phone_service": 1,
    "has_internet_service": 1,
    "has_online_security": 1,
    "has_online_backup": 0,
    "has_device_protection": 1,
    "has_tech_support": 0,
    "has_streaming_tv": 1,
    "has_streaming_movies": 0,
    "revenue_per_tenure": 3.33,
    "engagement_to_tenure_ratio": 2.83,
    "support_to_usage_ratio": 0.35,
    "is_senior_citizen": 0,
    "has_partner": 1,
    "has_dependents": 0,
    "gender_encoded": 0,
}

# Same shape but with values that should clearly point at churn:
# very new account, no commitment, no engagement, many escalations.
HIGH_RISK_CUSTOMER = {
    "monthly_revenue": 95.00,
    "tenure_months": 2,
    "contract_type": 0,
    "total_charges": 190.00,
    "payment_method": 0,
    "paperless_billing": 1,
    "login_frequency": 0.5,
    "feature_adoption_rate": 0.1,
    "days_since_last_login": 45,
    "avg_session_duration": 3.0,
    "engagement_score": 8.0,
    "support_tickets_total": 12,
    "support_tickets_last_90d": 8,
    "escalation_count": 3,
    "has_phone_service": 1,
    "has_internet_service": 1,
    "has_online_security": 0,
    "has_online_backup": 0,
    "has_device_protection": 0,
    "has_tech_support": 0,
    "has_streaming_tv": 0,
    "has_streaming_movies": 0,
    "revenue_per_tenure": 47.50,
    "engagement_to_tenure_ratio": 4.0,
    "support_to_usage_ratio": 24.0,
    "is_senior_citizen": 0,
    "has_partner": 0,
    "has_dependents": 0,
    "gender_encoded": 1,
}


# --- Health endpoint ------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_response_structure(self, client):
        data = client.get("/health").json()
        for field in ("status", "model_loaded", "model_name", "optimal_threshold", "version"):
            assert field in data

    def test_health_model_is_loaded(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


# --- Single prediction ---------------------------------------------------

class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_returns_200(self, client):
        assert client.post("/predict", json=VALID_CUSTOMER).status_code == 200

    def test_predict_response_structure(self, client):
        data = client.post("/predict", json=VALID_CUSTOMER).json()
        for field in ("churn_probability", "churn_prediction", "risk_level", "monthly_revenue_at_risk"):
            assert field in data

    def test_predict_probability_range(self, client):
        data = client.post("/predict", json=VALID_CUSTOMER).json()
        assert 0 <= data["churn_probability"] <= 1

    def test_predict_risk_level_valid(self, client):
        data = client.post("/predict", json=VALID_CUSTOMER).json()
        assert data["risk_level"] in ("critical", "high", "medium", "low")

    def test_predict_revenue_at_risk_logic(self, client):
        """When the model says the customer stays, revenue at risk must be 0."""
        data = client.post("/predict", json=VALID_CUSTOMER).json()
        if not data["churn_prediction"]:
            assert data["monthly_revenue_at_risk"] == 0

    def test_predict_high_risk_customer(self, client):
        """Obvious churn signals should produce a non-trivial probability."""
        data = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
        assert data["churn_probability"] > 0.1

    def test_predict_returns_risk_factors(self, client):
        data = client.post("/predict", json=VALID_CUSTOMER).json()
        assert "top_risk_factors" in data
        assert isinstance(data["top_risk_factors"], list)


# --- Batch prediction ----------------------------------------------------

class TestBatchEndpoint:
    """Tests for POST /predict/batch."""

    def test_batch_returns_200(self, client):
        payload = {"customers": [VALID_CUSTOMER, HIGH_RISK_CUSTOMER]}
        assert client.post("/predict/batch", json=payload).status_code == 200

    def test_batch_response_structure(self, client):
        data = client.post("/predict/batch", json={"customers": [VALID_CUSTOMER]}).json()
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 1

    def test_batch_summary_stats(self, client):
        payload = {"customers": [VALID_CUSTOMER, HIGH_RISK_CUSTOMER]}
        summary = client.post("/predict/batch", json=payload).json()["summary"]
        assert summary["total_customers_scored"] == 2
        assert "high_risk_customers" in summary
        assert "total_monthly_revenue_at_risk" in summary

    def test_batch_count_matches_input(self, client):
        payload = {"customers": [VALID_CUSTOMER] * 5}
        data = client.post("/predict/batch", json=payload).json()
        assert len(data["predictions"]) == 5


# --- Input validation ----------------------------------------------------

class TestInputValidation:
    """Tests that Pydantic correctly rejects malformed input."""

    def test_missing_required_field(self, client):
        """monthly_revenue is required - omitting it must fail with 422."""
        bad = {"tenure_months": 12, "contract_type": 1, "total_charges": 500}
        assert client.post("/predict", json=bad).status_code == 422

    def test_negative_revenue_rejected(self, client):
        bad = {**VALID_CUSTOMER, "monthly_revenue": -50}
        assert client.post("/predict", json=bad).status_code == 422

    def test_invalid_contract_type(self, client):
        """contract_type is bounded to 0-2."""
        bad = {**VALID_CUSTOMER, "contract_type": 5}
        assert client.post("/predict", json=bad).status_code == 422

    def test_empty_batch_rejected(self, client):
        """min_length=1 on the customers list rejects empty batches."""
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422
