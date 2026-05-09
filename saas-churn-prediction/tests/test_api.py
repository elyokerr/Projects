"""
API Tests
===========
Tests for the churn prediction API endpoints.

Tests cover:
    - Health check endpoint
    - Single customer prediction
    - Batch prediction
    - Input validation (bad data handling)
    - Edge cases

Run with:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app


# ─── Fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan events (model loading)."""
    with TestClient(app) as c:
        yield c


# ─── Sample Data ─────────────────────────────────────────────────

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


# ─── Health Check ────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "optimal_threshold" in data
        assert "version" in data

    def test_health_model_is_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


# ─── Single Prediction ──────────────────────────────────────────

class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_structure(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_level" in data
        assert "monthly_revenue_at_risk" in data

    def test_predict_probability_range(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert 0 <= data["churn_probability"] <= 1

    def test_predict_risk_level_valid(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert data["risk_level"] in ("critical", "high", "medium", "low")

    def test_predict_revenue_at_risk_logic(self, client):
        """If not predicted to churn, revenue at risk should be 0."""
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        if not data["churn_prediction"]:
            assert data["monthly_revenue_at_risk"] == 0

    def test_predict_high_risk_customer(self, client):
        """A customer with obvious churn signals should score higher."""
        response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        data = response.json()
        # High-risk customer should have a non-trivial churn probability
        assert data["churn_probability"] > 0.1

    def test_predict_returns_risk_factors(self, client):
        response = client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert "top_risk_factors" in data
        assert isinstance(data["top_risk_factors"], list)


# ─── Batch Prediction ───────────────────────────────────────────

class TestBatchEndpoint:
    """Tests for POST /predict/batch."""

    def test_batch_returns_200(self, client):
        payload = {"customers": [VALID_CUSTOMER, HIGH_RISK_CUSTOMER]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_response_structure(self, client):
        payload = {"customers": [VALID_CUSTOMER]}
        response = client.post("/predict/batch", json=payload)
        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 1

    def test_batch_summary_stats(self, client):
        payload = {"customers": [VALID_CUSTOMER, HIGH_RISK_CUSTOMER]}
        response = client.post("/predict/batch", json=payload)
        data = response.json()
        summary = data["summary"]
        assert summary["total_customers_scored"] == 2
        assert "high_risk_customers" in summary
        assert "total_monthly_revenue_at_risk" in summary

    def test_batch_count_matches_input(self, client):
        customers = [VALID_CUSTOMER] * 5
        payload = {"customers": customers}
        response = client.post("/predict/batch", json=payload)
        data = response.json()
        assert len(data["predictions"]) == 5


# ─── Validation ──────────────────────────────────────────────────

class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_missing_required_field(self, client):
        """monthly_revenue is required — omitting it should fail."""
        bad_customer = {
            "tenure_months": 12,
            "contract_type": 1,
            "total_charges": 500,
        }
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_negative_revenue_rejected(self, client):
        bad_customer = {**VALID_CUSTOMER, "monthly_revenue": -50}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_invalid_contract_type(self, client):
        bad_customer = {**VALID_CUSTOMER, "contract_type": 5}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_empty_batch_rejected(self, client):
        response = client.post("/predict/batch", json={"customers": []})
        assert response.status_code == 422
