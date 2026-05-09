"""
API Request & Response Schemas
================================
Pydantic models that define the data contracts for the churn prediction API.

These schemas:
- Validate incoming request data (type checking, value ranges)
- Define the shape of API responses
- Auto-generate OpenAPI/Swagger documentation

Why Pydantic?
- Industry standard for FastAPI input validation
- Provides automatic type coercion and error messages
- Generates interactive API docs at /docs
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ─── Request Schemas ─────────────────────────────────────────────


class CustomerFeatures(BaseModel):
    """
    Input features for a single customer churn prediction.

    These match the 45 features produced by the feature engineering pipeline.
    Only the most impactful features are required; others have sensible defaults.
    """

    # ── Subscription Features ──
    monthly_revenue: float = Field(
        ..., ge=0, description="Monthly recurring revenue (£)"
    )
    tenure_months: int = Field(
        ..., ge=0, description="Months as a customer"
    )
    contract_type: int = Field(
        ..., ge=0, le=2,
        description="Contract type: 0=month-to-month, 1=one-year, 2=two-year"
    )
    total_charges: float = Field(
        ..., ge=0, description="Total charges over lifetime (£)"
    )
    payment_method: int = Field(
        default=0, ge=0, le=3,
        description="Payment method: 0=electronic check, 1=mailed check, 2=bank transfer, 3=credit card"
    )
    paperless_billing: int = Field(
        default=1, ge=0, le=1, description="Uses paperless billing: 0=no, 1=yes"
    )

    # ── Usage & Engagement Features ──
    login_frequency: float = Field(
        default=5.0, ge=0, description="Average logins per month"
    )
    feature_adoption_rate: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Fraction of product features used (0-1)"
    )
    days_since_last_login: int = Field(
        default=3, ge=0, description="Days since the customer last logged in"
    )
    avg_session_duration: float = Field(
        default=15.0, ge=0, description="Average session duration in minutes"
    )
    engagement_score: float = Field(
        default=50.0, ge=0, le=100, description="Composite engagement score (0-100)"
    )

    # ── Support Features ──
    support_tickets_total: int = Field(
        default=2, ge=0, description="Total support tickets submitted"
    )
    support_tickets_last_90d: int = Field(
        default=0, ge=0, description="Support tickets in the last 90 days"
    )
    escalation_count: int = Field(
        default=0, ge=0, description="Number of escalated support tickets"
    )

    # ── Service Features ──
    has_phone_service: int = Field(default=1, ge=0, le=1)
    has_internet_service: int = Field(default=1, ge=0, le=1)
    has_online_security: int = Field(default=0, ge=0, le=1)
    has_online_backup: int = Field(default=0, ge=0, le=1)
    has_device_protection: int = Field(default=0, ge=0, le=1)
    has_tech_support: int = Field(default=0, ge=0, le=1)
    has_streaming_tv: int = Field(default=0, ge=0, le=1)
    has_streaming_movies: int = Field(default=0, ge=0, le=1)

    # ── Derived / Ratio Features ──
    revenue_per_tenure: float = Field(
        default=0.0, ge=0, description="Monthly revenue divided by tenure"
    )
    engagement_to_tenure_ratio: float = Field(
        default=0.0, ge=0, description="Engagement score relative to tenure"
    )
    support_to_usage_ratio: float = Field(
        default=0.0, ge=0, description="Support tickets relative to usage"
    )
    is_senior_citizen: int = Field(default=0, ge=0, le=1)
    has_partner: int = Field(default=0, ge=0, le=1)
    has_dependents: int = Field(default=0, ge=0, le=1)
    gender_encoded: int = Field(default=0, ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "examples": [{
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
            }]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions on multiple customers."""

    customers: List[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000,
        description="List of customers to score (max 1000 per request)"
    )


# ─── Response Schemas ────────────────────────────────────────────


class PredictionResponse(BaseModel):
    """Response for a single customer prediction."""

    churn_probability: float = Field(
        description="Probability of churning (0-1)"
    )
    churn_prediction: bool = Field(
        description="Whether the customer is predicted to churn at the optimal threshold"
    )
    risk_level: str = Field(
        description="Risk category: critical, high, medium, or low"
    )
    monthly_revenue_at_risk: float = Field(
        description="Monthly revenue that would be lost if this customer churns (£)"
    )
    top_risk_factors: Optional[List[str]] = Field(
        default=None,
        description="Top features contributing to the churn prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[PredictionResponse]
    summary: dict = Field(
        description="Aggregate stats: total scored, high-risk count, total revenue at risk"
    )


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""

    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded and ready")
    model_name: str = Field(description="Name of the loaded model")
    optimal_threshold: float = Field(description="Decision threshold in use")
    version: str = Field(description="API version")
