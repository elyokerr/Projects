"""
API request and response schemas.

Pydantic models that define the data contracts for the API. Defining
schemas this way gets us three things for free:

  - Automatic input validation with clear error messages
  - Auto-generated Swagger and ReDoc documentation
  - Type-safe code in the endpoint handlers

The CustomerFeatures schema covers the 29 most useful features. Any
features computed downstream that aren't in this schema get a default
of 0 inside the API service, which is correct for binary/flag features
and reasonable for ratio features in the absence of other information.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# --- Request schemas ------------------------------------------------------


class CustomerFeatures(BaseModel):
    """Features for a single customer churn prediction.

    Only four fields are required: monthly_revenue, tenure_months,
    contract_type, and total_charges. The rest have sensible defaults
    so the API can be called with minimal information when richer
    behavioral data isn't available.
    """

    # Subscription features (required)
    monthly_revenue: float = Field(..., ge=0, description="Monthly recurring revenue (£)")
    tenure_months: int = Field(..., ge=0, description="Months as a customer")
    contract_type: int = Field(
        ..., ge=0, le=2,
        description="Contract type: 0=month-to-month, 1=one-year, 2=two-year",
    )
    total_charges: float = Field(..., ge=0, description="Lifetime total charges (£)")

    # Subscription features (optional with defaults)
    payment_method: int = Field(
        default=0, ge=0, le=3,
        description="0=electronic check, 1=mailed check, 2=bank transfer, 3=credit card",
    )
    paperless_billing: int = Field(default=1, ge=0, le=1)

    # Usage and engagement
    login_frequency: float = Field(default=5.0, ge=0, description="Average logins per month")
    feature_adoption_rate: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Fraction of product features used (0-1)",
    )
    days_since_last_login: int = Field(default=3, ge=0)
    avg_session_duration: float = Field(default=15.0, ge=0, description="Minutes per session")
    engagement_score: float = Field(default=50.0, ge=0, le=100)

    # Support history
    support_tickets_total: int = Field(default=2, ge=0)
    support_tickets_last_90d: int = Field(default=0, ge=0)
    escalation_count: int = Field(default=0, ge=0)

    # Product entitlements
    has_phone_service: int = Field(default=1, ge=0, le=1)
    has_internet_service: int = Field(default=1, ge=0, le=1)
    has_online_security: int = Field(default=0, ge=0, le=1)
    has_online_backup: int = Field(default=0, ge=0, le=1)
    has_device_protection: int = Field(default=0, ge=0, le=1)
    has_tech_support: int = Field(default=0, ge=0, le=1)
    has_streaming_tv: int = Field(default=0, ge=0, le=1)
    has_streaming_movies: int = Field(default=0, ge=0, le=1)

    # Derived ratios
    revenue_per_tenure: float = Field(default=0.0, ge=0)
    engagement_to_tenure_ratio: float = Field(default=0.0, ge=0)
    support_to_usage_ratio: float = Field(default=0.0, ge=0)

    # Demographics
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
    """Request body for the batch endpoint."""

    customers: List[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000,
        description="One to one thousand customers to score",
    )


# --- Response schemas -----------------------------------------------------


class PredictionResponse(BaseModel):
    """Per-customer prediction returned by /predict and /predict/batch."""

    churn_probability: float = Field(description="Probability of churn (0 to 1)")
    churn_prediction: bool = Field(description="Decision at the optimized threshold")
    risk_level: str = Field(description="critical, high, medium, or low")
    monthly_revenue_at_risk: float = Field(
        description="Monthly revenue exposure if the customer churns (£)",
    )
    top_risk_factors: Optional[List[str]] = Field(
        default=None,
        description="Up to five features most strongly associated with this prediction",
    )


class BatchPredictionResponse(BaseModel):
    """Aggregate response for batch predictions."""

    predictions: List[PredictionResponse]
    summary: dict = Field(
        description="Aggregate statistics across the batch",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str
    optimal_threshold: float
    version: str
