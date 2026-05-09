"""
SaaS Churn Prediction API
============================
A FastAPI service that serves churn predictions from the trained model.

Endpoints:
    GET  /health         → Service status and model info
    POST /predict        → Single customer churn prediction
    POST /predict/batch  → Batch predictions (up to 1000 customers)

Run locally:
    uvicorn api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs  (Swagger UI)
    http://localhost:8000/redoc (ReDoc)

Why FastAPI?
    - Automatic request validation via Pydantic
    - Auto-generated interactive API documentation
    - Async support for high concurrency
    - Type hints throughout — clean, readable code
    - Industry standard for ML model serving in Python
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Add project root to path so we can import src modules ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from api.schemas import (
    CustomerFeatures,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)


# ─── Global Model State ─────────────────────────────────────────

model = None
scaler = None
metadata = None
feature_names = None
optimal_threshold = 0.5


# ─── Application Lifecycle ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup, clean up on shutdown."""
    global model, scaler, metadata, feature_names, optimal_threshold

    try:
        model_path = MODELS_DIR / "best_model.joblib"
        metadata_path = MODELS_DIR / "model_metadata.joblib"
        scaler_path = MODELS_DIR / "scaler.joblib"

        print("\n" + "=" * 60)
        print("LOADING MODEL ARTIFACTS")
        print("=" * 60)

        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"MODELS_DIR: {MODELS_DIR}")

        print(f"\nModel path: {model_path}")
        print(f"Exists: {model_path.exists()}")

        print(f"\nMetadata path: {metadata_path}")
        print(f"Exists: {metadata_path.exists()}")

        print(f"\nScaler path: {scaler_path}")
        print(f"Exists: {scaler_path.exists()}")

        if not model_path.exists():
            raise RuntimeError(
                f"Model not found at {model_path}. "
                "Run the training pipeline first (Phase 2)."
            )

        if not metadata_path.exists():
            raise RuntimeError(
                f"Metadata not found at {metadata_path}"
            )

        # ── Load artifacts ──
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)

        feature_names = metadata["feature_names"]
        optimal_threshold = metadata["optimal_threshold"]

        # ── Optional scaler ──
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print("\n✓ Scaler loaded")
        else:
            scaler = None
            print("\n• No scaler found (normal for tree-based models)")

        print(f"\n✓ Model loaded: {metadata['best_model']}")
        print(f"✓ Threshold: {optimal_threshold:.4f}")
        print(f"✓ Features expected: {len(feature_names)}")

        print("=" * 60 + "\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print("MODEL LOADING FAILED")
        print("=" * 60)
        print(type(e))
        print(str(e))
        print("=" * 60 + "\n")

        model = None
        metadata = None
        scaler = None
        feature_names = None

    yield  # Application runs

    # Cleanup
    print("Shutting down API...")


# ─── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(
    title="SaaS Churn Prediction API",
    description=(
        "Predict customer churn risk for a SaaS product. "
        "Returns churn probability, risk level, and revenue at risk. "
        "Built with FastAPI, scikit-learn, and XGBoost."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (for dashboard or frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper Functions ────────────────────────────────────────────

def _classify_risk(probability: float) -> str:
    """Map churn probability to a business risk level."""
    if probability >= 0.8:
        return "critical"
    elif probability >= 0.6:
        return "high"
    elif probability >= 0.4:
        return "medium"
    else:
        return "low"


def _prepare_features(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Convert a Pydantic model to a DataFrame matching the training features.

    The model expects features in the same order and names as training.
    Any features not provided by the API schema get a default value of 0.
    """
    customer_dict = customer.model_dump()

    # Build a DataFrame with all expected feature columns
    row = {fname: customer_dict.get(fname, 0) for fname in feature_names}
    df = pd.DataFrame([row], columns=feature_names)

    return df


def _predict_single(customer: CustomerFeatures) -> PredictionResponse:
    """Run prediction for a single customer."""
    df = _prepare_features(customer)

    # Apply scaler if the best model requires it (Logistic Regression)
    if scaler is not None and metadata["best_model"] == "Logistic Regression":
        features_array = scaler.transform(df)
    else:
        features_array = df.values

    probability = float(model.predict_proba(features_array)[0, 1])
    prediction = probability >= optimal_threshold
    risk_level = _classify_risk(probability)

    # Identify top risk factors (features with highest absolute values)
    feature_values = df.iloc[0]
    top_factors = (
        feature_values.abs()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        risk_level=risk_level,
        monthly_revenue_at_risk=round(
            customer.monthly_revenue if prediction else 0, 2
        ),
        top_risk_factors=top_factors,
    )


# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check if the API is running and the model is loaded.

    Returns service status, model info, and the decision threshold in use.
    Useful for monitoring and load balancer health checks.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=metadata["best_model"] if metadata else "not loaded",
        optimal_threshold=optimal_threshold,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn risk for a single customer.

    Accepts customer features and returns:
    - Churn probability (0-1)
    - Binary prediction at the optimal threshold
    - Risk level (critical / high / medium / low)
    - Monthly revenue at risk if the customer churns
    - Top 5 features driving the prediction

    Example use case: A customer success manager checks the risk
    for a specific account before a renewal call.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return _predict_single(customer)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn risk for multiple customers in one request.

    Accepts up to 1000 customers and returns individual predictions
    plus an aggregate summary (total at risk, revenue exposure).

    Example use case: Nightly batch scoring of the entire customer base,
    or scoring a segment for a targeted retention campaign.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    total_revenue_at_risk = 0
    high_risk_count = 0

    for customer in request.customers:
        try:
            pred = _predict_single(customer)
            predictions.append(pred)

            if pred.churn_prediction:
                total_revenue_at_risk += pred.monthly_revenue_at_risk
            if pred.risk_level in ("critical", "high"):
                high_risk_count += 1

        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Prediction failed for customer: {str(e)}",
            )

    summary = {
        "total_customers_scored": len(predictions),
        "high_risk_customers": high_risk_count,
        "total_monthly_revenue_at_risk": round(total_revenue_at_risk, 2),
        "estimated_annual_revenue_at_risk": round(total_revenue_at_risk * 12, 2),
    }

    return BatchPredictionResponse(predictions=predictions, summary=summary)


# ─── Run with: uvicorn api.main:app --reload ─────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
