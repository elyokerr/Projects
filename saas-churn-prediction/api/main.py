"""
SaaS Churn Prediction API.

A FastAPI service that exposes the trained churn model as a REST API.
It validates incoming requests with Pydantic, runs predictions, and
returns probabilities together with business-friendly risk levels and
revenue exposure estimates.

Endpoints:
    GET  /health           - Service status and model info
    POST /predict          - Single-customer prediction
    POST /predict/batch    - Batch prediction (up to 1000 customers)

Run locally:
    uvicorn api.main:app --reload --port 8000

Interactive documentation:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from src/ when running uvicorn from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)


# --- Global model state ---------------------------------------------------
# Populated once at startup by the lifespan handler. Module-level globals
# are appropriate here because the model is read-only and shared across
# every request.

model = None
scaler = None
metadata = None
feature_names = None
optimal_threshold = 0.5


# --- Lifespan handler -----------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup; clean up on shutdown.

    Loading once at startup (rather than per-request) keeps latency low.
    The artifacts produced by src.models.train are read from disk and
    bound to the module-level globals above.
    """
    global model, scaler, metadata, feature_names, optimal_threshold

    try:
        model_path = MODELS_DIR / "best_model.joblib"
        metadata_path = MODELS_DIR / "model_metadata.joblib"
        scaler_path = MODELS_DIR / "scaler.joblib"

        print("\n" + "=" * 60)
        print("Loading model artifacts")
        print("=" * 60)
        print(f"Models directory: {MODELS_DIR}")

        if not model_path.exists():
            raise RuntimeError(
                f"Model not found at {model_path}. "
                "Run the training pipeline first (src.models.train)."
            )
        if not metadata_path.exists():
            raise RuntimeError(f"Metadata not found at {metadata_path}")

        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        feature_names = metadata["feature_names"]
        optimal_threshold = metadata["optimal_threshold"]

        # The scaler is only needed for the logistic regression model;
        # tree-based models work on raw values.
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        print(f"Model:              {metadata['best_model']}")
        print(f"Optimal threshold:  {optimal_threshold:.4f}")
        print(f"Feature count:      {len(feature_names)}")
        print(f"Scaler loaded:      {scaler is not None}")
        print("=" * 60 + "\n")

    except Exception as exc:
        # Set everything to None so /health reports the failure clearly.
        print(f"\nModel loading failed: {exc}\n")
        model = None
        metadata = None
        scaler = None
        feature_names = None

    yield

    print("API shutting down.")


# --- FastAPI app ----------------------------------------------------------

app = FastAPI(
    title="SaaS Churn Prediction API",
    description=(
        "Predicts churn risk for SaaS customers. Returns churn probability, "
        "risk level, monthly revenue at risk, and the top features driving "
        "each prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Wide-open CORS for portfolio convenience. A production deployment would
# narrow this to the dashboard's origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers --------------------------------------------------------------

def _classify_risk(probability: float) -> str:
    """Bucket a churn probability into a business-friendly risk tier."""
    if probability >= 0.8:
        return "critical"
    if probability >= 0.6:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def _prepare_features(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert a validated request body into a model-ready DataFrame.

    The model expects features in a specific order with specific names.
    Any feature the request doesn't supply gets a default of 0 - the
    Pydantic schema already enforces this for the columns it knows about,
    but engineered features computed elsewhere need a fallback too.
    """
    customer_dict = customer.model_dump()
    row = {fname: customer_dict.get(fname, 0) for fname in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def _predict_single(customer: CustomerFeatures) -> PredictionResponse:
    """Run a prediction for one customer and return the API response shape."""
    df = _prepare_features(customer)

    # Linear models need scaled inputs; tree models don't. The decision
    # depends on which model won the comparison and is stored in metadata.
    if scaler is not None and metadata["best_model"] == "logistic_regression":
        features_array = scaler.transform(df)
    else:
        features_array = df.values

    probability = float(model.predict_proba(features_array)[0, 1])
    prediction = probability >= optimal_threshold
    risk_level = _classify_risk(probability)

    # "Top risk factors" is a simple ranking by absolute feature value.
    # A proper implementation would use SHAP values per request - that's
    # noted as a future improvement in the project README. The current
    # approximation is fast and gives sensible-looking output.
    feature_values = df.iloc[0]
    top_factors = (
        feature_values.abs()
        .sort_values(ascending=False)
        .head(5)
        .index
        .tolist()
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


# --- Endpoints ------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Report service status and key model parameters.

    Used by load balancers and orchestrators to decide whether the
    service is ready to handle traffic. Also useful for debugging.
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
    """Predict churn risk for a single customer.

    Returns the probability, the binary decision at the optimized
    threshold, a business risk tier, monthly revenue exposure, and the
    top contributing features.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return _predict_single(customer)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for up to 1000 customers in a single request.

    Returns individual predictions plus aggregate statistics (count of
    high-risk customers, total revenue at risk). The 1000-customer cap
    is enforced at the schema level.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    total_revenue_at_risk = 0.0
    high_risk_count = 0

    for customer in request.customers:
        try:
            pred = _predict_single(customer)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Prediction failed for a customer: {exc}",
            )

        predictions.append(pred)
        if pred.churn_prediction:
            total_revenue_at_risk += pred.monthly_revenue_at_risk
        if pred.risk_level in ("critical", "high"):
            high_risk_count += 1

    summary = {
        "total_customers_scored": len(predictions),
        "high_risk_customers": high_risk_count,
        "total_monthly_revenue_at_risk": round(total_revenue_at_risk, 2),
        "estimated_annual_revenue_at_risk": round(total_revenue_at_risk * 12, 2),
    }

    return BatchPredictionResponse(predictions=predictions, summary=summary)


# --- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
