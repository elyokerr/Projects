"""
Prediction module.

Two entry points:

    predict_single(features_dict)  - Score a single customer
    predict_batch()                - Score every customer in features.csv

Both apply the model's optimized threshold to produce a binary prediction
and bucket the probability into one of four business risk levels
(critical, high, medium, low) so the customer success team has a clear
prioritization signal.

The batch path is what feeds the Streamlit dashboard - it writes
scored_customers.csv with one row per customer including monthly revenue
exposure and the actual churn label (useful for sanity-checking).

Run as a module:
    python -m src.models.predict
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR


def load_model():
    """Load the trained model and metadata from disk."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    metadata = joblib.load(MODELS_DIR / "model_metadata.joblib")
    return model, metadata


def predict_single(customer_features: dict) -> dict:
    """Score one customer and return probability, prediction, and risk level.

    Args:
        customer_features: Mapping of feature_name -> value. Missing keys
            are not allowed; the API service is responsible for filling
            in defaults via Pydantic before calling here.

    Returns:
        Dict with churn_probability, churn_prediction (0/1), risk_level,
        and the threshold the prediction was made at.
    """
    model, metadata = load_model()
    threshold = metadata["optimal_threshold"]
    feature_names = metadata["feature_names"]

    # Construct a single-row DataFrame in the exact feature order the
    # model expects. Misordered features would silently produce wrong
    # predictions.
    X = pd.DataFrame([customer_features])[feature_names]

    probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(probability >= threshold)

    # Bucket probabilities into business-friendly tiers.
    if probability >= 0.7:
        risk_level = "critical"
    elif probability >= 0.5:
        risk_level = "high"
    elif probability >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": prediction,
        "risk_level": risk_level,
        "threshold_used": threshold,
    }


def predict_batch() -> pd.DataFrame:
    """Score every customer in the feature store and return a results frame.

    The output includes the actual churn label so downstream consumers
    (notebook EDA, dashboard sanity checks) can compare predictions
    against ground truth.
    """
    model, metadata = load_model()
    threshold = metadata["optimal_threshold"]
    feature_names = metadata["feature_names"]

    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    X = df[feature_names]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "customer_id": df["customer_id"],
        "churn_probability": np.round(probabilities, 4),
        "churn_prediction": predictions,
        "monthly_revenue": df["monthly_revenue"],
        "risk_level": pd.cut(
            probabilities,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["low", "medium", "high", "critical"],
        ),
        "actual_churn": df["churn"],
    })

    results = results.sort_values("churn_probability", ascending=False)

    print(f"  Scored {len(results):,} customers")
    print("  Risk distribution:")
    print(results["risk_level"].value_counts().to_string())
    mrr_at_risk = results[results["churn_prediction"] == 1]["monthly_revenue"].sum()
    print(f"\n  Revenue at risk (predicted churners): £{mrr_at_risk:,.2f}/month")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Batch Scoring")
    print("=" * 60)
    results = predict_batch()

    output_path = PROCESSED_DATA_DIR / "scored_customers.csv"
    results.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")
