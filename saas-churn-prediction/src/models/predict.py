"""
Prediction Module
==================
Handles inference for single customers and batch scoring.
Used by the FastAPI service (Phase 3) and dashboard (Phase 4).

Usage:
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
    """Load the trained model and metadata."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    metadata = joblib.load(MODELS_DIR / "model_metadata.joblib")
    return model, metadata


def predict_single(customer_features: dict) -> dict:
    """
    Predict churn probability for a single customer.

    Args:
        customer_features: dict of feature_name -> value

    Returns:
        dict with churn_probability, churn_prediction, risk_level
    """
    model, metadata = load_model()
    threshold = metadata["optimal_threshold"]
    feature_names = metadata["feature_names"]

    # Build feature vector in correct order
    X = pd.DataFrame([customer_features])[feature_names]

    probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(probability >= threshold)

    # Risk levels for business use
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
    """
    Score all customers in the feature store.
    Returns DataFrame with customer_id, probability, prediction, risk_level.
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
    print(f"  Risk distribution:")
    print(results["risk_level"].value_counts().to_string())
    print(f"\n  Revenue at risk (predicted churners): £{results[results['churn_prediction'] == 1]['monthly_revenue'].sum():,.2f}/month")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Batch Scoring")
    print("=" * 60)
    results = predict_batch()

    # Save scored results
    output_path = PROCESSED_DATA_DIR / "scored_customers.csv"
    results.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved to {output_path}")
