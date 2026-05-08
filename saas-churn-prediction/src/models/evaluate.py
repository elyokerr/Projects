"""
Model Evaluation & Explainability
===================================
Generates evaluation plots and SHAP-based model explanations.

Outputs:
    - ROC curves comparison
    - Precision-Recall curves
    - Confusion matrices
    - SHAP summary plot (global feature importance)
    - SHAP waterfall plot (individual predictions)

Usage:
    python -m src.models.evaluate
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE

# Plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

PLOTS_DIR = MODELS_DIR / "plots"


def load_test_data():
    """Load features and reproduce the same train/test split."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test


def plot_roc_curves(model, X_test, y_test):
    """Plot ROC curve for the best model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC = 0.50)")
    ax.set_title("ROC Curve — Churn Prediction Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ ROC curve saved")


def plot_precision_recall(model, X_test, y_test):
    """Plot Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name="XGBoost")
    ax.set_title("Precision-Recall Curve — Churn Prediction", fontsize=14, fontweight="bold")
    ax.axhline(y=y_test.mean(), color="r", linestyle="--", alpha=0.3, label=f"Baseline ({y_test.mean():.2f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Precision-Recall curve saved")


def plot_confusion_matrix(model, X_test, y_test, threshold: float):
    """Plot confusion matrix at the optimal threshold."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Active", "Churned"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix (threshold={threshold:.3f})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Confusion matrix saved")


def plot_feature_importance(model, feature_names: list):
    """Plot feature importance (top 20). Works for both tree and linear models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Feature Importance (Gain)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        xlabel = "Absolute Coefficient Magnitude"
    else:
        print("  ⚠ Model does not support feature importance plotting")
        return

    indices = np.argsort(importances)[-20:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color="#4A90D9")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel(xlabel)
    ax.set_title("Top 20 Features — Model Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Feature importance plot saved")


def generate_shap_explanations(model, X_test, feature_names: list):
    """
    Generate SHAP explanations — the gold standard for ML interpretability.

    This is a major differentiator in interviews. SHAP values show:
    - Which features drive predictions globally
    - How each feature pushes individual predictions toward churn/active
    """
    print("  Computing SHAP values (this may take a minute)...")

    # Use a sample for speed
    sample_size = min(500, len(X_test))
    X_sample = X_test.iloc[:sample_size]

    # Auto-detect explainer type based on model
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot (global importance with direction)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Feature Impact on Churn Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ SHAP summary plot saved")

    # Bar plot (absolute importance)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title("SHAP Mean Absolute Impact", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ SHAP bar plot saved")

    # Waterfall plot for a single churner prediction
    churner_idx = X_sample.index[
        model.predict_proba(X_sample)[:, 1] > 0.7
    ]
    if len(churner_idx) > 0:
        idx = 0  # first high-risk customer in sample
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray):
            base_val = float(base_val[0]) if len(base_val) == 1 else float(base_val[1])
        fig, ax = plt.subplots(figsize=(12, 8))
        shap_explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=base_val,
            data=X_sample.iloc[idx].values,
            feature_names=feature_names,
        )
        shap.waterfall_plot(shap_explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall — High-Risk Customer Example", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ SHAP waterfall plot saved")

    return shap_values


def run_evaluation():
    """Run the full evaluation and explanation pipeline."""
    print("=" * 60)
    print("Model Evaluation & Explainability")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading model and test data...")
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    metadata = joblib.load(MODELS_DIR / "model_metadata.joblib")
    X_test, y_test = load_test_data()
    threshold = metadata["optimal_threshold"]
    feature_names = metadata["feature_names"]
    print(f"  Model: {metadata['best_model']}")
    print(f"  Threshold: {threshold:.4f}")

    print("\n[2/4] Generating evaluation plots...")
    plot_roc_curves(model, X_test, y_test)
    plot_precision_recall(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test, threshold)
    plot_feature_importance(model, feature_names)

    print("\n[3/4] Generating SHAP explanations...")
    shap_values = generate_shap_explanations(model, X_test, feature_names)

    print("\n[4/4] Summary...")
    print(f"  All plots saved to: {PLOTS_DIR}")
    print(f"  Files generated:")
    for f in sorted(PLOTS_DIR.glob("*.png")):
        print(f"    - {f.name}")

    print(f"\n✓ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
