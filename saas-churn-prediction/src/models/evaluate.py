"""
Model evaluation and explainability.

Once the winning model is selected and serialized, this module produces
the visualizations that go into the dashboard and documentation:

    ROC and Precision-Recall curves
    Confusion matrix at the optimized threshold
    Feature importance (model-native)
    SHAP summary, bar, and waterfall plots

SHAP is the more substantial output. The summary plot shows global
feature impact with direction (red pushes toward churn, blue toward
retention). The waterfall plot explains a single high-risk prediction
in terms anyone can follow, which is what makes the predictions
actionable for a customer success team.

All plots are saved as PNGs to models/plots/.

Run as a module:
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
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE


# Visual styling - matched to the dashboard for a consistent look.
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

PLOTS_DIR = MODELS_DIR / "plots"


def load_test_data():
    """Reproduce the same train/test split used in training.

    The split is deterministic given the same RANDOM_STATE, so this
    reliably returns the exact test set the model was evaluated on.
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    return X_test, y_test


def plot_roc_curves(model, X_test, y_test) -> None:
    """ROC curve for the winning model with a random-baseline reference."""
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="Best model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC = 0.50)")
    ax.set_title("ROC Curve - Churn Prediction Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved roc_curve.png")


def plot_precision_recall(model, X_test, y_test) -> None:
    """Precision-Recall curve - more informative than ROC for imbalanced data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name="Best model")
    ax.set_title("Precision-Recall Curve - Churn Prediction", fontsize=14, fontweight="bold")
    ax.axhline(
        y=y_test.mean(), color="r", linestyle="--", alpha=0.3,
        label=f"Baseline ({y_test.mean():.2f})",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved precision_recall_curve.png")


def plot_confusion_matrix(model, X_test, y_test, threshold: float) -> None:
    """Confusion matrix at the optimized threshold (not the default 0.5)."""
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
        f"Confusion Matrix (threshold = {threshold:.3f})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrix.png")


def plot_feature_importance(model, feature_names: list) -> None:
    """Top-20 feature importance.

    Uses the native importance signal: gain for tree models, absolute
    coefficient magnitude for linear models. Skips quietly if the model
    exposes neither.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Feature Importance (Gain)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        xlabel = "Absolute Coefficient Magnitude"
    else:
        print("  Model does not expose feature importances - skipping plot")
        return

    indices = np.argsort(importances)[-20:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color="#4A90D9")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel(xlabel)
    ax.set_title("Top 20 Features by Model Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved feature_importance.png")


def generate_shap_explanations(model, X_test, feature_names: list):
    """Compute SHAP values and generate the three standard SHAP plots.

    SHAP values are the gold standard for ML explainability. They have
    strong theoretical foundations (Shapley values from game theory)
    and produce additive, locally accurate explanations.

    The summary plot shows global importance with direction. The bar
    plot strips out direction for a simpler view. The waterfall plot
    explains one specific prediction - this is what a customer success
    manager would look at when reviewing a flagged account.

    A sample of 500 test points is used for speed; using the full test
    set would take noticeably longer with no real change in conclusions.
    """
    print("  Computing SHAP values (sampling 500 test rows)...")

    sample_size = min(500, len(X_test))
    X_sample = X_test.iloc[:sample_size]

    # TreeExplainer is fast and exact for tree models; LinearExplainer
    # serves the same role for linear models. Picking automatically by
    # introspecting the model means we don't have to change this code
    # if a different model wins the comparison.
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # 1. Beeswarm summary - global importance plus direction.
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
    print("  Saved shap_summary.png")

    # 2. Bar plot - mean absolute SHAP value, ranked.
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
    print("  Saved shap_bar.png")

    # 3. Waterfall plot - explains a single high-probability prediction.
    # This is the plot a customer success manager would actually look at.
    high_risk_indices = X_sample.index[model.predict_proba(X_sample)[:, 1] > 0.7]
    if len(high_risk_indices) > 0:
        idx = 0
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
        plt.title(
            "SHAP Waterfall - High-Risk Customer Example",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved shap_waterfall.png")

    return shap_values


def run_evaluation() -> None:
    """Execute the full evaluation pipeline."""
    print("=" * 60)
    print("Model Evaluation and Explainability")
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
    generate_shap_explanations(model, X_test, feature_names)

    print("\n[4/4] Summary")
    print(f"  All plots saved to: {PLOTS_DIR}")
    for f in sorted(PLOTS_DIR.glob("*.png")):
        print(f"    {f.name}")

    print("\nEvaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
