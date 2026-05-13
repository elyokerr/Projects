"""
Model training pipeline.

Trains and compares three classifiers on the engineered feature set,
tracks every run with MLflow, optimizes the decision threshold for
business value, and serializes the winning model along with metadata
that downstream services (API, dashboard, batch scoring) rely on.

Models compared:
    Logistic Regression  - simple baseline with strong interpretability
    Random Forest        - moderate complexity, handles non-linearities
    XGBoost              - the industry workhorse for tabular data

The winner is chosen by ROC-AUC. The optimal threshold is selected to
maximize F1 on the test set, which gives a sensible default trade-off
between catching churners (recall) and avoiding false alarms (precision).
A different cost structure - say, a high cost of customer success time -
could justify a different threshold; the same machinery would apply.

Run as a module:
    python -m src.models.train
"""

import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

warnings.filterwarnings("ignore")


# --- Data loading ---------------------------------------------------------

def load_features() -> tuple[pd.DataFrame, pd.Series]:
    """Load the engineered feature set and split into X, y."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  Churn rate: {y.mean():.1%} ({y.sum():,} / {len(y):,})")
    return X, y


# --- Preprocessing --------------------------------------------------------

def preprocess(X_train, X_test):
    """Standardize numerical features.

    Logistic Regression needs scaled inputs to behave well; tree-based
    models do not, but applying the same transform to all models keeps
    the comparison apples-to-apples and lets us cache one scaler artifact.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# --- Model definitions ----------------------------------------------------

def get_models() -> dict:
    """Return the three candidate models with tuned default hyperparameters.

    Hyperparameters here were chosen based on a small manual sweep and
    standard rules of thumb. A proper hyperparameter search (Optuna,
    GridSearchCV) would be the next step in a production project; for
    a portfolio piece, manageable defaults plus MLflow tracking lets us
    revisit and tune later without redoing the scaffolding.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # Counter the 73/27 class imbalance
            random_state=RANDOM_STATE,
            C=0.5,                     # Moderate regularization
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.8,      # Approximate ratio of negatives to positives
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# --- Evaluation -----------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute and print headline metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n  {model_name} results")
    for k, v in metrics.items():
        print(f"    {k:>12}: {v}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Active', 'Churned'])}")

    return metrics


def find_optimal_threshold(model, X_test, y_test) -> tuple[float, dict]:
    """Find the decision threshold that maximizes F1.

    The default sklearn threshold of 0.5 is rarely optimal for business
    problems. By scanning the precision-recall curve and picking the
    threshold that maximizes F1, we get a better default trade-off
    between catching churners and avoiding false alarms.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # F1 for every threshold along the PR curve. The +1e-8 prevents
    # divide-by-zero when both precision and recall are 0 at a threshold.
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    y_pred_optimal = (y_prob >= best_threshold).astype(int)
    optimal_metrics = {
        "optimal_threshold": round(float(best_threshold), 4),
        "precision_at_optimal": round(precision_score(y_test, y_pred_optimal), 4),
        "recall_at_optimal": round(recall_score(y_test, y_pred_optimal), 4),
        "f1_at_optimal": round(f1_score(y_test, y_pred_optimal), 4),
    }

    print("\n  Threshold optimization")
    print(f"    Default (0.5) F1: {f1_score(y_test, model.predict(X_test)):.4f}")
    print(f"    Optimal threshold: {best_threshold:.4f}")
    print(f"    F1 at optimal: {optimal_metrics['f1_at_optimal']}")
    print(f"    Precision at optimal: {optimal_metrics['precision_at_optimal']}")
    print(f"    Recall at optimal: {optimal_metrics['recall_at_optimal']}")

    return best_threshold, optimal_metrics


def cross_validate_model(model, X, y, model_name: str) -> float:
    """Stratified 5-fold cross-validation for ROC-AUC stability check."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    mean_auc, std_auc = scores.mean(), scores.std()
    print(f"  {model_name} CV ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    return mean_auc


# --- Revenue impact -------------------------------------------------------

def estimate_revenue_impact(model, X_test, y_test, monthly_revenue_test, threshold: float):
    """Translate model performance into financial terms.

    "Recall is 0.82" is meaningless to a non-technical stakeholder.
    "The model identifies £64,000 of annual revenue we can save" is
    immediately actionable. This function converts confusion matrix
    cells into monetary impact using each customer's actual MRR.

    Assumes a 20% retention rate from intervention - i.e. if the
    customer success team reaches every flagged customer, they save
    one in five. Conservative but defensible.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    true_positives = (y_pred == 1) & (y_test == 1)
    false_negatives = (y_pred == 0) & (y_test == 1)

    mrr_at_risk_caught = monthly_revenue_test[true_positives].sum()
    mrr_at_risk_missed = monthly_revenue_test[false_negatives].sum()
    total_mrr_at_risk = monthly_revenue_test[y_test == 1].sum()

    retention_rate = 0.20
    estimated_saved = mrr_at_risk_caught * retention_rate

    print("\n  Revenue impact estimation")
    print(f"    Total MRR at risk: £{total_mrr_at_risk:,.2f}")
    print(f"    MRR identified by model: £{mrr_at_risk_caught:,.2f}")
    print(f"    MRR missed: £{mrr_at_risk_missed:,.2f}")
    print(f"    Estimated monthly savings (20% retention): £{estimated_saved:,.2f}")
    print(f"    Estimated annual savings: £{estimated_saved * 12:,.2f}")

    return {
        "total_mrr_at_risk": round(float(total_mrr_at_risk), 2),
        "mrr_identified": round(float(mrr_at_risk_caught), 2),
        "mrr_missed": round(float(mrr_at_risk_missed), 2),
        "estimated_monthly_savings": round(float(estimated_saved), 2),
        "estimated_annual_savings": round(float(estimated_saved * 12), 2),
    }


# --- Main pipeline --------------------------------------------------------

def run_training():
    """Execute the full training pipeline and persist artifacts."""
    print("=" * 60)
    print("Model Training Pipeline")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("\n[1/6] Loading features...")
    X, y = load_features()

    print("\n[2/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # Preserve raw monthly_revenue values for the revenue impact calc.
    # We need the unscaled numbers, so grab them before preprocessing.
    monthly_revenue_test = X_test["monthly_revenue"].values

    print("\n[3/6] Preprocessing...")
    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)
    feature_names = X.columns.tolist()

    print("\n[4/6] Training models...")
    models = get_models()
    results = {}
    best_model = None
    best_auc = 0

    for name, model in models.items():
        print(f"\n{'-' * 40}")
        print(f"  Training: {name}")
        print(f"{'-' * 40}")

        with mlflow.start_run(run_name=name):
            # Logistic regression uses scaled inputs; tree models work on
            # raw values. Tree models are insensitive to monotonic
            # transformations, so scaling would be redundant for them.
            if name == "logistic_regression":
                model.fit(X_train_scaled, y_train)
                metrics = evaluate_model(model, X_test_scaled, y_test, name)
                cv_auc = cross_validate_model(model, X_train_scaled, y_train, name)
            else:
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test, name)
                cv_auc = cross_validate_model(model, X_train, y_train, name)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_metric("cv_roc_auc", cv_auc)

            if name == "xgboost":
                mlflow.xgboost.log_model(model, name)
            else:
                mlflow.sklearn.log_model(model, name)

            results[name] = {**metrics, "cv_roc_auc": cv_auc}

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = (name, model)

    print(f"\n{'=' * 60}")
    print(f"[5/6] Optimizing threshold for best model: {best_model[0]}")
    print(f"{'=' * 60}")

    if best_model[0] == "logistic_regression":
        threshold, threshold_metrics = find_optimal_threshold(best_model[1], X_test_scaled, y_test)
        revenue = estimate_revenue_impact(
            best_model[1], X_test_scaled, y_test, monthly_revenue_test, threshold,
        )
    else:
        threshold, threshold_metrics = find_optimal_threshold(best_model[1], X_test, y_test)
        revenue = estimate_revenue_impact(
            best_model[1], X_test, y_test, monthly_revenue_test, threshold,
        )

    print("\n[6/6] Saving artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model[1], model_path)
    print(f"  Model:    {model_path}")

    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler:   {scaler_path}")

    # Metadata bundles everything downstream consumers need: the model
    # name, the threshold to use, the exact feature names in the order
    # the model expects, and the business impact estimates.
    metadata = {
        "best_model": best_model[0],
        "optimal_threshold": threshold,
        "feature_names": feature_names,
        "metrics": results[best_model[0]],
        "threshold_metrics": threshold_metrics,
        "revenue_impact": revenue,
    }
    metadata_path = MODELS_DIR / "model_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    print(f"  Metadata: {metadata_path}")

    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")
    print("\nModel comparison:")
    summary = pd.DataFrame(results).T
    print(summary.to_string())
    print(f"\nBest model: {best_model[0]} (ROC-AUC: {best_auc:.4f})")
    print(f"Optimal threshold: {threshold:.4f}")
    print(f"Estimated annual revenue savings: £{revenue['estimated_annual_savings']:,.2f}")
    print("\nTraining pipeline complete.")
    print(f"{'=' * 60}")

    return best_model, results, metadata


if __name__ == "__main__":
    run_training()
