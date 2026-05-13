"""
Feature engineering pipeline.

This stage joins everything together: the customer dimensions, the
subscription facts, the product usage flags, and the synthetic usage
and support data. The bulk of the heavy lifting happens in SQL via
features.sql - CTEs aggregate per-customer engagement and support
metrics, then a final SELECT assembles the analytics-ready feature set.

A small number of features are computed in Python afterwards because
they're awkward to express in SQL: ratio features, composite scores,
and one-hot encodings.

The end result is a 45-column feature_store table and a matching
features.csv file, ready for model training.

Run as a module:
    python -m src.data.feature_engineering
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROCESSED_DATA_DIR, SQL_DIR
from src.db import get_engine


def execute_sql_features(engine) -> pd.DataFrame:
    """Run the main feature engineering SQL and return the result as a DataFrame."""
    sql_path = SQL_DIR / "features.sql"
    with open(sql_path) as f:
        query = f.read()

    df = pd.read_sql(query, engine)
    print(f"  SQL features generated: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def add_python_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that are easier to express in Python than in SQL.

    These tend to fall into three buckets:
      - Ratios that require care around zero-denominators
      - Composite scores combining multiple features with weights
      - Categorical one-hot encodings of derived segments
    """
    # Average revenue per enabled module (a rough value-per-feature metric).
    df["revenue_per_module"] = np.where(
        df["modules_enabled"] > 0,
        (df["monthly_revenue"] / df["modules_enabled"]).round(2),
        df["monthly_revenue"],
    )

    # Lifetime value per month of tenure - a proxy for customer worth.
    df["lifetime_value_indicator"] = (
        df["total_revenue"] / (df["account_age_months"] + 1)
    ).round(2)

    # Composite engagement score: weighted blend of login frequency,
    # session length, and feature adoption, normalized to a 0-1 scale.
    df["engagement_intensity"] = (
        (df["avg_weekly_logins"] / df["avg_weekly_logins"].max()) * 0.4
        + (df["avg_session_minutes"] / df["avg_session_minutes"].max()) * 0.3
        + df["avg_feature_adoption"] * 0.3
    ).round(3)

    # Binary flag: has the customer ever contacted support?
    df["has_contacted_support"] = (df["total_tickets"] > 0).astype(int)

    # Ticket rate per month - a stronger signal than raw counts because
    # it controls for tenure.
    df["tickets_per_month"] = np.where(
        df["account_age_months"] > 0,
        (df["total_tickets"] / df["account_age_months"]).round(3),
        df["total_tickets"].astype(float),
    )

    # One-hot encode the account_segment categorical from the SQL stage.
    segment_dummies = pd.get_dummies(df["account_segment"], prefix="segment")
    for col in segment_dummies.columns:
        segment_dummies[col] = segment_dummies[col].astype(int)
    df = pd.concat([df, segment_dummies], axis=1)
    df = df.drop(columns=["account_segment"])

    print(f"  Python features added: now {df.shape[1]} total columns")
    return df


def validate_features(df: pd.DataFrame) -> None:
    """Quick sanity checks on the assembled feature set."""
    print("\nFeature Validation")

    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if len(nulls) == 0:
        print("  No null values")
    else:
        print(f"  Nulls found:\n{nulls}")

    churn_rate = df["churn"].mean()
    print(f"  Churn rate: {churn_rate:.1%} ({df['churn'].sum():,} / {len(df):,})")
    print(
        f"  Monthly revenue: "
        f"£{df['monthly_revenue'].min():.2f} – £{df['monthly_revenue'].max():.2f}"
    )
    print(
        f"  Account age: "
        f"{df['account_age_months'].min()} – {df['account_age_months'].max()} months"
    )
    print(f"  Mean weekly logins: {df['avg_weekly_logins'].mean():.1f}")
    print(f"  Total features (excluding customer_id and churn): {df.shape[1] - 2}")


def run_feature_engineering() -> pd.DataFrame:
    """Execute the full feature engineering pipeline."""
    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)

    print("\n[1/5] Connecting to database...")
    engine = get_engine()

    print("\n[2/5] Executing SQL feature queries...")
    df = execute_sql_features(engine)

    print("\n[3/5] Adding Python-computed features...")
    df = add_python_features(df)

    print("\n[4/5] Validating features...")
    validate_features(df)

    print("\n[5/5] Saving outputs...")
    df.to_sql("feature_store", engine, if_exists="replace", index=False)
    print(f"  feature_store table written ({len(df):,} rows)")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DATA_DIR / "features.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved to {csv_path}")

    print(
        f"\nFeature engineering complete: "
        f"{df.shape[1]} features for {df.shape[0]:,} customers"
    )
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_feature_engineering()
