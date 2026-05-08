"""
Data Ingestion Pipeline
========================
Loads the raw Telco Customer Churn CSV, cleans it, reshapes columns into
a realistic SaaS product context, and writes structured tables to the database.

Tables created:
    - customers        : core customer demographics and account info
    - subscriptions    : plan details, billing, contract type
    - product_usage    : which product modules each customer has enabled

Usage:
    python -m src.data.ingest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

# Allow running as module from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import RAW_CSV_PATH, RANDOM_STATE
from src.db import get_engine


# ─── Column Mapping ──────────────────────────────────────────────
SAAS_COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "is_senior",
    "Partner": "has_partner",
    "Dependents": "has_dependents",
    "tenure": "account_age_months",
    "PhoneService": "has_base_product",
    "MultipleLines": "has_multi_seat",
    "InternetService": "service_tier",
    "OnlineSecurity": "has_security_module",
    "OnlineBackup": "has_backup_module",
    "DeviceProtection": "has_protection_module",
    "TechSupport": "has_support_addon",
    "StreamingTV": "has_analytics_module",
    "StreamingMovies": "has_reporting_module",
    "Contract": "contract_type",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_revenue",
    "TotalCharges": "total_revenue",
    "Churn": "churn",
}


def load_raw_data(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV and perform initial validation."""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns from {path.name}")
    return df


def clean_and_reshape(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data, rename columns to SaaS context, fix types."""
    df = df.rename(columns=SAAS_COLUMN_MAP)

    df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors="coerce")
    mask = df["total_revenue"].isna()
    df.loc[mask, "total_revenue"] = df.loc[mask, "monthly_revenue"] * df.loc[mask, "account_age_months"]
    print(f"  Fixed {mask.sum()} blank total_revenue values")

    bool_columns = [
        "has_partner", "has_dependents", "has_base_product",
        "paperless_billing", "churn",
    ]
    for col in bool_columns:
        df[col] = (df[col] == "Yes").astype(int)

    tier_map = {"DSL": "basic", "Fiber optic": "premium", "No": "free"}
    df["service_tier"] = df["service_tier"].map(tier_map)

    module_columns = [
        "has_multi_seat", "has_security_module", "has_backup_module",
        "has_protection_module", "has_support_addon",
        "has_analytics_module", "has_reporting_module",
    ]
    for col in module_columns:
        df[col] = (df[col] == "Yes").astype(int)

    contract_map = {
        "Month-to-month": "monthly",
        "One year": "annual",
        "Two year": "biennial",
    }
    df["contract_type"] = df["contract_type"].map(contract_map)

    payment_map = {
        "Electronic check": "electronic_check",
        "Mailed check": "mailed_check",
        "Bank transfer (automatic)": "bank_transfer_auto",
        "Credit card (automatic)": "credit_card_auto",
    }
    df["payment_method"] = df["payment_method"].map(payment_map)

    np.random.seed(RANDOM_STATE)
    reference_date = pd.Timestamp("2024-12-01")
    df["signup_date"] = df["account_age_months"].apply(
        lambda m: reference_date - pd.DateOffset(months=int(m))
    )

    print(f"  Cleaned and reshaped {len(df):,} rows")
    return df


def write_customers_table(df: pd.DataFrame, engine) -> None:
    """Write the customers dimension table."""
    customers = df[[
        "customer_id", "gender", "is_senior", "has_partner",
        "has_dependents", "signup_date", "account_age_months",
    ]].copy()
    customers.to_sql("customers", engine, if_exists="replace", index=False)
    print(f"  ✓ customers table: {len(customers):,} rows")


def write_subscriptions_table(df: pd.DataFrame, engine) -> None:
    """Write the subscriptions fact table."""
    subscriptions = df[[
        "customer_id", "service_tier", "contract_type",
        "paperless_billing", "payment_method",
        "monthly_revenue", "total_revenue", "churn",
    ]].copy()
    subscriptions.to_sql("subscriptions", engine, if_exists="replace", index=False)
    print(f"  ✓ subscriptions table: {len(subscriptions):,} rows")


def write_product_usage_table(df: pd.DataFrame, engine) -> None:
    """Write the product module usage table."""
    product_usage = df[[
        "customer_id", "has_base_product", "has_multi_seat",
        "has_security_module", "has_backup_module",
        "has_protection_module", "has_support_addon",
        "has_analytics_module", "has_reporting_module",
    ]].copy()
    module_cols = [c for c in product_usage.columns if c.startswith("has_")]
    product_usage["modules_enabled"] = product_usage[module_cols].sum(axis=1)
    product_usage.to_sql("product_usage", engine, if_exists="replace", index=False)
    print(f"  ✓ product_usage table: {len(product_usage):,} rows")


def run_ingestion():
    """Execute the full ingestion pipeline."""
    print("=" * 60)
    print("SaaS Churn Data Ingestion Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading raw data...")
    df = load_raw_data()

    print("\n[2/4] Cleaning and reshaping to SaaS context...")
    df = clean_and_reshape(df)

    print("\n[3/4] Connecting to database...")
    engine = get_engine()

    print("\n[4/4] Writing tables to database...")
    write_customers_table(df, engine)
    write_subscriptions_table(df, engine)
    write_product_usage_table(df, engine)

    with engine.connect() as conn:
        for table in ["customers", "subscriptions", "product_usage"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  Validation — {table}: {count:,} rows")

    print("\n✓ Ingestion complete!")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_ingestion()
