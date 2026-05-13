"""
Data ingestion pipeline.

Loads the raw Telco Customer Churn CSV from Kaggle, cleans the data,
relabels columns into a SaaS product context, and writes three relational
tables to the configured database:

    customers       - demographics and account age
    subscriptions   - plan tier, contract, billing, revenue, churn label
    product_usage   - which product modules each customer has enabled

The reshape is intentional. The original dataset describes a telco
customer base; reframing it as a B2B SaaS company makes the rest of the
project (feature engineering, dashboards, API contract) tell a more
realistic story.

Run as a module:
    python -m src.data.ingest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

# Allow running this file as a module from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import RANDOM_STATE, RAW_CSV_PATH
from src.db import get_engine


# Mapping from raw Kaggle column names to the SaaS-flavored schema used
# throughout the project. The keys are exactly what the CSV provides;
# the values are the column names used everywhere downstream.
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
    """Load the raw CSV and report basic shape information."""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns from {path.name}")
    return df


def clean_and_reshape(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column renaming, type fixes, and SaaS-context relabeling.

    The original CSV has a handful of quirks:
      - TotalCharges is stored as a string with blank entries for new customers
      - Yes/No columns need to become integers
      - InternetService values aren't intuitive SaaS terminology
      - Contract and PaymentMethod use long-form strings

    All of those get normalized here so downstream code can rely on
    consistent types and naming.
    """
    df = df.rename(columns=SAAS_COLUMN_MAP)

    # TotalCharges arrives as a string with blanks for very new accounts.
    # Compute the missing values from monthly revenue and account age.
    df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors="coerce")
    missing = df["total_revenue"].isna()
    df.loc[missing, "total_revenue"] = (
        df.loc[missing, "monthly_revenue"] * df.loc[missing, "account_age_months"]
    )
    print(f"  Filled {missing.sum()} blank total_revenue values")

    # Convert Yes/No flags to integers for ML readiness.
    bool_columns = [
        "has_partner", "has_dependents", "has_base_product",
        "paperless_billing", "churn",
    ]
    for col in bool_columns:
        df[col] = (df[col] == "Yes").astype(int)

    # Relabel internet service values into SaaS tiers.
    df["service_tier"] = df["service_tier"].map({
        "DSL": "basic",
        "Fiber optic": "premium",
        "No": "free",
    })

    # Module add-ons: convert Yes/No to integers.
    module_columns = [
        "has_multi_seat", "has_security_module", "has_backup_module",
        "has_protection_module", "has_support_addon",
        "has_analytics_module", "has_reporting_module",
    ]
    for col in module_columns:
        df[col] = (df[col] == "Yes").astype(int)

    # Contract and payment values into short, snake_case identifiers.
    df["contract_type"] = df["contract_type"].map({
        "Month-to-month": "monthly",
        "One year": "annual",
        "Two year": "biennial",
    })
    df["payment_method"] = df["payment_method"].map({
        "Electronic check": "electronic_check",
        "Mailed check": "mailed_check",
        "Bank transfer (automatic)": "bank_transfer_auto",
        "Credit card (automatic)": "credit_card_auto",
    })

    # Derive a plausible signup date by working backwards from account age.
    # Useful for time-based filtering in the dashboard.
    np.random.seed(RANDOM_STATE)
    reference_date = pd.Timestamp("2024-12-01")
    df["signup_date"] = df["account_age_months"].apply(
        lambda m: reference_date - pd.DateOffset(months=int(m))
    )

    print(f"  Cleaned and reshaped {len(df):,} rows")
    return df


def write_customers_table(df: pd.DataFrame, engine) -> None:
    """Persist the customers dimension table."""
    customers = df[[
        "customer_id", "gender", "is_senior", "has_partner",
        "has_dependents", "signup_date", "account_age_months",
    ]].copy()
    customers.to_sql("customers", engine, if_exists="replace", index=False)
    print(f"  customers table written ({len(customers):,} rows)")


def write_subscriptions_table(df: pd.DataFrame, engine) -> None:
    """Persist the subscriptions fact table (one row per customer)."""
    subscriptions = df[[
        "customer_id", "service_tier", "contract_type",
        "paperless_billing", "payment_method",
        "monthly_revenue", "total_revenue", "churn",
    ]].copy()
    subscriptions.to_sql("subscriptions", engine, if_exists="replace", index=False)
    print(f"  subscriptions table written ({len(subscriptions):,} rows)")


def write_product_usage_table(df: pd.DataFrame, engine) -> None:
    """Persist the product module usage table.

    Adds a modules_enabled column that counts how many add-on modules
    the customer is paying for. Used directly as a feature later.
    """
    product_usage = df[[
        "customer_id", "has_base_product", "has_multi_seat",
        "has_security_module", "has_backup_module",
        "has_protection_module", "has_support_addon",
        "has_analytics_module", "has_reporting_module",
    ]].copy()

    module_cols = [c for c in product_usage.columns if c.startswith("has_")]
    product_usage["modules_enabled"] = product_usage[module_cols].sum(axis=1)

    product_usage.to_sql("product_usage", engine, if_exists="replace", index=False)
    print(f"  product_usage table written ({len(product_usage):,} rows)")


def run_ingestion() -> pd.DataFrame:
    """Execute the full ingestion pipeline end-to-end."""
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

    # Quick row-count validation to confirm the writes succeeded.
    with engine.connect() as conn:
        for table in ["customers", "subscriptions", "product_usage"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  Validation: {table} = {count:,} rows")

    print("\nIngestion complete.")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_ingestion()
