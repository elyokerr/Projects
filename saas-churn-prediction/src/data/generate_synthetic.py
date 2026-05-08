"""
Synthetic Engagement Data Generator
=====================================
Generates realistic SaaS usage and engagement data for each customer.
The generated data is intentionally correlated with churn status to produce
meaningful features, while adding enough noise to avoid data leakage.

Tables created:
    - usage_events     : weekly login counts, feature usage, session duration
    - support_tickets  : customer support interactions with severity/resolution

Usage:
    python -m src.data.generate_synthetic
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import RANDOM_STATE
from src.db import get_engine


np.random.seed(RANDOM_STATE)


def load_base_data(engine) -> pd.DataFrame:
    """Load customer + subscription data to correlate engagement with."""
    query = """
        SELECT
            c.customer_id,
            c.account_age_months,
            s.churn,
            s.service_tier,
            s.monthly_revenue,
            s.contract_type
        FROM customers c
        JOIN subscriptions s ON c.customer_id = s.customer_id
    """
    return pd.read_sql(query, engine)


def generate_usage_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate weekly usage metrics for each customer.

    Churned customers tend to have:
    - Fewer logins per week
    - Lower feature adoption rate
    - Shorter session durations
    - Declining usage trends
    """
    records = []

    for _, row in df.iterrows():
        is_churner = row["churn"] == 1
        tier = row["service_tier"]
        tenure = row["account_age_months"]

        if is_churner:
            base_logins = np.random.uniform(1, 8)
            base_session_min = np.random.uniform(3, 15)
            base_feature_pct = np.random.uniform(0.05, 0.35)
            trend = np.random.uniform(-0.3, -0.02)
        else:
            base_logins = np.random.uniform(4, 20)
            base_session_min = np.random.uniform(8, 45)
            base_feature_pct = np.random.uniform(0.20, 0.80)
            trend = np.random.uniform(-0.05, 0.15)

        tier_boost = {"premium": 1.3, "basic": 1.0, "free": 0.6}.get(tier, 1.0)

        for week in range(1, 5):
            week_factor = 1 + trend * (week / 4)
            noise = np.random.normal(1.0, 0.2)

            logins = max(0, int(base_logins * tier_boost * week_factor * noise))
            session_duration = max(0, round(base_session_min * tier_boost * week_factor * noise, 1))
            features_used = max(0, min(1.0, round(base_feature_pct * week_factor * noise, 2)))
            api_calls = max(0, int(logins * np.random.uniform(2, 10) * tier_boost))
            pages_viewed = max(0, int(logins * np.random.uniform(3, 12)))

            records.append({
                "customer_id": row["customer_id"],
                "week_number": week,
                "logins": logins,
                "avg_session_minutes": session_duration,
                "feature_adoption_rate": features_used,
                "api_calls": api_calls,
                "pages_viewed": pages_viewed,
            })

    result = pd.DataFrame(records)
    print(f"  Generated {len(result):,} usage event rows ({len(df):,} customers × 4 weeks)")
    return result


def generate_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate support ticket history for each customer.

    Churned customers tend to have:
    - More tickets (frustration)
    - Higher severity tickets
    - More unresolved tickets
    - Longer resolution times
    """
    records = []
    severity_levels = ["low", "medium", "high", "critical"]
    categories = [
        "billing_issue", "bug_report", "feature_request",
        "onboarding_help", "performance_complaint", "account_access",
    ]

    for _, row in df.iterrows():
        is_churner = row["churn"] == 1
        tenure = row["account_age_months"]

        if is_churner:
            n_tickets = np.random.choice(range(0, 8), p=[0.10, 0.15, 0.25, 0.20, 0.15, 0.08, 0.05, 0.02])
        else:
            n_tickets = np.random.choice(range(0, 8), p=[0.25, 0.30, 0.20, 0.12, 0.07, 0.03, 0.02, 0.01])

        for t in range(n_tickets):
            if is_churner:
                severity = np.random.choice(severity_levels, p=[0.15, 0.30, 0.35, 0.20])
            else:
                severity = np.random.choice(severity_levels, p=[0.35, 0.35, 0.20, 0.10])

            category = np.random.choice(categories)

            if is_churner:
                resolved = np.random.choice([0, 1], p=[0.35, 0.65])
            else:
                resolved = np.random.choice([0, 1], p=[0.10, 0.90])

            if resolved:
                base_hours = {"low": 4, "medium": 12, "high": 24, "critical": 48}[severity]
                resolution_hours = max(1, int(base_hours * np.random.uniform(0.5, 2.5)))
                if is_churner:
                    resolution_hours = int(resolution_hours * np.random.uniform(1.2, 2.0))
            else:
                resolution_hours = None

            days_ago = np.random.randint(1, min(90, max(30, tenure * 30)))

            records.append({
                "customer_id": row["customer_id"],
                "ticket_number": t + 1,
                "category": category,
                "severity": severity,
                "resolved": resolved,
                "resolution_hours": resolution_hours,
                "days_ago": days_ago,
            })

    result = pd.DataFrame(records)
    print(f"  Generated {len(result):,} support tickets for {result['customer_id'].nunique():,} customers")
    return result


def run_synthetic_generation():
    """Execute the full synthetic data generation pipeline."""
    print("=" * 60)
    print("Synthetic Engagement Data Generation")
    print("=" * 60)

    print("\n[1/4] Connecting to database and loading base data...")
    engine = get_engine()
    df = load_base_data(engine)
    print(f"  Loaded {len(df):,} customers")

    print("\n[2/4] Generating usage events...")
    usage_events = generate_usage_events(df)

    print("\n[3/4] Generating support tickets...")
    support_tickets = generate_support_tickets(df)

    print("\n[4/4] Writing to database...")
    usage_events.to_sql("usage_events", engine, if_exists="replace", index=False)
    print(f"  ✓ usage_events table: {len(usage_events):,} rows")

    support_tickets.to_sql("support_tickets", engine, if_exists="replace", index=False)
    print(f"  ✓ support_tickets table: {len(support_tickets):,} rows")

    print("\n─── Engagement Summary ───")
    churners = df[df["churn"] == 1]["customer_id"]
    active = df[df["churn"] == 0]["customer_id"]

    churn_logins = usage_events[usage_events["customer_id"].isin(churners)]["logins"].mean()
    active_logins = usage_events[usage_events["customer_id"].isin(active)]["logins"].mean()
    print(f"  Avg weekly logins — churners: {churn_logins:.1f}, active: {active_logins:.1f}")

    churn_tickets = support_tickets[support_tickets["customer_id"].isin(churners)].groupby("customer_id").size().mean()
    active_tickets = support_tickets[support_tickets["customer_id"].isin(active)].groupby("customer_id").size().mean()
    print(f"  Avg tickets — churners: {churn_tickets:.1f}, active: {active_tickets:.1f}")

    print("\n✓ Synthetic data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_synthetic_generation()
