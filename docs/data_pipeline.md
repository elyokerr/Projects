# Data Pipeline Documentation

> How raw customer data is transformed into 45 predictive features through a structured, reproducible pipeline.

---

## Overview

The data pipeline takes a single raw CSV file of customer records and transforms it into a rich, analysis-ready feature set stored in a relational database. This mirrors how data teams in industry work — raw data rarely arrives in a format suitable for modelling, and the feature engineering process is often where the most business value is created.

The pipeline has three stages, each handled by a separate Python module:

1. **Ingestion** — Load and reshape raw data into a SaaS-appropriate schema
2. **Synthetic Data Generation** — Create realistic engagement and support data that wouldn't exist in the original dataset
3. **Feature Engineering** — Combine SQL queries and Python transformations to produce 45 predictive features

---

## Stage 1: Data Ingestion (`src/data/ingest.py`)

### What it does

The raw dataset is the Telco Customer Churn dataset from Kaggle — a commonly used dataset for churn analysis. However, it's structured around a telecoms company, not a SaaS business. The ingestion step reshapes the data to simulate a B2B SaaS context.

### Transformations applied

- Column names are remapped to SaaS terminology (e.g., "MonthlyCharges" becomes monthly recurring revenue, "Contract" becomes subscription plan tier)
- Data types are cleaned and standardised (currency fields parsed, categorical values normalised)
- Records are written to a relational database with a proper schema

### Database schema

The `src/sql/init.sql` file defines the database structure:

- **customers** — Core customer records (demographics, account details)
- **subscriptions** — Plan type, contract length, billing information
- **usage_events** — How actively each customer uses the product (generated in Stage 2)
- **support_tickets** — Customer support interactions (generated in Stage 2)
- **churn_labels** — The target variable indicating whether a customer churned
- **features** — The final engineered feature set used for modelling

### Why this matters

In industry, data scientists rarely work with clean, single-table CSVs. Real data lives in relational databases with multiple tables that need to be joined and transformed. This step demonstrates understanding of database design, data modelling, and ETL (Extract, Transform, Load) processes.

---

## Stage 2: Synthetic Data Generation (`src/data/generate_synthetic.py`)

### What it does

The original Kaggle dataset doesn't include behavioural engagement data — things like how often customers log in, how many features they use, or how many support tickets they've submitted. In a real SaaS company, this data would come from product analytics tools and CRM systems.

This stage generates realistic synthetic data to fill that gap, creating:

- **Usage events** — Login frequency, feature adoption, session duration
- **Support tickets** — Ticket counts, response times, escalation rates

### How the synthetic data is generated

The generation is not random — it's correlated with churn behaviour. Customers who churned tend to have lower engagement scores, fewer logins, and more support tickets. This mirrors real-world patterns where declining product usage is one of the strongest churn signals.

### Why this matters

This demonstrates the ability to think critically about what data would exist in a real environment and to generate realistic test data when working with limited datasets. It also shows understanding of the relationship between customer behaviour and business outcomes.

---

## Stage 3: Feature Engineering (`src/data/feature_engineering.py` + `src/sql/features.sql`)

### What it does

This is the most technically substantial stage. It combines SQL-based feature queries with Python transformations to produce 45 predictive features across five categories.

### Feature categories

**Subscription Features** — Plan tier, contract length, monthly revenue, tenure, payment method. These capture the customer's commercial relationship with the product.

**Usage & Engagement Features** — Login frequency, feature adoption rate, days since last login, session trends. These capture how actively the customer is using the product. Declining engagement is often the earliest warning sign of churn.

**Support Features** — Total tickets, recent ticket velocity, time between tickets, escalation count. High support activity can indicate frustration, while zero support activity might mean the customer has disengaged entirely.

**Financial Features** — Revenue per tenure month, total contract value, price sensitivity indicators. These help the model understand the economic relationship and identify customers who may be churning due to cost.

**Behavioural Ratios & Trends** — Engagement-to-tenure ratios, support-to-usage ratios, trend indicators. These derived features capture patterns that single features cannot — for example, a customer with high tenure but declining engagement is a very different risk profile from a new customer with low engagement.

### SQL techniques used

The `src/sql/features.sql` file demonstrates several intermediate-to-advanced SQL techniques:

- **Common Table Expressions (CTEs)** — Breaking complex queries into readable, logical steps
- **Window Functions** — Calculating running averages, rankings, and time-based comparisons within customer groups
- **Aggregations with GROUP BY** — Summarising transaction-level data to customer-level features
- **CASE statements** — Creating categorical features and handling edge cases
- **JOINs across multiple tables** — Combining data from customers, subscriptions, usage, and support tables

### Why this matters

Feature engineering is consistently cited by hiring managers as the skill that separates strong candidates from average ones. Anyone can call `model.fit()` — the value is in knowing which features to create and why they matter to the business. Using SQL for feature engineering (rather than doing everything in pandas) demonstrates readiness for production environments where data lives in databases, not CSV files.

---

## Data Flow Diagram

```
telco_churn_raw.csv
       │
       ▼
  ┌─────────┐
  │ ingest  │──→ customers, subscriptions, churn_labels tables
  └─────────┘
       │
       ▼
  ┌──────────────────┐
  │ generate_synthetic│──→ usage_events, support_tickets tables
  └──────────────────┘
       │
       ▼
  ┌────────────────────┐
  │ feature_engineering │──→ features table → features.csv (45 columns)
  │  (SQL + Python)     │
  └────────────────────┘
       │
       ▼
  data/processed/features.csv  ←  Ready for model training
```

---

## Reproducibility

The entire pipeline is deterministic — running it again on the same raw data produces the same output. Random seeds are set for all synthetic data generation. The pipeline can be run either through the notebook (`notebooks/01_full_pipeline.ipynb`) or as standalone modules via the command line.
