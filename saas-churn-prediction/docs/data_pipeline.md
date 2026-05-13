# Data Pipeline

This document walks through how raw data becomes the 45-column feature set the model is trained on.

## Pipeline Overview

```
Raw CSV  ─►  Ingestion  ─►  Synthetic Data  ─►  Feature Engineering  ─►  feature_store
            (Python)         (Python)             (SQL + Python)
```

Each stage is an idempotent script that reads from and writes to the database. Re-running the whole pipeline from scratch takes about thirty seconds on a laptop.

## Stage 1: Ingestion (`src/data/ingest.py`)

**Input:** `data/raw/telco_churn_raw.csv` (7,043 rows, 21 columns from Kaggle)

**Output:** Three relational tables in the database

The Kaggle Telco Customer Churn dataset is the starting point. It describes telco customers but is structurally similar to a B2B SaaS customer base: monthly revenue, contract length, product add-ons, demographic flags, and a binary churn label. I relabel the columns to use SaaS-appropriate names (`MonthlyCharges` becomes `monthly_revenue`, `tenure` becomes `account_age_months`, `InternetService` becomes `service_tier` with values `free`, `basic`, `premium`).

This stage handles a handful of data quality issues that are typical of real-world data:

- `TotalCharges` is stored as a string with empty values for very new customers. I fill missing values by computing `monthly_revenue * account_age_months`.
- Yes/No columns become integers for ML readiness.
- Contract and payment method strings become snake_case identifiers.
- A synthetic `signup_date` is derived by working backwards from `account_age_months` against a fixed reference date. This makes the dataset feel less artificial when displayed in the dashboard.

The output is split across three tables to demonstrate relational modeling:

- `customers` - demographics and account age
- `subscriptions` - plan tier, contract, revenue, churn label
- `product_usage` - which product modules each customer has enabled

A real SaaS company would have these as separate sources of truth anyway (CRM, billing system, product analytics), so the split reflects how the data actually arrives.

## Stage 2: Synthetic Data Generation (`src/data/generate_synthetic.py`)

**Input:** The `customers` and `subscriptions` tables from Stage 1

**Output:** `usage_events` (one row per customer per week, four weeks) and `support_tickets` (variable count per customer)

The Telco dataset has no behavioral data. In a real SaaS company, behavioral signals - declining logins, low feature adoption, escalating support tickets - are typically the earliest churn indicators. Without them, a model trained only on demographic and contract features misses the most actionable patterns.

I generate behavioral data that's correlated with the existing churn label. The generation is deliberately noisy - the model still has to work for its performance:

- Engagement (logins, session duration, feature adoption) is sampled from different distributions for churners and active customers
- A trend term makes churners show declining engagement over the four-week window, while active customers stay flat or grow
- Support ticket volume, severity, and resolution rates skew worse for churners
- Premium tier customers are slightly more engaged on average regardless of churn status

The exact distributions are documented inline in `generate_synthetic.py`. The seed is fixed (`RANDOM_STATE = 42`), so re-running produces identical synthetic data.

**Why this matters for the modeling story:** the performance numbers reflect both real patterns (tenure, contract type, monthly revenue do genuinely predict churn) and patterns I deliberately built in (declining logins, unresolved tickets). The model isn't memorizing - the noise is high enough that it has to learn - but the synthetic correlations contribute meaningfully to recall. This is documented honestly here rather than tucked away.

## Stage 3: Feature Engineering (`src/data/feature_engineering.py`)

**Input:** All five tables from Stages 1 and 2

**Output:** `feature_store` table and `data/processed/features.csv`

This stage joins everything together and produces the analytics-ready feature set. Most of the heavy lifting happens in SQL via `src/sql/features.sql`. A small number of features are added in Python afterwards because they're awkward to express in SQL.

### The SQL stage

`features.sql` uses three CTEs to roll up the event-level data:

- `usage_agg` aggregates the weekly usage data to one row per customer (average logins, max logins, average session duration, etc.)
- `usage_trend` compares engagement in the second half of the window (weeks 3-4) against the first half (weeks 1-2). A negative trend is a strong churn signal even when overall engagement is still positive.
- `ticket_agg` aggregates support history (total tickets, critical tickets, resolution rate, etc.)

The final `SELECT` joins the three CTEs to the dimension tables. `LEFT JOIN`s mean customers with no usage events or support tickets still appear in the output, with NULLs handled by `COALESCE`.

A few interesting features are computed directly in the SQL:

- `account_segment` (new / established / mature) via a `CASE` on tenure
- `engagement_risk_flag` - a binary flag triggered by declining logins combined with low feature adoption
- `support_risk_flag` - a binary flag triggered by unresolved or critical tickets
- `rule_based_risk_score` - a simple count of risk factors, useful as a sanity check against the model

The rule-based score isn't used by the model (which would be circular) but provides a baseline comparison. The model should outperform it - if it doesn't, something is wrong.

### The Python stage

A small set of features are easier to express in pandas than in SQL:

- `revenue_per_module` - average revenue per enabled module, with a divide-by-zero guard
- `lifetime_value_indicator` - total revenue per month of tenure, a rough proxy for customer worth
- `engagement_intensity` - a weighted composite of login frequency, session length, and feature adoption, normalized to a 0-1 scale
- `tickets_per_month` - ticket rate normalized by tenure (a stronger signal than raw counts)
- One-hot encoding of `account_segment`

All of these could be done in SQL with enough effort, but the readability cost would be high. I made the call that the boundary between SQL and Python should be based on which language expresses the intent most clearly, not on dogmatic adherence to one or the other.

## Output

The final feature set has 45 columns and 7,043 rows. The `customer_id` column is preserved for joining back to scored predictions. The `churn` column is the target variable. The remaining 43 columns are features available to the model.

The output is written to both the database (`feature_store` table) and a CSV file (`data/processed/features.csv`). The CSV is what the training pipeline reads - it's faster to load and doesn't require a database connection, which makes it easier to run notebooks in Colab.

## Running the Pipeline

End-to-end with the Makefile:

```bash
make pipeline
```

Or step by step:

```bash
python -m src.data.ingest
python -m src.data.generate_synthetic
python -m src.data.feature_engineering
```

Each step prints validation output to confirm row counts and basic data quality. If a step fails, the error message points at which assertion failed and what to check.

## Data Validation

`feature_engineering.py` runs a small set of validation checks at the end:

- No null values in the feature set
- Expected churn rate (~26.5%)
- Sensible ranges for monthly revenue, account age, and login counts
- Total feature count matches expectations

These are intentionally lightweight. A production pipeline would use something like Great Expectations or dbt tests, with documented expectations for every column. For a portfolio project, the validation here is enough to catch the kinds of regressions I'd realistically introduce when modifying the code.
