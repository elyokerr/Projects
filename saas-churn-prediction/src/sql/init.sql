-- =============================================================
-- SaaS Churn Prediction - Database Schema
-- =============================================================
-- This file defines the PostgreSQL schema used by the project.
-- It serves two purposes:
--   1. Documentation: a single place to see what tables exist and how
--      they relate.
--   2. Optional initialization: can be run manually to create the
--      schema, though the Python ingestion scripts also create tables
--      automatically via SQLAlchemy `if_exists="replace"`.
--
-- The same schema is used for SQLite (local development) - the syntax
-- here uses standard SQL that works on both engines.
-- =============================================================


-- Drop existing tables for a clean reinitialization.
DROP TABLE IF EXISTS feature_store CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS usage_events CASCADE;
DROP TABLE IF EXISTS product_usage CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;


-- --- Core dimension and fact tables -------------------------------

CREATE TABLE customers (
    customer_id          VARCHAR(20) PRIMARY KEY,
    gender               VARCHAR(10),
    is_senior            INTEGER,
    has_partner          INTEGER,
    has_dependents       INTEGER,
    signup_date          DATE,
    account_age_months   INTEGER
);

CREATE TABLE subscriptions (
    customer_id          VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id),
    service_tier         VARCHAR(10),    -- free, basic, premium
    contract_type        VARCHAR(10),    -- monthly, annual, biennial
    paperless_billing    INTEGER,
    payment_method       VARCHAR(30),
    monthly_revenue      NUMERIC(10, 2),
    total_revenue        NUMERIC(10, 2),
    churn                INTEGER         -- Target variable: 1 = churned, 0 = active
);

CREATE TABLE product_usage (
    customer_id          VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id),
    has_base_product     INTEGER,
    has_multi_seat       INTEGER,
    has_security_module  INTEGER,
    has_backup_module    INTEGER,
    has_protection_module INTEGER,
    has_support_addon    INTEGER,
    has_analytics_module INTEGER,
    has_reporting_module INTEGER,
    modules_enabled      INTEGER         -- Sum of all has_* flags
);


-- --- Synthetic engagement tables (populated by generate_synthetic) -

CREATE TABLE usage_events (
    customer_id            VARCHAR(20) REFERENCES customers(customer_id),
    week_number            INTEGER,
    logins                 INTEGER,
    avg_session_minutes    NUMERIC(6, 1),
    feature_adoption_rate  NUMERIC(4, 2),
    api_calls              INTEGER,
    pages_viewed           INTEGER,
    PRIMARY KEY (customer_id, week_number)
);

CREATE TABLE support_tickets (
    customer_id        VARCHAR(20) REFERENCES customers(customer_id),
    ticket_number      INTEGER,
    category           VARCHAR(30),
    severity           VARCHAR(10),   -- low, medium, high, critical
    resolved           INTEGER,
    resolution_hours   INTEGER,
    days_ago           INTEGER,
    PRIMARY KEY (customer_id, ticket_number)
);


-- --- Feature store (populated by feature_engineering.py) ----------
-- Schema is dynamic based on the output of features.sql.
-- The table is created automatically by pandas.to_sql().


-- --- Indexes for query performance --------------------------------

CREATE INDEX idx_usage_customer ON usage_events(customer_id);
CREATE INDEX idx_tickets_customer ON support_tickets(customer_id);
CREATE INDEX idx_subscriptions_churn ON subscriptions(churn);
