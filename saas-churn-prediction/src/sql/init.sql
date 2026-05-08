-- =============================================================
-- SaaS Churn Prediction — Database Schema
-- =============================================================
-- Run this to initialise the PostgreSQL schema.
-- Tables are also created by the Python ingestion scripts,
-- but this file documents the intended schema for reference.
-- =============================================================

-- Drop existing tables (for clean re-runs)
DROP TABLE IF EXISTS feature_store CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS usage_events CASCADE;
DROP TABLE IF EXISTS product_usage CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ─── Core Tables ─────────────────────────────────────────────

CREATE TABLE customers (
    customer_id     VARCHAR(20) PRIMARY KEY,
    gender          VARCHAR(10),
    is_senior       INTEGER,
    has_partner     INTEGER,
    has_dependents  INTEGER,
    signup_date     DATE,
    account_age_months INTEGER
);

CREATE TABLE subscriptions (
    customer_id     VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id),
    service_tier    VARCHAR(10),    -- free, basic, premium
    contract_type   VARCHAR(10),    -- monthly, annual, biennial
    paperless_billing INTEGER,
    payment_method  VARCHAR(30),
    monthly_revenue NUMERIC(10, 2),
    total_revenue   NUMERIC(10, 2),
    churn           INTEGER         -- target variable
);

CREATE TABLE product_usage (
    customer_id         VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id),
    has_base_product    INTEGER,
    has_multi_seat      INTEGER,
    has_security_module INTEGER,
    has_backup_module   INTEGER,
    has_protection_module INTEGER,
    has_support_addon   INTEGER,
    has_analytics_module INTEGER,
    has_reporting_module INTEGER,
    modules_enabled     INTEGER
);

-- ─── Engagement Tables (synthetic) ──────────────────────────

CREATE TABLE usage_events (
    customer_id         VARCHAR(20) REFERENCES customers(customer_id),
    week_number         INTEGER,
    logins              INTEGER,
    avg_session_minutes NUMERIC(6, 1),
    feature_adoption_rate NUMERIC(4, 2),
    api_calls           INTEGER,
    pages_viewed        INTEGER,
    PRIMARY KEY (customer_id, week_number)
);

CREATE TABLE support_tickets (
    customer_id     VARCHAR(20) REFERENCES customers(customer_id),
    ticket_number   INTEGER,
    category        VARCHAR(30),
    severity        VARCHAR(10),
    resolved        INTEGER,
    resolution_hours INTEGER,
    days_ago        INTEGER,
    PRIMARY KEY (customer_id, ticket_number)
);

-- ─── Feature Store (populated by feature engineering) ────────
-- Schema is dynamic based on features.sql output
-- Created automatically by feature_engineering.py

-- ─── Indexes for query performance ──────────────────────────
CREATE INDEX idx_usage_customer ON usage_events(customer_id);
CREATE INDEX idx_tickets_customer ON support_tickets(customer_id);
CREATE INDEX idx_subscriptions_churn ON subscriptions(churn);
