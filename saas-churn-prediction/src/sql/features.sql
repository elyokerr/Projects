-- =============================================================
-- SaaS Churn Prediction — SQL Feature Engineering
-- =============================================================
-- This file contains the feature engineering logic written in SQL.
-- Demonstrates: CTEs, window functions, aggregations, CASE
-- expressions, and multi-table joins.
--
-- These queries are executed by feature_engineering.py and the
-- results are stored in the 'feature_store' table.
-- =============================================================


-- ─── Feature Set 1: Customer Demographics & Account ──────────
-- Basic customer-level attributes

-- (handled inline in the main query below)


-- ─── Feature Set 2: Subscription & Revenue Features ─────────
-- Revenue patterns, contract risk signals

-- (handled inline in the main query below)


-- ─── Feature Set 3: Product Adoption Features ───────────────
-- Module usage patterns and adoption depth

-- (handled inline in the main query below)


-- ─── Feature Set 4: Engagement Metrics (aggregated) ─────────
-- Usage intensity, trends, and session behaviour

-- (handled inline in the main query below)


-- ─── Feature Set 5: Support Ticket Features ─────────────────
-- Ticket volume, severity, resolution quality

-- (handled inline in the main query below)


-- =============================================================
-- MAIN FEATURE ENGINEERING QUERY
-- Combines all feature sets into a single analytics-ready table
-- =============================================================

WITH usage_agg AS (
    -- Aggregate weekly usage events into customer-level metrics
    SELECT
        customer_id,
        AVG(logins) AS avg_weekly_logins,
        MAX(logins) AS max_weekly_logins,
        MIN(logins) AS min_weekly_logins,
        AVG(avg_session_minutes) AS avg_session_minutes,
        AVG(feature_adoption_rate) AS avg_feature_adoption,
        SUM(api_calls) AS total_api_calls,
        SUM(pages_viewed) AS total_pages_viewed
    FROM usage_events
    GROUP BY customer_id
),

usage_trend AS (
    -- Calculate login trend: compare last 2 weeks vs first 2 weeks
    -- Positive = growing engagement, negative = declining
    SELECT
        customer_id,
        AVG(CASE WHEN week_number >= 3 THEN logins ELSE NULL END)
        - AVG(CASE WHEN week_number <= 2 THEN logins ELSE NULL END) AS login_trend,
        AVG(CASE WHEN week_number >= 3 THEN avg_session_minutes ELSE NULL END)
        - AVG(CASE WHEN week_number <= 2 THEN avg_session_minutes ELSE NULL END) AS session_trend
    FROM usage_events
    GROUP BY customer_id
),

ticket_agg AS (
    -- Aggregate support tickets into customer-level features
    SELECT
        customer_id,
        COUNT(*) AS total_tickets,
        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS critical_tickets,
        SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) AS high_severity_tickets,
        SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) AS unresolved_tickets,
        AVG(CASE WHEN resolved = 1 THEN resolution_hours ELSE NULL END) AS avg_resolution_hours,
        ROUND(
            CAST(SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) AS FLOAT)
            / NULLIF(COUNT(*), 0), 2
        ) AS resolution_rate
    FROM support_tickets
    GROUP BY customer_id
)

-- Final feature assembly
SELECT
    -- ─── Identifiers ───
    c.customer_id,

    -- ─── Demographics ───
    CASE WHEN c.gender = 'Male' THEN 1 ELSE 0 END AS is_male,
    c.is_senior,
    c.has_partner,
    c.has_dependents,

    -- ─── Account Features ───
    c.account_age_months,
    CASE
        WHEN c.account_age_months <= 6 THEN 'new'
        WHEN c.account_age_months <= 24 THEN 'established'
        ELSE 'mature'
    END AS account_segment,

    -- ─── Subscription Features ───
    CASE
        WHEN s.service_tier = 'free' THEN 0
        WHEN s.service_tier = 'basic' THEN 1
        WHEN s.service_tier = 'premium' THEN 2
    END AS service_tier_encoded,
    CASE
        WHEN s.contract_type = 'monthly' THEN 0
        WHEN s.contract_type = 'annual' THEN 1
        WHEN s.contract_type = 'biennial' THEN 2
    END AS contract_type_encoded,
    s.paperless_billing,
    s.monthly_revenue,
    s.total_revenue,
    CASE
        WHEN c.account_age_months > 0
        THEN ROUND(s.total_revenue / c.account_age_months, 2)
        ELSE s.monthly_revenue
    END AS avg_monthly_spend,

    -- ─── Product Adoption ───
    p.modules_enabled,
    p.has_security_module,
    p.has_backup_module,
    p.has_protection_module,
    p.has_support_addon,
    p.has_analytics_module,
    p.has_reporting_module,
    -- Adoption depth: percentage of available modules used
    ROUND(p.modules_enabled / 8.0, 2) AS module_adoption_rate,

    -- ─── Engagement Metrics ───
    COALESCE(u.avg_weekly_logins, 0) AS avg_weekly_logins,
    COALESCE(u.max_weekly_logins, 0) AS max_weekly_logins,
    COALESCE(u.min_weekly_logins, 0) AS min_weekly_logins,
    COALESCE(u.avg_session_minutes, 0) AS avg_session_minutes,
    COALESCE(u.avg_feature_adoption, 0) AS avg_feature_adoption,
    COALESCE(u.total_api_calls, 0) AS total_api_calls,
    COALESCE(u.total_pages_viewed, 0) AS total_pages_viewed,

    -- ─── Engagement Trends ───
    COALESCE(ut.login_trend, 0) AS login_trend,
    COALESCE(ut.session_trend, 0) AS session_trend,
    -- Engagement risk flag: declining logins + low adoption
    CASE
        WHEN COALESCE(ut.login_trend, 0) < -1
             AND COALESCE(u.avg_feature_adoption, 0) < 0.25
        THEN 1 ELSE 0
    END AS engagement_risk_flag,

    -- ─── Support Features ───
    COALESCE(t.total_tickets, 0) AS total_tickets,
    COALESCE(t.critical_tickets, 0) AS critical_tickets,
    COALESCE(t.high_severity_tickets, 0) AS high_severity_tickets,
    COALESCE(t.unresolved_tickets, 0) AS unresolved_tickets,
    COALESCE(t.avg_resolution_hours, 0) AS avg_resolution_hours,
    COALESCE(t.resolution_rate, 1.0) AS resolution_rate,
    -- Support risk flag: multiple unresolved or critical tickets
    CASE
        WHEN COALESCE(t.unresolved_tickets, 0) >= 2
             OR COALESCE(t.critical_tickets, 0) >= 1
        THEN 1 ELSE 0
    END AS support_risk_flag,

    -- ─── Composite Risk Score (rule-based, pre-ML) ───
    (
        CASE WHEN s.contract_type = 'monthly' THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(ut.login_trend, 0) < -1 THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(t.unresolved_tickets, 0) >= 2 THEN 1 ELSE 0 END
        + CASE WHEN p.modules_enabled <= 2 THEN 1 ELSE 0 END
        + CASE WHEN c.account_age_months <= 6 THEN 1 ELSE 0 END
    ) AS rule_based_risk_score,

    -- ─── Target ───
    s.churn

FROM customers c
JOIN subscriptions s ON c.customer_id = s.customer_id
JOIN product_usage p ON c.customer_id = p.customer_id
LEFT JOIN usage_agg u ON c.customer_id = u.customer_id
LEFT JOIN usage_trend ut ON c.customer_id = ut.customer_id
LEFT JOIN ticket_agg t ON c.customer_id = t.customer_id;
