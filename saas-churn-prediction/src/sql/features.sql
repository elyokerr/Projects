-- =============================================================
-- SaaS Churn Prediction - SQL Feature Engineering
-- =============================================================
-- A single CTE-based query that assembles the customer-level feature
-- set used for model training. It demonstrates the SQL patterns that
-- come up most often in a real analytics workload:
--
--   - Common Table Expressions (CTEs) for readable multi-step logic
--   - Aggregations to roll up event-level data to customer level
--   - Window-style trend comparisons (first half vs. second half)
--   - CASE expressions for derived categoricals and risk flags
--   - LEFT JOINs to combine multiple data sources gracefully
--   - COALESCE to handle customers with no engagement or ticket data
--
-- The output of this file is consumed by feature_engineering.py, which
-- adds a small number of additional Python-computed features and
-- writes the final feature_store table.
-- =============================================================


WITH usage_agg AS (
    -- Roll up the weekly usage_events data to one row per customer.
    SELECT
        customer_id,
        AVG(logins)                  AS avg_weekly_logins,
        MAX(logins)                  AS max_weekly_logins,
        MIN(logins)                  AS min_weekly_logins,
        AVG(avg_session_minutes)     AS avg_session_minutes,
        AVG(feature_adoption_rate)   AS avg_feature_adoption,
        SUM(api_calls)               AS total_api_calls,
        SUM(pages_viewed)            AS total_pages_viewed
    FROM usage_events
    GROUP BY customer_id
),

usage_trend AS (
    -- Compare engagement in the second half of the window (weeks 3-4)
    -- against the first half (weeks 1-2). A negative trend is a strong
    -- churn signal even when overall engagement is still positive.
    SELECT
        customer_id,
        AVG(CASE WHEN week_number >= 3 THEN logins END)
        - AVG(CASE WHEN week_number <= 2 THEN logins END)
            AS login_trend,
        AVG(CASE WHEN week_number >= 3 THEN avg_session_minutes END)
        - AVG(CASE WHEN week_number <= 2 THEN avg_session_minutes END)
            AS session_trend
    FROM usage_events
    GROUP BY customer_id
),

ticket_agg AS (
    -- Aggregate support history into customer-level summary features.
    SELECT
        customer_id,
        COUNT(*)                                                       AS total_tickets,
        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END)         AS critical_tickets,
        SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END)             AS high_severity_tickets,
        SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END)                  AS unresolved_tickets,
        AVG(CASE WHEN resolved = 1 THEN resolution_hours END)          AS avg_resolution_hours,
        ROUND(
            CAST(SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) AS FLOAT)
            / NULLIF(COUNT(*), 0), 2
        ) AS resolution_rate
    FROM support_tickets
    GROUP BY customer_id
)


-- Final assembly: join everything together with the customer dimension
-- and subscription facts. LEFT JOINs ensure customers with no usage
-- events or support tickets still appear, with NULLs handled by
-- COALESCE below.

SELECT
    c.customer_id,

    -- Demographics
    CASE WHEN c.gender = 'Male' THEN 1 ELSE 0 END AS is_male,
    c.is_senior,
    c.has_partner,
    c.has_dependents,

    -- Account
    c.account_age_months,
    CASE
        WHEN c.account_age_months <= 6  THEN 'new'
        WHEN c.account_age_months <= 24 THEN 'established'
        ELSE 'mature'
    END AS account_segment,

    -- Subscription features (integer-encoded for downstream ML use)
    CASE
        WHEN s.service_tier = 'free'    THEN 0
        WHEN s.service_tier = 'basic'   THEN 1
        WHEN s.service_tier = 'premium' THEN 2
    END AS service_tier_encoded,
    CASE
        WHEN s.contract_type = 'monthly'  THEN 0
        WHEN s.contract_type = 'annual'   THEN 1
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

    -- Product adoption
    p.modules_enabled,
    p.has_security_module,
    p.has_backup_module,
    p.has_protection_module,
    p.has_support_addon,
    p.has_analytics_module,
    p.has_reporting_module,
    ROUND(p.modules_enabled / 8.0, 2) AS module_adoption_rate,

    -- Engagement metrics (with COALESCE defaults for missing data)
    COALESCE(u.avg_weekly_logins,    0) AS avg_weekly_logins,
    COALESCE(u.max_weekly_logins,    0) AS max_weekly_logins,
    COALESCE(u.min_weekly_logins,    0) AS min_weekly_logins,
    COALESCE(u.avg_session_minutes,  0) AS avg_session_minutes,
    COALESCE(u.avg_feature_adoption, 0) AS avg_feature_adoption,
    COALESCE(u.total_api_calls,      0) AS total_api_calls,
    COALESCE(u.total_pages_viewed,   0) AS total_pages_viewed,

    -- Engagement trends
    COALESCE(ut.login_trend,   0) AS login_trend,
    COALESCE(ut.session_trend, 0) AS session_trend,

    -- Compound risk flag: declining logins combined with low adoption
    CASE
        WHEN COALESCE(ut.login_trend, 0) < -1
             AND COALESCE(u.avg_feature_adoption, 0) < 0.25
        THEN 1 ELSE 0
    END AS engagement_risk_flag,

    -- Support features
    COALESCE(t.total_tickets,         0)   AS total_tickets,
    COALESCE(t.critical_tickets,      0)   AS critical_tickets,
    COALESCE(t.high_severity_tickets, 0)   AS high_severity_tickets,
    COALESCE(t.unresolved_tickets,    0)   AS unresolved_tickets,
    COALESCE(t.avg_resolution_hours,  0)   AS avg_resolution_hours,
    COALESCE(t.resolution_rate,       1.0) AS resolution_rate,

    -- Compound risk flag: meaningful unresolved or critical ticket volume
    CASE
        WHEN COALESCE(t.unresolved_tickets, 0) >= 2
             OR COALESCE(t.critical_tickets, 0) >= 1
        THEN 1 ELSE 0
    END AS support_risk_flag,

    -- Composite rule-based risk score (pre-ML baseline).
    -- A simple count of risk factors - useful as a sanity check against
    -- the model's predictions, and a fallback if the model is unavailable.
    (
        CASE WHEN s.contract_type = 'monthly'                THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(ut.login_trend, 0) < -1           THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(t.unresolved_tickets, 0) >= 2     THEN 1 ELSE 0 END
      + CASE WHEN p.modules_enabled <= 2                     THEN 1 ELSE 0 END
      + CASE WHEN c.account_age_months <= 6                  THEN 1 ELSE 0 END
    ) AS rule_based_risk_score,

    -- Target variable
    s.churn

FROM customers c
JOIN subscriptions s   ON c.customer_id = s.customer_id
JOIN product_usage p   ON c.customer_id = p.customer_id
LEFT JOIN usage_agg u  ON c.customer_id = u.customer_id
LEFT JOIN usage_trend ut ON c.customer_id = ut.customer_id
LEFT JOIN ticket_agg t ON c.customer_id = t.customer_id;
