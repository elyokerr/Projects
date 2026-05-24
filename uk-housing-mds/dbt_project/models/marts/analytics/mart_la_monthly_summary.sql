SELECT
    local_authority_code,
    region_code,
    date_trunc('month', transfer_date)::DATE AS month_start,
    COUNT(*) AS transaction_count,
    MEDIAN(price_paid_gbp) AS median_price_gbp,
    AVG(price_paid_gbp) AS mean_price_gbp,
    SUM(CASE WHEN is_new_build THEN 1 ELSE 0 END) AS new_build_count
FROM {{ ref('fct_transactions') }}
WHERE local_authority_code IS NOT NULL
GROUP BY 1, 2, 3
