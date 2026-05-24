-- LA monthly median price vs HPI regional benchmark for the same month.
WITH la_monthly AS (
    SELECT
        local_authority_code,
        region_code,
        month_start,
        median_price_gbp
    FROM {{ ref('mart_la_monthly_summary') }}
),
hpi_regional AS (
    SELECT
        date_trunc('month', index_date)::DATE AS month_start,
        area_code AS region_code,
        average_price_gbp AS hpi_price_gbp
    FROM {{ ref('stg_hpi') }}
)
SELECT
    la.local_authority_code,
    la.region_code,
    la.month_start,
    la.median_price_gbp,
    hpi.hpi_price_gbp,
    (la.median_price_gbp - hpi.hpi_price_gbp) / hpi.hpi_price_gbp AS premium_to_benchmark
FROM la_monthly la
LEFT JOIN hpi_regional hpi USING (month_start, region_code)
