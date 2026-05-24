SELECT
    local_authority_code,
    region_code,
    month_start,
    transaction_count,
    median_price_gbp,
    mean_price_gbp,
    new_build_count
FROM main.mart_la_monthly_summary
