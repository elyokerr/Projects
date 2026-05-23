SELECT
    transaction_unique_id,
    transfer_date,
    price_paid_gbp,
    postcode,
    property_type,
    is_new_build,
    tenure,
    local_authority_code,
    region_code
FROM main.fct_transactions
