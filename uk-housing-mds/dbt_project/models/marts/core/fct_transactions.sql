{{ config(
    materialized='incremental',
    unique_key='transaction_unique_id',
    on_schema_change='fail'
) }}

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
FROM {{ ref('int_ppd_enriched') }}

{% if is_incremental() %}
WHERE transfer_date >= (
    SELECT COALESCE(MAX(transfer_date), DATE '1900-01-01') FROM {{ this }}
) - INTERVAL 60 DAY
{% endif %}
