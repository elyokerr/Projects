SELECT
    transaction_unique_id,
    CAST(price_paid AS BIGINT) AS price_paid_gbp,
    CAST(date_of_transfer AS DATE) AS transfer_date,
    UPPER(postcode) AS postcode,
    property_type,
    new_build_flag = 'Y' AS is_new_build,
    tenure,
    UPPER(district) AS district,
    UPPER(county) AS county,
    ppd_category_type
FROM {{ source('raw', 'ppd') }}
WHERE postcode IS NOT NULL
  AND price_paid > 0
