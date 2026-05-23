SELECT
    CAST(date AS DATE) AS index_date,
    area_code,
    region_name,
    CAST(average_price AS DOUBLE) AS average_price_gbp,
    CAST(index AS DOUBLE) AS hpi_value
FROM {{ source('raw', 'hpi') }}
