-- Flag months whose row count deviates >3σ from the trailing 3-month median.
WITH monthly AS (
    SELECT
        date_trunc('month', transfer_date)::DATE AS month_start,
        COUNT(*) AS cnt
    FROM {{ ref('fct_transactions') }}
    GROUP BY 1
),
windowed AS (
    SELECT
        month_start,
        cnt,
        AVG(cnt) OVER (
            ORDER BY month_start
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS trailing_mean,
        STDDEV_SAMP(cnt) OVER (
            ORDER BY month_start
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS trailing_sd
    FROM monthly
)
SELECT *
FROM windowed
WHERE trailing_sd IS NOT NULL
  AND trailing_sd > 0
  AND ABS(cnt - trailing_mean) > 3 * trailing_sd
