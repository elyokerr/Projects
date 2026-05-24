WITH RECURSIVE date_seq(d) AS (
    SELECT DATE '1995-01-01'
    UNION ALL
    SELECT d + INTERVAL 1 DAY FROM date_seq WHERE d < CURRENT_DATE
)
SELECT
    d AS date_day,
    EXTRACT(year FROM d) AS year,
    EXTRACT(month FROM d) AS month,
    EXTRACT(quarter FROM d) AS quarter,
    EXTRACT(dow FROM d) AS day_of_week,
    date_trunc('month', d)::DATE AS month_start
FROM date_seq
