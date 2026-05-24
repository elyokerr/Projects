SELECT *
FROM {{ ref('mart_premium_to_benchmark') }}
WHERE premium_to_benchmark < -0.8
   OR premium_to_benchmark > 5.0
