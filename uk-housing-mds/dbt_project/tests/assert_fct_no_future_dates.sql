-- Returns rows that violate the invariant. dbt considers an empty result PASS.
SELECT *
FROM {{ ref('fct_transactions') }}
WHERE transfer_date > CURRENT_DATE
