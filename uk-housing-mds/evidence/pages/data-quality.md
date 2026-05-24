---
title: Data Quality
---

Pipeline-health snapshot derived directly from the warehouse marts.

```sql corpus_size
SELECT COUNT(*) AS rows FROM housing.fct_transactions
```

```sql freshness
SELECT
    MAX(transfer_date) AS latest_transfer,
    DATE_DIFF('day', MAX(transfer_date), CURRENT_DATE) AS freshness_lag_days
FROM housing.fct_transactions
```

```sql distinct_postcodes
SELECT COUNT(DISTINCT postcode) AS postcodes FROM housing.fct_transactions
```

```sql distinct_las
SELECT COUNT(DISTINCT local_authority_code) AS las
FROM housing.fct_transactions
WHERE local_authority_code IS NOT NULL
```

<BigValue data={corpus_size} value=rows title="Transactions in warehouse" />
<BigValue data={freshness} value=freshness_lag_days title="Freshness lag (days)" />
<BigValue data={distinct_postcodes} value=postcodes title="Distinct postcodes" />
<BigValue data={distinct_las} value=las title="Distinct local authorities" />
<BigValue data={freshness} value=latest_transfer title="Latest transfer date" />

> **Note:** Full pipeline-health metrics (dbt test pass rate, Prefect run success rate, ingest row delta vs trailing median) are deferred until the `mart_pipeline_health` model exists. The metrics above are warehouse-state-only and surface freshness + corpus shape.
