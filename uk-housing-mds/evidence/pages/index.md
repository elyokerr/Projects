---
title: UK Housing Market Data Site
---

End-to-end batch ELT pipeline over HM Land Registry Price Paid Data, ONS NSPL, and UK HPI. Modelled with dbt against DuckDB (dev/CI) and BigQuery (prod), validated with Great Expectations + dbt tests, orchestrated with Prefect on GitHub Actions cron.

```sql txn_count
SELECT COUNT(*) AS transactions FROM housing.fct_transactions
```

```sql latest_date
SELECT MAX(transfer_date) AS latest FROM housing.fct_transactions
```

<BigValue data={txn_count} value=transactions title="Transactions in warehouse" />
<BigValue data={latest_date} value=latest title="Latest transfer date" />

## Transactions per month

```sql monthly_txns
SELECT
    month_start,
    SUM(transaction_count) AS transactions
FROM housing.mart_la_monthly_summary
GROUP BY 1
ORDER BY 1
```

<LineChart data={monthly_txns} x=month_start y=transactions title="Monthly transaction volume" />

## Explore

- [Regional trends](regional-trends)
- [Premium vs HPI benchmark](premium-vs-benchmark)
- [Data quality](data-quality)
