---
title: UK Housing Market Analytics
description: A scheduled batch ELT pipeline over HM Land Registry Price Paid Data, ONS NSPL, and UK HPI — modelled with dbt, validated with Great Expectations, orchestrated with Prefect.
hide_title: true
---

<div style="margin: 2rem 0 0.5rem 0;">
<h1 style="font-size: 2.5rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.5rem;">
UK Housing Market Analytics
</h1>
<p style="font-size: 1.15rem; color: var(--grey-600); max-width: 60rem; line-height: 1.5; margin-top: 0;">
A refreshable, benchmarked, geo-enriched view of UK residential property transactions — built from three public sources on an entirely free stack.
</p>
</div>

```sql txn_count
SELECT COUNT(*) AS transactions FROM housing.fct_transactions
```

```sql distinct_las
SELECT COUNT(DISTINCT local_authority_code) AS las
FROM housing.fct_transactions
WHERE local_authority_code IS NOT NULL
```

```sql distinct_postcodes
SELECT COUNT(DISTINCT postcode) AS postcodes FROM housing.fct_transactions
```

```sql date_range
SELECT
  MIN(transfer_date) AS earliest,
  MAX(transfer_date) AS latest,
  DATE_DIFF('day', MAX(transfer_date), CURRENT_DATE) AS freshness_days
FROM housing.fct_transactions
```

## At a glance

<Grid cols=4>
  <BigValue
    data={txn_count}
    value=transactions
    fmt='num0'
    title='Transactions'
  />
  <BigValue
    data={distinct_las}
    value=las
    fmt='num0'
    title='Local authorities'
  />
  <BigValue
    data={distinct_postcodes}
    value=postcodes
    fmt='num0'
    title='Distinct postcodes'
  />
  <BigValue
    data={date_range}
    value=latest
    title='Latest transaction'
  />
</Grid>

<Alert status="info">
This site is currently driven by a small synthetic fixture (~50 transactions) so the pipeline can be reviewed end-to-end without a 27 M-row backfill. Hero numbers, charts, and tables will reflect real Land Registry data once the production run completes.
</Alert>

## Transaction volume over time

```sql monthly_txns
SELECT
    month_start,
    SUM(transaction_count) AS transactions,
    MEDIAN(median_price_gbp) AS median_price
FROM housing.mart_la_monthly_summary
GROUP BY 1
ORDER BY 1
```

<AreaChart
  data={monthly_txns}
  x=month_start
  y=transactions
  yAxisTitle='Transactions per month'
  fillColor='#236aa4'
  fillOpacity=0.25
  lineColor='#236aa4'
  title='Monthly transaction volume'
  subtitle='Aggregated across all UK local authorities'
/>

## Where the money goes

<Grid cols=2>
<div>

### Top local authorities by activity

```sql top_las
SELECT
    local_authority_code AS local_authority,
    SUM(transaction_count) AS transactions,
    AVG(median_price_gbp) AS avg_median_price_gbp
FROM housing.mart_la_monthly_summary
WHERE local_authority_code IS NOT NULL
GROUP BY 1
ORDER BY transactions DESC
LIMIT 10
```

<BarChart
  data={top_las}
  x=local_authority
  y=transactions
  sort=false
  swapXY=true
  title=''
/>

</div>

<div>

### Median price by region

```sql region_prices
SELECT
    region_code AS region,
    AVG(median_price_gbp) AS median_price_gbp,
    SUM(transaction_count) AS transactions
FROM housing.mart_la_monthly_summary
WHERE region_code IS NOT NULL
GROUP BY 1
ORDER BY median_price_gbp DESC
```

<DataTable data={region_prices} rows=10>
  <Column id=region title='Region' />
  <Column id=median_price_gbp title='Median price' fmt='£#,##0' contentType='colorscale' colorScale='blues' />
  <Column id=transactions title='Tx' fmt='num0' />
</DataTable>

</div>
</Grid>

## Explore the marts

<Grid cols=3>
<div style="padding: 1.5rem; border: 1px solid var(--grey-200); border-radius: 0.75rem; background: var(--grey-50); height: 100%;">

### Regional trends
Monthly median price and transaction volume by UK region, selectable by region.

<LinkButton url='/regional-trends'>Open →</LinkButton>

</div>

<div style="padding: 1.5rem; border: 1px solid var(--grey-200); border-radius: 0.75rem; background: var(--grey-50); height: 100%;">

### Premium vs HPI benchmark
Where local authority median prices sit relative to the UK House Price Index.

<LinkButton url='/premium-vs-benchmark'>Open →</LinkButton>

</div>

<div style="padding: 1.5rem; border: 1px solid var(--grey-200); border-radius: 0.75rem; background: var(--grey-50); height: 100%;">

### Data quality
Pipeline health — freshness, corpus size, distinct postcodes &amp; local authorities.

<LinkButton url='/data-quality'>Open →</LinkButton>

</div>
</Grid>

---

<div style="margin-top: 2rem; padding: 1.5rem; border-radius: 0.75rem; background: var(--grey-50); border: 1px solid var(--grey-200);">

#### How this is built

End-to-end batch ELT over **HM Land Registry Price Paid Data**, **ONS National Statistics Postcode Lookup**, and the **UK House Price Index**. Three sources land as Parquet, validate through **Great Expectations** suites, load into a **DuckDB** development warehouse (or **BigQuery** in production via dbt profile switch), and propagate through five **dbt** layers — staging → intermediate → core dims/fact → analytics marts — with **38 dbt tests** plus three singular cross-table invariants. The entire flow runs in **Prefect** on a monthly **GitHub Actions** cron, with a quota-aware fallback from BigQuery to DuckDB.

[View the pipeline on GitHub](https://github.com/elyokerr/Projects/tree/main/uk-housing-mds) · [Read the design doc](https://github.com/elyokerr/Projects/blob/main/uk-housing-mds/docs/2026-05-23-uk-housing-mds-design.md)

</div>
