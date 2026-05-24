---
title: Premium vs HPI Benchmark
---

Each local authority's monthly median sale price compared to the regional HPI benchmark price. Positive premium means the LA traded above its regional HPI; negative means below.

```sql las
SELECT DISTINCT local_authority_code
FROM housing.mart_premium_to_benchmark
WHERE local_authority_code IS NOT NULL
ORDER BY local_authority_code
```

<Dropdown data={las} name=la value=local_authority_code>
    <DropdownOption value="%" valueLabel="All LAs" />
</Dropdown>

## LA median vs HPI benchmark

```sql la_vs_hpi
SELECT
    month_start,
    median_price_gbp AS la_median,
    hpi_price_gbp AS hpi_benchmark
FROM housing.mart_premium_to_benchmark
WHERE local_authority_code LIKE '${inputs.la.value}'
ORDER BY month_start
```

<LineChart
    data={la_vs_hpi}
    x=month_start
    y={["la_median", "hpi_benchmark"]}
    title="LA median price vs HPI benchmark"
    yFmt=gbp0
/>

## Top 10 LAs by current premium-to-benchmark

```sql top_premiums
WITH latest AS (
    SELECT MAX(month_start) AS m FROM housing.mart_premium_to_benchmark
)
SELECT
    local_authority_code,
    region_code,
    month_start,
    median_price_gbp,
    hpi_price_gbp,
    premium_to_benchmark
FROM housing.mart_premium_to_benchmark, latest
WHERE month_start = latest.m
  AND premium_to_benchmark IS NOT NULL
ORDER BY premium_to_benchmark DESC
LIMIT 10
```

<DataTable data={top_premiums}>
    <Column id=local_authority_code />
    <Column id=region_code />
    <Column id=month_start />
    <Column id=median_price_gbp fmt=gbp0 />
    <Column id=hpi_price_gbp fmt=gbp0 />
    <Column id=premium_to_benchmark fmt=pct1 />
</DataTable>
