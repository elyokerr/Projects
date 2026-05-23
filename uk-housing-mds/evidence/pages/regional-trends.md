---
title: Regional Trends
---

```sql regions
SELECT DISTINCT region_code
FROM housing.mart_la_monthly_summary
WHERE region_code IS NOT NULL
ORDER BY region_code
```

<Dropdown data={regions} name=region value=region_code defaultValue="%">
    <DropdownOption value="%" valueLabel="All regions" />
</Dropdown>

## Monthly median price by region

```sql region_monthly
SELECT
    month_start,
    region_code,
    AVG(median_price_gbp) AS median_price_gbp
FROM housing.mart_la_monthly_summary
WHERE region_code LIKE '${inputs.region.value}'
GROUP BY 1, 2
ORDER BY 1
```

<LineChart
    data={region_monthly}
    x=month_start
    y=median_price_gbp
    series=region_code
    title="Median price (GBP) by region, monthly"
    yFmt=gbp0
/>

## Transaction volume by local authority

```sql la_volume
SELECT
    local_authority_code,
    SUM(transaction_count) AS transactions
FROM housing.mart_la_monthly_summary
WHERE region_code LIKE '${inputs.region.value}'
GROUP BY 1
ORDER BY transactions DESC
LIMIT 25
```

<BarChart
    data={la_volume}
    x=local_authority_code
    y=transactions
    title="Total transactions by local authority"
    swapXY=true
/>
