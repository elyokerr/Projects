# Dashboard Screenshots

This directory holds screenshots of the Streamlit dashboard that are referenced from the main README and from `docs/dashboard.md`.

## What to capture

To match the references in the documentation, capture the following views:

- `01_kpi_row.png` - The five KPI cards at the top of the dashboard with no filters applied
- `02_risk_distribution.png` - The two side-by-side bar charts (risk level distribution and revenue at risk by tier)
- `03_probability_scatter.png` - The probability histogram with threshold line, and the revenue-vs-probability scatter plot
- `04_customer_drilldown.png` - The customer drilldown section with a high-risk customer selected
- `05_high_risk_table.png` - The color-coded high-risk customer table

## How to capture

1. Run the dashboard with `make dashboard`
2. Open <http://localhost:8501> in a browser
3. Set the browser window to 1440 pixels wide for consistent dimensions
4. Use your OS's screenshot tool to capture each section
5. Save the PNGs in this directory using the names above

## Image specifications

- Format: PNG
- Width: 1440 pixels recommended (the dashboard's layout looks best here)
- Compression: lossless

## Why these aren't committed

Screenshots are easy to regenerate but heavy in version control. The repo stays lean by treating this directory as a placeholder; populate it locally when needed for portfolio presentation.
