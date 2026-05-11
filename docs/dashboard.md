# Dashboard Documentation

## Overview

The Streamlit dashboard provides an interactive interface for customer success teams to monitor churn risk, identify at-risk accounts, and prioritise retention efforts. It reads the batch-scored data from Phase 2 and presents it through interactive Plotly charts with filtering and drilldown capabilities.

## Running the Dashboard

### Locally

```bash
# From the project root
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

### With Docker

```bash
docker-compose up dashboard
```

Opens at `http://localhost:8501`.

## Data Dependencies

The dashboard reads three files produced by earlier phases:

| File | Source | Purpose |
|---|---|---|
| `data/processed/scored_customers.csv` | Phase 2 batch scoring | Customer IDs, churn probabilities, risk levels, revenue |
| `data/processed/features.csv` | Phase 1 feature engineering | Full feature set for customer drilldown |
| `models/model_metadata.joblib` | Phase 2 training | Model name, threshold, metrics, revenue impact |

If any file is missing, the dashboard displays an error message explaining which phase needs to be run first.

## Dashboard Sections

### KPI Cards

Five metrics displayed at the top of the page for an at-a-glance summary:

- **Total Customers** — count of customers matching current filters
- **Predicted Churners** — customers above the optimal churn threshold
- **Churn Rate** — percentage of customers predicted to churn
- **MRR at Risk** — total monthly recurring revenue from predicted churners
- **Critical Risk** — count of customers with churn probability ≥ 80%

### Risk Distribution Chart

A bar chart showing how many customers fall into each risk level (low, medium, high, critical). Colour-coded from green to red. Helps the team understand the overall risk landscape.

### Revenue at Risk Chart

A bar chart showing the total monthly revenue at risk within each risk level. This is more useful than customer counts because it highlights where the biggest financial exposure is — a few high-revenue critical customers may matter more than many low-revenue ones.

### Churn Probability Distribution

A histogram of churn probabilities across all customers, with a vertical dashed line at the model's optimal threshold. Shows the distribution of model confidence and how many customers sit near the decision boundary.

### Revenue vs Churn Probability Scatter

An interactive scatter plot with churn probability on the x-axis and monthly revenue on the y-axis, coloured by risk level. The top-right quadrant (high probability, high revenue) contains the most important customers to retain.

### Model Performance Summary

Displays the model's key metrics (ROC-AUC, F1, precision, recall), the optimal threshold, and the estimated revenue impact. Provides transparency so stakeholders understand the model's reliability.

### Customer Drilldown

A dropdown selector (sorted by risk) that shows an individual customer's:
- Churn probability and risk level
- Monthly revenue and contract details
- Engagement metrics (login frequency, feature adoption, session duration)
- Top contributing features visualised as a horizontal bar chart

This is designed for account managers who want to understand *why* a specific customer is flagged.

### High-Risk Customer Table

A sortable, filterable table of all high-risk and critical-risk customers. Includes a download button to export the list as CSV — this is the actionable output that customer success teams would use to prioritise outreach.

## Sidebar Filters

Three interactive filters that update all charts and tables simultaneously:

- **Risk Level** — filter to a specific risk category or show all
- **Monthly Revenue Range** — slider to focus on a revenue band
- **Churn Probability Range** — slider to focus on a confidence band

## Design Decisions

**Plotly over matplotlib/seaborn:** Interactive charts (hover, zoom, pan) are more useful for stakeholders than static images. Plotly integrates natively with Streamlit.

**Sidebar filters:** Streamlit's sidebar pattern keeps filters visible without consuming main content space. All charts react to filter changes instantly.

**Download button:** The most important output for business users. A CSV of high-risk customers that can be imported into a CRM or used for manual outreach.

**Colour scheme:** Consistent green-to-red colour mapping across all charts reinforces the risk severity interpretation.

## Technologies

| Technology | Purpose |
|---|---|
| **Streamlit** | Dashboard framework — Python-native, rapid development |
| **Plotly** | Interactive, publication-quality visualisations |
| **pandas** | Data manipulation and filtering |
| **joblib** | Loading serialised model metadata |
