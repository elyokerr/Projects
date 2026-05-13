# Dashboard

A Streamlit dashboard aimed at customer success teams. The goal is to turn the model's output into something a non-technical stakeholder can use directly: at-a-glance KPIs, risk distribution charts, revenue exposure analysis, individual customer drilldown, and a downloadable list of high-risk accounts.

## Why Streamlit

For internal data tools, Streamlit hits a sweet spot:

- One Python file becomes a fully interactive web app
- Charts (Plotly, Altair, native) and tables render natively
- Caching with `@st.cache_data` makes interactions feel instant
- Deployment is a Docker container or a Streamlit Cloud free-tier deploy

For external-facing products you'd reach for React. For something a customer success team uses internally, Streamlit ships in a fraction of the time with no loss of usefulness.

## What the Dashboard Shows

The dashboard is organized into five sections, each addressing a question a customer success manager would actually ask.

### Section 1: KPI cards

Five cards across the top of the page:

- Total customers (after applying sidebar filters)
- Predicted churners
- Churn rate
- Monthly revenue at risk
- Critical risk count

These give an immediate read on the size of the problem and how the filters have narrowed the view.

### Section 2: Risk distribution and revenue at risk

Two bar charts side by side:

- **Risk Level Distribution.** Customer counts by tier (critical, high, medium, low). Color-coded green-to-red.
- **Revenue at Risk by Risk Level.** Same tiers, but summed monthly revenue instead of customer counts.

The pair matters because a "critical" account with £5/month MRR is less urgent than a "high" account with £500/month. Looking at both charts together prioritizes the right way.

### Section 3: Probability and revenue distributions

Two more charts:

- **Churn Probability Distribution.** Histogram of probabilities with the model's decision threshold drawn as a dashed line.
- **Revenue vs Churn Probability.** Scatter plot colored by risk level. Hovering reveals customer IDs.

The scatter is the one I find most useful in practice. It's the only view that combines probability and revenue in a single picture. Customers in the top-right corner (high probability, high revenue) are the ones a customer success manager would call first.

### Section 4: Model performance and business impact

Three columns of metadata about the underlying model:

- Model info (which model won, threshold, ROC-AUC, F1)
- Revenue impact estimates (total MRR at risk, MRR identified, annual savings)
- Threshold metrics (precision, recall, F1 at the optimized threshold)

This section makes the model's caveats visible. Anyone consuming the dashboard can see exactly what F1 score backs the predictions and what assumptions go into the revenue estimates.

### Section 5: Customer drilldown

A dropdown of all filtered customers (sorted by churn probability). Selecting one shows:

- Their prediction (probability, risk level, monthly revenue)
- Their profile (tenure, contract, total charges, payment method)
- Their engagement (login frequency, feature adoption, support tickets)
- A horizontal bar chart of the top features for this customer

The bar chart uses the same feature-magnitude approximation as the API's `top_risk_factors` field. A future improvement would be to call the API for real SHAP values per customer; for now, the magnitude view is good enough to support a conversation about why a specific customer is flagged.

### Section 6: High-risk customer table

A sortable, color-coded table of the top 50 high-risk customers, with a CSV download button.

This is the most important output for non-technical users. A customer success manager opens the dashboard once a week, applies whatever filters are relevant ("only customers with monthly revenue above £50"), and downloads the CSV to feed into their CRM workflow. The CSV is the artifact that gets used; the dashboard is the interface for producing it.

## Sidebar Filters

Three filters that all the charts respond to:

- **Risk level** - All, critical, high, medium, low
- **Monthly revenue range** - Two-handle slider on the actual data range
- **Churn probability range** - Two-handle slider, 0 to 1

The filtered customer count and total are displayed at the bottom of the sidebar so it's clear how aggressive the current filter is.

## Design Choices

A few decisions worth calling out:

### Color consistency

A `COLOR_MAP` constant defines the four risk-tier colors once and is reused across every chart. Green for low, amber for medium, orange for high, red for critical. Color is doing real work here - a reader scanning the page should be able to read severity from color alone, without needing to look at the legend.

### Direct file access, not API calls

The dashboard reads `scored_customers.csv` and `model_metadata.joblib` directly from disk. It does not call the API.

This is intentional. The dashboard's job is to render predictions that already exist, not to compute new ones. Coupling it to the API would add latency, fragility, and a dependency that doesn't earn its keep. The training pipeline writes the CSV, the dashboard reads the CSV. Simple.

The downside is that the dashboard doesn't reflect predictions for customers added since the last batch scoring run. For this use case (weekly customer success workflow), that's fine - retraining and rescoring on a weekly schedule matches the cadence of the retention workflow anyway.

### Caching

Every data load is wrapped in `@st.cache_data`. Without caching, every interaction (changing a filter, selecting a customer in the dropdown) would re-read the CSV from disk. With caching, the data is loaded once on first render and held in memory until the underlying file changes.

The `@st.cache_data` decorator is smart enough to invalidate the cache when the file's modification time changes, so re-running the training pipeline automatically refreshes the dashboard's data without needing to restart it.

### Graceful degradation

The dashboard handles a few realistic failure modes:

- If `scored_customers.csv` doesn't exist, it shows a clear error and stops, pointing at how to fix it
- If `features.csv` doesn't exist, the customer drilldown is hidden but the rest of the dashboard still works
- If `model_metadata.joblib` doesn't exist, the model performance section is skipped

The goal is for the dashboard to be useful in as many partial states as possible.

## Running the Dashboard

Without Docker:

```bash
make dashboard
```

Or directly:

```bash
streamlit run dashboard/app.py
```

The dashboard opens at <http://localhost:8501>.

With Docker:

```bash
docker compose up dashboard
```

This brings up the dashboard plus the API (the dashboard depends on the API health check for ordered startup, even though it doesn't actually call the API at runtime).

## Screenshots

Screenshots live in `docs/screenshots/` and are referenced from the main README. If you fork this project, populate that directory with your own captures - I'd recommend capturing the KPI row, the risk distribution charts, and the customer drilldown with a high-risk customer selected.

## Future Work

A few additions I'd make if I were taking this further:

- **Real SHAP values for the customer drilldown.** Either pre-compute SHAP values during batch scoring and store them alongside the predictions, or call the API on demand.
- **Time series view.** Track risk distribution and revenue at risk over time, week by week, to see whether retention efforts are working in aggregate.
- **Segmentation.** Break down predictions by account segment (new / established / mature), service tier, or other dimensions that would inform different retention strategies.
- **Intervention tracking.** Log when a customer success manager has reached out to a flagged account so the dashboard can show "already contacted" status.
