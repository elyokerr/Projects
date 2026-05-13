"""
SaaS Churn Risk Dashboard.

An interactive Streamlit dashboard aimed at customer success teams.
The goal is to turn the model's output into something a non-technical
stakeholder can use directly: at-a-glance KPIs, risk distribution
charts, revenue exposure analysis, individual customer drilldown, and
a downloadable list of high-risk accounts.

The dashboard reads three files produced by upstream phases:

    data/processed/scored_customers.csv  - per-customer predictions
    data/processed/features.csv          - full feature set (for drilldown)
    models/model_metadata.joblib         - threshold, metrics, revenue impact

If any of those are missing the dashboard fails fast with a clear error
message pointing to the missing file.

Run locally:
    streamlit run dashboard/app.py

The dashboard opens at http://localhost:8501.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --- Path setup -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --- Page config ----------------------------------------------------------

st.set_page_config(
    page_title="SaaS Churn Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Cached data loaders --------------------------------------------------
# Streamlit re-runs the entire script on every interaction, so caching
# data loads is essential for responsiveness.

@st.cache_data
def load_scored_customers() -> pd.DataFrame:
    """Load the batch-scored customer predictions."""
    path = DATA_DIR / "scored_customers.csv"
    if not path.exists():
        st.error(
            f"Scored customers file not found at {path}. "
            "Run the model training and batch scoring pipeline first."
        )
        st.stop()
    df = pd.read_csv(path)
    risk_order = ["low", "medium", "high", "critical"]
    df["risk_level"] = pd.Categorical(df["risk_level"], categories=risk_order, ordered=True)
    return df


@st.cache_data
def load_features() -> pd.DataFrame | None:
    """Load the full feature set for customer drilldown.

    Returns None if the file is missing - the dashboard can still render
    the high-level views without it.
    """
    path = DATA_DIR / "features.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_model_metadata() -> dict:
    """Load model metadata (threshold, metrics, revenue impact)."""
    path = MODELS_DIR / "model_metadata.joblib"
    if not path.exists():
        st.error("Model metadata not found. Run the training pipeline first.")
        st.stop()
    return joblib.load(path)


# --- Data ----------------------------------------------------------------

scored_df = load_scored_customers()
features_df = load_features()
metadata = load_model_metadata()


# --- Sidebar filters -----------------------------------------------------

st.sidebar.title("Filters")
st.sidebar.markdown("---")

risk_options = ["All"] + sorted(
    scored_df["risk_level"].dropna().astype(str).unique().tolist()
)
selected_risk = st.sidebar.selectbox("Risk Level", risk_options, index=0)

min_rev = float(scored_df["monthly_revenue"].min())
max_rev = float(scored_df["monthly_revenue"].max())
rev_range = st.sidebar.slider(
    "Monthly Revenue Range (£)",
    min_value=min_rev,
    max_value=max_rev,
    value=(min_rev, max_rev),
    step=1.0,
)

prob_range = st.sidebar.slider(
    "Churn Probability Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.05,
)

# Apply filters across all views.
filtered_df = scored_df.copy()
if selected_risk != "All":
    filtered_df = filtered_df[filtered_df["risk_level"] == selected_risk]
filtered_df = filtered_df[
    (filtered_df["monthly_revenue"] >= rev_range[0])
    & (filtered_df["monthly_revenue"] <= rev_range[1])
    & (filtered_df["churn_probability"] >= prob_range[0])
    & (filtered_df["churn_probability"] <= prob_range[1])
]

st.sidebar.markdown("---")
st.sidebar.metric("Customers Shown", f"{len(filtered_df):,}")
st.sidebar.metric("Total Customers", f"{len(scored_df):,}")


# --- Header --------------------------------------------------------------

st.title("SaaS Churn Risk Dashboard")
st.markdown(
    "Monitor customer churn risk, identify at-risk accounts, and prioritize "
    "retention efforts based on predicted revenue impact."
)
st.markdown("---")


# --- KPI cards -----------------------------------------------------------

total_customers = len(filtered_df)
predicted_churners = filtered_df[filtered_df["churn_prediction"] == 1]
churn_rate = (len(predicted_churners) / total_customers * 100) if total_customers else 0
mrr_at_risk = predicted_churners["monthly_revenue"].sum()
critical_count = len(filtered_df[filtered_df["risk_level"] == "critical"])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Predicted Churners", f"{len(predicted_churners):,}")
col3.metric("Churn Rate", f"{churn_rate:.1f}%")
col4.metric("MRR at Risk", f"£{mrr_at_risk:,.0f}")
col5.metric("Critical Risk", f"{critical_count:,}")

st.markdown("---")


# --- Color scheme used across all charts ---------------------------------
# Consistent green-to-red mapping reinforces the risk severity reading.

COLOR_MAP = {
    "low": "#2ecc71",       # Green
    "medium": "#f39c12",    # Amber
    "high": "#e67e22",      # Orange
    "critical": "#e74c3c",  # Red
}


# --- Row 1: risk distribution and revenue at risk ------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Risk Level Distribution")
    risk_counts = (
        filtered_df["risk_level"]
        .value_counts()
        .reindex(["low", "medium", "high", "critical"], fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["Risk Level", "Count"]

    fig_risk = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        color_discrete_map=COLOR_MAP,
        text="Count",
    )
    fig_risk.update_layout(showlegend=False, height=400, xaxis_title="", yaxis_title="Number of Customers")
    fig_risk.update_traces(textposition="outside")
    st.plotly_chart(fig_risk, use_container_width=True)

with col_right:
    st.subheader("Revenue at Risk by Risk Level")
    rev_by_risk = (
        filtered_df[filtered_df["churn_prediction"] == 1]
        .groupby("risk_level", observed=True)["monthly_revenue"]
        .sum()
        .reindex(["low", "medium", "high", "critical"], fill_value=0)
        .reset_index()
    )
    rev_by_risk.columns = ["Risk Level", "Monthly Revenue at Risk (£)"]

    fig_rev = px.bar(
        rev_by_risk,
        x="Risk Level",
        y="Monthly Revenue at Risk (£)",
        color="Risk Level",
        color_discrete_map=COLOR_MAP,
        text_auto=",.0f",
    )
    fig_rev.update_layout(showlegend=False, height=400, xaxis_title="", yaxis_title="Monthly Revenue at Risk (£)")
    fig_rev.update_traces(textposition="outside")
    st.plotly_chart(fig_rev, use_container_width=True)


# --- Row 2: probability distribution and revenue scatter ----------------

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("Churn Probability Distribution")
    fig_hist = px.histogram(
        filtered_df,
        x="churn_probability",
        nbins=30,
        color_discrete_sequence=["#3498db"],
        labels={"churn_probability": "Churn Probability"},
    )
    threshold = metadata.get("optimal_threshold", 0.5)
    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Threshold: {threshold:.2f}",
        annotation_position="top right",
    )
    fig_hist.update_layout(height=400, xaxis_title="Churn Probability", yaxis_title="Number of Customers")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right2:
    st.subheader("Revenue vs Churn Probability")
    fig_scatter = px.scatter(
        filtered_df,
        x="churn_probability",
        y="monthly_revenue",
        color="risk_level",
        color_discrete_map=COLOR_MAP,
        hover_data=["customer_id"],
        opacity=0.6,
        labels={
            "churn_probability": "Churn Probability",
            "monthly_revenue": "Monthly Revenue (£)",
            "risk_level": "Risk Level",
        },
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)


# --- Model performance summary -------------------------------------------

st.markdown("---")
st.subheader("Model Performance and Business Impact")

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.markdown("**Model Info**")
    st.write(f"- Model: `{metadata.get('best_model', 'N/A')}`")
    st.write(f"- Optimal threshold: `{metadata.get('optimal_threshold', 0):.4f}`")
    metrics = metadata.get("metrics", {})
    if metrics:
        st.write(f"- ROC-AUC: `{metrics.get('roc_auc', 0):.4f}`")
        st.write(f"- F1 score: `{metrics.get('f1', 0):.4f}`")

with perf_col2:
    st.markdown("**Revenue Impact (Full Dataset)**")
    rev_impact = metadata.get("revenue_impact", {})
    if rev_impact:
        st.write(f"- Total MRR at risk: `£{rev_impact.get('total_mrr_at_risk', 0):,.2f}`")
        st.write(f"- MRR identified by model: `£{rev_impact.get('mrr_identified', 0):,.2f}`")
        st.write(f"- Estimated annual savings: `£{rev_impact.get('estimated_annual_savings', 0):,.2f}`")

with perf_col3:
    st.markdown("**Threshold Metrics**")
    thresh_metrics = metadata.get("threshold_metrics", {})
    if thresh_metrics:
        precision = thresh_metrics.get("precision_at_optimal")
        recall = thresh_metrics.get("recall_at_optimal")
        f1 = thresh_metrics.get("f1_at_optimal")
        st.write(f"- Precision: `{float(precision):.4f}`" if precision is not None else "- Precision: N/A")
        st.write(f"- Recall: `{float(recall):.4f}`" if recall is not None else "- Recall: N/A")
        st.write(f"- F1: `{float(f1):.4f}`" if f1 is not None else "- F1: N/A")


# --- Customer drilldown --------------------------------------------------

st.markdown("---")
st.subheader("Customer Drilldown")
st.markdown(
    "Select a customer to see their full risk profile and the features "
    "most strongly associated with their prediction."
)

if features_df is not None:
    drilldown_df = filtered_df.merge(features_df, on="customer_id", how="left", suffixes=("", "_feat"))
    customer_ids = filtered_df.sort_values("churn_probability", ascending=False)["customer_id"].tolist()

    if customer_ids:
        selected_customer = st.selectbox(
            "Select customer (sorted by churn risk)",
            customer_ids,
            format_func=lambda x: (
                f"Customer {x} - "
                f"{str(filtered_df[filtered_df['customer_id'] == x]['risk_level'].values[0]).upper()} risk"
            ),
        )

        if selected_customer:
            cust_row = drilldown_df[drilldown_df["customer_id"] == selected_customer].iloc[0]

            detail_col1, detail_col2, detail_col3 = st.columns(3)

            with detail_col1:
                st.markdown("**Prediction**")
                st.metric("Churn Probability", f"{cust_row['churn_probability']:.1%}")
                st.metric("Risk Level", cust_row["risk_level"].upper())
                st.metric("Monthly Revenue", f"£{cust_row['monthly_revenue']:.2f}")

            with detail_col2:
                st.markdown("**Customer Profile**")
                st.write(f"- Tenure: `{int(cust_row.get('tenure_months', 0))}` months")
                contract_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
                st.write(f"- Contract: `{contract_map.get(int(cust_row.get('contract_type', 0)), 'Unknown')}`")
                st.write(f"- Total charges: `£{cust_row.get('total_charges', 0):,.2f}`")
                st.write(f"- Payment method: `{int(cust_row.get('payment_method', 0))}`")

            with detail_col3:
                st.markdown("**Engagement**")
                st.write(f"- Login frequency: `{cust_row.get('login_frequency', 0):.1f}`")
                st.write(f"- Feature adoption: `{cust_row.get('feature_adoption_rate', 0):.0%}`")
                st.write(f"- Days since login: `{int(cust_row.get('days_since_last_login', 0))}`")
                st.write(f"- Engagement score: `{cust_row.get('engagement_score', 0):.1f}`")
                st.write(f"- Support tickets (90d): `{int(cust_row.get('support_tickets_last_90d', 0))}`")

            # Top contributing features visualized as a horizontal bar chart.
            # This is a simple feature-magnitude view; a future improvement
            # would be to display per-prediction SHAP values from the API.
            st.markdown("**Top Features for This Customer**")
            skip_cols = {
                "customer_id", "churn", "churn_probability", "churn_prediction",
                "monthly_revenue", "risk_level", "actual_churn",
            }
            feature_cols = [
                c for c in drilldown_df.columns
                if c not in skip_cols and not c.endswith("_feat")
            ]

            if feature_cols:
                cust_features = cust_row[feature_cols].astype(float)
                top_features = cust_features.abs().sort_values(ascending=False).head(10)

                fig_features = go.Figure()
                fig_features.add_trace(
                    go.Bar(
                        x=top_features.values,
                        y=top_features.index,
                        orientation="h",
                        marker_color=[
                            "#e74c3c" if cust_features[f] > 0 else "#2ecc71"
                            for f in top_features.index
                        ],
                    )
                )
                fig_features.update_layout(
                    height=350,
                    xaxis_title="Feature Value (absolute)",
                    yaxis_title="",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_features, use_container_width=True)
else:
    st.info("Full feature data not found - upload `features.csv` to enable customer drilldown.")


# --- High-risk customer table --------------------------------------------

st.markdown("---")
st.subheader("High-Risk Customers - Action Required")

high_risk = (
    filtered_df[filtered_df["risk_level"].isin(["critical", "high"])]
    .sort_values("churn_probability", ascending=False)
)

if len(high_risk) > 0:
    display_cols = [
        "customer_id", "churn_probability", "risk_level",
        "monthly_revenue", "churn_prediction",
    ]
    st.dataframe(
        high_risk[display_cols]
        .head(50)
        .style.format({
            "churn_probability": "{:.1%}",
            "monthly_revenue": "£{:,.2f}",
        })
        .map(
            lambda x: "background-color: #fadbd8" if x == "critical" else (
                "background-color: #fdebd0" if x == "high" else ""
            ),
            subset=["risk_level"],
        ),
        use_container_width=True,
        height=500,
    )

    # Downloadable CSV - the most important output for non-technical users.
    csv = high_risk[display_cols].to_csv(index=False)
    st.download_button(
        label="Download High-Risk Customers (CSV)",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )
else:
    st.success("No high-risk customers match the current filters.")


# --- Footer --------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "SaaS Churn Prediction Dashboard · "
    f"Model: {metadata.get('best_model', 'N/A')} · "
    f"Threshold: {metadata.get('optimal_threshold', 0):.4f}"
    "</div>",
    unsafe_allow_html=True,
)
