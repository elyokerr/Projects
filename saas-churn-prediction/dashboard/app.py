"""
SaaS Churn Risk Dashboard
============================
An interactive Streamlit dashboard for customer success teams to:
- Monitor churn risk across the customer base
- Identify high-risk customers for proactive retention
- Understand revenue at risk
- Drill into individual customer risk profiles
- Explore which features drive churn predictions

Run locally:
    streamlit run dashboard/app.py

Why Streamlit?
    - Fastest path from data to interactive web app
    - Python-native — no JS/HTML required
    - Industry standard for data science dashboards
    - Easy to deploy (Streamlit Cloud, Docker, etc.)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Project path setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ─── Page Config ─────────────────────────────────────────────────

st.set_page_config(
    page_title="SaaS Churn Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Data Loading (cached) ──────────────────────────────────────

@st.cache_data
def load_scored_customers():
    """Load batch-scored customer data from Phase 2."""
    path = DATA_DIR / "scored_customers.csv"
    if not path.exists():
        st.error(
            f"Scored customers file not found at {path}. "
            "Run the Phase 2 notebook first to generate batch scores."
        )
        st.stop()
    df = pd.read_csv(path)
    # Ensure risk_level is ordered for charts
    risk_order = ["low", "medium", "high", "critical"]
    df["risk_level"] = pd.Categorical(df["risk_level"], categories=risk_order, ordered=True)
    return df


@st.cache_data
def load_features():
    """Load the full feature set for drilldown analysis."""
    path = DATA_DIR / "features.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_model_metadata():
    """Load model metadata (threshold, metrics, revenue impact)."""
    path = MODELS_DIR / "model_metadata.joblib"
    if not path.exists():
        st.error("Model metadata not found. Run Phase 2 first.")
        st.stop()
    return joblib.load(path)


# ─── Load Data ──────────────────────────────────────────────────

scored_df = load_scored_customers()
features_df = load_features()
metadata = load_model_metadata()


# ─── Sidebar Filters ────────────────────────────────────────────

st.sidebar.title("🎯 Filters")
st.sidebar.markdown("---")

# Risk level filter
risk_options = ["All"] + sorted(
    scored_df["risk_level"].dropna().astype(str).unique().tolist()
)
selected_risk = st.sidebar.selectbox("Risk Level", risk_options, index=0)

# Revenue range filter
min_rev = float(scored_df["monthly_revenue"].min())
max_rev = float(scored_df["monthly_revenue"].max())
rev_range = st.sidebar.slider(
    "Monthly Revenue Range (£)",
    min_value=min_rev,
    max_value=max_rev,
    value=(min_rev, max_rev),
    step=1.0,
)

# Churn probability filter
prob_range = st.sidebar.slider(
    "Churn Probability Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.05,
)

# Apply filters
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


# ─── Header ─────────────────────────────────────────────────────

st.title("📊 SaaS Churn Risk Dashboard")
st.markdown(
    "Monitor customer churn risk, identify at-risk accounts, "
    "and prioritise retention efforts based on predicted revenue impact."
)
st.markdown("---")


# ─── KPI Cards ──────────────────────────────────────────────────

# Calculate metrics on filtered data
total_customers = len(filtered_df)
predicted_churners = filtered_df[filtered_df["churn_prediction"] == 1]
churn_rate = len(predicted_churners) / total_customers * 100 if total_customers > 0 else 0
mrr_at_risk = predicted_churners["monthly_revenue"].sum()
arr_at_risk = mrr_at_risk * 12
critical_count = len(filtered_df[filtered_df["risk_level"] == "critical"])

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Customers", f"{total_customers:,}")
with col2:
    st.metric("Predicted Churners", f"{len(predicted_churners):,}")
with col3:
    st.metric("Churn Rate", f"{churn_rate:.1f}%")
with col4:
    st.metric("MRR at Risk", f"£{mrr_at_risk:,.0f}")
with col5:
    st.metric("Critical Risk", f"{critical_count:,}")

st.markdown("---")


# ─── Row 1: Risk Distribution + Revenue Impact ─────────────────

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

    colour_map = {
        "low": "#2ecc71",
        "medium": "#f39c12",
        "high": "#e67e22",
        "critical": "#e74c3c",
    }

    fig_risk = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        color_discrete_map=colour_map,
        text="Count",
    )
    fig_risk.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="",
        yaxis_title="Number of Customers",
    )
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
        color_discrete_map=colour_map,
        text_auto=",.0f",
    )
    fig_rev.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="",
        yaxis_title="Monthly Revenue at Risk (£)",
    )
    fig_rev.update_traces(textposition="outside")
    st.plotly_chart(fig_rev, use_container_width=True)


# ─── Row 2: Churn Probability Distribution + Scatter ────────────

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
    # Add threshold line
    threshold = metadata.get("optimal_threshold", 0.5)
    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Threshold: {threshold:.2f}",
        annotation_position="top right",
    )
    fig_hist.update_layout(
        height=400,
        xaxis_title="Churn Probability",
        yaxis_title="Number of Customers",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right2:
    st.subheader("Revenue vs Churn Probability")

    fig_scatter = px.scatter(
        filtered_df,
        x="churn_probability",
        y="monthly_revenue",
        color="risk_level",
        color_discrete_map=colour_map,
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


# ─── Model Performance Summary ──────────────────────────────────

st.markdown("---")
st.subheader("📈 Model Performance & Business Impact")

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.markdown("**Model Info**")
    st.write(f"- Model: `{metadata.get('best_model', 'N/A')}`")
    st.write(f"- Optimal Threshold: `{metadata.get('optimal_threshold', 'N/A'):.4f}`")

    metrics = metadata.get("metrics", {})
    if metrics:
        st.write(f"- ROC-AUC: `{metrics.get('roc_auc', 'N/A'):.4f}`")
        st.write(f"- F1 Score: `{metrics.get('f1', 'N/A'):.4f}`")

with perf_col2:
    st.markdown("**Revenue Impact (Full Dataset)**")
    rev_impact = metadata.get("revenue_impact", {})
    if rev_impact:
        st.write(f"- Total MRR at Risk: `£{rev_impact.get('total_mrr_at_risk', 0):,.2f}`")
        st.write(f"- MRR Identified by Model: `£{rev_impact.get('mrr_identified', 0):,.2f}`")
        st.write(f"- Est. Annual Savings: `£{rev_impact.get('estimated_annual_savings', 0):,.2f}`")

with perf_col3:
    st.markdown("**Threshold Metrics**")
    thresh_metrics = metadata.get("threshold_metrics", {})

    if thresh_metrics:
        precision = thresh_metrics.get("precision")
        recall = thresh_metrics.get("recall")
        f1 = thresh_metrics.get("f1")

        st.write(
            f"- Precision at Threshold: `{float(precision):.4f}`"
            if precision is not None
            else "- Precision at Threshold: N/A"
        )

        st.write(
            f"- Recall at Threshold: `{float(recall):.4f}`"
            if recall is not None
            else "- Recall at Threshold: N/A"
        )

        st.write(
            f"- F1 at Threshold: `{float(f1):.4f}`"
            if f1 is not None
            else "- F1 at Threshold: N/A"
        )


# ─── Customer Drilldown ─────────────────────────────────────────

st.markdown("---")
st.subheader("🔍 Customer Drilldown")
st.markdown("Select a customer to see their full risk profile and contributing features.")

if features_df is not None:
    # Merge scored data with features for drilldown
    drilldown_df = filtered_df.merge(features_df, on="customer_id", how="left", suffixes=("", "_feat"))

    # Customer selector
    customer_ids = filtered_df.sort_values("churn_probability", ascending=False)["customer_id"].tolist()

    selected_customer = st.selectbox(
    "Select Customer (sorted by churn risk)",
    customer_ids,
    format_func=lambda x: (
        f"Customer {x} — "
        f"{str(filtered_df[filtered_df['customer_id'] == x]['risk_level'].values[0]).upper()} risk"
    ),
)

    if selected_customer:
        cust_row = drilldown_df[drilldown_df["customer_id"] == selected_customer].iloc[0]

        detail_col1, detail_col2, detail_col3 = st.columns(3)

        with detail_col1:
            st.markdown("**Prediction**")
            prob = cust_row["churn_probability"]
            st.metric("Churn Probability", f"{prob:.1%}")
            st.metric("Risk Level", cust_row["risk_level"].upper())
            st.metric(
                "Monthly Revenue",
                f"£{cust_row['monthly_revenue']:.2f}",
            )

        with detail_col2:
            st.markdown("**Customer Profile**")
            st.write(f"- Tenure: `{int(cust_row.get('tenure_months', 0))}` months")
            contract_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
            st.write(f"- Contract: `{contract_map.get(int(cust_row.get('contract_type', 0)), 'Unknown')}`")
            st.write(f"- Total Charges: `£{cust_row.get('total_charges', 0):,.2f}`")
            st.write(f"- Payment Method: `{int(cust_row.get('payment_method', 0))}`")

        with detail_col3:
            st.markdown("**Engagement**")
            st.write(f"- Login Frequency: `{cust_row.get('login_frequency', 0):.1f}`")
            st.write(f"- Feature Adoption: `{cust_row.get('feature_adoption_rate', 0):.0%}`")
            st.write(f"- Days Since Login: `{int(cust_row.get('days_since_last_login', 0))}`")
            st.write(f"- Engagement Score: `{cust_row.get('engagement_score', 0):.1f}`")
            st.write(f"- Support Tickets (90d): `{int(cust_row.get('support_tickets_last_90d', 0))}`")

        # Feature importance for this customer
        st.markdown("**Top Features for This Customer**")

        # Show the most important features (by absolute value)
        feature_cols = [c for c in drilldown_df.columns if c not in [
            "customer_id", "churn", "churn_probability", "churn_prediction",
            "monthly_revenue", "risk_level", "actual_churn",
        ] and not c.endswith("_feat")]

        if feature_cols:
            cust_features = cust_row[feature_cols].astype(float)
            top_features = cust_features.abs().sort_values(ascending=False).head(10)

            fig_features = go.Figure()
            fig_features.add_trace(
                go.Bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation="h",
                    marker_color=["#e74c3c" if cust_features[f] > 0 else "#2ecc71" for f in top_features.index],
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
    st.info("Full feature data not found — upload `features.csv` to enable customer drilldown.")


# ─── High-Risk Customer Table ───────────────────────────────────

st.markdown("---")
st.subheader("⚠️ High-Risk Customers — Action Required")

high_risk = filtered_df[filtered_df["risk_level"].isin(["critical", "high"])].sort_values(
    "churn_probability", ascending=False
)

if len(high_risk) > 0:
    display_cols = ["customer_id", "churn_probability", "risk_level", "monthly_revenue", "churn_prediction"]
    st.dataframe(
        high_risk[display_cols]
        .head(50)
        .style.format(
            {
                "churn_probability": "{:.1%}",
                "monthly_revenue": "£{:,.2f}",
            }
        )
        .applymap(
            lambda x: "background-color: #fadbd8" if x == "critical" else (
                "background-color: #fdebd0" if x == "high" else ""
            ),
            subset=["risk_level"],
        ),
        use_container_width=True,
        height=500,
    )

    # Download button
    csv = high_risk[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download High-Risk Customers CSV",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )
else:
    st.success("No high-risk customers match the current filters.")


# ─── Footer ─────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "SaaS Churn Prediction Dashboard · Built with Streamlit & Plotly · "
    f"Model: {metadata.get('best_model', 'N/A')} · "
    f"Threshold: {metadata.get('optimal_threshold', 0):.4f}"
    "</div>",
    unsafe_allow_html=True,
)
