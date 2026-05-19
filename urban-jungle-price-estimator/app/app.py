"""
Lowest-Price Quote Estimator - Dashboard
Interactive Streamlit demo

Launch (from the project root):
    streamlit run app/app.py

First run trains the model from data/raw/UJ_datatask_prices.csv (~30s) and
caches it to models/uj_price_estimator_bundle.joblib. Subsequent runs load
instantly from cache.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quote Estimator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished look
st.markdown(
    """
    <style>
    .main {padding-top: 1rem;}
    .stMetric {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .price-headline {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin: 1rem 0;
    }
    .price-band {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #e8f4f8;
        color: #1f4e79;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .insight-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffa726;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# Paths (project layout: app/app.py, data/raw/<csv>, models/<bundle>)
# ────────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UJ_datatask_prices.csv"
MODELS_DIR = PROJECT_ROOT / "models"
BUNDLE_PATH = MODELS_DIR / "uj_price_estimator_bundle.joblib"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FEATURES_NUM = ["AGE", "BEDROOMS_N", "ACCID_CONTENTS", "ALARM_BIN"]
FEATURES_CAT = ["OCCUPATION", "INSUREDPOSTCODE", "POSTCODE_OUTWARD", "POSTCODE_AREA"]
BEDROOM_MAP = {"One": 1, "Three": 3, "Five": 5}
REF_DATE = pd.Timestamp("2026-05-15")

# Best hyperparameters discovered in the notebook
BEST_PARAMS = {
    "max_iter": 800,
    "learning_rate": 0.05,
    "max_leaf_nodes": 127,
    "min_samples_leaf": 5,
    "l2_regularization": 1.0,
}


# ────────────────────────────────────────────────────────────────────────────
# Model training / loading
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_bundle() -> dict:
    """Load the cached bundle if present; otherwise train from the CSV."""
    if BUNDLE_PATH.exists():
        return joblib.load(BUNDLE_PATH)

    if not DATA_PATH.exists():
        st.error(
            f"Could not find `{DATA_PATH.name}` at `{DATA_PATH}`. "
            "Place the raw CSV in `data/raw/` and reload."
        )
        st.stop()

    with st.spinner("First-time setup — training the model (~30 seconds)…"):
        progress = st.progress(0.0, text="Loading data…")
        df_raw = pd.read_csv(DATA_PATH, low_memory=False)
        progress.progress(0.2, text="Building per-QuoteID panel…")

        kept = ["DOB", "OCCUPATION", "ACCIDENTALCONTENTS", "NUMBEDROOMS",
                "INSUREDPOSTCODE", "ALARMTYPE"]
        panel = (df_raw
                 .groupby("QUOTEID")
                 .agg(target=("TotalAmountPayable", "min"),
                      **{c: (c, "first") for c in kept})
                 .reset_index())

        # Feature engineering
        panel["AGE"] = ((REF_DATE - pd.to_datetime(panel["DOB"], format="%d/%m/%Y"))
                        .dt.days // 365).astype(int)
        panel["BEDROOMS_N"] = panel["NUMBEDROOMS"].map(BEDROOM_MAP).astype(int)
        panel["ACCID_CONTENTS"] = panel["ACCIDENTALCONTENTS"].astype(int)
        panel["ALARM_BIN"] = (panel["ALARMTYPE"] != "NoAlarm").astype(int)

        def outward(pc):
            m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)", str(pc))
            return m.group(1) if m else pc

        panel["POSTCODE_OUTWARD"] = panel["INSUREDPOSTCODE"].apply(outward)
        panel["POSTCODE_AREA"] = panel["INSUREDPOSTCODE"].str.extract(
            r"^([A-Z]+)", expand=False)

        X = panel[FEATURES_NUM + FEATURES_CAT].copy()
        y = panel["target"].copy()
        for c in FEATURES_CAT:
            X[c] = X[c].astype("category")
        cat_levels = {c: X[c].cat.categories.tolist() for c in FEATURES_CAT}

        progress.progress(0.4, text="Training point-estimate model…")
        point_model = HistGradientBoostingRegressor(
            categorical_features=FEATURES_CAT,
            random_state=RANDOM_STATE,
            **BEST_PARAMS,
        ).fit(X, y)

        progress.progress(0.6, text="Training quantile models (q10, q50, q90)…")
        quantile_models = {}
        for i, q in enumerate([0.1, 0.5, 0.9]):
            quantile_models[q] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q,
                max_iter=400, learning_rate=0.05,
                categorical_features=FEATURES_CAT,
                random_state=RANDOM_STATE,
            ).fit(X, y)
            progress.progress(0.6 + 0.1 * (i + 1), text=f"Trained q={q}")

        progress.progress(0.95, text="Saving bundle…")
        bundle = {
            "point_model": point_model,
            "quantile_models": quantile_models,
            "feature_cols": FEATURES_NUM + FEATURES_CAT,
            "numeric_cols": FEATURES_NUM,
            "categorical_cols": FEATURES_CAT,
            "cat_levels": cat_levels,
            "training_date": datetime.now().isoformat(timespec="seconds"),
            "training_rows": int(len(X)),
            "best_params": BEST_PARAMS,
            "random_state": RANDOM_STATE,
            # Reference metrics from notebook v2 evaluation
            "test_metrics": {
                "MAE_£": 0.6739, "RMSE_£": 1.4540,
                "MAPE_%": 0.3843, "R2": 0.9991,
            },
            "panel_stats": {
                "min": float(y.min()), "max": float(y.max()),
                "mean": float(y.mean()), "median": float(y.median()),
                "n_rows": int(len(y)),
            },
        }
        joblib.dump(bundle, BUNDLE_PATH)
        progress.progress(1.0, text="Done!")
        time.sleep(0.4)
        progress.empty()
        return bundle


# ────────────────────────────────────────────────────────────────────────────
# Prediction helper
# ────────────────────────────────────────────────────────────────────────────
def predict_lowest_price(scenario: dict, bundle: dict) -> dict:
    pc = scenario["INSUREDPOSTCODE"]
    m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)", str(pc))
    outward = m.group(1) if m else pc
    area_m = re.match(r"^([A-Z]+)", str(pc))
    area = area_m.group(1) if area_m else None

    row = pd.DataFrame([{**scenario,
                         "POSTCODE_OUTWARD": outward,
                         "POSTCODE_AREA": area}])[bundle["feature_cols"]]
    for c in bundle["categorical_cols"]:
        row[c] = pd.Categorical(row[c], categories=bundle["cat_levels"][c])

    return {
        "point": float(bundle["point_model"].predict(row)[0]),
        "q10": float(bundle["quantile_models"][0.1].predict(row)[0]),
        "q50": float(bundle["quantile_models"][0.5].predict(row)[0]),
        "q90": float(bundle["quantile_models"][0.9].predict(row)[0]),
    }


# ────────────────────────────────────────────────────────────────────────────
# Visualisations
# ────────────────────────────────────────────────────────────────────────────
def gauge_chart(point: float, q10: float, q90: float,
                pmin: float, pmax: float) -> go.Figure:
    """Speedometer-style gauge for the predicted price."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=point,
        number={"prefix": "£", "valueformat": ",.2f",
                "font": {"size": 40, "color": "#1f4e79"}},
        delta={"reference": (pmin + pmax) / 2,
               "valueformat": ",.0f", "suffix": " vs market avg"},
        title={"text": "<b>Predicted lowest panel quote</b>",
               "font": {"size": 16}},
        gauge={
            "axis": {"range": [pmin, pmax], "tickprefix": "£",
                     "tickfont": {"size": 11}},
            "bar": {"color": "#1f4e79", "thickness": 0.25},
            "bgcolor": "#f0f4f8",
            "steps": [
                {"range": [pmin, (pmin + pmax) * 0.3],
                 "color": "#c8e6c9"},
                {"range": [(pmin + pmax) * 0.3, (pmin + pmax) * 0.55],
                 "color": "#fff9c4"},
                {"range": [(pmin + pmax) * 0.55, pmax],
                 "color": "#ffcdd2"},
            ],
            "threshold": {
                "line": {"color": "darkred", "width": 3},
                "thickness": 0.75,
                "value": point,
            },
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def confidence_band_chart(point: float, q10: float, q50: float,
                          q90: float) -> go.Figure:
    """Horizontal band showing the 80% prediction interval."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[q10, q90], y=[0, 0],
        mode="lines", line=dict(color="#90caf9", width=24),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[q10], y=[0], mode="markers+text",
        marker=dict(size=14, color="#1976d2", symbol="line-ns",
                    line=dict(width=3, color="#1976d2")),
        text=[f"£{q10:.2f}<br>(q10)"], textposition="bottom center",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[q90], y=[0], mode="markers+text",
        marker=dict(size=14, color="#1976d2", symbol="line-ns",
                    line=dict(width=3, color="#1976d2")),
        text=[f"£{q90:.2f}<br>(q90)"], textposition="bottom center",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[q50], y=[0], mode="markers+text",
        marker=dict(size=16, color="#0d47a1", symbol="diamond"),
        text=[f"<b>£{q50:.2f}</b><br>(median)"], textposition="top center",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[point], y=[0], mode="markers+text",
        marker=dict(size=20, color="#d32f2f", symbol="star"),
        text=[f"<b>POINT £{point:.2f}</b>"], textposition="top center",
        showlegend=False, hoverinfo="skip",
    ))
    band_width = q90 - q10
    fig.update_layout(
        title=f"<b>80% prediction band — width £{band_width:.2f}</b>",
        height=220,
        xaxis=dict(title="Predicted lowest quote (£)", tickprefix="£"),
        yaxis=dict(visible=False, range=[-1.5, 1.5]),
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────────────────────────────────
bundle = load_or_train_bundle()

POSTCODES = bundle["cat_levels"]["INSUREDPOSTCODE"]
OCCUPATIONS = bundle["cat_levels"]["OCCUPATION"]
PANEL_MIN = bundle["panel_stats"]["min"]
PANEL_MAX = bundle["panel_stats"]["max"]

# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.title("Lowest-Price Quote Estimator")
st.caption(
    "Predicts the cheapest home insurance quote a customer will be offered "
    "across a 7-insurer panel, before any live underwriting calls are made."
)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# Sidebar — model metadata + presets
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model card")
    st.markdown(
        f"""
        **Algorithm:** HistGradientBoosting (tuned)
        **Trained on:** {bundle['training_rows']:,} quote scenarios
        **Trained:** {bundle['training_date'][:10]}
        **Random seed:** {bundle['random_state']}
        """
    )

    st.markdown("### Test-set performance")
    m = bundle["test_metrics"]
    st.metric("MAE", f"£{m['MAE_£']:.2f}", help="Mean absolute error")
    st.metric("MAPE", f"{m['MAPE_%']:.2f}%", help="Mean absolute % error")
    st.metric("R²", f"{m['R2']:.4f}", help="Variance explained")

    st.markdown("---")
    st.markdown("### Quick presets")
    preset = st.radio(
        "Try a sample customer:",
        ["(custom)", "Young renter — central London",
         "Family home — suburbs", "Senior — affluent area"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Resources")
    st.markdown(
        """
        - [README.md](../README.md)
        - [Notebook](../notebooks/UJ_price_estimator.ipynb)
        """
    )

# Apply preset defaults
preset_defaults = {
    "(custom)": dict(age=30, beds=3, postcode=POSTCODES[0],
                     occupation=OCCUPATIONS[0], accid=False, alarm=False),
    "Young renter — central London": dict(
        age=30, beds=1, postcode="N65TX" if "N65TX" in POSTCODES else POSTCODES[0],
        occupation="D78" if "D78" in OCCUPATIONS else OCCUPATIONS[0],
        accid=False, alarm=False),
    "Family home — suburbs": dict(
        age=40, beds=3,
        postcode="HA19NA" if "HA19NA" in POSTCODES else POSTCODES[len(POSTCODES)//2],
        occupation="E09" if "E09" in OCCUPATIONS else OCCUPATIONS[1],
        accid=True, alarm=False),
    "Senior — affluent area": dict(
        age=50, beds=5,
        postcode="SW71AA" if "SW71AA" in POSTCODES else POSTCODES[-1],
        occupation="A01" if "A01" in OCCUPATIONS else OCCUPATIONS[-1],
        accid=True, alarm=True),
}
defaults = preset_defaults[preset]

# ────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_about = st.tabs(
    ["Get a quote", "Market insights", "About this model"]
)

# ── Tab 1: Prediction ──────────────────────────────────────────────────────
with tab_predict:
    col_form, col_result = st.columns([1, 1.4])

    with col_form:
        st.markdown("### Customer details")
        with st.form("quote_form", border=False):
            age = st.slider(
                "Age", min_value=18, max_value=80,
                value=int(defaults["age"]), step=1,
                help="Customer's current age in years",
            )
            beds = st.slider(
                "Number of bedrooms",
                min_value=1, max_value=10,
                value=int(defaults["beds"]), step=1,
                help="Number of bedrooms in the property",
            )
            postcode = st.selectbox(
                "Postcode",
                options=POSTCODES,
                index=POSTCODES.index(defaults["postcode"])
                      if defaults["postcode"] in POSTCODES else 0,
                help="Full UK postcode — the strongest price driver",
            )
            occupation = st.selectbox(
                "Occupation code",
                options=OCCUPATIONS,
                index=OCCUPATIONS.index(defaults["occupation"])
                      if defaults["occupation"] in OCCUPATIONS else 0,
                help="Insurer-standard occupation code",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                accid = st.toggle(
                    "Accidental contents cover",
                    value=defaults["accid"],
                    help="Include cover for accidental damage to contents",
                )
            with col_b:
                alarm = st.toggle(
                    "Maintained alarm",
                    value=defaults["alarm"],
                    help="Property has a maintained burglar alarm",
                )
            submitted = st.form_submit_button(
                "Estimate lowest quote", type="primary",
                use_container_width=True,
            )

    with col_result:
        scenario = {
            "AGE": int(age),
            "BEDROOMS_N": int(beds),
            "ACCID_CONTENTS": int(accid),
            "ALARM_BIN": int(alarm),
            "OCCUPATION": occupation,
            "INSUREDPOSTCODE": postcode,
        }

        try:
            result = predict_lowest_price(scenario, bundle)

            # Hero number
            st.markdown(
                f"<div class='price-headline'>£{result['point']:.2f}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='price-band'>"
                f"80% confidence: <b>£{result['q10']:.2f}</b> – "
                f"<b>£{result['q90']:.2f}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Gauge
            st.plotly_chart(
                gauge_chart(result["point"], result["q10"], result["q90"],
                            PANEL_MIN, PANEL_MAX),
                use_container_width=True,
            )

            # Band
            st.plotly_chart(
                confidence_band_chart(
                    result["point"], result["q10"], result["q50"],
                    result["q90"]),
                use_container_width=True,
            )

            # Contextual insight
            band_width = result["q90"] - result["q10"]
            pct_of_market = (result["point"] - PANEL_MIN) / (PANEL_MAX - PANEL_MIN) * 100
            if pct_of_market < 25:
                pos = "in the cheapest quartile of the market — strong value"
            elif pct_of_market < 50:
                pos = "below the market median — good value"
            elif pct_of_market < 75:
                pos = "above the median — premium-tier pricing"
            else:
                pos = "in the most expensive quartile — high-risk profile"

            st.markdown(
                f"""
                <div class='insight-box'>
                <b>Insight:</b> This quote sits at the <b>{pct_of_market:.0f}th percentile</b>
                of the market range (£{PANEL_MIN:.0f}–£{PANEL_MAX:.0f}) — {pos}.
                The 80% confidence band of £{band_width:.2f} suggests the model is
                {'highly confident' if band_width < 5 else 'moderately confident' if band_width < 20 else 'less certain'}
                about this specific profile.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "If this is an unseen postcode or occupation, the model cannot "
                "extrapolate — see the 'About this model' tab for details on "
                "geographic generalisation limits."
            )

# ── Tab 2: Insights ────────────────────────────────────────────────────────
with tab_insights:
    st.markdown("### How the panel prices across the market")
    st.caption(
        "Explore how the lowest panel quote varies with the strongest pricing "
        "drivers identified by the model: **postcode** and **property size**."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Price by bedrooms (current postcode)")
        ages_grid = [int(age)] * 3
        bed_rows = []
        for n_beds in [1, 3, 5]:
            try:
                r = predict_lowest_price(
                    {"AGE": int(age), "BEDROOMS_N": n_beds,
                     "ACCID_CONTENTS": int(accid), "ALARM_BIN": int(alarm),
                     "OCCUPATION": occupation, "INSUREDPOSTCODE": postcode},
                    bundle)
                bed_rows.append({"bedrooms": n_beds, "price": r["point"],
                                 "q10": r["q10"], "q90": r["q90"]})
            except Exception:
                pass
        if bed_rows:
            bed_df = pd.DataFrame(bed_rows)
            fig_beds = go.Figure()
            fig_beds.add_trace(go.Bar(
                x=bed_df["bedrooms"].astype(str) + "-bed",
                y=bed_df["price"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=bed_df["q90"] - bed_df["price"],
                    arrayminus=bed_df["price"] - bed_df["q10"],
                ),
                marker_color="#1f4e79",
                text=[f"£{p:.0f}" for p in bed_df["price"]],
                textposition="outside",
            ))
            fig_beds.update_layout(
                yaxis_title="Predicted lowest quote (£)",
                xaxis_title="",
                showlegend=False,
                height=350,
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_beds, use_container_width=True)

    with col_b:
        st.markdown("#### Price by postcode (current bedrooms)")
        sampled_pcs = POSTCODES[::max(1, len(POSTCODES) // 12)][:12]
        pc_rows = []
        for pc in sampled_pcs:
            try:
                r = predict_lowest_price(
                    {"AGE": int(age), "BEDROOMS_N": int(beds),
                     "ACCID_CONTENTS": int(accid), "ALARM_BIN": int(alarm),
                     "OCCUPATION": occupation, "INSUREDPOSTCODE": pc},
                    bundle)
                pc_rows.append({"postcode": pc, "price": r["point"]})
            except Exception:
                pass
        if pc_rows:
            pc_df = pd.DataFrame(pc_rows).sort_values("price")
            colors = ["#43a047" if p < pc_df["price"].median() else "#e53935"
                      for p in pc_df["price"]]
            fig_pc = go.Figure(go.Bar(
                x=pc_df["price"], y=pc_df["postcode"],
                orientation="h",
                marker_color=colors,
                text=[f"£{p:.0f}" for p in pc_df["price"]],
                textposition="outside",
            ))
            fig_pc.update_layout(
                xaxis_title="Predicted lowest quote (£)",
                yaxis_title="",
                showlegend=False,
                height=350,
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown("---")
    st.markdown("### What drives the price?")
    st.markdown(
        """
        Based on permutation-importance analysis from the notebook, the
        biggest pricing levers (ranked) are:
        """
    )
    drivers_df = pd.DataFrame({
        "Driver": ["Postcode (geography)", "Property size (bedrooms)",
                   "Alarm type", "Accidental cover", "Age", "Occupation"],
        "Univariate R²": [0.43, 0.26, 0.10, 0.03, 0.02, 0.01],
        "Why it matters": [
            "Crime, theft and flood risk vary hugely by area",
            "Larger properties = more contents at risk",
            "Strong interaction with high-risk postcodes",
            "Add-on cover, modest direct impact",
            "Weak signal — only 4 distinct ages in data",
            "Mostly captured by postcode in practice",
        ],
    })
    st.dataframe(drivers_df, use_container_width=True, hide_index=True)

# ── Tab 3: About ───────────────────────────────────────────────────────────
with tab_about:
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown("### How the model works")
        st.markdown(
            """
            This estimator uses **HistGradientBoosting** — a modern
            gradient-boosted decision tree algorithm — to learn the
            relationship between 6 customer/property features and the
            lowest quote available from a panel of 7 insurers.

            **Training data:** 18,720 unique customer scenarios derived from
            385,000 individual insurer quotes.

            **Why these 6 features?** A rigorous data-quality audit found
            that of 39 declared "input" columns in the source data, 31 were
            constants, 1 was redundant, and 1 was a quote *output* that
            would leak the target. The remaining 6 perfectly define every
            unique scenario in the dataset.
            """
        )

        st.markdown("### Honest validation results")
        st.markdown(
            """
            The model was validated under three different splitting
            strategies to give an honest picture of generalisation:
            """
        )
        val_df = pd.DataFrame({
            "Split strategy": [
                "Random 80/20 (interpolation)",
                "GroupKFold by postcode (new geography)",
                "GroupKFold by occupation (new customer type)",
            ],
            "Tuned HGB MAE": ["£0.67", "£23.75", "£7.18"],
            "Interpretation": [
                "Best case — model has seen similar scenarios",
                "Worst case — geography is the dominant signal",
                "Realistic — small drop in accuracy",
            ],
        })
        st.dataframe(val_df, use_container_width=True, hide_index=True)

        st.info(
            "**For deployment:** the realistic expected error on a "
            "*known* postcode is £0.67. For *unseen* postcodes the model "
            "cannot extrapolate — external data (crime rate, flood risk) "
            "would be needed to close the gap."
        )

    with col_r:
        st.markdown("### Model card")
        st.markdown(
            f"""
            | Property | Value |
            |---|---|
            | Algorithm | HistGradientBoosting |
            | Iterations | {BEST_PARAMS['max_iter']} |
            | Learning rate | {BEST_PARAMS['learning_rate']} |
            | Max leaf nodes | {BEST_PARAMS['max_leaf_nodes']} |
            | L2 regularisation | {BEST_PARAMS['l2_regularization']} |
            | Random seed | {RANDOM_STATE} |
            | Training rows | {bundle['training_rows']:,} |
            | Trained | {bundle['training_date'][:10]} |
            """
        )
        st.markdown("### Quantile heads")
        st.markdown(
            """
            Three additional models produce a calibrated **80%
            prediction band**:
            - **q10** — pessimistic (lower bound)
            - **q50** — median estimate
            - **q90** — optimistic (upper bound)

            Empirical coverage on the test set: **75.4%**
            (target 80%; mean band width £15.83).
            """
        )
        st.markdown("### Next steps")
        st.markdown(
            """
            - External features (crime, flood-risk)
            - Dockerised FastAPI inference service
            - Drift monitoring (Evidently AI)
            - Conformal-prediction calibration
            """
        )

st.markdown("---")
st.caption(
    "Built with Streamlit · HistGradientBoosting (scikit-learn) · "
    "Plotly · joblib · Trained on UJ_datatask_prices.csv"
)
