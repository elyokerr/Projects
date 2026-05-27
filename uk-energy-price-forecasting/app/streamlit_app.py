"""UK Day-Ahead Electricity Price Forecasting — Streamlit app (Phase 7).

Views
-----
1. Day-ahead forecast — fan chart for a chosen midnight origin.
2. Backtest explorer   — rolling-origin metrics + per-horizon MAE + ablation table.

Runs with zero secrets using the committed synthetic fixture panel.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest.rolling_origin import (
    BacktestResult,
    build_ablation_table,
    generate_origins,
    run_backtest,
)
from src.build.fixtures import load_fixture_panel
from src.metrics.pinball import mean_pinball
from src.metrics.point_metrics import mae
from src.models.baselines import SeasonalNaive
from src.models.global_ml import GlobalLGBM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZON = 48
QUANTILES = (0.1, 0.5, 0.9)
BACKTEST_DAYS = 5  # origins over the last ~5 fixture days


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_panel():
    """Load and cache the fixture PanelBundle."""
    return load_fixture_panel()


@st.cache_resource
def get_fitted_lgbm(_bundle, train_end_str: str):
    """Fit and cache a GlobalLGBM model.

    The leading underscore on ``_bundle`` prevents Streamlit from trying to
    hash the PanelBundle (not hashable).  The ``train_end_str`` key ensures
    a new model is trained if the training cutoff changes.
    """
    train_end = pd.Timestamp(train_end_str)
    model = GlobalLGBM(quantiles=QUANTILES)
    model.fit(_bundle, train_end=train_end)
    return model


# ---------------------------------------------------------------------------
# Pure helper functions (testable outside Streamlit)
# ---------------------------------------------------------------------------


def compute_forecast(
    bundle,
    model,
    origin: pd.Timestamp,
    horizon: int = HORIZON,
    quantiles: tuple = QUANTILES,
) -> dict:
    """Return a dict with forecast arrays and metadata for one origin.

    Parameters
    ----------
    bundle : PanelBundle
    model   : fitted model with predict_quantiles()
    origin  : pd.Timestamp — last observed step (tz-naive)
    horizon : int
    quantiles : tuple of float

    Returns
    -------
    dict with keys:
        "timestamps"  — pd.DatetimeIndex of forecast steps
        "quantiles"   — {q: np.ndarray}
        "actuals"     — np.ndarray of length horizon (observed values)
        "origin"      — the origin timestamp
    """
    target_index = bundle.target.time_index
    target_values = bundle.target.values().flatten()

    pos = target_index.get_loc(origin)
    forecast_timestamps = target_index[pos + 1 : pos + 1 + horizon]
    actuals = target_values[pos + 1 : pos + 1 + horizon]

    preds = model.predict_quantiles(bundle, origin, horizon, quantiles)

    return {
        "timestamps": forecast_timestamps,
        "quantiles": preds,
        "actuals": actuals,
        "origin": origin,
    }


def compute_window_metrics(actuals: np.ndarray, preds: dict) -> dict:
    """Pinball loss + MAE for a single forecast window.

    Parameters
    ----------
    actuals : np.ndarray  shape (horizon,)
    preds   : {q: np.ndarray}

    Returns
    -------
    dict with "pinball" and "mae" keys.
    """
    pinball_val = mean_pinball(actuals, preds)
    q_median = min(preds.keys(), key=lambda q: abs(q - 0.5))
    mae_val = mae(actuals, preds[q_median])
    return {"pinball": pinball_val, "mae": mae_val}


def run_demo_backtest(bundle, model, model_name: str | None = None) -> BacktestResult:
    """Run a small rolling-origin backtest over the last ~BACKTEST_DAYS days.

    Parameters
    ----------
    bundle     : PanelBundle
    model      : fitted or fittable model
    model_name : str or None

    Returns
    -------
    BacktestResult
    """
    target_index = bundle.target.time_index
    # Reserve enough room: origins must have >=HORIZON steps after them.
    end_limit = target_index[-(HORIZON + 1)]
    backtest_start = target_index[-(BACKTEST_DAYS * 48 + HORIZON + 1)]

    origins = generate_origins(
        target_index,
        start=backtest_start,
        step="1D",
        end=end_limit,
    )

    return run_backtest(
        model,
        bundle,
        origins,
        horizon=HORIZON,
        quantiles=QUANTILES,
        refit=False,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------


def fan_chart(
    fc: dict,
    title: str = "Day-ahead forecast",
) -> go.Figure:
    """Build a Plotly fan chart from a compute_forecast() result dict."""
    timestamps = fc["timestamps"]
    preds = fc["quantiles"]
    actuals = fc["actuals"]

    q_lo, q_med, q_hi = 0.1, 0.5, 0.9
    x = list(timestamps)

    fig = go.Figure()

    # Shaded band q0.1—q0.9
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=list(preds[q_hi]) + list(preds[q_lo])[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% interval (q0.1–q0.9)",
            hoverinfo="skip",
        )
    )

    # Median forecast line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=list(preds[q_med]),
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            name="Median forecast (q0.5)",
        )
    )

    # Actual out-turn
    fig.add_trace(
        go.Scatter(
            x=x,
            y=list(actuals),
            mode="lines",
            line=dict(color="#d62728", width=2, dash="dot"),
            name="Actual out-turn",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Settlement period",
        yaxis_title="Price (£/MWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    return fig


def horizon_mae_chart(result: BacktestResult, title: str = "MAE by forecast horizon") -> go.Figure:
    """Per-horizon MAE (steps 1..HORIZON) from a BacktestResult."""
    q_median = min(result.quantiles, key=lambda q: abs(q - 0.5))
    median_fc = result.forecasts[q_median]  # (n_origins, horizon)

    # MAE at each horizon step
    horizon_mae_vals = np.mean(np.abs(result.actuals - median_fc), axis=0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(horizon_mae_vals) + 1)),
            y=list(horizon_mae_vals),
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color="#1f77b4", width=2),
            name="MAE",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Horizon step (settlement periods, 30 min each)",
        yaxis_title="MAE (£/MWh)",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit layout helpers — called inside the script-run context
# ---------------------------------------------------------------------------


def _build_model(bundle, model_label: str, train_end: pd.Timestamp):
    """Return a fitted model instance for the given label."""
    if model_label.startswith("Seasonal"):
        m = SeasonalNaive(season=48)
        m.fit(bundle)
        return m
    # LightGBM — use cached fit keyed by train_end
    return get_fitted_lgbm(bundle, str(train_end))


def _render_day_ahead_tab(bundle, model_label: str, tab):
    """Render the day-ahead forecast view."""
    with tab:
        target_index = bundle.target.time_index

        # Generate all midnight origins, leave last HORIZON steps as buffer.
        all_origins = generate_origins(
            target_index,
            start=target_index[0],
            step="1D",
            end=target_index[-(HORIZON + 1)],
        )

        if not all_origins:
            st.error("No valid forecast origins found in the fixture panel.")
            return

        # Origin selector — default to last valid origin
        origin_labels = [str(o) for o in all_origins]
        default_idx = len(origin_labels) - 1
        selected_label = st.selectbox(
            "Forecast origin (midnight)",
            options=origin_labels,
            index=default_idx,
            key="origin_selector",
        )
        origin = pd.Timestamp(selected_label)

        # Fit / retrieve model
        train_end = origin  # train up to origin
        with st.spinner("Fitting model (first run only)…"):
            model = _build_model(bundle, model_label, train_end)

        # Compute forecast
        fc = compute_forecast(bundle, model, origin, HORIZON, QUANTILES)
        metrics = compute_window_metrics(fc["actuals"], fc["quantiles"])

        # Metrics row
        col1, col2 = st.columns(2)
        col1.metric("Pinball loss (this window)", f"{metrics['pinball']:.2f} £/MWh")
        col2.metric("MAE — median forecast", f"{metrics['mae']:.2f} £/MWh")

        # Fan chart
        title = f"Day-ahead forecast — origin {selected_label} | {model_label}"
        fig = fan_chart(fc, title=title)
        st.plotly_chart(fig, use_container_width=True)

        if model_label.startswith("Seasonal"):
            st.caption(
                "Seasonal-naive replicates yesterday's prices verbatim — "
                "the uncertainty band has zero width (all quantiles are equal)."
            )


def _render_backtest_tab(bundle, model_label: str, tab):
    """Render the backtest explorer view."""
    with tab:
        st.subheader("Rolling-origin backtest")
        st.caption(
            f"Evaluating over the last ~{BACKTEST_DAYS} fixture days "
            f"({BACKTEST_DAYS} midnight origins, horizon = {HORIZON} steps = 24 h)."
        )

        # ---- Run naive backtest (always fast) --------------------------------
        @st.cache_data
        def _naive_backtest(_bundle_key):
            _b = load_panel()
            m = SeasonalNaive(season=48)
            return run_demo_backtest(_b, m, model_name="seasonal_naive")

        naive_result = _naive_backtest(id(bundle))

        # ---- Run selected model's backtest -----------------------------------
        is_lgbm = not model_label.startswith("Seasonal")

        @st.cache_data
        def _lgbm_backtest(_bundle_key):
            _b = load_panel()
            target_index = _b.target.time_index
            train_end = target_index[-(BACKTEST_DAYS * 48 + HORIZON + 2)]
            m = get_fitted_lgbm(_b, str(train_end))
            return run_demo_backtest(_b, m, model_name="GlobalLGBM")

        if is_lgbm:
            with st.spinner("Running LightGBM backtest (~45 s first run)…"):
                lgbm_result = _lgbm_backtest(id(bundle))
            active_result = lgbm_result
        else:
            active_result = naive_result

        # ---- Metrics tiles for active model ----------------------------------
        st.subheader(f"Metrics — {model_label}")
        m_dict = active_result.metrics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pinball", f"{m_dict['pinball']:.3f}")
        c2.metric("Coverage 80%", f"{m_dict['coverage_80']:.2%}")
        c3.metric("CRPS", f"{m_dict['crps']:.3f}")
        c4.metric("MAE", f"{m_dict['mae']:.2f} £/MWh")

        # ---- Per-horizon MAE chart -------------------------------------------
        st.plotly_chart(
            horizon_mae_chart(active_result, title=f"MAE by horizon — {model_label}"),
            use_container_width=True,
        )

        # ---- Ablation table --------------------------------------------------
        st.subheader("Ablation table")
        if is_lgbm:
            results_dict = {
                "seasonal_naive": naive_result,
                "GlobalLGBM": lgbm_result,
            }
            ablation_df = build_ablation_table(results_dict, baseline="seasonal_naive")
            st.dataframe(ablation_df.style.format("{:.4f}"), use_container_width=True)
        else:
            results_dict = {"seasonal_naive": naive_result}
            ablation_df = build_ablation_table(results_dict, baseline="seasonal_naive")
            st.dataframe(ablation_df.style.format("{:.4f}"), use_container_width=True)
            st.info(
                "Switch to **Global LightGBM** in the sidebar to add the LGBM row. "
                "TiDE / TFT / Chronos rows are populated from the Colab training run "
                "(see notebooks/phase6_deep_models.ipynb)."
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="UK Electricity Price Forecasting",
        page_icon="⚡",
        layout="wide",
    )

    st.title("UK Day-Ahead Electricity Price Forecasting")
    st.info(
        "Running on the committed synthetic fixture panel (no API keys needed). "
        "See data/README.md for using live Elexon + ENTSO-E data."
    )

    # Sidebar model selector
    model_label = st.sidebar.selectbox(
        "Model",
        [
            "Seasonal-naive (instant)",
            "Global LightGBM (probabilistic, ~45s first run)",
        ],
    )

    bundle = load_panel()

    tab_forecast, tab_backtest = st.tabs(["Day-ahead forecast", "Backtest explorer"])

    _render_day_ahead_tab(bundle, model_label, tab_forecast)
    _render_backtest_tab(bundle, model_label, tab_backtest)


if __name__ == "__main__":
    main()
