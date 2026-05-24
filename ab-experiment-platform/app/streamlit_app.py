"""Streamlit front-end for ab-experiment-platform.

This file is a THIN CLIENT: it contains no statistics logic.
All computation is delegated to the src.abtest library.
"""

# --- sys.path prepend so imports work on Streamlit Community Cloud ---
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.abtest import (
    ExperimentData,
    aa_test,
    apply_cuped,
    beta_binomial,
    decide,
    msprt_stream,
    naive_peeking_fpr,
    power_for_sample_size,
    required_sample_size,
    simulate_conversion,
    srm_check,
    two_proportion_z,
    welch_t,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="A/B Experiment Platform", layout="wide")
st.title("A/B Experiment Platform")
st.caption(
    "A self-contained toolkit for designing, validating, analysing, "
    "and monitoring online experiments — powered by the ab-experiment-platform library."
)

# ---------------------------------------------------------------------------
# Sidebar — data source
# ---------------------------------------------------------------------------
st.sidebar.header("Data source")
data_source = st.sidebar.radio(
    "Choose data source", ["Simulated data", "Upload CSV"], index=0
)

data: ExperimentData | None = None
metric: str | None = None
covariate: str | None = None

if data_source == "Simulated data":
    st.sidebar.subheader("Simulation parameters")
    n_per_arm = st.sidebar.number_input("n per arm", min_value=100, max_value=500_000, value=5000, step=500)
    base_rate = st.sidebar.number_input("Base rate", min_value=0.01, max_value=0.99, value=0.20, step=0.01, format="%.2f")
    absolute_lift = st.sidebar.number_input("Absolute lift", min_value=-0.5, max_value=0.5, value=0.02, step=0.005, format="%.3f")
    covariate_corr = st.sidebar.number_input("Covariate correlation", min_value=0.0, max_value=1.0, value=0.0, step=0.1, format="%.1f")
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

    ed = simulate_conversion(
        n_per_arm=int(n_per_arm),
        base_rate=float(base_rate),
        absolute_lift=float(absolute_lift),
        covariate_corr=float(covariate_corr),
        seed=int(seed),
    )
    data = ed
    metric = "converted"
    covariate = "pre_covariate" if covariate_corr > 0 else None
    st.sidebar.success(f"Simulated {int(n_per_arm)*2:,} rows ({int(n_per_arm):,} per arm).")

else:  # Upload CSV
    st.sidebar.subheader("Upload your data")
    uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        cols = list(df_raw.columns)

        variant_col = st.sidebar.selectbox("Variant column", cols)
        metric_col = st.sidebar.selectbox("Metric column", cols)
        cov_options = ["(none)"] + cols
        cov_sel = st.sidebar.selectbox("Covariate column (optional)", cov_options)
        ctrl_label = st.sidebar.text_input("Control label", value="control")
        treat_label = st.sidebar.text_input("Treatment label", value="treatment")

        try:
            cov_col = cov_sel if cov_sel != "(none)" else None
            data = ExperimentData(
                df_raw,
                variant_col=variant_col,
                metric_cols=[metric_col],
                control_label=ctrl_label,
                treatment_label=treat_label,
                covariate_col=cov_col,
            )
            metric = metric_col
            covariate = cov_col
            st.sidebar.success(f"Loaded {len(df_raw):,} rows.")
        except ValueError as exc:
            st.sidebar.error(str(exc))
    else:
        st.sidebar.info("Upload a CSV to get started.")

# Store in session state
if data is not None:
    st.session_state["data"] = data
    st.session_state["metric"] = metric
    st.session_state["covariate"] = covariate

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_design, tab_health, tab_analyse, tab_monitor = st.tabs(
    ["Design", "Health checks", "Analyse", "Monitor"]
)

# ===========================================================================
# TAB 1 — DESIGN  (no loaded data required)
# ===========================================================================
with tab_design:
    st.header("Sample-size & power calculator")
    st.markdown(
        "Compute the required sample size for a given MDE and display the "
        "power curve without needing any loaded experiment data."
    )

    col1, col2 = st.columns(2)
    with col1:
        d_baseline = st.number_input("Baseline rate", min_value=0.01, max_value=0.99, value=0.20, step=0.01, format="%.2f", key="d_base")
        d_mde = st.number_input("MDE (absolute)", min_value=0.001, max_value=0.5, value=0.02, step=0.005, format="%.3f", key="d_mde")
        d_alpha = st.number_input("Alpha (type-I error)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f", key="d_alpha")
    with col2:
        d_power = st.number_input("Target power (1-β)", min_value=0.50, max_value=0.99, value=0.80, step=0.05, format="%.2f", key="d_power")
        d_daily = st.number_input(
            "Daily traffic per arm (0 = skip duration)",
            min_value=0, max_value=1_000_000, value=0, step=100, key="d_daily"
        )

    daily_arg = int(d_daily) if d_daily > 0 else None
    res = required_sample_size(
        baseline_rate=float(d_baseline),
        mde_absolute=float(d_mde),
        alpha=float(d_alpha),
        power=float(d_power),
        daily_traffic_per_arm=daily_arg,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Per-arm sample size", f"{res.sample_size_per_arm:,}")
    m2.metric("Total sample size", f"{res.total_sample_size:,}")
    if res.duration_days is not None:
        m3.metric("Estimated duration (days)", f"{res.duration_days:.1f}")
    else:
        m3.metric("Estimated duration (days)", "—")

    # Power curve
    ns = np.linspace(500, res.sample_size_per_arm * 2, 30).astype(int)
    powers = [
        power_for_sample_size(
            baseline_rate=float(d_baseline),
            mde_absolute=float(d_mde),
            n_per_arm=int(n),
            alpha=float(d_alpha),
        )
        for n in ns
    ]

    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=ns.tolist(), y=powers, mode="lines+markers", name="Power"))
    fig_power.add_hline(y=float(d_power), line_dash="dash", line_color="red",
                        annotation_text=f"Target {d_power:.0%}")
    fig_power.add_vline(x=res.sample_size_per_arm, line_dash="dot", line_color="grey",
                        annotation_text=f"n={res.sample_size_per_arm:,}")
    fig_power.update_layout(
        title="Power vs. sample size per arm",
        xaxis_title="Sample size per arm",
        yaxis_title="Power",
        yaxis={"tickformat": ".0%"},
        height=400,
    )
    st.plotly_chart(fig_power, use_container_width=True)

# ===========================================================================
# TAB 2 — HEALTH CHECKS
# ===========================================================================
with tab_health:
    st.header("Health checks")

    if "data" not in st.session_state:
        st.info("Load or simulate data in the sidebar first.")
    else:
        _data: ExperimentData = st.session_state["data"]

        # SRM check
        st.subheader("Sample Ratio Mismatch (SRM)")
        srm = srm_check(_data)
        if srm.passed:
            st.success(f"SRM check PASSED — {srm.detail}  (p = {srm.p_value:.4f})")
        else:
            st.error(f"SRM check FAILED — {srm.detail}  (p = {srm.p_value:.4f})")

        st.markdown(
            f"**Statistic:** {srm.statistic:.4f} &nbsp;|&nbsp; "
            f"**p-value:** {srm.p_value:.4f} &nbsp;|&nbsp; "
            f"**Passed:** {'Yes' if srm.passed else 'No'}"
        )

        # A/A calibration
        st.subheader("A/A test calibration")
        st.markdown(
            "Runs many A/A simulations at the control rate to estimate the "
            "empirical false-positive rate. Should be close to α = 0.05."
        )
        if st.button("Run A/A calibration (300 simulations)"):
            _metric = st.session_state["metric"]
            ctrl_mean = float(_data.control[_metric].mean())
            with st.spinner("Running A/A simulations…"):
                fpr = aa_test(
                    base_rate=ctrl_mean,
                    n_per_arm=2000,
                    n_simulations=300,
                    alpha=0.05,
                    seed=0,
                )
            c1, c2 = st.columns(2)
            c1.metric("Empirical FPR", f"{fpr:.3f}")
            c2.metric("Nominal α", "0.050")

            fig_aa = go.Figure()
            fig_aa.add_trace(go.Bar(x=["Empirical FPR", "Nominal α"], y=[fpr, 0.05],
                                    marker_color=["steelblue", "salmon"]))
            fig_aa.update_layout(
                title="A/A calibration: empirical FPR vs nominal α",
                yaxis_title="False-positive rate",
                yaxis={"range": [0, max(0.15, fpr * 1.5)]},
                height=350,
            )
            st.plotly_chart(fig_aa, use_container_width=True)

# ===========================================================================
# TAB 3 — ANALYSE
# ===========================================================================
with tab_analyse:
    st.header("Experiment analysis")

    if "data" not in st.session_state:
        st.info("Load or simulate data in the sidebar first.")
    else:
        _data = st.session_state["data"]
        _metric = st.session_state["metric"]
        _covariate = st.session_state["covariate"]

        # Determine test type: use two_proportion_z for binary, welch_t otherwise
        unique_vals = _data.df[_metric].dropna().unique()
        is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})

        if is_binary:
            freq_result = two_proportion_z(_data, _metric)
            test_name = "Two-proportion z-test"
        else:
            freq_result = welch_t(_data, _metric)
            test_name = "Welch's t-test"

        # ----------------------------------------------------------------
        # Frequentist results
        # ----------------------------------------------------------------
        st.subheader(f"Frequentist analysis — {test_name}")

        fa, fb, fc, fd = st.columns(4)
        fa.metric("Control mean", f"{freq_result.control_mean:.4f}")
        fb.metric("Treatment mean", f"{freq_result.treatment_mean:.4f}")
        fc.metric("Absolute effect", f"{freq_result.absolute_effect:+.4f}")
        fd.metric("Relative effect", f"{freq_result.relative_effect:+.2%}")

        ga, gb, gc = st.columns(3)
        ga.metric("95% CI", f"[{freq_result.ci_low:+.4f}, {freq_result.ci_high:+.4f}]")
        gb.metric("p-value", f"{freq_result.p_value:.4f}")
        gc.metric("Significant", "Yes" if freq_result.significant else "No")

        st.markdown(f"**Verdict:** {freq_result.verdict}")

        # CI plot (horizontal)
        fig_ci = go.Figure()
        fig_ci.add_trace(
            go.Scatter(
                x=[freq_result.absolute_effect],
                y=["Unadjusted"],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[freq_result.ci_high - freq_result.absolute_effect],
                    arrayminus=[freq_result.absolute_effect - freq_result.ci_low],
                ),
                mode="markers",
                marker=dict(size=12, color="steelblue"),
                name="Unadjusted",
            )
        )
        fig_ci.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No effect")
        fig_ci.update_layout(
            title="Confidence interval for absolute effect",
            xaxis_title="Absolute effect",
            height=250,
        )
        st.plotly_chart(fig_ci, use_container_width=True)

        # ----------------------------------------------------------------
        # Bayesian analysis
        # ----------------------------------------------------------------
        st.subheader("Bayesian analysis — Beta-Binomial posterior")
        if not is_binary:
            st.warning(
                "Beta-Binomial is designed for binary metrics. "
                "Results may be unreliable for continuous data."
            )

        bayes_result = beta_binomial(_data, _metric, seed=0)

        ba, bb, bc = st.columns(3)
        ba.metric("P(treatment > control)", f"{bayes_result.prob_treatment_better:.3f}")
        bb.metric("Expected loss (treatment)", f"{bayes_result.expected_loss_treatment:.5f}")
        bc.metric("Expected loss (control)", f"{bayes_result.expected_loss_control:.5f}")

        bd, be = st.columns(2)
        bd.metric("95% Credible interval", f"[{bayes_result.cred_low:+.4f}, {bayes_result.cred_high:+.4f}]")
        be.markdown(f"**Verdict:** {bayes_result.verdict}")

        # Probability gauge
        prob = bayes_result.prob_treatment_better
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "P(treatment > control) %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "steelblue"},
                    "steps": [
                        {"range": [0, 50], "color": "salmon"},
                        {"range": [50, 95], "color": "lightyellow"},
                        {"range": [95, 100], "color": "lightgreen"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "value": 95},
                },
                number={"suffix": "%", "valueformat": ".1f"},
            )
        )
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ----------------------------------------------------------------
        # CUPED (if covariate is available)
        # ----------------------------------------------------------------
        if _covariate:
            st.subheader("CUPED — variance reduction")
            cuped_result = apply_cuped(_data, _metric, covariate=_covariate)

            ca, cb, cc = st.columns(3)
            ca.metric("Variance reduction", f"{cuped_result.variance_reduction:.1%}")
            cb.metric("Adjusted absolute effect", f"{cuped_result.adjusted_absolute_effect:+.4f}")
            cc.metric(
                "Adjusted 95% CI",
                f"[{cuped_result.adjusted_ci_low:+.4f}, {cuped_result.adjusted_ci_high:+.4f}]",
            )

            # Overlay unadjusted vs adjusted CI
            fig_cuped = go.Figure()
            for label, eff, lo, hi, color in [
                ("Unadjusted", freq_result.absolute_effect, freq_result.ci_low, freq_result.ci_high, "steelblue"),
                ("CUPED-adjusted", cuped_result.adjusted_absolute_effect, cuped_result.adjusted_ci_low, cuped_result.adjusted_ci_high, "darkorange"),
            ]:
                fig_cuped.add_trace(
                    go.Scatter(
                        x=[eff],
                        y=[label],
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=[hi - eff],
                            arrayminus=[eff - lo],
                        ),
                        mode="markers",
                        marker=dict(size=12, color=color),
                        name=label,
                    )
                )
            fig_cuped.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No effect")
            fig_cuped.update_layout(
                title="Unadjusted vs CUPED-adjusted confidence intervals",
                xaxis_title="Absolute effect",
                height=280,
            )
            st.plotly_chart(fig_cuped, use_container_width=True)

        # ----------------------------------------------------------------
        # Decision
        # ----------------------------------------------------------------
        st.subheader("Decision")
        dec_col1, dec_col2 = st.columns(2)
        with dec_col1:
            mde_input = st.number_input(
                "MDE for practical significance (absolute)",
                min_value=0.001, max_value=0.5, value=0.02, step=0.005, format="%.3f",
                key="dec_mde",
            )
        with dec_col2:
            guardrails_ok = st.checkbox("Guardrails OK", value=True, key="dec_guardrails")

        decision = decide(freq_result, mde_absolute=float(mde_input), guardrails_ok=guardrails_ok)

        if decision.recommendation == "ship":
            banner_color = "#d4edda"
            banner_text_color = "#155724"
            banner_icon = "SHIP"
        elif decision.recommendation == "no_ship":
            banner_color = "#f8d7da"
            banner_text_color = "#721c24"
            banner_icon = "NO SHIP"
        else:
            banner_color = "#e2e3e5"
            banner_text_color = "#383d41"
            banner_icon = "INCONCLUSIVE"

        st.markdown(
            f"""
            <div style="background:{banner_color};color:{banner_text_color};
                        padding:20px;border-radius:8px;margin-top:10px;">
                <h2 style="margin:0;">{banner_icon}</h2>
                <p style="margin:8px 0 0 0;">{decision.rationale}</p>
                <ul style="margin:8px 0 0 0;">
                    <li>Statistically significant: {'Yes' if decision.statistically_significant else 'No'}</li>
                    <li>Practically significant (|effect| ≥ MDE): {'Yes' if decision.practically_significant else 'No'}</li>
                    <li>Guardrails OK: {'Yes' if decision.guardrails_ok else 'No'}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 4 — MONITOR (sequential / peeking)
# ===========================================================================
with tab_monitor:
    st.header("Sequential monitoring")
    st.markdown(
        "Simulate the mSPRT sequential likelihood ratio test and compare it "
        "with the inflated false-positive rate from naive peeking."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        mon_base = st.number_input("Base rate", min_value=0.01, max_value=0.99, value=0.20, step=0.01, format="%.2f", key="mon_base")
        mon_lift = st.number_input("Lift (absolute)", min_value=0.0, max_value=0.5, value=0.02, step=0.005, format="%.3f", key="mon_lift")
    with col_b:
        mon_nmax = st.number_input("n_max (per arm)", min_value=1000, max_value=200_000, value=20_000, step=1000, key="mon_nmax")
        mon_tau = st.number_input("tau (mSPRT mixing variance)", min_value=0.001, max_value=0.5, value=0.05, step=0.005, format="%.3f", key="mon_tau")
    with col_c:
        mon_alpha = st.number_input("Alpha", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f", key="mon_alpha")
        mon_seed = st.number_input("Seed", min_value=0, max_value=99999, value=42, step=1, key="mon_seed")

    if st.button("Run sequential analysis"):
        with st.spinner("Running mSPRT and naive-peeking simulation…"):
            seq = msprt_stream(
                base_rate=float(mon_base),
                lift=float(mon_lift),
                n_max=int(mon_nmax),
                alpha=float(mon_alpha),
                tau=float(mon_tau),
                seed=int(mon_seed),
            )
            naive_fpr = naive_peeking_fpr(
                base_rate=float(mon_base),
                n_max=int(mon_nmax),
                look_every=500,
                n_sims=100,
                alpha=float(mon_alpha),
                seed=int(mon_seed),
            )

        # mSPRT summary
        st.subheader("mSPRT result")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Crossed threshold?", "Yes" if seq.crossed else "No")
        s2.metric("Stop index", str(seq.stop_index) if seq.stop_index is not None else "—")
        s3.metric("Final statistic", f"{seq.final_statistic:.3f}")
        s4.metric("Threshold (1/α)", f"{seq.threshold:.1f}")

        # Reconstruct the mSPRT likelihood-ratio path for plotting
        # (uses the same math as the library — no extra statistics logic, just array ops)
        _rng = np.random.default_rng(int(mon_seed))
        _c = _rng.binomial(1, float(mon_base), int(mon_nmax)).astype(float)
        _t = _rng.binomial(1, float(mon_base) + float(mon_lift), int(mon_nmax)).astype(float)
        _ns = np.arange(1, int(mon_nmax) + 1)
        _d = np.cumsum(_t) / _ns - np.cumsum(_c) / _ns
        _var = float(mon_base) * (1 - float(mon_base)) * 2
        _tau = float(mon_tau)
        _lr = np.sqrt(_var / (_var + _ns * _tau**2)) * np.exp(
            (_ns**2 * _tau**2 * _d**2) / (2 * _var * (_var + _ns * _tau**2))
        )

        fig_seq = go.Figure()
        fig_seq.add_trace(
            go.Scatter(x=_ns.tolist(), y=_lr.tolist(), mode="lines",
                       name="Likelihood ratio", line=dict(color="steelblue"))
        )
        fig_seq.add_hline(y=seq.threshold, line_dash="dash", line_color="red",
                          annotation_text=f"Threshold = {seq.threshold:.1f}")
        if seq.stop_index is not None:
            fig_seq.add_vline(x=seq.stop_index, line_dash="dot", line_color="green",
                              annotation_text=f"Stopped at n={seq.stop_index:,}")
        fig_seq.update_layout(
            title="mSPRT likelihood-ratio over time",
            xaxis_title="Observations (per arm)",
            yaxis_title="Likelihood ratio",
            yaxis_type="log",
            height=400,
        )
        st.plotly_chart(fig_seq, use_container_width=True)

        # Peeking comparison
        st.subheader("Naive peeking vs mSPRT: false-positive rate comparison")
        n1, n2 = st.columns(2)
        n1.metric(
            "Naive peeking FPR",
            f"{naive_fpr:.3f}",
            delta=f"{naive_fpr - float(mon_alpha):+.3f} vs α",
            delta_color="inverse",
        )
        n2.metric("mSPRT guarantee", f"≤ {float(mon_alpha):.3f}")

        st.markdown(
            f"""
            **Why naive peeking inflates FPR:** repeatedly testing at fixed α = {mon_alpha:.2f}
            each time you look accumulates type-I error. With {int(mon_nmax)//500} looks, the
            actual FPR is approximately **{naive_fpr:.1%}** — far above the nominal {float(mon_alpha):.0%}.
            The mSPRT controls FPR at any stopping time by construction, so you can
            peek continuously without inflating the false-positive rate.
            """
        )
