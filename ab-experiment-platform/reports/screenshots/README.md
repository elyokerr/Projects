# App screenshots

Drop screenshots of the running Streamlit app here. Capture each at a sensible window width
(~1400px) so the charts are legible. Recommended filenames below match what the project README
can embed.

Launch the app first: `streamlit run app/streamlit_app.py`, sidebar → **Simulated data**
(`n_per_arm=5000, base_rate=0.20, absolute_lift=0.02, covariate_corr=0.5, seed=42`).

| # | Filename | Tab | What to show |
|---|---|---|---|
| 1 | `01_design_power.png` | Design | Sample-size metrics (per-arm N, total, duration) + the power-vs-sample-size curve. |
| 2 | `02_health_srm.png` | Health | The green SRM "pass" badge with its p-value, and the A/A calibration result (FPR ≈ 5%). |
| 3 | `03_analyse_decision.png` | Analyse | Frequentist effect + CI chart, the Bayesian P(treatment better), the CUPED variance-reduction line, and the **SHIP** decision banner. |
| 4 | `04_monitor_sequential.png` | Monitor | **The headline** - naive-peeking FPR (~31%) vs mSPRT (~4%), plus the mSPRT statistic crossing the 1/α threshold (set `lift=0.05` so it stops early). |

Optional:

| # | Filename | What to show |
|---|---|---|
| 5 | `05_landing.png` | The app landing view with the sidebar data-source controls (good as a hero image). |
| 6 | `06_analyse_bayesian.png` | A close-up of the Bayesian posterior / expected-loss panel, if you want it separate from shot 3. |
