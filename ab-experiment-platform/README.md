# A/B Experiment Platform

**Design, validate, and analyse online experiments the right way.**

A Python library, Streamlit application, and validated simulation suite covering the full
experiment lifecycle: power design → pre-flight health checks → frequentist and Bayesian analysis
→ CUPED variance reduction → sequential (always-valid) testing → decision logic.

---

## Hero results

Correctness is verified by simulation against known ground truth (see `notebooks/06_simulation_validation.ipynb`).

| Metric | Result |
|---|---|
| Sequential monitoring false-positive rate - naive peeking | **~31%** at nominal 5% |
| Sequential monitoring false-positive rate - mSPRT | **~4%** at nominal 5% |
| CUPED variance reduction (correlated pre-period covariate) | **~10%** |
| Frequentist CI coverage (nominal 95%, 500 simulated experiments) | **~96%** |
| Power calibration: predicted vs empirical at n=2941/arm | **80.0% predicted vs ~79% empirical** |

---

## The business problem

Most A/B tests fail silently:

- **Underpowered** - launch decisions made on experiments with 20–40% power.
- **Peeking** - analysts check p-values daily and stop early, inflating false-positive rates from
  5% to 30%+.
- **Sample-ratio mismatch** - undetected assignment bugs bias every downstream metric.
- **Statistical vs practical significance** - a p<0.05 result on a 0.01 pp lift is not a ship decision.

This platform encodes the correct lifecycle so that each of these failure modes has an explicit,
tested mitigation.

---

## What this demonstrates

| Module | Responsibility |
|---|---|
| `src/abtest/design.py` | Two-proportion power curves; required sample size via Cohen's h (statsmodels `NormalIndPower`) |
| `src/abtest/health.py` | Sample-ratio mismatch (chi-square GoF); A/A test |
| `src/abtest/frequentist.py` | Two-proportion z-test, Welch t-test, Mann-Whitney U; Bonferroni/BH correction |
| `src/abtest/bayesian.py` | Beta-Binomial conjugate posterior; P(treatment better) + expected loss |
| `src/abtest/cuped.py` | CUPED covariate adjustment; variance-reduction measurement |
| `src/abtest/sequential.py` | mSPRT (mixture likelihood ratio, always-valid); O'Brien-Fleming alpha-spending |
| `src/abtest/decision.py` | Structured ship/no-ship/inconclusive recommendation with guardrails |
| `src/abtest/simulate.py` | Ground-truth data generator for simulation-based correctness validation |

The simulation suite in `notebooks/06_simulation_validation.ipynb` runs 500 experiments at the
designed operating point and checks CI coverage, null FPR, and power calibration against
analytically predicted values.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app/streamlit_app.py

# Run the unit test suite (34 tests)
pytest tests -q

# Run the end-to-end smoke test (slower - ~5 s)
RUN_SLOW=1 pytest tests/test_end_to_end.py -q
```

Docker (no local Python required):

```bash
docker build -t ab-experiment-platform .
docker run -p 8501:8501 ab-experiment-platform
# then open http://localhost:8501
```

---

## Project structure

```
ab-experiment-platform/
├── src/abtest/               # Core library
│   ├── data.py               # ExperimentData dataclass (the shared contract)
│   ├── design.py             # Power analysis and sample-size calculation
│   ├── health.py             # SRM and A/A checks
│   ├── frequentist.py        # z-test, Welch t, Mann-Whitney, correction
│   ├── bayesian.py           # Beta-Binomial conjugate analysis
│   ├── cuped.py              # CUPED variance reduction
│   ├── sequential.py         # mSPRT, O'Brien-Fleming
│   ├── decision.py           # Decision logic
│   ├── simulate.py           # Ground-truth simulator
│   └── results.py            # Result dataclasses
├── app/
│   └── streamlit_app.py      # Interactive Streamlit interface
├── notebooks/
│   ├── 01_eda_cookie_cats.ipynb           # Cookie Cats EDA + group sizes + SRM
│   ├── 02_design_power.ipynb              # Power curves, sample-size explorer
│   ├── 03_frequentist_bayesian.ipynb      # z-test + Beta-Binomial on 7-day retention
│   ├── 04_cuped_variance_reduction.ipynb  # CUPED variance reduction (simulated data)
│   ├── 05_sequential_testing.ipynb        # mSPRT vs naive peeking comparison
│   └── 06_simulation_validation.ipynb     # 500-experiment simulation-based validation
├── tests/                    # pytest suite (34 unit tests + 1 e2e smoke)
├── docs/
│   ├── methodology.md        # Method formulas and assumptions
│   └── 2026-05-24-ab-experiment-platform-design.md
├── reports/figures/          # Power curve and validation summary figures
├── data/
│   ├── README.md             # Cookie Cats dataset download instructions
│   └── raw/                  # gitignored - place cookie_cats.csv here
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci-ab-experiment-platform.yml
```

---

## Methodology

The library follows a strict experiment lifecycle enforced by the `ExperimentData` dataclass, which
acts as the shared contract between all modules.

1. **Design** - `design.py` computes the minimum detectable effect and required sample size using
   a two-proportion z-test power formula via statsmodels' `NormalIndPower.solve_power`. Effect size
   is expressed as Cohen's h (`proportion_effectsize`).

2. **Health checks** - before any analysis, `health.py` runs a chi-square goodness-of-fit test
   to detect sample-ratio mismatch (flagged at p<0.001) and an optional A/A calibration.

3. **Frequentist analysis** - `frequentist.py` exposes three tests (two-proportion z, Welch t,
   Mann-Whitney U) for conversion, continuous, and non-parametric cases respectively.
   Multiple-comparison correction uses Bonferroni or Benjamini-Hochberg via statsmodels.

4. **Bayesian analysis** - `bayesian.py` models conversions with a Beta-Binomial conjugate
   posterior. Outputs: P(treatment better) and expected loss under each decision.

5. **CUPED** - `cuped.py` applies the Deng et al. covariate adjustment. The adjustment
   coefficient theta is estimated from the experiment data; adjusted outcomes are unbiased.
   Variance reduction is measured relative to the unadjusted estimator.

6. **Sequential testing** - `sequential.py` implements the mixture Sequential Probability Ratio
   Test (mSPRT), which provides an always-valid p-value: the probability of ever exceeding the
   threshold under H0 is bounded by alpha regardless of when you look. O'Brien-Fleming
   alpha-spending boundaries are also implemented for group-sequential designs.

7. **Decision** - `decision.py` combines statistical significance, practical significance (MDE
   threshold), and guardrail checks into a structured `Decision` with a `recommendation` field
   (`ship`, `no_ship`, or `inconclusive`).

8. **Simulator** - `simulate.py` generates synthetic experiments with known ground truth
   (specified lift, baseline rate, and covariate correlation), enabling simulation-based
   correctness validation of every module.

See [docs/methodology.md](docs/methodology.md) for formulas and assumptions.

---

## Tech stack

| Technology | Role |
|---|---|
| Python 3.11 | Runtime; stable ABI for scientific packages |
| NumPy | Vectorised simulation and array operations |
| SciPy | Statistical distributions, chi-square and z-test primitives |
| statsmodels | Power analysis (`NormalIndPower`), multiple-testing correction |
| pandas | Tabular data handling for `ExperimentData` |
| Streamlit | Reactive web UI without a backend server |
| Plotly | Interactive figures in the app and notebooks |
| matplotlib | Static publication-quality figures for reports |
| pytest | Unit and integration test runner |
| ruff | Fast Python linter (enforces consistent style) |
| Docker | Reproducible containerised deployment of the Streamlit app |
| GitHub Actions | CI: lint + test on every push and PR via `paths:` filter |
| Streamlit Community Cloud | Zero-cost public deployment target |

---

## Limitations and next steps

- **Cookie Cats case study** - any notebook analysing the Cookie Cats dataset requires downloading
  the CSV from Kaggle (see `data/README.md`). The download cannot be automated without a Kaggle
  API key.

- **CUPED on simulated data only** - the Cookie Cats dataset has no pre-period covariate, so the
  CUPED notebook uses simulated data where the covariate correlation is known. Applying CUPED in
  practice requires a pre-experiment baseline metric for each unit.

- **Single-metric focus** - all tests operate on a single primary metric. Ratio metrics (e.g.
  revenue per user = total revenue / users, where numerator and denominator are both random)
  require the delta method; this is a documented future extension.

- **Heterogeneous treatment effects** - the library estimates average treatment effects only.
  Subgroup uplift modelling (X-learner, causal forests) is a natural next step documented in
  the design doc.

- **Fixed randomisation unit** - the library assumes user-level randomisation. Cluster
  randomisation (e.g. page-level or session-level) with inflated variance is not yet handled.
