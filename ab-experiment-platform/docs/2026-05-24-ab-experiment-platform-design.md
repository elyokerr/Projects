# A/B Experiment Platform — Online Experimentation & Causal Analysis Toolkit

**Design document · 2026-05-24**

## 1. Project overview

A reusable online-experimentation toolkit that covers the full A/B test lifecycle: designing an experiment (required sample size and duration), validating its health (sample-ratio mismatch and A/A checks), analysing it with both frequentist and Bayesian methods, reducing variance with CUPED, and turning the result into a defensible ship / no-ship decision. It additionally supports continuous monitoring through always-valid sequential testing, so an experiment can be stopped early without inflating the false-positive rate.

The toolkit is library-first: all statistics live in a pure, framework-agnostic Python package (`src/abtest/`) built around a small set of explicit data and result objects. A Streamlit application and a set of notebooks are thin clients over that package. A built-in data simulator generates experiments with a known ground-truth effect, which powers both the demonstrations and the correctness tests.

## 2. Problem statement

Controlled experiments are the standard way product teams decide whether a change works, but they are routinely run incorrectly. Common failures are: tests that are underpowered (too few units to detect the effect that matters); "peeking" at results and stopping as soon as significance appears (which inflates the false-positive rate far above the nominal 5%); ignoring sample-ratio mismatch (an unequal split that silently invalidates the comparison); and conflating statistical significance with practical significance.

The platform encodes the correct workflow end to end. It tells the analyst how large and how long the experiment must be *before* it runs, checks the experiment's integrity, analyses the outcome with calibrated methods, and combines statistical evidence, the pre-registered minimum detectable effect, and guardrail metrics into a single recommendation. The always-valid sequential module makes continuous monitoring statistically safe rather than a source of false positives.

## 3. Users

| User | How they consume the project |
|---|---|
| Product / growth analyst | Uses the Streamlit app to design an experiment, upload its data, run health checks and analysis, and read the ship / no-ship recommendation — no statistics expertise required. |
| Data scientist | Imports `src/abtest/` as a tested Python library and calls the pure analysis functions directly inside their own pipeline or notebook. |
| Engineer reviewing the repository | Reads the module boundaries, the data/result contracts, and the simulation-based validation tests that assert the methods are correct. |

## 4. Datasets

| Source | Contents | Role | Volume |
|---|---|---|---|
| Cookie Cats mobile-game A/B test | Real experiment moving a progression gate from level 30 to level 40, with per-player 1-day and 7-day retention (binary) outcomes | Headline real-data case study | ~90,000 players |
| Built-in simulator (`simulate.py`) | Synthetic experiments with configurable base rate, true treatment effect, effect heterogeneity, and pre-period covariate correlation | Ground-truth validation, CUPED and sequential demonstrations | Configurable |

The Cookie Cats dataset is a public, well-documented A/B test distributed as a single CSV; its published finding is that the 7-day retention difference between the two gate positions is not statistically significant. That honest, non-significant result anchors a disciplined ship / no-ship narrative. The simulator is framed around a UK e-commerce checkout experiment (conversion rate with a pre-period spend covariate) and provides the known-effect data that real datasets cannot, enabling direct verification of estimator correctness, variance reduction, and false-positive control.

`data/raw/` holds the Cookie Cats CSV (gitignored); `data/README.md` documents its source and licence. Simulated data is generated on demand and never committed.

## 5. Tech stack

| Layer | Tool | Justification |
|---|---|---|
| Core numerics | NumPy, SciPy | Vectorised statistics, distributions, and hypothesis tests. |
| Statistical models | statsmodels | Power and proportion utilities, multiple-comparisons correction. |
| Data handling | pandas | Tidy per-unit frames and the `ExperimentData` contract. |
| Application | Streamlit | Pure-Python interactive UI consistent with the rest of the portfolio; deploys free. |
| Visualisation | Plotly | Interactive confidence intervals, posterior distributions, and sequential-boundary plots. |
| Testing | pytest | Unit, property/simulation, and app-smoke layers. |
| Lint | ruff | Python style consistency. |
| Packaging / parity | Docker | Reproducible local run; host-independent deployment. |
| CI | GitHub Actions | Free for public repositories; runs lint and the full test suite on every push. |
| Hosting | Streamlit Community Cloud | Free, purpose-built for Streamlit; deploys directly from the repository. |

The entire stack is free, CPU-only, and runs on a basic laptop; no GPU or Colab is required.

## 6. Architecture

All statistical logic is implemented as pure functions that accept an `ExperimentData` object and return a typed result object. The application and notebooks construct `ExperimentData`, call these functions, and render the returned results — they hold no statistical logic themselves.

```
                ┌─────────────────────────────────────────┐
                │            src/abtest/ (library)          │
                │                                           │
   ExperimentData ─▶ design.py     ─▶ PowerResult           │
                │   health.py      ─▶ HealthCheckResult      │
                │   frequentist.py ─▶ FrequentistResult      │
                │   bayesian.py    ─▶ BayesianResult         │
                │   cuped.py       ─▶ CupedResult            │
                │   sequential.py  ─▶ SequentialResult       │
                │   decision.py    ─▶ Decision               │
                │   simulate.py    ─▶ ExperimentData          │
                └─────────────────────────────────────────┘
                        ▲                       ▲
                        │                       │
              ┌─────────┴────────┐     ┌────────┴─────────┐
              │  Streamlit app   │     │    notebooks     │
              │  (4 tabs, thin)  │     │  (thin clients)  │
              └──────────────────┘     └──────────────────┘
```

**Data contract.** `ExperimentData` wraps a tidy per-unit frame with: `unit_id`, `variant` (control / treatment), one or more metric columns (binary or continuous), an optional `pre_period_covariate` (for CUPED), and an optional `assignment_timestamp` (for sequential analysis). Construction validates the schema, the variant labels, and missingness.

**Result objects.** Each analysis returns a small dataclass carrying the effect estimate, an interval, the relevant test statistic or posterior summary, and a plain-English verdict, so every consumer renders results uniformly.

**Module responsibilities.**

- `data.py` — the `ExperimentData` contract, CSV loaders, and input validation.
- `design.py` — statistical power, minimum detectable effect, required sample size, and experiment-duration estimation.
- `health.py` — sample-ratio-mismatch test (chi-square) and an A/A test runner.
- `frequentist.py` — two-proportion z-test, Welch's t-test, Mann-Whitney U, confidence intervals, and multiple-comparisons correction (Bonferroni and Benjamini-Hochberg).
- `bayesian.py` — Beta-Binomial conjugate model for conversion metrics: P(treatment > control), expected loss, and credible intervals.
- `cuped.py` — CUPED variance reduction using the pre-period covariate.
- `sequential.py` — always-valid inference via mixture SPRT and group-sequential (O'Brien-Fleming) boundaries for valid early stopping.
- `decision.py` — combines statistical significance, practical significance against the minimum detectable effect, and guardrail-metric checks into a `ship` / `no-ship` / `inconclusive` recommendation.
- `simulate.py` — ground-truth experiment generator parametrised by base rate, true effect, heterogeneity, and pre-period correlation.

**Application.** The Streamlit app has four tabs mirroring the lifecycle — **Design** (calculators), **Health** (sample-ratio mismatch and A/A on uploaded data), **Analyse** (frequentist, Bayesian, and CUPED side by side), and **Monitor** (sequential testing). It validates any uploaded CSV against the `ExperimentData` schema before analysis.

## 7. Repository structure

```
ab-experiment-platform/
├── README.md                    ← 9-section overview
├── requirements.txt             ← Python 3.11 pinned dependencies
├── .gitignore
├── Dockerfile
├── src/
│   └── abtest/
│       ├── __init__.py
│       ├── data.py
│       ├── design.py
│       ├── health.py
│       ├── frequentist.py
│       ├── bayesian.py
│       ├── cuped.py
│       ├── sequential.py
│       ├── decision.py
│       └── simulate.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   ├── 01_eda_cookie_cats.ipynb
│   ├── 02_design_power.ipynb
│   ├── 03_frequentist_bayesian.ipynb
│   ├── 04_cuped_variance_reduction.ipynb
│   ├── 05_sequential_testing.ipynb
│   └── 06_simulation_validation.ipynb
├── data/
│   ├── raw/                     ← Cookie Cats CSV (gitignored)
│   └── README.md
├── reports/
│   └── figures/
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_design.py
│   ├── test_health.py
│   ├── test_frequentist.py
│   ├── test_bayesian.py
│   ├── test_cuped.py
│   ├── test_sequential.py
│   ├── test_decision.py
│   └── test_simulate.py
├── docs/
│   ├── 2026-05-24-ab-experiment-platform-design.md
│   └── methodology.md
└── .github/
    └── workflows/
        └── ci.yml
```

## 8. Statistical methods

**Design.** Given a baseline metric, a minimum detectable effect, a significance level, and target power, `design.py` returns the required per-variant sample size and, given an expected traffic rate, an experiment duration. The same routines produce power curves.

**Health checks.** Sample-ratio mismatch is tested with a chi-square goodness-of-fit test against the intended allocation; a detected mismatch flags the experiment's results as untrustworthy. The A/A runner checks that, with no treatment, the false-positive rate matches the nominal level.

**Frequentist analysis.** Binary metrics use a two-proportion z-test; continuous metrics use Welch's t-test, with Mann-Whitney U available for non-normal data. All report effect size and a confidence interval. When several metrics are tested together, p-values are corrected with Bonferroni (family-wise) or Benjamini-Hochberg (false-discovery-rate) procedures.

**Bayesian analysis.** For conversion metrics, a Beta-Binomial conjugate model yields the posterior probability that treatment beats control, the expected loss of each decision, and credible intervals — a decision-oriented complement to the frequentist test.

**Variance reduction.** CUPED uses a pre-experiment covariate correlated with the outcome to remove pre-existing between-unit variance, tightening the confidence interval without biasing the effect estimate.

**Sequential testing.** Mixture SPRT and group-sequential (O'Brien-Fleming) boundaries provide always-valid inference: significance can be evaluated continuously and the experiment stopped as soon as a boundary is crossed, while the overall false-positive rate stays at or below the nominal level.

**Decision.** `decision.py` combines three inputs — statistical significance, practical significance (whether the estimated effect exceeds the pre-registered minimum detectable effect), and guardrail-metric checks — into a single `ship` / `no-ship` / `inconclusive` recommendation with the supporting evidence attached.

## 9. Validation methodology

Because the simulator produces data with a known true effect, the platform's own correctness is directly testable rather than assumed.

1. **Estimator recovery.** For data generated with a known effect δ, each estimator's interval covers δ and the point estimate falls within tolerance, across a grid of base rates and sample sizes.
2. **False-positive control under the null.** With δ = 0 over many seeds, the frequentist test rejects at approximately the significance level α, confirming calibration.
3. **Power.** At the designed minimum detectable effect, the empirical rejection rate matches the power predicted by `design.py`.
4. **Variance reduction.** With a correlated pre-period covariate, CUPED produces a strictly smaller variance than the naïve estimator while recovering the same effect.
5. **Sequential validity.** Under continuous monitoring with peeking at δ = 0, the always-valid sequential test holds the false-positive rate at or below α, whereas a naïve repeatedly-evaluated t-test demonstrably exceeds it. This contrast is the headline result.
6. **Real-data cross-check.** The platform reproduces the published Cookie Cats finding: the 7-day retention difference between gate positions is not statistically significant, with a bootstrap confidence interval consistent with the published analysis.

Headline metrics reported in the README: the naïve-peeking false-positive rate versus the sequential-test false-positive rate; the CUPED variance-reduction percentage on the simulated covariate; estimator interval coverage; and the reproduced Cookie Cats effect with its confidence interval.

## 10. Error handling

Validation occurs at the system boundary — data ingestion and the application — while internal calls between the library's own pure functions trust their inputs.

- **Data ingestion.** Unknown variant labels, empty variant groups, all-missing metric columns, and single-variant data are rejected with clear messages rather than raising raw exceptions.
- **Statistical guardrails.** Tiny samples, zero-variance metrics, a detected sample-ratio mismatch, and a CUPED request with no available covariate produce explicit warnings (and, for CUPED, a documented fallback to the naïve estimator) instead of silent or misleading output.
- **Application.** Uploaded CSVs are validated against the `ExperimentData` schema before any analysis runs; schema violations surface as actionable in-UI errors.

## 11. Testing strategy

Testing uses pytest with one test file per module and three layers:

- **Unit tests** check closed-form cases with known answers (for example, the two-proportion z-test reproduces a hand-computed statistic).
- **Property / simulation tests** implement the six validation checks in Section 9, using fixed random seeds so results are deterministic and fast enough to run in CI.
- **Application smoke tests** confirm the Streamlit thin-client functions return without error on a fixture dataset.

`tests/conftest.py` prepends the project root to `sys.path` so tests run from any working directory. The suite must pass and ruff must report no findings before merge; CI enforces both on every push.

## 12. Deployment

- **Streamlit Community Cloud** hosts the public application, deployed directly from the repository; the README carries the live URL and screenshots.
- **Docker** provides local parity and host independence: `docker build` then `docker run -p 8501:8501` serves the same application.
- **GitHub Actions** runs ruff and the full pytest suite — including the simulation-based validation tests — on every push, so statistical correctness is continuously enforced.

## 13. Scaling path

- **Larger experiments.** The analysis functions are vectorised and additionally accept pre-aggregated sufficient statistics (counts and sums) in place of per-unit frames, so memory stays flat as unit counts grow.
- **Many concurrent experiments.** The library is stateless and drops behind a FastAPI service or a batch job without modification; an experiment-metadata store (DuckDB or Postgres) would hold definitions and results. This is out of scope for the current project.
- **More metric types.** The result-object contract means adding ratio metrics (via the delta method) or count metrics is a new module rather than a change to existing ones.

## 14. Definition of Done

1. `src/abtest/` is complete — `data`, `design`, `health`, `frequentist`, `bayesian`, `cuped`, `sequential`, `decision`, and `simulate` — with pure functions and type hints throughout.
2. All three test layers pass; the six validation checks are green in CI; ruff reports no findings.
3. The six notebooks run top to bottom on the registered project kernel.
4. The Streamlit app works across all four tabs on both the Cookie Cats data and a simulated upload, and is deployed to Streamlit Community Cloud with a working public link.
5. The 9-section README has real headline metrics filled in with no placeholders; `docs/methodology.md` explains each method and its assumptions; `data/README.md` documents the Cookie Cats source.
6. The Dockerfile builds and runs; CI is green on `main`.
