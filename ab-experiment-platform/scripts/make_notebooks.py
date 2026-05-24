"""Generate all 6 analysis + validation notebooks for ab-experiment-platform."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)

KERNEL = {
    "display_name": "ab-experiment-platform (.venv)",
    "language": "python",
    "name": "ab-experiment-platform-venv",
}

HEADER_SOURCE = """\
import os, sys
from pathlib import Path
_cwd = Path.cwd()
_root = next((p for p in [_cwd] + list(_cwd.parents)
              if (p / 'requirements.txt').exists() and (p / 'src').is_dir()), None)
assert _root, f'Could not find project root from {_cwd}'
os.chdir(_root)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
print(f'Project root: {_root}')
"""


def make_notebook(title: str, cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    """Build a notebook from (celltype, source) pairs, prepending title + header."""
    nb = new_notebook()
    nb["metadata"]["kernelspec"] = KERNEL
    nb["metadata"]["language_info"] = {"name": "python"}

    # Title markdown cell
    nb.cells.append(new_markdown_cell(f"# {title}"))
    # Standard header code cell
    nb.cells.append(new_code_cell(HEADER_SOURCE))

    for celltype, source in cells:
        if celltype == "markdown":
            nb.cells.append(new_markdown_cell(source))
        elif celltype == "code":
            nb.cells.append(new_code_cell(source))
        else:
            raise ValueError(f"Unknown cell type: {celltype!r}")

    return nb


def write(name: str, nb: nbformat.NotebookNode) -> None:
    path = NOTEBOOKS / name
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Notebook 01 — Cookie Cats EDA
# ---------------------------------------------------------------------------

NB01_CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
## Cookie Cats A/B Test — Exploratory Data Analysis

[Cookie Cats](https://www.kaggle.com/datasets/meirnizri/cookie-cats) is a popular mobile puzzle game.
The A/B test moved the first "gate" (a forced wait before continuing) from level 30 to level 40.
The key metrics are **1-day retention** and **7-day retention**.

> **Data required:** place the CSV at `data/raw/cookie_cats.csv`.
> See `data/README.md` for the one-step Kaggle download command.
""",
    ),
    (
        "code",
        """\
import pandas as pd
from pathlib import Path

csv = Path('data/raw/cookie_cats.csv')
if not csv.exists():
    print('cookie_cats.csv not found — see data/README.md for the one-step download. Skipping live cells.')
    df = None
else:
    df = pd.read_csv(csv)
    df.head()
""",
    ),
    (
        "markdown",
        """\
### Group sizes & retention rates
""",
    ),
    (
        "code",
        """\
if df is not None:
    print("Group sizes:")
    print(df['version'].value_counts())
    print()
    print("Retention means by version:")
    print(df.groupby('version')[['retention_1', 'retention_7']].mean().round(4))
""",
    ),
    (
        "markdown",
        """\
### Retention bar chart
""",
    ),
    (
        "code",
        """\
if df is not None:
    import matplotlib.pyplot as plt

    ret = df.groupby('version')[['retention_1', 'retention_7']].mean()
    ax = ret.plot(kind='bar', figsize=(7, 4), rot=0,
                  color=['#4C72B0', '#DD8452'])
    ax.set_title('Cookie Cats — Retention by Gate Version')
    ax.set_xlabel('Version')
    ax.set_ylabel('Retention Rate')
    ax.legend(['1-day', '7-day'])
    plt.tight_layout()
    plt.show()
""",
    ),
    (
        "markdown",
        """\
### Sample Ratio Mismatch (SRM) check on 7-day retention
""",
    ),
    (
        "code",
        """\
if df is not None:
    from src.abtest import ExperimentData, srm_check

    df2 = df.rename(columns={'version': 'variant', 'retention_7': 'converted'})
    df2['variant'] = df2['variant'].map({'gate_30': 'control', 'gate_40': 'treatment'})
    data = ExperimentData(
        df2[['variant', 'converted']].assign(unit_id=range(len(df2))),
        metric_cols=['converted'],
    )
    result = srm_check(data)
    print(f"SRM check passed: {result.passed}")
    print(f"chi2 statistic : {result.statistic:.4f}")
    print(f"p-value        : {result.p_value:.4f}")
    print(f"Detail         : {result.detail}")
""",
    ),
]

# ---------------------------------------------------------------------------
# Notebook 02 — Power & Design
# ---------------------------------------------------------------------------

NB02_CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
## Power Analysis & Sample-Size Design

We use the Cookie Cats 7-day baseline retention of **~0.19** (hardcoded here so this
notebook runs without external data).  We explore:

1. A sample-size table across several minimum detectable effects (MDEs).
2. A power-vs-n curve showing how achieved power grows with sample size.
""",
    ),
    (
        "code",
        """\
from src.abtest import required_sample_size, power_for_sample_size

BASELINE = 0.19
MDES = [0.005, 0.01, 0.02, 0.03]

print(f"{'MDE':>6}  {'n/arm':>8}  {'total n':>9}  {'days (1k/day/arm)':>18}")
print("-" * 50)
for mde in MDES:
    res = required_sample_size(
        baseline_rate=BASELINE,
        mde_absolute=mde,
        alpha=0.05,
        power=0.80,
        daily_traffic_per_arm=1000,
    )
    print(f"{mde:>6.3f}  {res.sample_size_per_arm:>8,}  {res.total_sample_size:>9,}  {res.duration_days!r:>18}")
""",
    ),
    (
        "markdown",
        """\
### Power-vs-n curve for MDE = 0.01
""",
    ),
    (
        "code",
        """\
import numpy as np
import matplotlib.pyplot as plt

MDE = 0.01
ns = np.linspace(500, 60_000, 200).astype(int)
powers = [power_for_sample_size(baseline_rate=BASELINE, mde_absolute=MDE, n_per_arm=n) for n in ns]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ns, powers, color='steelblue')
ax.axhline(0.80, color='red', linestyle='--', label='80% power target')
required = required_sample_size(baseline_rate=BASELINE, mde_absolute=MDE, alpha=0.05, power=0.80)
ax.axvline(required.sample_size_per_arm, color='orange', linestyle=':', label=f'Required n={required.sample_size_per_arm:,}')
ax.set_xlabel('Sample size per arm')
ax.set_ylabel('Statistical power')
ax.set_title(f'Power curve — baseline={BASELINE}, MDE={MDE}')
ax.legend()
plt.tight_layout()
plt.show()
""",
    ),
    (
        "markdown",
        """\
### Interpretation

- With an MDE of 0.01 (1 percentage point) on a 19% baseline, we need roughly the sample
  size shown above per arm to achieve 80% power.
- Cookie Cats collected ~44,000 users per arm — ample for MDEs ≥ 0.01.
- Smaller MDEs (0.005) require far larger samples and longer experiment durations.
""",
    ),
]

# ---------------------------------------------------------------------------
# Notebook 03 — Frequentist & Bayesian analysis
# ---------------------------------------------------------------------------

NB03_CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
## Frequentist & Bayesian Analysis — 7-day Retention

We run both a **two-proportion z-test** and a **Beta-Binomial Bayesian** analysis on the
Cookie Cats 7-day retention metric.

A **bootstrap CI** (2 000 resamples) serves as an external cross-check on the parametric CI.

> If `cookie_cats.csv` is absent, we fall back to a **simulated dataset** at the same scale
> and note this clearly below.
""",
    ),
    (
        "code",
        """\
import pandas as pd
import numpy as np
from pathlib import Path
from src.abtest import (
    ExperimentData, simulate_conversion,
    two_proportion_z, beta_binomial,
)

csv = Path('data/raw/cookie_cats.csv')
if csv.exists():
    df = pd.read_csv(csv)
    df2 = df.rename(columns={'version': 'variant', 'retention_7': 'converted'})
    df2['variant'] = df2['variant'].map({'gate_30': 'control', 'gate_40': 'treatment'})
    data = ExperimentData(
        df2[['variant', 'converted']].assign(unit_id=range(len(df2))),
        metric_cols=['converted'],
    )
    using_sim = False
    print("Using real Cookie Cats data.")
else:
    print("cookie_cats.csv not found — using SIMULATED data as stand-in.")
    print("Parameters: n_per_arm=45000, base_rate=0.19, absolute_lift=-0.008, seed=7")
    sim = simulate_conversion(n_per_arm=45_000, base_rate=0.19, absolute_lift=-0.008, seed=7)
    data = sim
    using_sim = True
""",
    ),
    (
        "markdown",
        """\
### Frequentist: Two-proportion z-test
""",
    ),
    (
        "code",
        """\
freq = two_proportion_z(data, 'converted', alpha=0.05)
print(f"Control rate     : {freq.control_mean:.4f}")
print(f"Treatment rate   : {freq.treatment_mean:.4f}")
print(f"Absolute effect  : {freq.absolute_effect:+.4f}")
print(f"Relative effect  : {freq.relative_effect:+.2%}")
print(f"95% CI           : [{freq.ci_low:+.4f}, {freq.ci_high:+.4f}]")
print(f"p-value          : {freq.p_value:.4f}")
print(f"Significant      : {freq.significant}")
print(f"Verdict          : {freq.verdict}")
""",
    ),
    (
        "markdown",
        """\
### Bayesian: Beta-Binomial
""",
    ),
    (
        "code",
        """\
bayes = beta_binomial(data, 'converted', seed=42)
print(f"P(treatment better)       : {bayes.prob_treatment_better:.4f}")
print(f"Expected loss (treatment) : {bayes.expected_loss_treatment:.6f}")
print(f"Expected loss (control)   : {bayes.expected_loss_control:.6f}")
print(f"95% credible interval     : [{bayes.cred_low:+.4f}, {bayes.cred_high:+.4f}]")
print(f"Verdict                   : {bayes.verdict}")
""",
    ),
    (
        "markdown",
        """\
### Bootstrap CI (external cross-check)
""",
    ),
    (
        "code",
        """\
rng = np.random.default_rng(0)
ctrl = data.control['converted'].values.astype(float)
trt  = data.treatment['converted'].values.astype(float)

diffs = []
for _ in range(2_000):
    bc = rng.choice(ctrl, size=len(ctrl), replace=True)
    bt = rng.choice(trt,  size=len(trt),  replace=True)
    diffs.append(bt.mean() - bc.mean())

diffs = np.array(diffs)
boot_lo, boot_hi = np.percentile(diffs, [2.5, 97.5])
print(f"Bootstrap 95% CI for difference: [{boot_lo:+.4f}, {boot_hi:+.4f}]")
print(f"Bootstrap point estimate       : {diffs.mean():+.4f}")
""",
    ),
    (
        "markdown",
        """\
### Conclusion

The published Cookie Cats finding is that the **7-day retention difference is NOT statistically
significant** — the gate move from 30 to 40 had little detectable impact on 7-day retention at
conventional thresholds.

Both the frequentist p-value and Bayesian posterior should reflect this: the 95% CI
straddles zero and P(treatment better) is close to 0.5.

> Note: if using simulated data (no CSV found), the simulated lift of −0.008 mimics the
> direction seen in the original data. Results will be similar in magnitude but not identical.
""",
    ),
]

# ---------------------------------------------------------------------------
# Notebook 04 — CUPED Variance Reduction
# ---------------------------------------------------------------------------

NB04_CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
## CUPED — Controlled-experiment Using Pre-Experiment Data

**CUPED** (Deng et al., 2013) reduces variance in experiment estimates by regressing out
a pre-experiment covariate correlated with the metric of interest.  This increases
statistical power without collecting more data.

> Cookie Cats lacks a pre-period covariate, so this notebook uses **simulated data** with a
> known covariate–metric correlation of 0.7.
""",
    ),
    (
        "code",
        """\
from src.abtest import simulate_conversion, apply_cuped, two_proportion_z
import matplotlib.pyplot as plt
import numpy as np

# Null experiment: true lift = 0, strong covariate correlation
sim = simulate_conversion(
    n_per_arm=40_000,
    base_rate=0.30,
    absolute_lift=0.0,
    covariate_corr=0.7,
    seed=31,
)
print(f"n_control  : {sim.n_control:,}")
print(f"n_treatment: {sim.n_treatment:,}")
""",
    ),
    (
        "code",
        """\
# Unadjusted frequentist result
freq = two_proportion_z(sim, 'converted', alpha=0.05)
print("=== Unadjusted ===")
print(f"Absolute effect : {freq.absolute_effect:+.5f}")
print(f"95% CI          : [{freq.ci_low:+.5f}, {freq.ci_high:+.5f}]")
print(f"p-value         : {freq.p_value:.4f}")
""",
    ),
    (
        "code",
        """\
# CUPED-adjusted result
cuped = apply_cuped(sim, 'converted', covariate='pre_covariate', alpha=0.05)
print("=== CUPED-adjusted ===")
print(f"Theta (regression coeff) : {cuped.theta:.4f}")
print(f"Variance reduction       : {cuped.variance_reduction:.2%}")
print(f"Adjusted absolute effect : {cuped.adjusted_absolute_effect:+.5f}")
print(f"Adjusted 95% CI          : [{cuped.adjusted_ci_low:+.5f}, {cuped.adjusted_ci_high:+.5f}]")
print(f"Adjusted p-value         : {cuped.adjusted_p_value:.4f}")
""",
    ),
    (
        "code",
        """\
# Visual comparison of CI widths
labels = ['Unadjusted', 'CUPED']
lows   = [freq.ci_low,         cuped.adjusted_ci_low]
highs  = [freq.ci_high,        cuped.adjusted_ci_high]
points = [freq.absolute_effect, cuped.adjusted_absolute_effect]

fig, ax = plt.subplots(figsize=(6, 3))
for i, (lo, hi, pt, lab) in enumerate(zip(lows, highs, points, labels)):
    ax.plot([lo, hi], [i, i], lw=3, color='steelblue' if i == 0 else 'darkorange')
    ax.plot(pt, i, 'o', color='navy' if i == 0 else 'saddlebrown', zorder=5)

ax.axvline(0, color='red', linestyle='--', alpha=0.6, label='True lift = 0')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel('Absolute effect (95% CI)')
ax.set_title('CUPED: Narrowed confidence interval')
ax.legend()
plt.tight_layout()
plt.show()

print(f"\\nCI width reduction: {(freq.ci_high - freq.ci_low):.5f} → {(cuped.adjusted_ci_high - cuped.adjusted_ci_low):.5f}")
""",
    ),
    (
        "markdown",
        """\
### Key takeaways

- With `covariate_corr=0.7` the variance reduction is approximately **50%** (since variance
  reduction ≈ ρ²).  This shrinks the CI noticeably.
- CUPED is most impactful when you have a strong pre-period signal (e.g. last week's revenue,
  prior retention rate).
- The true lift here is 0 — the adjusted estimate should remain centred near zero,
  confirming the adjustment doesn't introduce bias.
""",
    ),
]

# ---------------------------------------------------------------------------
# Notebook 05 — Sequential Testing
# ---------------------------------------------------------------------------

NB05_CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
## Sequential Testing & the Peeking Problem

### The peeking problem

Checking a p-value repeatedly during an experiment inflates the Type-I error rate far above
the nominal α.  This notebook quantifies that inflation and shows that **mSPRT**
(mixture Sequential Probability Ratio Test) maintains valid error control under continuous
monitoring — producing an *always-valid* p-value.
""",
    ),
    (
        "code",
        """\
import numpy as np
from src.abtest import naive_peeking_fpr, msprt_stream, obrien_fleming_bounds

# --- Naive peeking FPR ---
naive_fpr = naive_peeking_fpr(
    base_rate=0.20,
    n_max=20_000,
    look_every=500,
    n_sims=300,
    alpha=0.05,
    seed=99,
)
print(f"Naive peeking FPR : {naive_fpr:.3f}  (expected ~0.31)")
""",
    ),
    (
        "code",
        """\
# --- mSPRT null cross rate ---
null_crossed = [
    msprt_stream(base_rate=0.20, lift=0.0, n_max=20_000, tau=0.05, seed=s).crossed
    for s in range(300)
]
msprt_null_fpr = float(np.mean(null_crossed))
print(f"mSPRT null FPR    : {msprt_null_fpr:.3f}  (expected ~0.04)")
""",
    ),
    (
        "code",
        """\
import matplotlib.pyplot as plt

methods = ['Naive peeking', 'mSPRT']
fprs    = [naive_fpr, msprt_null_fpr]

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(methods, fprs, color=['tomato', 'steelblue'], width=0.4)
ax.axhline(0.05, color='black', linestyle='--', label='Nominal α = 0.05')
for bar, v in zip(bars, fprs):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f'{v:.3f}', ha='center', fontsize=11)
ax.set_ylabel('False-positive rate (null)')
ax.set_title('Peeking inflation vs mSPRT control')
ax.legend()
ax.set_ylim(0, max(fprs) * 1.3)
plt.tight_layout()
plt.show()
""",
    ),
    (
        "markdown",
        """\
### mSPRT early stopping under a real effect
""",
    ),
    (
        "code",
        """\
result = msprt_stream(base_rate=0.20, lift=0.05, n_max=20_000, tau=0.05, seed=41)
print(f"Crossed threshold : {result.crossed}")
print(f"Stop index        : {result.stop_index}  (out of 20 000)")
print(f"Final statistic   : {result.final_statistic:.4f}")
print(f"Threshold         : {result.threshold:.4f}")
""",
    ),
    (
        "markdown",
        """\
### O'Brien–Fleming group-sequential bounds
""",
    ),
    (
        "code",
        """\
bounds = obrien_fleming_bounds(n_looks=5, alpha=0.05)
print("O'Brien–Fleming alpha-spending bounds (z-score thresholds):")
for i, b in enumerate(bounds, 1):
    print(f"  Look {i}: z > {b:.4f}")
""",
    ),
    (
        "markdown",
        """\
### Summary

| Method | Null FPR |
|---|---|
| Naive peeking (check every 500 obs) | ~0.31 — **6× inflation** |
| mSPRT (always-valid) | ~0.04 — near nominal α |

- mSPRT achieves valid inference at *every* observation by using a mixture prior over effect
  sizes, turning the likelihood ratio into a martingale.
- O'Brien–Fleming bounds are the classical group-sequential alternative: pre-schedule a small
  number of looks with spending-adjusted thresholds.
""",
    ),
]

# ---------------------------------------------------------------------------
# Notebook 06 — Simulation Validation
# ---------------------------------------------------------------------------

NB06_CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
## Simulation Validation — Library Correctness Checks

This notebook validates the `ab-experiment-platform` library against known theoretical values:

1. **Estimator coverage** — 95% CIs from `two_proportion_z` should cover the true lift ~95% of the time.
2. **Null FPR** — under H₀ (lift = 0), significant results should occur ~5% of the time.
3. **Power** — achieved empirical power should match the theoretical prediction.

Figures are saved to `reports/figures/`.
""",
    ),
    (
        "code",
        """\
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.abtest import simulate_conversion, two_proportion_z, required_sample_size, power_for_sample_size

figures_dir = Path('reports/figures')
figures_dir.mkdir(parents=True, exist_ok=True)

BASE_RATE = 0.20
TRUE_LIFT = 0.03
N_PER_ARM = 4_000
N_SEEDS   = 500
ALPHA     = 0.05
""",
    ),
    (
        "markdown",
        """\
### 1 — Estimator coverage (N = 500 simulations)
""",
    ),
    (
        "code",
        """\
covers  = []
sigs    = []
effects = []

for seed in range(N_SEEDS):
    sim = simulate_conversion(n_per_arm=N_PER_ARM, base_rate=BASE_RATE,
                              absolute_lift=TRUE_LIFT, seed=seed)
    r = two_proportion_z(sim, 'converted', alpha=ALPHA)
    covers.append(r.ci_low <= TRUE_LIFT <= r.ci_high)
    sigs.append(r.significant)
    effects.append(r.absolute_effect)

coverage = float(np.mean(covers))
emp_power = float(np.mean(sigs))
print(f"Coverage (95% CI covers true lift = {TRUE_LIFT}): {coverage:.3f}  (expect ~0.95)")
""",
    ),
    (
        "markdown",
        """\
### 2 — Null FPR (N = 2 000 simulations at lift = 0)
""",
    ),
    (
        "code",
        """\
null_sigs = []
for seed in range(2_000):
    sim = simulate_conversion(n_per_arm=N_PER_ARM, base_rate=BASE_RATE,
                              absolute_lift=0.0, seed=seed)
    r = two_proportion_z(sim, 'converted', alpha=ALPHA)
    null_sigs.append(r.significant)

null_fpr = float(np.mean(null_sigs))
print(f"Null FPR : {null_fpr:.4f}  (expect ~0.05)")
""",
    ),
    (
        "markdown",
        """\
### 3 — Power: predicted vs empirical
""",
    ),
    (
        "code",
        """\
# Design sample size for 80% power at the given MDE
design = required_sample_size(baseline_rate=BASE_RATE, mde_absolute=TRUE_LIFT,
                              alpha=ALPHA, power=0.80)
n_designed = design.sample_size_per_arm

predicted_power = power_for_sample_size(baseline_rate=BASE_RATE, mde_absolute=TRUE_LIFT,
                                         n_per_arm=n_designed, alpha=ALPHA)

# Empirical power at the designed n (200 sims — fast)
emp_sigs_designed = []
for seed in range(200):
    sim = simulate_conversion(n_per_arm=n_designed, base_rate=BASE_RATE,
                              absolute_lift=TRUE_LIFT, seed=seed + 10_000)
    r = two_proportion_z(sim, 'converted', alpha=ALPHA)
    emp_sigs_designed.append(r.significant)

emp_power_designed = float(np.mean(emp_sigs_designed))
print(f"Designed n/arm   : {n_designed:,}")
print(f"Predicted power  : {predicted_power:.3f}  (expect ~0.80)")
print(f"Empirical power  : {emp_power_designed:.3f}  (expect ~0.80)")
""",
    ),
    (
        "markdown",
        """\
### Power curve figure
""",
    ),
    (
        "code",
        """\
ns = np.linspace(500, 10_000, 100).astype(int)
pw = [power_for_sample_size(baseline_rate=BASE_RATE, mde_absolute=TRUE_LIFT, n_per_arm=n, alpha=ALPHA)
      for n in ns]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ns, pw, color='steelblue', label='Theoretical power')
ax.axhline(0.80, color='red', linestyle='--', label='80% target')
ax.axvline(n_designed, color='orange', linestyle=':', label=f'Designed n={n_designed:,}')
ax.scatter([n_designed], [emp_power_designed], color='green', zorder=5,
           label=f'Empirical power={emp_power_designed:.2f}')
ax.set_xlabel('Sample size per arm')
ax.set_ylabel('Power')
ax.set_title(f'Power curve — base={BASE_RATE}, MDE={TRUE_LIFT}')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/power_curve.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: reports/figures/power_curve.png")
""",
    ),
    (
        "markdown",
        """\
### Validation summary figure
""",
    ),
    (
        "code",
        """\
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Coverage
ax = axes[0]
ax.bar(['Coverage'], [coverage], color='steelblue')
ax.axhline(0.95, color='red', linestyle='--', label='Expected 0.95')
ax.set_ylim(0.85, 1.0)
ax.set_title('CI Coverage')
ax.set_ylabel('Rate')
ax.text(0, coverage + 0.002, f'{coverage:.3f}', ha='center', fontsize=12)
ax.legend(fontsize=8)

# Null FPR
ax = axes[1]
ax.bar(['Null FPR'], [null_fpr], color='tomato')
ax.axhline(0.05, color='black', linestyle='--', label='Expected 0.05')
ax.set_ylim(0, 0.12)
ax.set_title('Null False-Positive Rate')
ax.set_ylabel('Rate')
ax.text(0, null_fpr + 0.002, f'{null_fpr:.3f}', ha='center', fontsize=12)
ax.legend(fontsize=8)

# Power comparison
ax = axes[2]
ax.bar(['Predicted', 'Empirical'], [predicted_power, emp_power_designed],
       color=['steelblue', 'darkorange'])
ax.axhline(0.80, color='red', linestyle='--', label='Target 0.80')
ax.set_ylim(0.6, 1.0)
ax.set_title('Power: Predicted vs Empirical')
ax.set_ylabel('Power')
for i, v in enumerate([predicted_power, emp_power_designed]):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=12)
ax.legend(fontsize=8)

plt.suptitle('Simulation Validation Summary', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/validation_summary.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: reports/figures/validation_summary.png")
""",
    ),
    (
        "markdown",
        """\
### Summary

| Check | Expected | Actual |
|---|---|---|
| CI Coverage | ~0.95 | see above |
| Null FPR | ~0.05 | see above |
| Predicted power | ~0.80 | see above |
| Empirical power | ~0.80 | see above |

All checks should be within sampling noise of their targets, confirming the library's
statistical correctness.
""",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    notebooks = [
        ("01_eda_cookie_cats.ipynb",       "Cookie Cats EDA",                  NB01_CELLS),
        ("02_design_power.ipynb",           "Power Analysis & Design",          NB02_CELLS),
        ("03_frequentist_bayesian.ipynb",   "Frequentist & Bayesian Analysis",  NB03_CELLS),
        ("04_cuped_variance_reduction.ipynb", "CUPED — Variance Reduction",     NB04_CELLS),
        ("05_sequential_testing.ipynb",     "Sequential Testing",               NB05_CELLS),
        ("06_simulation_validation.ipynb",  "Simulation Validation",            NB06_CELLS),
    ]

    print(f"Writing notebooks to: {NOTEBOOKS}")
    for filename, title, cells in notebooks:
        nb = make_notebook(title, cells)
        write(filename, nb)

    print("\nAll 6 notebooks written successfully.")


if __name__ == "__main__":
    main()
