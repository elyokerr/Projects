# Methodology

This document records the formula and assumptions behind each method in `src/abtest/`. All methods
operate on the shared `ExperimentData` contract (a tidy per-unit frame with a `variant` column,
one or more metric columns, and optional pre-period covariate / timestamp columns).

## Design - power and sample size (`design.py`)

For a two-proportion test with baseline rate `p1` and a target `p2 = p1 + MDE`, the effect size is
Cohen's h via statsmodels `proportion_effectsize(p2, p1)`. The required per-arm sample size is solved
with `NormalIndPower().solve_power(effect_size=h, alpha, power, ratio=1.0, alternative="two-sided")`.
Duration is `n_per_arm / daily_traffic_per_arm` when traffic is supplied.

**Assumptions:** independent units, equal allocation, a normal approximation to the binomial that is
valid away from rates near 0 or 1.

## Health checks (`health.py`)

**Sample-ratio mismatch (SRM).** A chi-square goodness-of-fit test compares observed arm counts to
the intended allocation (`scipy.stats.chisquare`). A mismatch is flagged at `p < 0.001` (the industry
norm), which marks the experiment's results as untrustworthy because a broken split confounds the
comparison.

**A/A test.** Repeatedly splits a single population with no treatment and measures the rejection rate
of the two-proportion z-test; under correct calibration this approaches the nominal `alpha`.

## Frequentist analysis (`frequentist.py`)

**Two-proportion z-test.** Pooled standard error
`SE = sqrt(p̄(1−p̄)(1/n_c + 1/n_t))`, `z = (p_t − p_c) / SE`, two-sided p-value from the normal CDF.
The reported confidence interval uses the unpooled SE.

**Welch's t-test.** `scipy.stats.ttest_ind(equal_var=False)` for continuous metrics, with a CI from
the t-distribution using the Welch–Satterthwaite degrees of freedom.

**Mann-Whitney U.** `scipy.stats.mannwhitneyu` (two-sided) for non-parametric / non-normal metrics;
no parametric CI is reported.

**Multiple comparisons.** `statsmodels.stats.multitest.multipletests` with Bonferroni
(family-wise error rate) or Benjamini-Hochberg (`fdr_bh`, false-discovery rate).

## Bayesian analysis (`bayesian.py`)

For a conversion metric, each arm's rate has a Beta posterior `Beta(prior_a + successes,
prior_b + failures)` (conjugate to the Binomial). Drawing samples from both posteriors gives:
`P(treatment > control)`, the expected loss of each decision `E[max(other − chosen, 0)]`, and a 95%
credible interval on the difference. Defaults use a uniform `Beta(1, 1)` prior.

**Assumptions:** independent units, a binary outcome, a Beta prior.

## CUPED variance reduction (`cuped.py`)

CUPED (Deng et al.) uses a pre-experiment covariate `X` correlated with the outcome `Y` to remove
pre-existing variance:

```
theta  = Cov(Y, X) / Var(X)
Y_adj  = Y − theta · (X − mean(X))
```

The treatment effect is the difference in adjusted means; it is unbiased because `E[X − mean(X)] = 0`
in a randomised experiment. The variance of `Y_adj` is lower than `Y` by approximately `rho²`, where
`rho` is the correlation between `Y` and `X`. Variance reduction is reported as
`1 − Var(Y_adj) / Var(Y)`.

## Sequential testing (`sequential.py`)

**mSPRT (mixture Sequential Probability Ratio Test).** For a difference in means with per-observation
variance `σ²`, a normal mixing prior `N(0, τ²)` on the true effect, and `n` observations with observed
mean difference `d`:

```
Λ_n = sqrt( σ² / (σ² + n·τ²) ) · exp( n²·τ²·d² / (2·σ²·(σ² + n·τ²)) )
```

Significance is declared the first time `Λ_n ≥ 1/alpha`. This is *always valid*: the probability of
ever crossing under the null is bounded, `P(∃n : Λ_n ≥ 1/alpha | H0) ≤ alpha`, so continuous
monitoring and early stopping do not inflate the false-positive rate. The mixing variance `τ` is set
on the order of the minimum detectable effect.

**O'Brien-Fleming group-sequential boundaries.** For `K` pre-planned looks at information fractions
`t_k = k/K`, the two-sided alpha-spending function `α*(t) = 2 − 2·Φ(z_{α/2}/√t)` is differenced into
the incremental alpha spent at each look, then converted to a z-boundary. Early boundaries are large
(hard to cross), relaxing toward `z_{α/2}` at the final look.

## Decision (`decision.py`)

`decide` combines three inputs into a recommendation:

- **Statistical significance** - from the frequentist result.
- **Practical significance** - the estimated effect is positive and at least the minimum detectable
  effect.
- **Guardrails** - no guardrail metric has regressed.

The recommendation is `ship` only when all three hold; `no_ship` when significant but a guardrail
regressed or the effect is below the MDE; and `inconclusive` when the result is not significant.

## Simulator (`simulate.py`)

`simulate_conversion` generates synthetic experiments with known ground truth. The baseline
conversion probability varies with a latent factor *additively on the probability scale*
(`base_p = base_rate + covariate_corr · k · latent`), and the treatment adds a fixed absolute lift.
This lets a pre-period covariate (`latent + noise`) correlate with the outcome **without** distorting
the additive treatment effect - a logit-scale shift would not preserve an additive lift and would bias
CUPED's recovered effect. Known ground truth enables simulation-based validation of estimator
recovery, false-positive calibration, power, and sequential validity (see
`notebooks/06_simulation_validation.ipynb`).
