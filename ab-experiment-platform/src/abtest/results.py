from dataclasses import dataclass


@dataclass
class PowerResult:
    baseline_rate: float
    mde_absolute: float
    alpha: float
    power: float
    sample_size_per_arm: int
    total_sample_size: int
    duration_days: float | None = None


@dataclass
class HealthCheckResult:
    check: str
    passed: bool
    statistic: float
    p_value: float
    detail: str


@dataclass
class FrequentistResult:
    metric: str
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    test: str
    verdict: str


@dataclass
class BayesianResult:
    metric: str
    prob_treatment_better: float
    expected_loss_treatment: float
    expected_loss_control: float
    cred_low: float
    cred_high: float
    verdict: str


@dataclass
class CupedResult:
    metric: str
    theta: float
    variance_reduction: float
    adjusted_absolute_effect: float
    adjusted_ci_low: float
    adjusted_ci_high: float
    adjusted_p_value: float


@dataclass
class SequentialResult:
    method: str
    crossed: bool
    stop_index: int | None
    final_statistic: float
    threshold: float
    detail: str


@dataclass
class Decision:
    recommendation: str
    statistically_significant: bool
    practically_significant: bool
    guardrails_ok: bool
    rationale: str
