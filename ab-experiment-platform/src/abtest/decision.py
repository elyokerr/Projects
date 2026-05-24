from __future__ import annotations

from src.abtest.results import Decision, FrequentistResult


def decide(result: FrequentistResult, *, mde_absolute: float,
           guardrails_ok: bool) -> Decision:
    stat_sig = result.significant
    practical = abs(result.absolute_effect) >= mde_absolute and result.absolute_effect > 0
    if not stat_sig:
        rec, why = "inconclusive", "effect not statistically significant"
    elif not guardrails_ok:
        rec, why = "no_ship", "a guardrail metric regressed"
    elif not practical:
        rec, why = "no_ship", "significant but below the minimum detectable effect"
    else:
        rec, why = "ship", "significant, above MDE, guardrails intact"
    return Decision(recommendation=rec, statistically_significant=stat_sig,
                    practically_significant=practical, guardrails_ok=guardrails_ok,
                    rationale=why)
