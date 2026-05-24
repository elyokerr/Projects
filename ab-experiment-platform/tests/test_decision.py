from src.abtest.decision import decide
from src.abtest.results import FrequentistResult


def _fr(effect, p, sig):
    return FrequentistResult("m", 0.2, 0.2 + effect, effect, effect / 0.2,
                             effect - 0.01, effect + 0.01, p, sig,
                             "two_proportion_z",
                             "significant difference" if sig
                             else "no significant difference")


def test_ship_when_sig_and_above_mde_and_guardrails_ok():
    d = decide(_fr(0.03, 0.001, True), mde_absolute=0.02, guardrails_ok=True)
    assert d.recommendation == "ship"


def test_no_ship_when_significant_but_below_mde():
    d = decide(_fr(0.005, 0.001, True), mde_absolute=0.02, guardrails_ok=True)
    assert d.recommendation == "no_ship"


def test_no_ship_when_guardrail_breached():
    d = decide(_fr(0.03, 0.001, True), mde_absolute=0.02, guardrails_ok=False)
    assert d.recommendation == "no_ship"


def test_inconclusive_when_not_significant():
    d = decide(_fr(0.03, 0.30, False), mde_absolute=0.02, guardrails_ok=True)
    assert d.recommendation == "inconclusive"
