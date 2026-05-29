from __future__ import annotations

from src.agent.nodes import is_grounded
from src.agent.state import Gap, GapReport


def _report(gaps: list[Gap]) -> GapReport:
    return GapReport(role="DS", location="London", n_postings=10, gaps=gaps)


def test_is_grounded_empty_gaps_is_grounded():
    report = _report([])
    assert is_grounded(report, valid_skills={"python", "sql"}) is True


def test_is_grounded_all_gaps_valid_and_demanded():
    gaps = [
        Gap(skill="python", demand_count=5, demand_fraction=0.5),
        Gap(skill="sql", demand_count=3, demand_fraction=0.3),
    ]
    report = _report(gaps)
    assert is_grounded(report, valid_skills={"python", "sql", "java"}) is True


def test_is_grounded_ungrounded_when_skill_invalid_or_zero_demand():
    # skill not in valid set
    report_invalid = _report([Gap(skill="hallucinated", demand_count=4, demand_fraction=0.4)])
    assert is_grounded(report_invalid, valid_skills={"python", "sql"}) is False

    # zero demand
    report_zero = _report([Gap(skill="python", demand_count=0, demand_fraction=0.0)])
    assert is_grounded(report_zero, valid_skills={"python", "sql"}) is False
