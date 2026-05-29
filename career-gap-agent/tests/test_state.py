from __future__ import annotations

from src.agent.state import AgentState, Gap, GapReport, Posting


def test_dataclasses_construct():
    posting = Posting(
        title="Data Scientist",
        company="Acme",
        description="We need Python and SQL.",
        location="London",
    )
    assert posting.title == "Data Scientist"
    assert posting.salary_min is None

    gap = Gap(skill="python", demand_count=5, demand_fraction=0.5)
    assert gap.demand_count == 5

    report = GapReport(role="Data Scientist", location="London", n_postings=10)
    assert report.gaps == []
    assert report.summary == ""


def test_agent_state_importable_and_usable():
    state: AgentState = {
        "messages": [],
        "role": "Data Scientist",
        "location": "London",
        "iterations": 0,
    }
    assert state["role"] == "Data Scientist"
    # total=False: partial dicts are valid
    partial: AgentState = {"iterations": 1}
    assert partial["iterations"] == 1
