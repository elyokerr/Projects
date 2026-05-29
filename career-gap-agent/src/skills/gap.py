from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from src.agent.state import Gap


def aggregate_demand(per_posting_skills: Sequence[Sequence[str]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for skills in per_posting_skills:
        c.update(set(skills))  # count each skill once per posting
    return dict(c)


def compute_gap(cv_skills: Sequence[str], demand: dict[str, int], n_postings: int) -> list[Gap]:
    have = {s.lower() for s in cv_skills}
    gaps = [
        Gap(skill=skill, demand_count=count, demand_fraction=count / max(n_postings, 1))
        for skill, count in demand.items()
        if skill.lower() not in have
    ]
    gaps.sort(key=lambda g: (-g.demand_count, g.skill))
    return gaps
