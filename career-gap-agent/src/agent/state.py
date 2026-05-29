from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Posting:
    title: str
    company: str
    description: str
    location: str
    salary_min: float | None = None
    salary_max: float | None = None


@dataclass
class Gap:
    skill: str
    demand_count: int  # how many postings required it
    demand_fraction: float  # demand_count / n_postings


@dataclass
class GapReport:
    role: str
    location: str
    n_postings: int
    gaps: list[Gap] = field(default_factory=list)
    summary: str = ""
