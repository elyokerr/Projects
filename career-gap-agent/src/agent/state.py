from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


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


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    role: str
    location: str
    cv_text: str
    postings: list[Posting]
    cv_skills: list[str]
    demand: dict[str, int]
    report: GapReport | None
    iterations: int
