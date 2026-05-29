"""Coarse LLM-facing tools for the career-gap agent.

The agent exposes exactly two tools so the planner's decision space stays
small and reliable. The ``@tool``-decorated callables here are used ONLY to
generate tool-call schemas for ``llm.bind_tools([...])``. The actual work runs
inside ``src.agent.nodes.tool_node`` against an ``AgentDeps`` instance, because
getting structured tool output into LangGraph state via the prebuilt
``ToolNode`` is awkward.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.tools import tool

from src.agent.state import GapReport
from src.data.adzuna_client import search_jobs as adzuna_search_jobs
from src.skills.esco_matcher import EscoMatcher
from src.skills.extractor import extract_skill_phrases
from src.skills.gap import aggregate_demand, compute_gap


@dataclass
class AgentDeps:
    """Runtime dependencies injected into the tool node."""

    app_id: str | None
    app_key: str | None
    matcher: EscoMatcher
    llm: Callable[[str], str]  # skill-extraction text completion fn


@tool
def search_jobs(role: str, location: str) -> str:
    """Search current job postings for a given role and location.

    Use this first to gather the live demand signal before any analysis.
    """
    # Schema-only stub; real execution happens in tool_node.
    raise NotImplementedError


@tool
def analyse_postings() -> str:
    """Analyse the gathered postings against the CV to produce a skill-gap report.

    Call this only after search_jobs has gathered postings.
    """
    # Schema-only stub; real execution happens in tool_node.
    raise NotImplementedError


TOOLS = [search_jobs, analyse_postings]


def run_search_jobs(role: str, location: str, deps: AgentDeps) -> tuple[str, dict]:
    """Execute the search_jobs tool. Returns (summary, state_update)."""
    postings = adzuna_search_jobs(
        role,
        location,
        app_id=deps.app_id,
        app_key=deps.app_key,
    )
    summary = (
        f"Found {len(postings)} postings for '{role}' in '{location}'."
        if postings
        else f"Found no postings for '{role}' in '{location}'."
    )
    return summary, {"postings": postings, "role": role, "location": location}


def run_analyse_postings(state: dict, deps: AgentDeps) -> tuple[str, dict]:
    """Execute the analyse_postings tool. Returns (summary, state_update)."""
    postings = state.get("postings") or []
    role = state.get("role", "")
    location = state.get("location", "")
    cv_text = state.get("cv_text", "")

    per_posting_skills: list[list[str]] = []
    for posting in postings:
        phrases = extract_skill_phrases(posting.description, deps.llm)
        per_posting_skills.append(deps.matcher.match(phrases))

    cv_phrases = extract_skill_phrases(cv_text, deps.llm) if cv_text else []
    cv_skills = deps.matcher.match(cv_phrases) if cv_phrases else []

    demand = aggregate_demand(per_posting_skills)
    n_postings = len(postings)
    gaps = compute_gap(cv_skills, demand, n_postings)

    top = ", ".join(g.skill for g in gaps[:3]) if gaps else "none"
    summary = (
        f"Analysed {n_postings} postings: {len(demand)} distinct in-demand skills, "
        f"{len(gaps)} gaps relative to the CV (top: {top})."
    )
    report = GapReport(
        role=role,
        location=location,
        n_postings=n_postings,
        gaps=gaps,
        summary=summary,
    )
    return summary, {"report": report, "cv_skills": cv_skills, "demand": demand}
