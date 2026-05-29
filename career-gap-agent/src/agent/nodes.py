"""Graph nodes and the groundedness check for the career-gap agent."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.state import AgentState, GapReport
from src.agent.tools import (
    TOOLS,
    AgentDeps,
    run_analyse_postings,
    run_search_jobs,
)

MAX_ITERATIONS = 6


def is_grounded(report: GapReport, valid_skills: set[str]) -> bool:
    if not report.gaps:
        return True
    valid_lower = {s.lower() for s in valid_skills}
    return all(g.demand_count > 0 and g.skill.lower() in valid_lower for g in report.gaps)


def planner_node(state: AgentState, llm: Any) -> dict:
    """Increment iterations and let the planner LLM decide the next action."""
    iterations = state.get("iterations", 0) + 1
    ai = llm.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [ai], "iterations": iterations}


def tool_node(state: AgentState, deps: AgentDeps) -> dict:
    """Custom tool node: execute the last AIMessage's tool calls against deps.

    Appends one ToolMessage per call and returns structured state updates
    (postings / report / cv_skills / demand) so they land in graph state.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    messages: list[ToolMessage] = []
    update: dict = {}

    for call in tool_calls:
        name = call["name"]
        args = call.get("args", {}) or {}
        call_id = call.get("id", "")

        if name == "search_jobs":
            summary, state_update = run_search_jobs(
                args.get("role", state.get("role", "")),
                args.get("location", state.get("location", "")),
                deps,
            )
        elif name == "analyse_postings":
            # analyse needs whatever postings have accumulated so far
            merged = {**state, **update}
            summary, state_update = run_analyse_postings(merged, deps)
        else:
            summary, state_update = f"Unknown tool: {name}", {}

        update.update(state_update)
        messages.append(ToolMessage(content=summary, tool_call_id=call_id))

    update["messages"] = messages
    return update


def reflection_node(state: AgentState, valid_skills: set[str]) -> dict:
    """Check groundedness; if ungrounded and under the cap, push a correction."""
    report = state.get("report")
    if report is None or is_grounded(report, valid_skills):
        return {}
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return {}
    correction = HumanMessage(
        content=(
            "The gap report contains skills that are not grounded in the ESCO "
            "taxonomy or have zero demand. Re-run analyse_postings and only "
            "report skills that appear in the postings."
        )
    )
    return {"messages": [correction]}


def has_tool_calls(message: AIMessage) -> bool:
    return bool(getattr(message, "tool_calls", None))
