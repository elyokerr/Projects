"""LangGraph wiring for the career-gap agent.

planner -> (tools | reflection) ; tools -> planner ;
reflection -> (planner | END), gated on groundedness + iteration cap.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    MAX_ITERATIONS,
    has_tool_calls,
    is_grounded,
    planner_node,
    reflection_node,
    tool_node,
)
from src.agent.state import AgentState
from src.agent.tools import AgentDeps


def build_graph(llm: Any, deps: AgentDeps):
    """Build and compile the agent graph.

    ``llm`` is the PLANNER chat model (distinct from ``deps.llm``, the
    skill-extraction text fn).
    """
    valid_skills = set(deps.matcher.index.labels)

    graph = StateGraph(AgentState)

    graph.add_node("planner", lambda state: planner_node(state, llm))
    graph.add_node("tools", lambda state: tool_node(state, deps))
    graph.add_node("reflection", lambda state: reflection_node(state, valid_skills))

    graph.set_entry_point("planner")

    def route_after_planner(state: AgentState) -> str:
        last = state["messages"][-1]
        if has_tool_calls(last) and state.get("iterations", 0) < MAX_ITERATIONS:
            return "tools"
        return "reflection"

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tools": "tools", "reflection": "reflection"},
    )

    graph.add_edge("tools", "planner")

    def route_after_reflection(state: AgentState) -> str:
        # reflection_node appends a corrective message iff ungrounded & under cap.
        report = state.get("report")
        if report is None:
            return END
        if is_grounded(report, valid_skills):
            return END
        if state.get("iterations", 0) >= MAX_ITERATIONS:
            return END
        return "planner"

    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"planner": "planner", END: END},
    )

    return graph.compile()
