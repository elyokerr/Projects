from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import build_graph
from src.agent.tools import AgentDeps


class StubLLM:
    """No-network planner LLM.

    invoke #1 -> search_jobs tool call
    invoke #2 -> analyse_postings tool call
    invoke #3+ -> plain final AIMessage

    ``bind_tools`` returns self so the call chain in planner_node works.
    """

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):  # noqa: ARG002 — schema binding is a no-op for the stub
        return self

    def invoke(self, messages):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_jobs",
                        "args": {"role": "data scientist", "location": "London"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyse_postings",
                        "args": {},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Here is your skill-gap report. Focus on the top gaps.")


def _snapshot_deps(matcher) -> AgentDeps:
    return AgentDeps(
        app_id=None,
        app_key=None,
        matcher=matcher,
        llm=lambda p: '["python", "sql"]',
    )


def test_graph_runs_end_to_end_with_stub(esco_matcher):
    deps = _snapshot_deps(esco_matcher)
    graph = build_graph(StubLLM(), deps)

    init = {
        "messages": [HumanMessage(content="Find my skill gaps for data scientist in London.")],
        "role": "data scientist",
        "location": "London",
        "cv_text": "I have experience with python.",
        "iterations": 0,
    }
    final = graph.invoke(init)

    assert final["report"] is not None
    assert final["iterations"] <= 6
