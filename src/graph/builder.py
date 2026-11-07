"""Graph builders: parallel-checks audit graph and LLM-driven audit graph."""
 
from typing import Any, Callable
 
from langgraph.graph import END, START, StateGraph
 
from src.graph.llm import get_cached_llm
from src.graph.nodes import (
    deep_checks_node,
    load_csv_node,
    make_agent_checks_node,
    make_check_node,
    merge_findings_node,
    profile_node,
    report_node,
    router_node,
)
from src.graph.state import GraphState
from src.graph.synthesis import make_synthesize_findings_node
from src.tools.checks import get_check, get_checks
 
CHECKS: list[tuple[str, Callable[..., Any]]] = [
    (spec.name, spec.run) for spec in get_checks(parallel=True)
]
 
 
def _go_unless_error(next_node: str) -> Callable[[GraphState], str]:
    def route(state: GraphState) -> str:
        return "report" if state.get("error") else next_node
 
    route.__name__ = f"route_to_{next_node}_unless_error"
    return route
 
 
def _fan_out_unless_error(names: list[str]) -> Callable[[GraphState], list[str]]:
    def route(state: GraphState) -> list[str]:
        return ["report"] if state.get("error") else list(names)
 
    route.__name__ = "fan_out_checks_unless_error"
    return route
 
 
def build_graph(checkpointer=None):
    """load_csv -> profile -> [checks] -> merge_findings -> router -> deep_checks -> report.
 
    An error after load_csv or profile skips straight to report.
    """
    builder = StateGraph(GraphState)
 
    builder.add_node("load_csv", load_csv_node)
    builder.add_node("profile", profile_node)
    builder.add_node("merge_findings", merge_findings_node)
    builder.add_node("router", router_node)
    builder.add_node("deep_checks", deep_checks_node)
    builder.add_node("report", report_node)
 
    check_node_names: list[str] = []
    for name, fn in CHECKS:
        node_name = f"check_{name}"
        builder.add_node(node_name, make_check_node(get_check(name), run_fn=fn))
        check_node_names.append(node_name)
 
    builder.add_edge(START, "load_csv")
    builder.add_conditional_edges(
        "load_csv",
        _go_unless_error("profile"),
        {"profile": "profile", "report": "report"},
    )
 
    if check_node_names:
        builder.add_conditional_edges(
            "profile",
            _fan_out_unless_error(check_node_names),
            ["report", *check_node_names],
        )
        builder.add_edge(check_node_names, "merge_findings")
    else:
        builder.add_conditional_edges(
            "profile",
            _go_unless_error("merge_findings"),
            {"merge_findings": "merge_findings", "report": "report"},
        )
 
    builder.add_edge("merge_findings", "router")
    builder.add_conditional_edges(
        "router",
        lambda state: "deep_checks" if state.get("leakage_suspected") else "report",
        {"deep_checks": "deep_checks", "report": "report"},
    )
    builder.add_edge("deep_checks", "report")
    builder.add_edge("report", END)
 
    return builder.compile(checkpointer=checkpointer)
 
 
def build_agent_graph(llm=None, checkpointer=None):
    """load_csv -> profile -> agent_checks -> deep_checks -> synthesize -> report.
 
    An error after load_csv or profile skips straight to report.
    """
    if llm is None:
        llm = get_cached_llm()
 
    builder = StateGraph(GraphState)
 
    builder.add_node("load_csv", load_csv_node)
    builder.add_node("profile", profile_node)
    builder.add_node("agent_checks", make_agent_checks_node(llm))
    builder.add_node("deep_checks", deep_checks_node)
    builder.add_node("synthesize", make_synthesize_findings_node(llm))
    builder.add_node("report", report_node)
 
    builder.add_edge(START, "load_csv")
    builder.add_conditional_edges(
        "load_csv",
        _go_unless_error("profile"),
        {"profile": "profile", "report": "report"},
    )
    builder.add_conditional_edges(
        "profile",
        _go_unless_error("agent_checks"),
        {"agent_checks": "agent_checks", "report": "report"},
    )
    builder.add_edge("agent_checks", "deep_checks")
    builder.add_edge("deep_checks", "synthesize")
    builder.add_edge("synthesize", "report")
    builder.add_edge("report", END)
 
    return builder.compile(checkpointer=checkpointer)
 
 
__all__ = ["build_graph", "build_agent_graph", "CHECKS"]
