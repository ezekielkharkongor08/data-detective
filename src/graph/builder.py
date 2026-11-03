"""Graph builder for M2/M3.

Wires: load_csv -> profile -> [5 checks in parallel] -> merge_findings ->
router -> [deep_checks?] -> report.

Uses static fan-out (normal edges) and joins via add_edge([...], ...).
No Send is used.
"""

from langgraph.graph import StateGraph, START, END

from src.graph.llm import get_cached_llm
from src.graph.nodes import (
    deep_checks_node,
    load_csv_node,
    make_agent_checks_node,
    make_check_node,
    make_synthesize_findings_node,
    merge_findings_node,
    profile_node,
    report_node,
    router_node,
)
from src.graph.state import GraphState
from src.tools.checks import (
    bias_check,
    duplicates_check,
    id_like_check,
    imbalance_check,
    missing_check,
    schema_check,
)

# Registry of all M2 parallel checks. The builder loops over this to add
# nodes and edges. Each entry: (check_name, check_function)
CHECKS: list[tuple[str, object]] = [
    ("bias", bias_check),
    ("duplicates", duplicates_check),
    ("missing_values", missing_check),
    ("schema", schema_check),
    ("class_imbalance", imbalance_check),
    ("id_like", id_like_check),
]


def build_graph():
    """Build and compile the M2/M3 audit graph.

    Flow:
        load_csv -> profile -> [check1..check5] -> merge_findings -> router
                                                              -> deep_checks (if leakage suspected)
                                                              -> report (otherwise)
        deep_checks -> report -> END

    A conditional edge after load_csv skips the checks entirely if an error
    was set (e.g., missing CSV file). A conditional edge after the router
    runs deep leakage checks only when a finding suggests potential leakage.
    """
    builder = StateGraph(GraphState)

    # Core nodes
    builder.add_node("load_csv", load_csv_node)
    builder.add_node("profile", profile_node)
    builder.add_node("merge_findings", merge_findings_node)
    builder.add_node("router", router_node)
    builder.add_node("deep_checks", deep_checks_node)
    builder.add_node("report", report_node)

    # Dynamically add a wrapped node for each check in the registry
    check_node_names: list[str] = []
    for name, fn in CHECKS:
        node_name = f"check_{name}"
        builder.add_node(node_name, make_check_node(fn))
        check_node_names.append(node_name)

    # Entry point
    builder.set_entry_point("load_csv")

    # Conditional edge: if load_csv set an error, skip straight to report.
    builder.add_conditional_edges(
        "load_csv",
        lambda state: "report" if state.get("error") else "profile",
        {
            "profile": "profile",
            "report": "report",
        },
    )

    # Fan-out: profile -> each check (5 normal edges)
    for node_name in check_node_names:
        builder.add_edge("profile", node_name)

    # Fan-in: all checks -> merge_findings -> router
    builder.add_edge(check_node_names, "merge_findings")
    builder.add_edge("merge_findings", "router")

    # Conditional edge from router: deep_checks if leakage suspected,
    # otherwise go straight to report.
    builder.add_conditional_edges(
        "router",
        lambda state: "deep_checks" if state.get("leakage_suspected") else "report",
        {
            "deep_checks": "deep_checks",
            "report": "report",
        },
    )

    # Deep checks -> report -> END
    builder.add_edge("deep_checks", "report")
    builder.add_edge("report", END)

    return builder.compile()


def build_agent_graph(llm=None):
    """Build and compile the M4 LLM-driven audit graph.

    Flow:
        load_csv -> profile -> agent_checks (ReAct, chooses tools)
                 -> synthesize (structured summary) -> report

    A conditional edge after load_csv skips straight to the report if the
    input file is missing.
    """
    if llm is None:
        llm = get_cached_llm()

    builder = StateGraph(GraphState)

    builder.add_node("load_csv", load_csv_node)
    builder.add_node("profile", profile_node)
    builder.add_node("agent_checks", make_agent_checks_node(llm))
    builder.add_node("synthesize", make_synthesize_findings_node(llm))
    builder.add_node("report", report_node)

    builder.set_entry_point("load_csv")

    # Skip the agent entirely when loading failed.
    builder.add_conditional_edges(
        "load_csv",
        lambda state: "report" if state.get("error") else "profile",
        {
            "profile": "profile",
            "report": "report",
        },
    )

    builder.add_edge("profile", "agent_checks")
    builder.add_edge("agent_checks", "synthesize")
    builder.add_edge("synthesize", "report")
    builder.add_edge("report", END)

    return builder.compile()


__all__ = ["build_graph", "build_agent_graph", "CHECKS"]
