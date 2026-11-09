"""LangGraph builder for the Visualizer Workbench (V2).

Builds a small ReAct graph: entry -> route -> [visualize | exit], where the
gate after ``route`` decides whether the agent runs plot tools at all.
V1 is not modified.
"""
 
from typing import Any, Optional
 
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
 
from src.graph.llm import get_cached_llm
from src.workbench.visualizer.state import GraphState
from src.workbench.visualizer.visualizer_nodes import (
    entry_node,
    route_node,
    agent_node,
    collect_findings_node,
)
from src.workbench.visualizer.visualizer_tools import ALL_TOOLS
 
 
def build_visualizer_graph(llm: Any | None = None, checkpointer=None):
    """Build and compile the V2 visualizer graph."""
    if llm is None:
        llm = get_cached_llm()
 
    builder = StateGraph(GraphState)
 
    builder.add_node("entry", entry_node)
    builder.add_node("route", route_node)
    builder.add_node("agent", agent_node(llm))
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("collect", collect_findings_node)
 
    builder.set_entry_point("entry")
 
    builder.add_edge("entry", "route")
 
    builder.add_conditional_edges(
        "route",
        lambda state: "visualize" if state.get("want_plots") else "exit",
        {
            "visualize": "agent",
            "exit": END,
        },
    )
 
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "collect"},
    )
 
    builder.add_edge("tools", "agent")
    builder.add_edge("collect", END)
 
    graph = builder.compile(checkpointer=checkpointer)
    graph.name = "Visualizer Workbench"
    return graph
 
 
__all__ = ["build_visualizer_graph"]
 
