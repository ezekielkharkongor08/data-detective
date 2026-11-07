"""Graph nodes for the Visualizer Workbench (V2).

Each node takes the current GraphState and returns a dict of field updates
to merge into the state. Calculation logic is kept in pure helper functions
so it can be tested without LangGraph.
"""

from __future__ import annotations

import json
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.workbench.visualizer.state import GraphState
from src.workbench.visualizer.visualizer_tools import ALL_TOOLS


_BUILD_SYSTEM_PROMPT = (
    "You are Data Detective Visualizer, a careful charting assistant.\n\n"
    "You receive V1 audit findings (check_name, severity, evidence, "
    "affected_columns, confidence) and a CSV path. You never see the full "
    "dataset and you do not compute statistics yourself.\n\n"
    "Your job: choose the right plot tools and call them. The ONLY tools "
    "are:\n{tool_lines}\n\n"
    "Rules:\n"
    "- Call EVERY relevant plot in ONE response, using multiple tool calls "
    "at once.\n"
    "- Pass the csv_path exactly as given.\n"
    "- Call each tool at most once.\n"
    "- Only plot columns that appear in the findings."
).format(
    tool_lines="\n".join(f"- {t.name}: {t.description}" for t in ALL_TOOLS)
)


def _build_system_prompt() -> str:
    return _BUILD_SYSTEM_PROMPT


def entry_node(state: GraphState) -> GraphState:
    """Validate input and set the ``want_plots`` flag."""
    csv_path = state.get("csv_path")
    findings = state.get("findings_in") or []

    if not csv_path:
        return {"error": "No csv_path provided; cannot visualize."}

    if not findings:
        return {
            "error": "No findings provided; nothing to visualize.",
            "want_plots": False,
        }

    request = {
        "dataset": csv_path,
        "target_column": state.get("target_column"),
        "findings": findings,
    }
    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=json.dumps(request, default=str)),
    ]
    return {"want_plots": True, "error": None, "messages": messages}


def route_node(state: GraphState) -> GraphState:
    """Routing node. The conditional edge after it reads ``want_plots``."""
    return {}


def agent_node(llm: BaseChatModel) -> Callable[[GraphState], dict]:
    """Factory for the ReAct agent node that chooses and runs plot tools."""
    llm_with_tools = None  # bound on first use, then reused

    def _agent(state: GraphState) -> dict:
        nonlocal llm_with_tools
        if llm_with_tools is None:
            llm_with_tools = llm.bind_tools(ALL_TOOLS)
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    return _agent


def collect_findings_node(state: GraphState) -> GraphState:
    """Extract and deduplicate plot results from tool messages."""
    messages = state.get("messages") or []
    plot_paths: list[str] = list(state.get("plot_paths") or [])
    seen: set[str] = set(plot_paths)

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content) if isinstance(message.content, str) else {}
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        plot_file = payload.get("plot_file")
        if plot_file and payload.get("status") == "ok" and plot_file not in seen:
            seen.add(plot_file)
            plot_paths.append(plot_file)

    return {"plot_paths": plot_paths}


__all__ = [
    "entry_node",
    "route_node",
    "agent_node",
    "collect_findings_node",
    "ALL_TOOLS",
]