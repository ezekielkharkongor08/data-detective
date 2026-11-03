"""ReAct check agent.

A small ReAct loop built with LangGraph:

    agent (LLM decides) -> tools (runs a check) -> agent ... -> END

The LLM sees only the compact profile (never the CSV) and chooses which
checks to call. Each tool returns Finding objects as JSON; we collect them
back into ``Finding`` objects when the loop ends.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from src.graph.tools import ALL_TOOLS, _parse
from src.schemas.finding import Finding


class AgentState(TypedDict):
    """State for the ReAct subgraph: only the message list."""

    messages: Annotated[list, add_messages]


_SYSTEM_PROMPT = """You are Data Detective, a careful data-quality auditor.

You receive a COMPACT PROFILE of a CSV dataset (columns, dtypes, missing
counts, sample rows). You never see the full dataset and you do not compute
statistics yourself.

Your job: choose the right check tools and call them. The ONLY tools are:
- run_duplicates_check
- run_missing_check  (finds NaN, blanks, and '?' markers)
- run_schema_check
- run_imbalance_check
- run_id_like_check
- run_leakage_check

Rules:
- Call EVERY relevant check in ONE response, using multiple tool calls at once.
- Always call run_duplicates_check, run_missing_check, and run_schema_check.
  Missing values may be encoded as '?' even when the profile shows 0 NaN.
- If a target_column is given, call run_imbalance_check, run_id_like_check,
  and run_leakage_check with that target_column.
- Pass the csv_path exactly as given.
- Call each tool at most once.
"""


def build_check_agent(
    llm: BaseChatModel,
    tools: Optional[list] = None,
) -> Any:
    """Build and compile the ReAct subgraph that chooses and runs checks."""
    tools = tools or ALL_TOOLS
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def _collect_findings(messages: list) -> list[Finding]:
    """Extract and deduplicate findings from tool messages."""
    findings: list[Finding] = []
    seen: set[tuple] = set()

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            parsed = _parse(message.content)
        except Exception:
            # One malformed tool result must not lose the others.
            continue
        for finding in parsed:
            key = (
                finding.check_name,
                tuple(sorted(finding.affected_columns)),
                finding.evidence,
            )
            if key not in seen:
                seen.add(key)
                findings.append(finding)

    return findings


def run_checks_with_agent(
    csv_path: str,
    target_column: Optional[str],
    profile: dict[str, Any],
    llm: BaseChatModel,
    tools: Optional[list] = None,
) -> tuple[list[Finding], list]:
    """Run the ReAct agent on a dataset profile and return its findings.

    Returns:
        (findings, messages) where findings are deduplicated ``Finding``
        objects produced by the tools the agent chose to call.
    """
    agent = build_check_agent(llm, tools=tools)

    profile_json = json.dumps(profile, default=str)
    request = {
        "dataset": csv_path,
        "target_column": target_column,
        "profile": profile_json,
    }

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(request)),
    ]

    result = agent.invoke({"messages": messages})
    all_messages = result.get("messages", [])
    return _collect_findings(all_messages), all_messages


__all__ = ["AgentState", "build_check_agent", "run_checks_with_agent"]
