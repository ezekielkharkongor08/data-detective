"""ReAct check agent.

A small ReAct loop: agent (LLM decides) -> tools (runs a check) -> END.
The LLM sees only the compact profile and chooses which checks to call;
tool results are collected back into ``Finding`` objects when it ends.
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

from src.graph.tools import ALL_TOOLS, _make_tool, _parse
from src.schemas.finding import Finding
from src.tools.checks import error_finding, get_checks, info_note, run_check


class AgentState(TypedDict):
    """State for the ReAct subgraph: only the message list."""

    messages: Annotated[list, add_messages]


def _build_system_prompt() -> str:
    """Build the ReAct system prompt from the check registry.

    Tool names, descriptions, and the "always / target" guidance all come
    from check metadata, so adding a check updates the prompt automatically.
    """
    specs = get_checks(agent_tool=True)
    tool_lines = "\n".join(
        f"- {spec.agent_tool_name}: {spec.description}" for spec in specs
    )
    always = [spec.agent_tool_name for spec in specs if spec.always]
    target_aware = [
        spec.agent_tool_name for spec in specs if spec.target_hint
    ]

    rules = [
        "- Call EVERY relevant check in ONE response, using multiple tool "
        "calls at once.",
    ]
    if always:
        rules.append(f"- Always call: {', '.join(always)}.")
    if target_aware:
        rules.append(
            "- If a target_column is given, also call: "
            f"{', '.join(target_aware)} with that target_column."
        )
    rules.extend(
        [
            "- Pass the csv_path exactly as given.",
            "- Call each tool at most once.",
        ]
    )

    return (
        "You are Data Detective, a careful data-quality auditor.\n\n"
        "You receive a COMPACT PROFILE of a CSV dataset (columns, dtypes, "
        "missing counts, sample rows). You never see the full dataset and you "
        "do not compute statistics yourself.\n\n"
        "Your job: choose the right check tools and call them. The ONLY tools "
        f"are:\n{tool_lines}\n\n"
        "Rules:\n" + "\n".join(rules) + "\n"
    )


_SYSTEM_PROMPT = _build_system_prompt()


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


def _called_check_names(messages: list) -> set[str]:
    """Return the agent-tool names the LLM actually invoked.

    Read from ``ToolMessage.name`` in the message history, which the model
    cannot forge, so this is the ground truth for which checks ran.
    """
    called: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage) and message.name:
            called.add(message.name)
    return called


def run_uncalled_agent_checks(
    messages: list,
    csv_path: str,
    target_column: Optional[str],
    profile: dict[str, Any],
    check_params: Optional[dict[str, dict]] = None,
) -> list[Finding]:
    """Run every ``agent_tool`` check the agent never called, plus one note.

    Each missing check runs through the normal ``run_check`` path with the
    same state the agent tools use (csv_path, target_column, profile), plus
    the caller's per-check params so user-supplied values (for example
    ``--sensitive``/``--favorable``) are honored. The agent keeps full choice
    of tools; this guard only guarantees completeness.

    Returns the missing checks' findings and, when any were run, one INFO
    finding naming them.
    """
    called = _called_check_names(messages)
    missing = [
        spec
        for spec in get_checks(agent_tool=True)
        if spec.agent_tool_name not in called
    ]

    findings: list[Finding] = []
    state = {
        "csv_path": csv_path,
        "target_column": target_column,
        "profile": profile,
        "check_params": check_params or {},
    }
    for spec in missing:
        try:
            findings += run_check(spec, state)
        except Exception as exc:  # noqa: BLE001 — one check must not stop the guard
            findings.append(error_finding(spec.name, exc))

    if missing:
        names = ", ".join(spec.name for spec in missing)
        findings.append(
            info_note(
                "agent",
                f"Checks not selected by the agent were run automatically: {names}",
            )
        )
    return findings


def run_checks_with_agent(
    csv_path: str,
    target_column: Optional[str],
    profile: dict[str, Any],
    llm: BaseChatModel,
    tools: Optional[list] = None,
    check_params: Optional[dict[str, dict]] = None,
) -> tuple[list[Finding], list]:
    """Run the ReAct agent on a dataset profile and return its findings.

    Deduplicated ``Finding`` objects produced by the tools the agent chose;
    also returns the full agent message history.

    ``csv_path``/``target_column``/``profile`` are bound into each tool from
    agent state, so the LLM controls only which tool to call and optional
    check parameters; it cannot redirect a check to a different file or drop
    the semantic profile. Caller-supplied ``check_params`` (from graph state)
    are injected into the bound tools; user values win over LLM values.
    """
    if tools is None:
        from src.tools.checks import get_checks

        tools = [
            _make_tool(
                spec,
                csv_path=csv_path,
                target_column=target_column,
                profile=profile,
                check_params=check_params,
            )
            for spec in get_checks(agent_tool=True)
        ]
    agent = build_check_agent(llm, tools=tools)

    # The LLM receives a compact digest, never the full profile. The full
    # profile stays in state, the DB, and the report.
    from src.graph.llm_context import build_llm_digest

    profile_digest = build_llm_digest(profile)
    request = {
        "dataset": csv_path,
        "target_column": target_column,
        "profile": profile_digest,
    }

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(request)),
    ]

    result = agent.invoke({"messages": messages})
    all_messages = result.get("messages", [])
    return _collect_findings(all_messages), all_messages


__all__ = [
    "AgentState",
    "build_check_agent",
    "run_checks_with_agent",
    "run_uncalled_agent_checks",
]
