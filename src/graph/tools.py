"""LangChain tools generated from the check registry.

Every registered check that declares ``agent_tool=True`` becomes one tool.
The tool name, description, and argument schema all come from the check's
metadata, so adding a check never requires editing this file.

Tools return the checks' findings as JSON strings so they can travel through
the agent's message history. The LLM only ever receives these results and the
compact profile — never the full CSV (design rule #1).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from src.schemas.finding import Finding
from src.tools.checks import CheckSpec, error_finding, get_checks, run_check


def _serialize(findings: list[Finding]) -> str:
    """Serialize findings to a JSON string for the agent message history."""
    return json.dumps([f.model_dump(mode="json") for f in findings])


def _parse(content: Any) -> list[Finding]:
    """Parse a tool message payload back into Finding objects."""
    if isinstance(content, str):
        data = json.loads(content)
    elif isinstance(content, list):
        # Some providers return content blocks; join text parts.
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
        data = json.loads(text) if text else []
    else:
        data = content or []
    return [Finding.model_validate(item) for item in data]


def _field_type(spec: CheckSpec, key: str) -> type:
    """Return the schema type for one tool argument.

    Optional arguments are wrapped in ``Optional`` so the model may omit them.
    """
    typ = spec.param_type(key)
    return typ if key in spec.required else Optional[typ]


def _args_schema(spec: CheckSpec) -> type:
    """Build a pydantic model describing the tool's arguments.

    Every check receives ``csv_path`` and ``target_column``; target-required
    checks require the target, the rest accept it as an optional hint.
    """
    required_keys = list(spec.required)
    if "csv_path" not in required_keys:
        required_keys.insert(0, "csv_path")

    optional_keys = list(spec.optional)
    if "target_column" not in required_keys and "target_column" not in optional_keys:
        optional_keys.insert(0, "target_column")

    fields: dict[str, tuple[type, Any]] = {}

    for key in required_keys:
        fields[key] = (
            _field_type(spec, key),
            Field(..., description=f"Required input for the {spec.name} check."),
        )
    for key in optional_keys:
        fields[key] = (
            _field_type(spec, key),
            Field(None, description=f"Optional parameter for the {spec.name} check."),
        )

    return create_model(f"{spec.name.title().replace('_', '')}Args", **fields)


def _make_runner(spec: CheckSpec):
    """Create the function behind one generated tool."""

    def _run(**kwargs: Any) -> str:
        # Build the same minimal state the registry expects, so tools and the
        # graph share one contract (required inputs, params, target gating).
        state: dict[str, Any] = {
            "csv_path": kwargs.get("csv_path"),
            "target_column": kwargs.get("target_column"),
        }
        for key in spec.required:
            if key not in state:
                state[key] = kwargs.get(key)
        params = {
            key: kwargs[key]
            for key in spec.optional
            if kwargs.get(key) is not None
        }
        state["check_params"] = {spec.name: params}

        try:
            findings = run_check(spec, state)
        except Exception as exc:  # noqa: BLE001 — a tool must not crash the agent
            findings = [error_finding(spec.name, exc)]
        return _serialize(findings)

    _run.__name__ = spec.agent_tool_name
    return _run


def _make_tool(spec: CheckSpec) -> StructuredTool:
    """Build one LangChain tool from a registered check."""
    description = spec.description or f"Run the {spec.name} check."
    return StructuredTool.from_function(
        func=_make_runner(spec),
        name=spec.agent_tool_name,
        description=(
            f"{description} Returns findings as JSON "
            "(an empty list means clean)."
        ),
        args_schema=_args_schema(spec),
    )


# Tools the agent may choose from. Order follows the registry.
ALL_TOOLS: list[StructuredTool] = [
    _make_tool(spec) for spec in get_checks(agent_tool=True)
]

# Keep module-level names for backwards compatibility
# (for example ``src.graph.tools.run_duplicates_check``).
for _tool in ALL_TOOLS:
    globals()[_tool.name] = _tool

__all__ = [
    *(tool.name for tool in ALL_TOOLS),
    "ALL_TOOLS",
    "_parse",
    "_serialize",
]
