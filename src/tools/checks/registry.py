"""Check registry: the single place a check is declared.

Adding a check is one file: write a function returning ``list[Finding]`` and
decorate it with ``@register_check(...)``. The graph builder, agent tools,
LLM prompt, and tests all read this registry, so no other file needs editing.
``name`` and ``run`` are set automatically; the other metadata fields
(required, optional, target_types, param_types, parallel, agent_tool,
fallback, accepted_semantic) control how and where the check is used.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from src.schemas.finding import Finding, Severity
from src.tools.csv_reader import read_csv_robust
from src.tools.profiling import get_semantic_lookup


@dataclass(frozen=True)
class CheckSpec:
    """Everything the graph, tools, and tests need to know about one check."""

    name: str
    run: Callable[..., list[Finding]]
    required: tuple[str, ...] = ("csv_path",)
    optional: tuple[str, ...] = ()
    target_types: frozenset[str] = frozenset({"any"})
    severity_key: str = ""
    description: str = ""
    always: bool = False
    target_hint: bool = False
    param_types: tuple[tuple[str, type], ...] = ()
    parallel: bool = True
    agent_tool: bool = True
    fallback: bool = True
    tool_name: str | None = None
    accepted_semantic: frozenset[str] = frozenset({"any"})

    @property
    def agent_tool_name(self) -> str:
        """Name of the LangChain tool generated for the ReAct agent."""
        return self.tool_name or f"run_{self.name}_check"

    def param_type(self, key: str) -> type:
        """Return the Python type used for one of this check's parameters.

        Checks can declare types in ``param_types``. Common parameters have
        sensible defaults so most checks need to declare nothing.
        """
        for name, typ in self.param_types:
            if name == key:
                return typ
        if key == "csv_path":
            return str
        if key == "target_column":
            return str | None
        return str if key in self.required else str | None


_REGISTRY: dict[str, CheckSpec] = {}
_DISCOVERED = False


def _first_line(text: str | None) -> str:
    """Return the first non-empty line of a docstring, or an empty string."""
    if not text:
        return ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def register_check(
    *,
    name: str,
    required: Iterable[str] = ("csv_path",),
    optional: Iterable[str] = (),
    target_types: Iterable[str] = ("any",),
    severity_key: str = "",
    description: str = "",
    always: bool = False,
    target_hint: bool = False,
    param_types: Iterable[tuple[str, type]] = (),
    parallel: bool = True,
    agent_tool: bool = True,
    fallback: bool = True,
    tool_name: str | None = None,
    accepted_semantic: Iterable[str] = ("any",),
) -> Callable[[Callable[..., list[Finding]]], Callable[..., list[Finding]]]:
    """Register a check function under a stable name.

    Returns the original function so direct imports keep working.

    Raises:
        ValueError: if the same name is registered by a different function.
    """

    def decorator(fn: Callable[..., list[Finding]]) -> Callable[..., list[Finding]]:
        if name in _REGISTRY:
            # Re-importing the same module is a no-op.
            if _REGISTRY[name].run is fn:
                return fn
            raise ValueError(f"Duplicate check name: {name!r}")

        _REGISTRY[name] = CheckSpec(
            name=name,
            run=fn,
            required=tuple(required),
            optional=tuple(optional),
            target_types=frozenset(target_types),
            severity_key=severity_key,
            description=description or _first_line(fn.__doc__),
            always=always,
            target_hint=target_hint,
            param_types=tuple(param_types),
            parallel=parallel,
            agent_tool=agent_tool,
            fallback=fallback,
            tool_name=tool_name,
            accepted_semantic=frozenset(accepted_semantic),
        )
        return fn

    return decorator


def discover_checks() -> None:
    """Import every module in ``src/tools/checks/`` so its decorators run.

    Files whose name starts with ``_`` are skipped, which is why helper or
    template files can live in the package safely. Importing is cached by
    Python, so calling this more than once is cheap and safe.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return

    package = importlib.import_module("src.tools.checks")
    package_dir = Path(package.__file__).parent

    for path in sorted(package_dir.glob("*.py")):
        if path.name.startswith("_") or path.stem == "registry":
            continue
        importlib.import_module(f"{package.__name__}.{path.stem}")

    _DISCOVERED = True


def get_checks(
    *,
    parallel: bool | None = None,
    agent_tool: bool | None = None,
    fallback: bool | None = None,
) -> list[CheckSpec]:
    """Return every registered check, optionally filtered by capability."""
    discover_checks()
    specs = list(_REGISTRY.values())

    if parallel is not None:
        specs = [spec for spec in specs if spec.parallel == parallel]
    if agent_tool is not None:
        specs = [spec for spec in specs if spec.agent_tool == agent_tool]
    if fallback is not None:
        specs = [spec for spec in specs if spec.fallback == fallback]

    return specs


def get_check(name: str) -> CheckSpec:
    """Return one registered check by name."""
    discover_checks()
    try:
        return _REGISTRY[name]
    except KeyError:
        registered = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown check: {name!r}. Registered checks: {registered}"
        ) from None


def info_note(check_name: str, message: str) -> Finding:
    """Build an informational note explaining why a check was skipped."""
    return Finding(
        check_name=check_name,
        severity=Severity.INFO,
        evidence=message,
        affected_columns=[],
        confidence=1.0,
    )


def error_finding(check_name: str, exc: BaseException) -> Finding:
    """Build a HIGH finding when a check raises unexpectedly."""
    return Finding(
        check_name=check_name,
        severity=Severity.HIGH,
        evidence=f"Check failed with error: {type(exc).__name__}: {exc}",
        affected_columns=[],
        confidence=0.0,
    )


def run_check(
    spec: CheckSpec,
    state: dict,
    run_fn: Callable[..., list[Finding]] | None = None,
) -> list[Finding]:
    """Run one check using parameters from graph state.

    Loads the CSV once centrally and applies the registry contract: missing
    inputs and unsupported target types become informational notes, CSV read
    failures become HIGH error findings, and optional params come from
    ``state["check_params"][spec.name]``. Unexpected check exceptions are
    left to the caller so the rest of the run survives.
    """
    csv_path = state.get("csv_path")
    if not csv_path or (isinstance(csv_path, str) and not csv_path.strip()):
        return [
            info_note(
                spec.name,
                "Skipped: required input 'csv_path' was not provided.",
            )
        ]

    # Load the CSV once, centrally, so every check does not read from disk.
    try:
        df = read_csv_robust(csv_path)
    except Exception as exc:
        return [error_finding(spec.name, exc)]

    kwargs: dict[str, object] = {
        # Every check now receives the pre-loaded DataFrame. A check that does
        # not need the target simply ignores it; one that cannot work without
        # it declares ``target_column`` in ``required``.
        "df": df,
        "target_column": state.get("target_column"),
    }

    # Validate remaining required inputs (csv_path already validated above).
    for key in spec.required:
        if key == "csv_path":
            continue
        value = state.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return [
                info_note(
                    spec.name,
                    f"Skipped: required input '{key}' was not provided.",
                )
            ]
        kwargs.setdefault(key, value)

    params = (state.get("check_params") or {}).get(spec.name) or {}
    for key in spec.optional:
        if key in params and params[key] is not None:
            kwargs[key] = params[key]

    if spec.target_types and "any" not in spec.target_types:
        profile = state.get("profile")
        # Only gate when the caller actually knows the target type (the graph
        # profile always does). Agent tools run without a profile; the check
        # itself still validates its inputs.
        if isinstance(profile, dict) and profile.get("target_type"):
            target_type = profile["target_type"]
            if target_type not in spec.target_types:
                allowed = ", ".join(sorted(spec.target_types))
                return [
                    info_note(
                        spec.name,
                        f"Skipped: target type '{target_type}' is not supported "
                        f"(supports: {allowed}).",
                    )
                ]

    # Semantic gating: a column is eligible only if its semantic type is
    # accepted AND it is not ambiguous. When accepted_semantic is {"any"}
    # the full df is passed unchanged (checks like duplicates and bias need
    # every column). Otherwise the df is filtered to eligible columns plus
    # the target column (kept so target-type gating still works), and one
    # informational note names the skipped columns, grouped by reason.
    skip_note: Finding | None = None
    if spec.accepted_semantic and "any" not in spec.accepted_semantic:
        profile = state.get("profile")
        lookup = get_semantic_lookup(profile) if isinstance(profile, dict) else {}
        target_column = state.get("target_column")

        if not lookup:
            # No semantic profile available: do not silently drop every
            # column. Run the check ungated and say so in one INFO note.
            skip_note = info_note(
                spec.name,
                f"Semantic profile unavailable for the {spec.name} check; "
                f"running ungated on all columns.",
            )
        else:
            eligible: list[str] = []
            wrong_type: list[str] = []
            ambiguous_type: list[str] = []
            for col in df.columns:
                sem = lookup.get(col, {}) or {}
                col_type = sem.get("type", "unknown")
                ambiguous = sem.get("ambiguous", True)
                if col == target_column:
                    # Always keep the target column for target-type gating.
                    eligible.append(col)
                elif ambiguous:
                    ambiguous_type.append(col)
                elif col_type not in spec.accepted_semantic:
                    wrong_type.append(col)
                else:
                    eligible.append(col)
            skipped = wrong_type + ambiguous_type
            if skipped:
                df = df[eligible]
                kwargs["df"] = df
                skip_note = info_note(
                    spec.name,
                    _skip_note_message(spec.name, wrong_type, ambiguous_type),
                )

    findings = (run_fn or spec.run)(**kwargs)
    if skip_note is not None:
        findings = [skip_note] + findings
    return findings


def _skip_note_message(
    spec_name: str, wrong_type: list[str], ambiguous_type: list[str]
) -> str:
    """Build the informational skip note grouping excluded columns by reason.

    Column names are capped with ``and N more`` so the note stays short.
    """
    parts: list[str] = []
    if wrong_type:
        parts.append(f"wrong type: {_cap_names(wrong_type)}")
    if ambiguous_type:
        parts.append(f"ambiguous type: {_cap_names(ambiguous_type)}")
    total = len(wrong_type) + len(ambiguous_type)
    return (
        f"Skipped {total} column(s) not eligible for the {spec_name} check: "
        + "; ".join(parts)
        + "."
    )


def _cap_names(cols: list[str], limit: int = 6) -> str:
    """Cap a column list with ``and N more`` when it exceeds the limit."""
    if len(cols) <= limit:
        return ", ".join(cols)
    return f"{', '.join(cols[:limit])} and {len(cols) - limit} more"


__all__ = [
    "CheckSpec",
    "register_check",
    "discover_checks",
    "get_checks",
    "get_check",
    "run_check",
    "info_note",
    "error_finding",
]
