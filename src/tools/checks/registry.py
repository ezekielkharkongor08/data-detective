"""Check registry: the single place a check is declared.

Adding a check is one file. Create ``src/tools/checks/my_check.py``, write a
function that returns ``list[Finding]``, and decorate it with
``@register_check(...)``. The graph builder, the parallel fan-out, the agent
tools, the LLM prompt, and the contract test all read this registry, so no
other file needs editing.

Metadata a check declares
-------------------------
``name``
    Stable identifier. Used in findings, ``check_params`` keys, and skip notes.
``run``
    The function that returns ``list[Finding]`` (supplied by the decorator).
``required``
    State keys that must be present before the check runs (for example
    ``csv_path`` or ``target_column``). A missing required key skips the check
    with one informational note.
``optional``
    Param names read from ``state["check_params"][name]`` (for example
    ``sensitive_columns`` or ``favorable_value``).
``target_types``
    Target types the check supports. ``{"any"}`` disables gating.
``severity_key``
    Key into the severity-threshold config (kept for the thresholds step).
``description``
    One line used to generate the agent tool and the LLM prompt.
``always``
    Tell the agent to call this check on every dataset (used to build the
    LLM prompt). Checks that only need ``csv_path`` usually set this.
``target_hint``
    Tell the agent to call this check when a target column is provided.
    Used only to build the LLM prompt guidance.
``param_types``
    Optional ``(name, python_type)`` pairs used to build the generated agent
    tool's argument schema. Unknown params default to ``str``.
``parallel``
    Include in the M2 parallel fan-out.
``agent_tool``
    Expose to the ReAct agent.
``fallback``
    Include in the deterministic fallback used when the LLM is unavailable.
``tool_name``
    Optional override for the generated agent tool name.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from src.schemas.finding import Finding, Severity
from src.tools.csv_reader import read_csv_robust


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

    The CSV is loaded exactly once here and passed to the check as ``df``,
    so individual checks no longer read from disk and no longer duplicate
    the CSV-load try/except boilerplate.

    The function applies the registry contract in one place:

    1. A missing ``csv_path`` becomes one informational note.
    2. A CSV read failure becomes one HIGH error finding.
    3. A missing other required input becomes one informational note.
    4. Optional params come from ``state["check_params"][spec.name]``.
    5. An unsupported target type becomes one informational note.

    Unexpected exceptions from the check itself are deliberately left to the
    caller so a graph node can convert them into an error finding without
    losing the rest of the run.

    Args:
        spec: The check to run.
        state: Graph state (or a minimal dict in tests).
        run_fn: Optional override for the function to call. Used by tests that
            patch a check without mutating the frozen spec.

    Returns:
        The findings from the check, or a single informational/error note.
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

    return (run_fn or spec.run)(**kwargs)


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
