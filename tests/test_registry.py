"""Contract tests for the check registry.

Validates that exactly 8 checks are registered (the parallel M2/M3 fan-out:
bias, duplicates, missing_values, schema, class_imbalance, id_like,
skewness, outliers), that every check returns valid findings, and that the
registry wiring (graph fan-out, agent tools) stays in sync.
"""

import pytest

import os
import tempfile

import pandas as pd

from src.schemas.finding import Finding, Severity
from src.tools.checks import CheckSpec, get_checks, run_check
from src.tools.profiling import profile_dataset

EXPECTED_CHECK_NAMES = {
    "bias",
    "duplicates",
    "missing_values",
    "schema",
    "class_imbalance",
    "id_like",
    "skewness",
    "outliers",
}


def _write_csv(df, name="registry.csv"):
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), name)
    df.to_csv(path, index=False)
    return path


def _state(csv_path, profile):
    return {
        "csv_path": csv_path,
        "target_column": "target",
        "profile": profile,
        "check_params": {},
    }


def _assert_valid_finding(finding: Finding) -> None:
    assert isinstance(finding, Finding)
    assert finding.check_name
    assert finding.severity in set(Severity)
    assert isinstance(finding.evidence, str) and finding.evidence.strip()
    assert isinstance(finding.affected_columns, list)
    assert all(isinstance(col, str) for col in finding.affected_columns)
    assert 0.0 <= finding.confidence <= 1.0


# ── registry size and names ─────────────────────────────────────────────

def test_registry_has_exactly_eight_checks():
    """The parallel check fan-out registers exactly 8 checks."""
    specs = get_checks(parallel=True)
    names = {spec.name for spec in specs}
    assert names == EXPECTED_CHECK_NAMES, f"got {sorted(names)}"


def test_registry_names_and_tool_names_are_unique():
    """Check names and generated tool names must not collide."""
    specs = get_checks()
    assert len({spec.name for spec in specs}) == len(specs)
    assert len({spec.agent_tool_name for spec in specs}) == len(specs)


def test_builder_checks_match_registry():
    """The graph fan-out is exactly the parallel checks in the registry."""
    from src.graph.builder import CHECKS

    parallel = get_checks(parallel=True)
    assert {name for name, _ in CHECKS} == {spec.name for spec in parallel}
    assert len(CHECKS) == len(parallel)


def test_agent_tools_match_registry():
    """The agent tools are exactly the registry's agent-tool checks."""
    from src.graph.tools import ALL_TOOLS

    expected = {spec.agent_tool_name for spec in get_checks(agent_tool=True)}
    assert {tool.name for tool in ALL_TOOLS} == expected


# ── per-check contract ──────────────────────────────────────────────────

def _make_dataset():
    df = pd.DataFrame(
        {
            "id": list(range(24)),
            "target": [i % 2 for i in range(24)],
            "leak": [i % 2 for i in range(24)],
            "num": [i % 5 for i in range(24)],
            "cat": ["a", "b", "c"] * 8,
            "missing": [i if i % 4 else None for i in range(24)],
            "text_number": ["1", "2"] * 12,
        }
    )
    csv_path = _write_csv(df)
    return csv_path, profile_dataset(csv_path, target_column="target")


@pytest.mark.parametrize("spec", get_checks(parallel=True), ids=lambda spec: spec.name)
def test_check_returns_valid_findings(spec: CheckSpec):
    """Every parallel check returns a list of valid Finding objects."""
    csv_path, profile = _make_dataset()
    findings = run_check(spec, _state(csv_path, profile))
    assert isinstance(findings, list)
    for finding in findings:
        _assert_valid_finding(finding)


@pytest.mark.parametrize("spec", get_checks(parallel=True), ids=lambda spec: spec.name)
def test_check_never_crashes_on_tiny_dataframe(spec: CheckSpec):
    """A tiny dataframe must never crash a check."""
    df = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0], "cat": ["a", "b", "a"]})
    csv_path = _write_csv(df, "tiny.csv")
    profile = profile_dataset(csv_path, target_column="target")
    findings = run_check(spec, _state(csv_path, profile))
    assert isinstance(findings, list)
    for finding in findings:
        _assert_valid_finding(finding)


@pytest.mark.parametrize("spec", get_checks(parallel=True), ids=lambda spec: spec.name)
def test_missing_required_input_is_one_info_note(spec: CheckSpec):
    """A missing required input becomes one informational note, not a crash."""
    if not spec.required:
        return  # skip checks with no required inputs
    csv_path, profile = _make_dataset()
    state = _state(csv_path, profile)
    state.pop(spec.required[0])
    findings = run_check(spec, state)
    assert len(findings) == 1
    assert findings[0].check_name == spec.name
    assert findings[0].severity == Severity.INFO
    assert "required input" in findings[0].evidence


@pytest.mark.parametrize("spec", get_checks(parallel=True), ids=lambda spec: spec.name)
def test_unsupported_target_type_skips_cleanly(spec: CheckSpec):
    """A check with target-type limits skips cleanly on an unsupported type."""
    if "any" in spec.target_types:
        return  # skip checks that accept every target type
    csv_path, profile = _make_dataset()
    state = _state(csv_path, profile)
    state["profile"] = {**profile, "target_type": "unsupported-type"}
    findings = run_check(spec, state)
    assert len(findings) == 1
    assert findings[0].check_name == spec.name
    assert findings[0].severity == Severity.INFO
    assert "not supported" in findings[0].evidence