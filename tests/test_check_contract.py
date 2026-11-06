"""Contract tests for every registered check.

The parametrization loops over ``get_checks()``, so a newly added check is
covered automatically without editing this file.

The contract is simple:

1. A check returns ``list[Finding]`` with valid fields.
2. A missing required input becomes one informational note.
3. An unsupported target type becomes one informational note.
4. A tiny dataframe never crashes the audit.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from src.schemas.finding import Finding, Severity
from src.tools.checks import CheckSpec, get_checks, run_check
from src.tools.profiling import profile_dataset


def _write_csv(df: pd.DataFrame, name: str = "contract.csv") -> str:
    path = os.path.join(tempfile.mkdtemp(), name)
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def dataset():
    """A small dataset that exercises every built-in check without crashing."""
    n_rows = 24
    df = pd.DataFrame(
        {
            "id": list(range(n_rows)),
            "target": [i % 2 for i in range(n_rows)],
            "leak": [i % 2 for i in range(n_rows)],
            "num": [i % 5 for i in range(n_rows)],
            "cat": ["a", "b", "c"] * 8,
            "missing": [i if i % 4 else None for i in range(n_rows)],
            "text_number": ["1", "2"] * 12,
        }
    )
    csv_path = _write_csv(df)
    return csv_path, profile_dataset(csv_path, target_column="target")


@pytest.fixture(scope="module")
def tiny_dataset():
    """The smallest sane dataframe: three rows, three columns."""
    df = pd.DataFrame(
        {"id": [1, 2, 3], "target": [0, 1, 0], "cat": ["a", "b", "a"]}
    )
    csv_path = _write_csv(df, "tiny.csv")
    return csv_path, profile_dataset(csv_path, target_column="target")


def _state(csv_path: str, profile: dict) -> dict:
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


# ── per-check contract ────────────────────────────────────────────────

@pytest.mark.parametrize("spec", get_checks(), ids=lambda spec: spec.name)
def test_check_returns_valid_findings(spec: CheckSpec, dataset):
    """Every check returns a list of valid Finding objects."""
    csv_path, profile = dataset

    findings = run_check(spec, _state(csv_path, profile))

    assert isinstance(findings, list)
    for finding in findings:
        _assert_valid_finding(finding)


@pytest.mark.parametrize("spec", get_checks(), ids=lambda spec: spec.name)
def test_check_never_crashes_on_tiny_dataframe(spec: CheckSpec, tiny_dataset):
    """A tiny dataframe must never crash a check."""
    csv_path, profile = tiny_dataset

    findings = run_check(spec, _state(csv_path, profile))

    assert isinstance(findings, list)
    for finding in findings:
        _assert_valid_finding(finding)


@pytest.mark.parametrize("spec", get_checks(), ids=lambda spec: spec.name)
def test_missing_required_input_is_one_info_note(spec: CheckSpec, dataset):
    """A missing required input becomes one informational note, not a crash."""
    csv_path, profile = dataset
    if not spec.required:
        pytest.skip("check declares no required inputs")

    state = _state(csv_path, profile)
    state.pop(spec.required[0])

    findings = run_check(spec, state)

    assert len(findings) == 1
    assert findings[0].check_name == spec.name
    assert findings[0].severity == Severity.INFO
    assert "required input" in findings[0].evidence


@pytest.mark.parametrize("spec", get_checks(), ids=lambda spec: spec.name)
def test_unsupported_target_type_skips_cleanly(spec: CheckSpec, dataset):
    """A check with target-type limits skips cleanly on an unsupported type."""
    if "any" in spec.target_types:
        pytest.skip("check supports every target type")

    csv_path, profile = dataset
    state = _state(csv_path, profile)
    state["profile"] = {**profile, "target_type": "unsupported-type"}

    findings = run_check(spec, state)

    assert len(findings) == 1
    assert findings[0].check_name == spec.name
    assert findings[0].severity == Severity.INFO
    assert "not supported" in findings[0].evidence


# ── registry wiring ───────────────────────────────────────────────────

def test_target_type_gating_mechanism(dataset):
    """The registry itself skips a restricted check on an unsupported type."""
    csv_path, profile = dataset

    def _must_not_run(**kwargs):
        raise AssertionError("restricted check should not run")

    spec = CheckSpec(
        name="synthetic",
        run=_must_not_run,
        target_types=frozenset({"binary"}),
    )
    state = _state(csv_path, profile)
    state["profile"] = {**profile, "target_type": "continuous"}

    findings = run_check(spec, state)

    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


def test_registry_names_and_tool_names_are_unique():
    """Check names and generated tool names must not collide."""
    specs = get_checks()

    assert len({spec.name for spec in specs}) == len(specs)
    assert len({spec.agent_tool_name for spec in specs}) == len(specs)


def test_builder_checks_are_generated_from_registry():
    """The graph fan-out is exactly the parallel checks in the registry."""
    from src.graph.builder import CHECKS

    parallel = get_checks(parallel=True)

    assert {name for name, _ in CHECKS} == {spec.name for spec in parallel}
    assert len(CHECKS) == len(parallel)


def test_agent_tools_are_generated_from_registry():
    """The agent tools are exactly the registry's agent-tool checks."""
    from src.graph.tools import ALL_TOOLS

    expected = {spec.agent_tool_name for spec in get_checks(agent_tool=True)}

    assert {tool.name for tool in ALL_TOOLS} == expected


__all__ = [
    "test_check_returns_valid_findings",
    "test_check_never_crashes_on_tiny_dataframe",
    "test_missing_required_input_is_one_info_note",
    "test_unsupported_target_type_skips_cleanly",
    "test_target_type_gating_mechanism",
    "test_registry_names_and_tool_names_are_unique",
    "test_builder_checks_are_generated_from_registry",
    "test_agent_tools_are_generated_from_registry",
]
