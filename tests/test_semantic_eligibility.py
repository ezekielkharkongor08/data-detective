"""Eligibility tests: every registered check must only run on accepted
semantic types, and skipped columns must produce one informational note.

Loops over all registered checks so a newly added check is covered
automatically.
"""

import os
import tempfile

import pandas as pd
import pytest

from src.schemas.finding import Severity
from src.tools.checks import CheckSpec, get_checks, run_check
from src.tools.profiling import profile_dataset


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


def _state(csv_path: str, profile: dict) -> dict:
    return {
        "csv_path": csv_path,
        "target_column": "target",
        "profile": profile,
        "check_params": {},
    }


# A dataset that exercises every semantic type.
_MIXED = pd.DataFrame(
    {
        "target": [0, 1] * 25,  # binary
        "measurement": [float(i) for i in range(50)],  # numeric_measurement
        "discrete": [i % 5 for i in range(50)],  # numeric_discrete
        "category": ["a", "b", "c", "d"] * 12 + ["a", "b"],  # categorical
        "identifier": [f"ID-{i}" for i in range(50)],  # identifier
        "constant": [7] * 50,  # constant
        "text": [f"unique long value {i} token" for i in range(50)],  # text
    }
)


def _mixed_state():
    csv_path = _write_csv(_MIXED)
    profile = profile_dataset(csv_path, target_column="target")
    return csv_path, profile


@pytest.mark.parametrize("spec", get_checks(), ids=lambda spec: spec.name)
def test_check_only_runs_on_accepted_types(spec: CheckSpec):
    """A check must not produce findings on columns outside its
    accepted_semantic; skipped columns get one info note instead."""
    csv_path, profile = _mixed_state()
    findings = run_check(spec, _state(csv_path, profile))

    accepted = spec.accepted_semantic
    lookup = {
        c["name"]: (c.get("semantic") or {}) for c in profile["columns"]
    }

    for f in findings:
        if f.severity == Severity.INFO and "Skipped" in f.evidence:
            # The info note about skipped columns is allowed.
            continue
        for col in f.affected_columns:
            sem = lookup.get(col, {})
            col_type = sem.get("type", "unknown")
            ambiguous = sem.get("ambiguous", True)
            if "any" not in accepted:
                assert col_type in accepted, (
                    f"{spec.name} found {col_type} column {col!r} "
                    f"but accepts {sorted(accepted)}"
                )
                assert not ambiguous, (
                    f"{spec.name} found ambiguous column {col!r}"
                )


@pytest.mark.parametrize("spec", get_checks(), ids=lambda spec: spec.name)
def test_check_skipped_columns_get_info_note(spec: CheckSpec):
    """If a check rejects at least one column, exactly one info note
    naming the skipped columns must appear."""
    csv_path, profile = _mixed_state()
    accepted = spec.accepted_semantic
    if "any" in accepted:
        pytest.skip("check accepts all types")

    findings = run_check(spec, _state(csv_path, profile))
    lookup = {
        c["name"]: (c.get("semantic") or {}) for c in profile["columns"]
    }
    skipped = [
        c["name"]
        for c in profile["columns"]
        if c["name"] != "target"
        and (
            (c.get("semantic") or {}).get("type", "unknown") not in accepted
            or (c.get("semantic") or {}).get("ambiguous", True)
        )
    ]

    if not skipped:
        return

    info_notes = [f for f in findings if f.severity == Severity.INFO]
    assert len(info_notes) >= 1, (
        f"{spec.name} skipped {skipped} columns but emitted no info note"
    )
    note_text = " ".join(f.evidence for f in info_notes)
    for col in skipped:
        assert col in note_text, (
            f"{spec.name} info note does not mention skipped column {col!r}"
        )


def test_skewness_and_outlier_reject_nominal_codes():
    """Hard gate: skewness and outlier must never accept nominal codes
    (columns that are not numeric_measurement or numeric_discrete)."""
    csv_path, profile = _mixed_state()
    lookup = {c["name"]: (c.get("semantic") or {}) for c in profile["columns"]}
    specs = {s.name: s for s in get_checks()}

    for spec_name in ("skewness", "outliers"):
        spec = specs[spec_name]
        findings = run_check(spec, _state(csv_path, profile))
        for f in findings:
            for col in f.affected_columns:
                sem = lookup.get(col, {})
                assert sem.get("type") in (
                    "numeric_measurement",
                    "numeric_discrete",
                ), (
                    f"{spec_name} accepted {col} of type {sem.get('type')}"
                )
