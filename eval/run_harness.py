"""Multi-dataset evaluation harness for Data Detective.

Audits each dataset in ``eval/datasets.yaml`` and checks invariants: no
crash, no duplicate findings, valid finding fields/severities, deterministic
repeat runs, and bias skipped when no sensitive columns are given.

Usage: python -m eval.run_harness
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import yaml

from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph


def _finding_key(finding: dict[str, Any]) -> tuple:
    """Return a hashable key for a finding.

    Two findings are the same when they share check_name, affected columns
    (order-insensitive), and evidence.
    """
    cols = tuple(sorted(finding.get("affected_columns", [])))
    return (
        str(finding.get("check_name", "")),
        cols,
        str(finding.get("evidence", "")),
    )


def _find_duplicates(findings: list[dict[str, Any]]) -> dict[tuple, int]:
    """Return {key: count} for findings that appear more than once."""
    counts: dict[tuple, int] = {}
    for f in findings:
        key = _finding_key(f)
        counts[key] = counts.get(key, 0) + 1
    return {key: count for key, count in counts.items() if count > 1}


def _check_findings_valid(findings: list[dict[str, Any]]) -> list[str]:
    """Return error strings for invalid findings.

    A valid finding has non-empty check_name, a known severity, non-empty
    evidence, affected_columns as a list of strings, and confidence in [0,1].
    """
    valid_severities = {"info", "low", "medium", "high", "critical"}
    errors: list[str] = []
    for i, f in enumerate(findings):
        prefix = f"finding[{i}]"
        if not isinstance(f.get("check_name"), str) or not f["check_name"].strip():
            errors.append(f"{prefix}: missing check_name")
        sev = f.get("severity")
        if sev not in valid_severities:
            errors.append(f"{prefix}: invalid severity {sev!r}")
        if not isinstance(f.get("evidence"), str) or not f["evidence"].strip():
            errors.append(f"{prefix}: missing evidence")
        cols = f.get("affected_columns")
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            errors.append(f"{prefix}: affected_columns must be a list of strings")
        conf = f.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            errors.append(f"{prefix}: confidence out of range: {conf!r}")
    return errors


def _audit_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    """Audit one dataset and return invariant-check results.

    ``spec`` holds ``name``, ``path``, ``target``, ``sensitive``, and
    ``favorable``. The result has ``passed``, ``errors``, ``findings_count``,
    and ``check_names``.
    """
    errors: list[str] = []
    name = spec.get("name", "unknown")
    path = spec.get("path", "")
    target = spec.get("target")
    sensitive = spec.get("sensitive", []) or []
    favorable = spec.get("favorable")

    if not path or not os.path.exists(path):
        return {
            "passed": False,
            "errors": [f"{name}: dataset path not found: {path}"],
            "findings_count": 0,
            "check_names": [],
        }

    # Bias check only runs when sensitive columns are given.
    check_params: dict[str, dict] = {}
    if sensitive:
        check_params["bias"] = {"sensitive_columns": sensitive}
        if favorable is not None:
            check_params["bias"]["favorable_value"] = favorable

    # Two identical runs verify determinism.
    results: list[dict[str, Any]] = []
    for run in range(2):
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": f"{name}-run-{run}"}}
        try:
            state = graph.invoke(
                {
                    "csv_path": path,
                    "target_column": target,
                    "check_params": check_params,
                },
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 — harness must not crash
            errors.append(f"{name}: graph raised {type(exc).__name__}: {exc}")
            return {
                "passed": False,
                "errors": errors,
                "findings_count": 0,
                "check_names": [],
            }
        results.append(state)

    findings_a = results[0].get("findings") or []
    findings_b = results[1].get("findings") or []
    keys_a = sorted(_finding_key(f) for f in findings_a)
    keys_b = sorted(_finding_key(f) for f in findings_b)
    if keys_a != keys_b:
        errors.append(f"{name}: findings are not deterministic across runs")

    findings = findings_a

    validity_errors = _check_findings_valid(findings)
    for err in validity_errors:
        errors.append(f"{name}: {err}")

    dupes = _find_duplicates(findings)
    for key, count in dupes.items():
        errors.append(f"{name}: duplicate finding {key} (x{count})")

    check_names = sorted({f.get("check_name", "") for f in findings})

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "findings_count": len(findings),
        "check_names": check_names,
    }


def run_all(dataset_path: str = "eval/datasets.yaml") -> dict[str, Any]:
    """Audit every dataset in ``dataset_path`` and summarize results."""
    with open(dataset_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    datasets = data.get("datasets", [])
    per_dataset: dict[str, dict[str, Any]] = {}
    overall_passed = True
    total_findings = 0

    for ds in datasets:
        result = _audit_dataset(ds)
        per_dataset[ds["name"]] = result
        if not result["passed"]:
            overall_passed = False
        total_findings += result["findings_count"]

    return {
        "passed": overall_passed,
        "datasets": per_dataset,
        "total_findings": total_findings,
    }


if __name__ == "__main__":
    summary = run_all()
    print(f"Overall: {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"Total findings: {summary['total_findings']}")
    for name, result in summary["datasets"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  {name}: {status} "
            f"(findings={result['findings_count']}, "
            f"checks={result['check_names']})"
        )
        for err in result["errors"]:
            print(f"    - {err}")
