"""Tests for the M4 ReAct agent graph (LLM mocked, no live model needed)."""

import json
import os
import tempfile

import pandas as pd
from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import MemorySaver

from src.graph.agent import run_checks_with_agent
from src.graph.builder import build_agent_graph
from src.graph.nodes import make_synthesize_findings_node
from src.schemas.finding import Finding, Severity
from src.schemas.report import AuditReport, SynthesisResult


def _write_csv(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.mkdtemp(), "sample.csv")
    df.to_csv(path, index=False)
    return path


class _FakeToolLLM:
    """Scripted tool-calling model: first call requests tools, then stops."""

    def __init__(self, csv_path: str, tool_calls: list[dict]):
        self.csv_path = csv_path
        self.tool_calls = tool_calls
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": {**args, "csv_path": self.csv_path},
                        "id": f"call_{i}",
                    }
                    for i, (name, args) in enumerate(self.tool_calls)
                ],
            )
        return AIMessage(content="Checks complete.")


_DEFAULT_SUMMARY = (
    "This dataset contains 4 rows and 2 columns, with column names a and b. "
    "Column a is numeric with missing values and basic statistics showing a "
    "min of 1.0, max of 2.0, mean of 1.3333, and std of 0.5774, while column "
    "b is categorical with 3 unique values including x, x, and y. The dataset "
    "has quality problems worth fixing: duplicate rows were detected, missing "
    "values appear in multiple columns, and the target distribution indicates "
    "a potential class imbalance issue that needs class weights or resampling "
    "during training. An ID-like column was identified that should not be used "
    "as a feature because it has a unique value for every row, which could "
    "lead to overfitting if included in a model. Concrete next steps include "
    "imputing or dropping missing values, removing exact duplicate rows, "
    "investigating the source of data gaps, applying class weights for the "
    "imbalanced target, and excluding identifier columns from the feature set "
    "before training any machine learning model."
)
_DEFAULT_PLAIN_TEXT = (
    "Plain text detailed summary. " * 6
)


class _FakeStructuredLLM:
    """Returns a canned structured summary."""

    def __init__(self, summary: str = _DEFAULT_SUMMARY):
        self.summary = summary

    def invoke(self, messages):
        return SynthesisResult(summary=self.summary)


class FakeAgentLLM:
    """Duck-typed LLM supporting both bind_tools and with_structured_output."""

    def __init__(
        self,
        csv_path: str,
        tool_calls: list[dict],
        structured_summary: str = _DEFAULT_SUMMARY,
        plain_text: str = _DEFAULT_PLAIN_TEXT,
    ):
        self.csv_path = csv_path
        self.tool_calls = tool_calls
        self.structured_summary = structured_summary
        self.plain_text = plain_text

    def bind_tools(self, tools):
        return _FakeToolLLM(self.csv_path, self.tool_calls)

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self.structured_summary)

    def invoke(self, messages):
        # Used by the plain-text fallback when JSON mode returns a heading.
        return AIMessage(content=self.plain_text)


class FailingLLM:
    """LLM that always fails (simulates Ollama not running)."""

    def bind_tools(self, tools):
        raise RuntimeError("no llm available")

    def with_structured_output(self, schema):
        raise RuntimeError("no llm available")


# ── direct agent run ──────────────────────────────────────────────────

def test_run_checks_with_agent_collects_findings():
    """Tool results are parsed back into Finding objects and deduplicated."""
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})
    csv = _write_csv(df)

    llm = FakeAgentLLM(
        csv,
        tool_calls=[("run_duplicates_check", {})],
    )
    findings, messages = run_checks_with_agent(
        csv_path=csv,
        target_column=None,
        profile={"n_rows": 4, "n_cols": 2},
        llm=llm,
    )

    assert any(f.check_name == "duplicates" for f in findings)
    assert len(messages) >= 1


def test_agent_calls_multiple_tools():
    """Multiple tool calls produce findings from several checks."""
    df = pd.DataFrame(
        {"a": [1, 1, 2, None], "b": ["x", "x", "y", "z"], "target": [0, 0, 1, 1]}
    )
    csv = _write_csv(df)

    llm = FakeAgentLLM(
        csv,
        tool_calls=[
            ("run_duplicates_check", {}),
            ("run_missing_check", {}),
        ],
    )
    findings, _ = run_checks_with_agent(
        csv_path=csv,
        target_column="target",
        profile={},
        llm=llm,
    )

    names = {f.check_name for f in findings}
    assert "duplicates" in names
    assert "missing_values" in names


# ── full agent graph ──────────────────────────────────────────────────

def test_agent_graph_end_to_end():
    """The M4 graph loads, profiles, runs the agent, summarizes, reports."""
    df = pd.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "y", "z"]})
    csv = _write_csv(df)

    llm = FakeAgentLLM(csv, tool_calls=[("run_duplicates_check", {})])
    graph = build_agent_graph(llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m4-e2e"}}
    final_state = graph.invoke({"csv_path": csv}, config=config)


    assert final_state.get("error") is None, final_state.get("error")
    report = final_state.get("report")
    report_text = final_state.get("report_text")
    assert report is not None
    # JSON is valid.
    json.loads(report)
    # Plain text is rendered from the same JSON object.
    assert report_text is not None
    assert "Summary:" in report_text
    assert "The dataset has quality problems worth fixing" in report_text
    assert final_state.get("overall_severity") == "high"
    assert any(f["check_name"] == "duplicates" for f in final_state.get("findings", []))


def test_agent_graph_missing_file_skips_agent():
    """A missing CSV skips the agent and still renders an error report."""
    llm = FakeAgentLLM("unused.csv", tool_calls=[])
    graph = build_agent_graph(llm=llm)
    final_state = graph.invoke({"csv_path": "does_not_exist.csv"})

    assert final_state.get("error") is not None
    assert final_state.get("report") is not None
    report_text = final_state.get("report_text")
    assert report_text is not None
    assert "Error:" in report_text


def test_agent_graph_falls_back_when_llm_fails():
    """A failing LLM falls back to deterministic checks instead of crashing."""
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})
    csv = _write_csv(df)

    graph = build_agent_graph(llm=FailingLLM(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m4-fallback"}}
    final_state = graph.invoke({"csv_path": csv}, config=config)


    assert final_state.get("error") is None, final_state.get("error")
    findings = final_state.get("findings", [])
    # Deterministic fallback still finds duplicates.
    assert any(f["check_name"] == "duplicates" for f in findings)
    # And records that the agent was unavailable.
    assert any(f["check_name"] == "agent" for f in findings)
    # Summary falls back to the highest severity rather than None.
    assert final_state.get("overall_severity") in {"high", "medium", "low", "info", "critical"}
    assert final_state.get("report") is not None


# ── config wiring ─────────────────────────────────────────────────────

def test_build_agent_graph_uses_injected_llm():
    """The builder accepts an injected LLM (tests never hit a live model)."""
    llm = FakeAgentLLM("x.csv", tool_calls=[])
    graph = build_agent_graph(llm=llm)
    assert graph is not None


def test_render_indents_multi_line_summary():
    """A detailed multi-line summary keeps its indentation on every line."""
    from src.graph.nodes import render_text_report

    report = render_text_report(
        AuditReport(
            dataset_id="test_ds",
            rows=4,
            cols=2,
            summary="Overview line.\n\n- First risk\n- Second risk",
        )
    )

    assert "  Overview line." in report
    assert "  - First risk" in report
    assert "  - Second risk" in report


def test_overall_severity_is_computed_in_python():
    """Severity comes from the findings, not from the LLM."""
    llm = FakeAgentLLM("x.csv", tool_calls=[])
    node = make_synthesize_findings_node(llm)

    finding = Finding(
        check_name="leakage",
        severity=Severity.CRITICAL,
        evidence="test evidence",
        affected_columns=["col"],
        confidence=0.95,
    )
    out = node({"findings": [finding.model_dump(mode="json")], "profile": {}})

    assert out["overall_severity"] == "critical"


def test_heading_only_summary_falls_back_to_plain_text():
    """A heading-only JSON summary is replaced by a plain-text answer."""
    llm = FakeAgentLLM("x.csv", tool_calls=[], structured_summary="Overview")
    node = make_synthesize_findings_node(llm)

    out = node({"findings": [], "profile": {}})

    assert len(out["summary"]) >= 80
    assert out["summary"].startswith("(Automatic summary - LLM output unavailable: reply too short)")


def test_agent_zero_findings_falls_back_to_deterministic():
    """If the agent calls no tools, deterministic checks still run."""
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})
    csv = _write_csv(df)

    llm = FakeAgentLLM(csv, tool_calls=[])
    graph = build_agent_graph(llm=llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-m4-zero"}}
    final_state = graph.invoke({"csv_path": csv}, config=config)


    assert any(
        f["check_name"] == "duplicates" for f in final_state.get("findings", [])
    )


# ── Step 5: synthesis prompt requirements ─────────────────────────────────

class _CapturingLLM:
    """LLM that captures the system message and returns a canned summary."""

    def __init__(self, summary: str = "Summary bullet one.\nSummary bullet two."):
        self.captured_system: str | None = None
        self.summary = summary

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        for m in messages:
            if m.__class__.__name__ == "SystemMessage":
                self.captured_system = m.content
        from src.schemas.report import SynthesisResult

        return SynthesisResult(summary=self.summary)


def test_synthesis_prompt_requires_prioritized_bullets():
    """The synthesis prompt instructs 3-5 prioritized bullets, no verbatim text,
    and a dataset description overview section."""
    llm = _CapturingLLM()
    node = make_synthesize_findings_node(llm)

    findings = [
        {
            "check_name": "bias",
            "severity": "critical",
            "evidence": "Disparate impact 0.358 for sex groups",
            "affected_columns": ["sex"],
            "confidence": 0.9,
        }
    ]
    node({"findings": findings, "profile": {"n_rows": 100, "n_cols": 10}})

    system = llm.captured_system or ""
    assert "3 to 5 bullets" in system
    assert "most severe first" in system
    assert "Do not paste the evidence text" in system
    assert "Use your own words" in system
    assert "Overview:" in system
    assert "Key issues:" in system


def test_synthesis_prompt_mentions_bias_only_when_bias_finding():
    """The synthesis prompt constrains bias mentions to bias findings."""
    llm = _CapturingLLM()
    node = make_synthesize_findings_node(llm)

    findings = [
        {
            "check_name": "missing_values",
            "severity": "low",
            "evidence": "Column 'age' has 1 missing value",
            "affected_columns": ["age"],
            "confidence": 1.0,
        }
    ]
    node({"findings": findings, "profile": {"n_rows": 100, "n_cols": 10}})

    system = llm.captured_system or ""
    assert "Talk about bias or fairness only if a finding has the check name 'bias'" in system
    assert "Never call a column sensitive or protected yourself" in system


__all__ = [
    "test_run_checks_with_agent_collects_findings",
    "test_agent_calls_multiple_tools",
    "test_agent_graph_end_to_end",
    "test_agent_graph_missing_file_skips_agent",
    "test_agent_graph_falls_back_when_llm_fails",
    "test_build_agent_graph_uses_injected_llm",
    "test_render_indents_multi_line_summary",
    "test_overall_severity_is_computed_in_python",
    "test_heading_only_summary_falls_back_to_plain_text",
    "test_agent_zero_findings_falls_back_to_deterministic",
    "test_synthesis_prompt_requires_prioritized_bullets",
    "test_synthesis_prompt_mentions_bias_only_when_bias_finding",
]
