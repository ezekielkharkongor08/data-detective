# Data Detective — Codebase Audit Report

**Project:** Data Detective (LangGraph CSV audit agent)  
**Audit date:** Thu Nov 05 2026  
**Scope:** Full source tree (excluding virtualenv `myenv/`, `__pycache__`, and `.git`)

---

## 1. Executive Summary

The project is a functional, milestone-driven LangGraph agent that audits CSVs for data quality (missing values, duplicates, imbalance, ID-like columns, schema issues, bias, and target leakage). It is well-structured with a clean registry-driven design, good test coverage (17 test files), and a separated persistence layer (LangGraph checkpoints + SQLAlchemy business tables). It has progressed through milestones M0–M6 conceptually, with M7 (FastAPI/Streamlit) only stubbed.

**Key observation:** The code is cohesive and follows its own design rules (LLM never sees the full CSV; state never holds a DataFrame; every finding is a structured `Finding` object). The main risks are operational/infra-related rather than architectural: the `.env` file contains a real password, `tests/` is listed in `.gitignore` (excluding the test suite from version control), the CSV is read from disk once per check, and several files are stubbed or orphaned.

---

## 2. File Tree & Architecture

### Root level

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point: parses args, builds the graph with PostgresSaver, invokes it, persists results to business tables, prints the report (text/JSON/both), supports `--list` and `--thread-id`. |
| `requirements.txt` | Full dependency list (pandas, sklearn, langgraph, langchain-ollama, fastapi, streamlit, psycopg, langgraph-checkpoint-postgres, pytest, etc.). |
| `pyproject.toml` | Minimal packaging config; `dependencies = []` (inconsistent with requirements.txt). |
| `README.md` | Empty (0 lines). |
| `AGENTS.md` | AI-assistant working rules (the "how" document). |
| `DATA_DETECTIVE_PROJECT_BRIEF.md` | Project brief/source of truth (the "what" document): milestones, design principles, planned checks, stack. |
| `.env` | Local config with **actual credentials** (Postgres user/password/database URL). Should not be committed. |
| `.env.example` | Template for `.env` (no real secrets). |
| `.gitignore` | Ignores Python bytecode, venv, `.env`, and **`tests/`** (see technical debt). |

### `src/`

| File | Purpose |
|------|---------|
| `src/__init__.py` | Package marker. |
| `src/config.py` | Pydantic-settings `Settings` class reading `.env`/env vars; builds `DATABASE_URL` and `postgres_conn_string`; cached via `lru_cache`. |
| `src/check_config.py` | Shared frozen dataclass thresholds (`CheckConfig`) for all checks: bias, leakage, imbalance, schema, missing, duplicates, target_type. Module-level singleton `check_config`. |
| `src/schemas/finding.py` | `Finding` Pydantic model + `Severity` enum (info/low/medium/high/critical). `Finding.__lt__` sorts highest severity first. |
| `src/schemas/report.py` | `SynthesisResult` (LLM structured output) and `AuditReport` (single source of truth, serialized to JSON + rendered to text). |
| `src/schemas/__init__.py` | Package marker. |
| `src/api/__init__.py` | Stub only (`"""FastAPI app (M7)."""`). FastAPI is in requirements but not implemented. |
| `src/db/connection.py` | SQLAlchemy engine, `SessionLocal`, `get_db()` generator, `init_db()`. Reads URL from `.env`. |
| `src/db/models.py` | ORM models: `Dataset`, `AuditRun`, `Finding`, `Report` (PostgreSQL JSONB columns). Separate from LangGraph checkpoints. |
| `src/db/repository.py` | CRUD: `ensure_schema`, `list_runs`, `get_run`, `save_run`, `register_dataset`. `save_run` upserts findings + report. |
| `src/db/__init__.py` | Re-exports connection/models/repository symbols. |
| `src/graph/state.py` | `GraphState` TypedDict: IDs, paths, profile, check_params, findings (with `merge_findings` reducer), leakage flags, deep_check_results, summary, overall_severity, report, report_text, error. |
| `src/graph/reducers.py` | Idempotent `merge_findings` reducer + dedupe key (check_name + sorted columns + SHA-256 of evidence). Prevents duplicate findings on re-invoke. |
| `src/graph/nodes.py` | All graph nodes: `load_csv_node`, `profile_node`, `merge_findings_node`, `router_node`, `deep_checks_node`, `make_agent_checks_node`, `make_synthesize_findings_node`, `make_check_node`, `render_text_report`, `_run_all_checks_deterministically`. Also holds the LLM synthesis prompt and fallback logic. |
| `src/graph/builder.py` | `build_graph()` (M2/M3 deterministic graph) and `build_agent_graph()` (M4 ReAct graph). `CHECKS` list built from registry. |
| `src/graph/llm.py` | LLM factory: `get_llm()` (ChatOllama), `get_structured_llm()` (unused), `get_cached_llm()` (lru_cache). |
| `src/graph/agent.py` | ReAct subgraph (`build_check_agent`) + `run_checks_with_agent()`. LLM sees only compact profile, chooses tools. `_collect_findings()` dedupes tool results. |
| `src/graph/tools.py` | Dynamically generates `StructuredTool`s from the check registry (`ALL_TOOLS`). |
| `src/graph/__init__.py` | Exports `build_graph`, `build_agent_graph`, `GraphState`. |
| `src/tools/csv_reader.py` | `read_csv_robust()`: encoding fallback (utf-8, utf-8-sig, latin-1, cp1252) + delimiter detection (comma, tab, semicolon, pipe). Used by every check and the profiler. |
| `src/tools/profiling.py` | `profile_dataset()`: compact summary (rows, cols, per-column dtype/missing/stats, duplicate_rows, target_type, sample_rows). No LLM. |
| `src/tools/target_type.py` | `detect_target_type()`: binary/multiclass/continuous/boolean/unknown. |
| `src/tools/missing_markers.py` | `count_missing()` + `MISSING_MARKERS` frozenset (shared definition of "missing" across profiling and checks). Configurable via env var. |
| `src/tools/__init__.py` | Package marker. |
| `src/tools/checks/registry.py` | `CheckSpec` dataclass, `register_check()` decorator, `discover_checks()` (auto-imports all `*.py`), `get_checks()`, `get_check()`, `run_check()` (applies contract: required inputs, optional params, target-type gating), `info_note()`, `error_finding()`. |
| `src/tools/checks/schema_check.py` | Check `schema`: flags text columns that are mostly numeric (MEDIUM). `always=True`. |
| `src/tools/checks/missing_check.py` | Check `missing_values`: flags NaN + text markers per column (LOW/HIGH/CRITICAL). `always=True`, custom `tool_name=run_missing_check`. |
| `src/tools/checks/duplicates_check.py` | Check `duplicates`: flags exact duplicate rows (MEDIUM/HIGH >10%). `always=True`. |
| `src/tools/checks/imbalance_check.py` | Check `class_imbalance`: flags skewed target (MEDIUM <30%, HIGH <10%). `target_hint=True`, requires `target_column`. |
| `src/tools/checks/id_like_check.py` | Check `id_like`: flags unique-per-row non-target columns (MEDIUM). `target_hint=True`. |
| `src/tools/checks/bias_check.py` | Check `bias`: disparate impact + chi-square across user-specified sensitive groups (CRITICAL/HIGH/MEDIUM). `fallback=False`, requires `target_column`, target_types binary/boolean. |
| `src/tools/checks/leakage_check.py` | Check `target_leakage`: per-column CV model (ROC AUC / R²) to detect leakage (CRITICAL/HIGH/MEDIUM). `parallel=False`, `target_hint=True`, requires `target_column`. |
| `src/tools/checks/__init__.py` | Imports all check functions + registry helpers (triggers registration on import). |

### `eval/`

| File | Purpose |
|------|---------|
| `eval/run_harness.py` | Multi-dataset harness (MemorySaver). Audits each dataset in `datasets.yaml` and checks invariants: no crash, no duplicate findings, valid fields/severities, deterministic repeat runs, single invoke, bias skipped without `--sensitive`. |
| `eval/datasets.yaml` | Dataset registry: `adult` (target income, sensitive sex/race, favorable >50K), `telco` (target Churn, sensitive gender, favorable Yes), `bank_loan` (target loan_approved, sensitive gender/race, favorable 1). |
| `eval/__init__.py` | Package marker (says M8). |

### `tests/` (17 files)

Unit + graph-level tests for every check, the registry contract, reducers, profiling, missing markers, CSV reader, config, db repository, M1–M4 graphs, M6 agent, and the eval harness. Each test writes its own temp CSV.

### `data/raw/` (read-only CSVs)

| File | Notes |
|------|-------|
| `adult.csv` | Adult Census Income (used in harness). |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Telco churn (used in harness). |
| `bank_loan.csv` | Bank loan (used in harness). |
| `f1_race_results.csv` | Present but **not referenced** in `datasets.yaml` or any code. |
| `loan_approval_dataset.csv` | Present but **not referenced** in `datasets.yaml` or any code. |

### `docs/`

| File | Purpose |
|------|---------|
| `docs/ADDING_A_CHECK.md` | Contributor guide for adding checks via the registry. |
| `docs/check_template.py` | Template check (lives in `docs/` so it is never auto-discovered/registered). |

---

## 3. LangGraph Pipeline

There are **two compiled graphs**, both using `GraphState` as the state schema.

### State schema — `GraphState` (`src/graph/state.py`)

A `TypedDict(total=False)` with these fields:

| Field | Type | Purpose |
|-------|------|---------|
| `dataset_id` | str | Derived from CSV filename stem (`path.stem`). |
| `csv_path` | str | Path to the CSV being audited. |
| `target_column` | Optional[str] | Target column name (needed for imbalance/leakage/bias). |
| `profile` | Optional[dict] | Compact dataset profile from `profile_dataset()`. |
| `check_params` | dict[str, dict] | Per-check parameters keyed by check name (e.g. `check_params["bias"] = {"sensitive_columns": [...]}`). |
| `findings` | Annotated[list[dict], merge_findings] | Merged findings from all checks, stored as plain dicts for JSON serialization. Reducer deduplicates. |
| `leakage_suspected` | Optional[bool] | Set by router when id_like or schema findings exist. |
| `suspect_columns` | Optional[list[str]] | Column names flagged by router. |
| `deep_check_results` | Optional[list[dict]] | Results from the deep leakage check. |
| `summary` | Optional[str] | LLM-produced plain-language summary. |
| `overall_severity` | Optional[str] | Computed in Python from findings (not trusted to LLM). |
| `report` | Optional[str] | JSON-serialized `AuditReport` (canonical output). |
| `report_text` | Optional[str] | Plain-text rendering of the same report. |
| `error` | Optional[str] | Error message if a node failed. |

All fields optional so nodes return only what they update. Findings are stored as dicts (via `Finding.model_dump(mode="json")`) so the LangGraph checkpointer can serialize them to JSON without a custom Pydantic serializer.

---

### Graph A: `build_graph()` — M2/M3 deterministic pipeline

Flow: `load_csv` → `profile` → [N parallel checks] → `merge_findings` → `router` → (conditional) `deep_checks` → `report` → `END`

**Nodes:**

1. **`load_csv`** — Validates the CSV exists; sets `csv_path`, `dataset_id`, clears `error`. Conditional edge: if error → skip straight to `report`.
2. **`profile`** — Calls `profile_dataset(csv_path, target_column)`; sets `profile`. Loads the CSV from disk once here.
3. **Parallel check nodes** (`check_schema`, `check_missing_values`, `check_duplicates`, `check_class_imbalance`, `check_id_like`, `check_bias`) — One node per registered check with `parallel=True`. Each calls `run_check(spec, state)` which loads the CSV **again** from disk via `read_csv_robust()`. Normal fan-out edges from `profile` to each.
4. **`merge_findings`** — No-op passthrough; the actual merge happens via the `merge_findings` reducer on the `findings` field. Join edge from all check nodes.
5. **`router`** — Examines merged findings. Sets `leakage_suspected=True` and `suspect_columns` if any `id_like` or `schema` finding exists (excluding the target column).
6. **`deep_checks`** (conditional) — Runs only if `leakage_suspected`. Runs `target_leakage` check across **all** non-target columns (not just suspect columns). Sets `deep_check_results` and merges leakage findings into `findings`.
7. **`report`** — Builds `AuditReport` from profile + findings + deep_check_results; sets `report` (JSON) and `report_text` (rendered text).

**Edges:** `load_csv` →(conditional) `profile` or `report`; `profile` → each check → `merge_findings` → `router` →(conditional) `deep_checks` or `report`; `deep_checks` → `report` → `END`.

---

### Graph B: `build_agent_graph()` — M4 ReAct LLM pipeline

Flow: `load_csv` → `profile` → `agent_checks` → `synthesize` → `report` → `END`

**Nodes:**

1. **`load_csv`** — Same as Graph A.
2. **`profile`** — Same as Graph A.
3. **`agent_checks`** — Runs `run_checks_with_agent()`: a ReAct subgraph where the LLM (with bound tools from the registry) chooses which checks to call. The LLM sees only the compact profile (never the CSV). On LLM failure, falls back to `_run_all_checks_deterministically()` (runs all `fallback=True` checks). If the LLM returns zero findings, deterministic checks run anyway (safety net).
4. **`synthesize`** — `make_synthesize_findings_node(llm)`: builds a compact JSON payload (profile + top 15 findings + severity facts), asks the LLM for a summary. Attempt 1: structured output (`SynthesisResult`). Attempt 2: plain text if result too short (<40 words). Attempt 3: `_fallback_summary()` (no LLM). `overall_severity` computed in Python from findings.
5. **`report`** — Same as Graph A.

**Conditional edge** after `load_csv`: if error → `report`; else → `profile`.

---

### The ReAct subgraph (`build_check_agent` in `src/graph/agent.py`)

A nested `StateGraph` with `AgentState` (messages only, `add_messages` reducer):
- `agent` node: LLM with bound tools → returns `AIMessage` with tool calls.
- `tools` node: `ToolNode(ALL_TOOLS)` executes the chosen check tools.
- Conditional edge via `tools_condition`: `agent` → `tools` → `agent` loop until no more tool calls, then `END`.

---

## 4. Active Features & Checks

Seven checks are registered in `src/tools/checks/`. Six run in the parallel fan-out; one (`target_leakage`) runs only in the router-triggered deep-checks step.

### How each check receives data

Every check has the standard signature `(csv_path, target_column=None, **params)`. The registry's `run_check()` injects:
- `csv_path` and `target_column` from `GraphState`.
- Optional params from `state["check_params"][check_name]`.
- Each check independently calls `read_csv_robust(csv_path)` to load the CSV from disk (no DataFrame is passed through state).

| Check | Name | Severity when triggered | Parallel fan-out? | Agent tool? | Fallback? | Required inputs | Optional params | Notes |
|-------|------|------------------------|-------------------|-------------|-----------|-----------------|-----------------|-------|
| Schema | `schema` | MEDIUM (text-as-number >80%) | Yes | Yes | Yes | `csv_path` | — | `always=True` |
| Missing values | `missing_values` | LOW / HIGH / CRITICAL (>20% / >50%) | Yes | Yes | Yes (`tool_name=run_missing_check`) | `csv_path` | — | `always=True`; counts NaN + text markers (`?`, `NA`, etc.) |
| Duplicates | `duplicates` | MEDIUM / HIGH (>10% rows) | Yes | Yes | Yes | `csv_path` | — | `always=True` |
| Class imbalance | `class_imbalance` | MEDIUM (<30% minority) / HIGH (<10%) | Yes | Yes | Yes | `csv_path`, `target_column` | — | `target_hint=True`; returns empty if target missing/balanced |
| ID-like | `id_like` | MEDIUM (unique per row) | Yes | Yes | Yes | `csv_path` | — | `target_hint=True`; skips target column |
| Bias / fairness | `bias` | CRITICAL/HIGH/MEDIUM (DI <0.8 + p<0.01) | Yes | Yes | **No** (`fallback=False`) | `csv_path`, `target_column` | `sensitive_columns`, `favorable_value` | target_types binary/boolean; user-specified columns only; returns INFO skip note if no sensitive columns |
| Target leakage (deep) | `target_leakage` | CRITICAL/HIGH/MEDIUM (AUC/R² thresholds) | **No** | Yes | Yes | `csv_path`, `target_column` | `suspicious_columns` | `target_hint=True`; `parallel=False`; runs in `deep_checks` node only when router suspects leakage; uses 5-fold CV with imputation/encoding inside a Pipeline |

**CLI parameter mapping** (`main.py` `_build_check_params`): only `--sensitive` and `--favorable` are mapped into `check_params["bias"]`. No CLI mechanism exists to pass params to other checks.

---

## 5. Infrastructure & IO

### PostgresSaver checkpointer integration (M5)

In `main.py`, the function `_persisted_graph(use_agent)`:

1. Loads settings via `get_settings()`.
2. Creates `PostgresSaver.from_conn_string(settings.postgres_conn_string)` — the psycopg-native connection string (the `postgresql+psycopg://` SQLAlchemy dialect is stripped to `postgresql://`).
3. Calls `saver.setup()` — idempotent; creates LangGraph's `checkpoints` and `checkpoint_writes` tables.
4. Builds `build_graph(checkpointer=saver)` or `build_agent_graph(checkpointer=saver)`.
5. The graph is invoked with `config = {"configurable": {"thread_id": run_id}}` where `run_id` is a UUID (or `--thread-id`).

After the graph completes, `main.py` separately calls `save_run(...)` from `src/db/repository.py` to persist to **business tables** (`audit_runs`, `findings`, `reports`, `datasets`). These are **separate from** LangGraph's checkpoint tables, as documented in `src/db/models.py` and `src/db/repository.py`.

**Two persistence systems run in parallel:**
- LangGraph `PostgresSaver` → `checkpoints` / `checkpoint_writes` tables (enables resume/replay by thread_id).
- SQLAlchemy business tables → `audit_runs`, `findings`, `reports`, `datasets` (for the `--list` CLI and human-approval workflow).

`ensure_schema()` (which calls SQLAlchemy `init_db()`) is invoked at startup in `main.py` to create business tables.

### How the LLM summary is called

In `make_synthesize_findings_node` (`src/graph/nodes.py`):

1. Builds a compact JSON payload via `_build_payload()`: profile (sample_rows trimmed to 2), up to 15 findings sorted by severity (evidence trimmed to 400 chars), and severity facts (counts, highest severity, total, not-shown count).
2. **Attempt 1 — Structured output:** `llm.with_structured_output(SynthesisResult)` invoked with system + human messages. `SynthesisResult.summary` must be ≥150 words per docstring (but code uses `MIN_SUMMARY_WORDS = 40` as the usability threshold — a mismatch).
3. **Attempt 2 — Plain text fallback:** If the structured result is too short, `llm.invoke()` with a plain-text system prompt.
4. **Attempt 3 — No-LLM fallback:** `_fallback_summary()` builds a deterministic text summary from findings if the LLM is unavailable or unhelpful.
5. `overall_severity` is **always** computed in Python via `_highest_severity(findings)`, never trusted to the LLM.

The LLM is built by `get_cached_llm()` (`src/graph/llm.py`) → `ChatOllama` with model/base_url/temperature from `.env`. `get_structured_llm()` exists but is **unused** (nodes.py calls `llm.with_structured_output()` directly).

The LLM is subject to design rule #1: it never receives the full CSV — only the compact profile and check results.

---

## 6. Observed Inefficiencies & Technical Debt

### A. CSV read from disk once per check (major inefficiency)

Every check independently calls `read_csv_robust(csv_path)`. A typical audit reads the CSV **7–8 times**: once by `profile_node`, once by each of the 6 parallel checks, and once more by `deep_checks_node` if leakage is suspected. For large files this is wasteful I/O. The design rule says state must not hold a DataFrame, but a single load + cache, or a transient DataFrame passed between nodes, could eliminate most redundant reads.

### B. Identical try/except boilerplate duplicated in every check

`schema_check`, `missing_check`, `duplicates_check`, `imbalance_check`, `id_like_check`, `bias_check`, and `leakage_check` all repeat the same pattern:

```python
try:
    df = read_csv_robust(csv_path)
except Exception as exc:
    return [Finding(check_name=..., severity=Severity.HIGH, evidence=f"Failed to read CSV: {exc}", ...)]
```

This could be centralized in `run_check()` or a shared helper.

### C. `.gitignore` excludes `tests/` from version control

Line 15 of `.gitignore` is `tests/`. The entire 17-file test suite is therefore **not tracked by git**. This is almost certainly accidental and would prevent CI from running tests.

### D. `.env` committed with real credentials

`.env` contains `POSTGRES_PASSWORD=08062006` and a full `DATABASE_URL` with embedded credentials. Per AGENTS.md rule 4, `.env` and secrets must never be committed. It is currently in the working tree.

### E. Orphaned/stale artifacts

- `src/db/__pycache__/checkpointer.cpython-314.pyc` exists, but there is **no `src/db/checkpointer.py`** source file. A checkpointer module was deleted but its bytecode remains.
- `tests/__pycache__/test_resume_no_duplicates.*.pyc` exists, but there is **no `tests/test_resume_no_duplicates.py`** source file. The M6 (human-in-the-loop) test file appears to have been removed.
- `tests/` is gitignored (see C), so even existing tests may not be backed up.

### F. Inconsistent packaging config

`pyproject.toml` has `dependencies = []` while `requirements.txt` lists 25+ dependencies. The project name is also `langgraph` (generic) rather than `data-detective`.

### G. Empty/stubbed components

- `README.md` is 0 lines — no project documentation.
- `src/api/__init__.py` is a stub — FastAPI (M7) is in requirements but not implemented.
- No Streamlit UI despite being planned for M7 and listed in requirements.
- `src/graph/llm.py` has `get_structured_llm()` which is never called.
- `eval/__init__.py` says M8 but `run_harness.py` says "Step 7".

### H. Redundant severity-ordering logic

- `Finding.__lt__()` in `finding.py` sorts by severity (CRITICAL first).
- `_TRUSTED_SEVERITY_ORDER` and `_SEVERITY_RANK` in `nodes.py` are two separate severity-rank dicts serving overlapping purposes.
- `render_text_report` sorts via `sorted(report.findings)` (using `__lt__`), while `_sort_findings` in `nodes.py` sorts via `_SEVERITY_RANK`. Two code paths for the same ordering.

### I. `deep_checks_node` ignores `suspect_columns`

The router computes `suspect_columns` (from id_like/schema findings), but `deep_checks_node` runs `target_leakage` across **all** non-target columns regardless. `suspect_columns` is set in state but never consumed by the deep check — only the boolean `leakage_suspected` gates the conditional edge. The comment acknowledges this ("The router decides whether to run deep checks; the check itself examines every column"), but the suspect-columns computation is effectively dead data.

### J. Mismatched word-count requirements

- `SynthesisResult.summary` docstring: "at least 150 words".
- `_SYSTEM_PROMPT`: "Keep the whole summary under 220 words."
- Code threshold `MIN_SUMMARY_WORDS = 40`.
These three numbers conflict and could cause the LLM output to be rejected or accepted unpredictably.

### K. Other minor issues

- `main.py` docstring lists `--thread-id` example twice (lines 11–12).
- `main.py` `_build_check_params` only maps CLI flags to `check_params["bias"]`; no CLI support for other checks' optional params.
- `bias_check.py` (497 lines) and `leakage_check.py` (389 lines) are the largest files; both contain repeated CSV-load + error-finding boilerplate.
- `leakage_check` reports a hardcoded "Baseline R²: 0.0000" in evidence rather than computing an actual baseline.
- `datasets.yaml` includes only 3 of the 5 CSVs in `data/raw/` (`f1_race_results.csv` and `loan_approval_dataset.csv` are orphaned).
- No human-in-the-loop pause despite M6 being designed for it: `main.py` docstring explicitly says "The graph runs to completion in a single invoke; there is no human-in-the-loop pause." The `approval_status` DB column exists but is always "pending".
- `conn`/`checkpointer` import mismatch: requirements lists `langgraph-checkpoint-postgres>=3.0.0` but code imports `from langgraph.checkpoint.postgres import PostgresSaver`.
- `Finding.confidence` is typed `float` (0.0–1.0) in Pydantic but stored as `Integer` (0–100) in the SQLAlchemy `Finding` model, with conversion `int(float(conf) * 100)` in `save_run`.

---

## 7. Milestone Alignment vs Brief

| Milestone | Status per brief (last updated 2026-10-31) | Observed state |
|-----------|-------------------------------------------|----------------|
| M0 Setup | Not started (per brief) | Complete: repo exists, CSV loads via `profile_dataset()`, robust CSV reader. |
| M1 First graph | Not started | Complete: `build_graph()` with load→profile→checks→report. |
| M2 Parallel checks + reducers | Not started | Complete: 6 parallel checks + idempotent `merge_findings` reducer. |
| M3 Conditional routing | Not started | Complete: router → deep_checks leakage path. |
| M4 LLM agent (ReAct) | Not started | Complete: `build_agent_graph()` with ReAct subgraph, structured output, fallback. |
| M5 Persistence | Not started | Complete: PostgresSaver checkpointer + SQLAlchemy business tables + `--list` CLI. |
| M7 Backend (FastAPI) | Not started | **Stub only** — `src/api/__init__.py` empty; no endpoints. |
| M8 Evaluation + tracing | Not started | Partial: `eval/run_harness.py` with invariants + `datasets.yaml`; no LangSmith/Langfuse integration, no planted-problem scoring. |
| M9 Polish | Not started | Not started: empty README, no diagram/demo. |

The brief says M0 is "not started" but the code clearly implements through M5. The brief has not been updated to reflect the work done (last updated 2026-10-31, while today is 2026-11-05).

---

**End of report.** No code was read, modified, or deleted during this audit beyond reading files for analysis.