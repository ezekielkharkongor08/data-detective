# Data Detective — Project Brief

> The core document of this project. It says what we are building, why, how, and what comes next.
> Update the **Status and Next Step** section after every work session.

Last updated: 2026-10-31

---

## 1. Status and Next Step

- **Current milestone:** M0 (not started)
- **Next step:** Create the repo, load the Telco Churn CSV with pandas, and write a plain `profile_dataset()` function (no LLM yet).
- **Blocked on:** nothing

Update this section at the end of every session. It is the first thing to read when coming back to the project.

---

## 2. One-line pitch

An AI agent (built with LangGraph) that audits a CSV dataset the way a careful data scientist would, and tells you whether the data deserves a model, before you train one.

## 3. Why we are building this

- First agent-building project, built side by side with the CampusX "Agentic AI using LangGraph" playlist, so every course topic gets applied in a real project.
- Goes beyond the common "chat with your data" and basic RAG projects. Most people build the model; few build the tool that checks the data first.
- Covers three areas at once: LangGraph/LangChain core concepts, backend infra (API, database), and data science.
- A strong CV project with a measurable result (how many planted problems the agent catches).

## 4. Working rules (important for AI assistants such as OpenCode)

- **The student writes the code.** The assistant helps by explaining, reviewing, suggesting, and debugging. It does not generate the whole project unasked.
- Explain things in simple English. Say *why*, not only *what*.
- Work milestone by milestone (section 9). Do not jump ahead to later milestones.
- Plan first, then build only after approval. Ask before large changes.
- Keep code small and readable. Prefer plain pandas and scikit-learn for the checks; use the LLM only where reasoning or explanation is needed.

## 5. The problem

Models fail because of bad data, not bad algorithms. Typical hidden problems:

- Numbers stored as text, odd missing-value markers (such as `?`), blank cells
- Duplicate rows
- Heavy class imbalance
- ID-like columns used as features
- **Target leakage**: a column that secretly contains the answer
- Label noise
- Bias across groups (for example by sex or race)
- Drift between train and test data

Today these checks are done by hand, inconsistently. Data Detective runs them automatically, proves each claim with evidence, and writes a clear report.

## 6. What it does

**Input:** a CSV file plus the name of the target column.
**Output:** a data-quality report card with findings ranked by severity, evidence for each finding, and suggested fixes.

### Checks planned (v1 first, rest later)

| Check | What it looks for | Version |
|---|---|---|
| Schema and types | Numbers stored as text, mixed types, bad dates | v1 |
| Missing values | NaN, blanks, markers like `?` | v1 |
| Duplicates | Repeated rows | v1 |
| Class imbalance | Skewed target | v1 |
| ID-like columns | Unique-per-row columns | v1 |
| Target leakage | A feature that predicts the target too well | v1 |
| Bias across groups | Different outcomes by sensitive column | v2 |
| Label noise | Labels that look wrong | v2 |
| Drift | Different distributions between two splits | v2 |

## 7. Core design principles

1. **The LLM never sees the whole CSV.** It sees a compact summary (column names, types, missing counts, basic stats, a few sample rows).
2. **Python tools do the real work.** pandas and scikit-learn compute; the LLM decides what to check, reads the results, and explains them.
3. **Keep state small.** LangGraph state holds a dataset ID, a file path, and findings, never the DataFrame itself. Tools load the CSV from disk.
4. **Every finding is structured and has evidence.** Use a Pydantic model such as `Finding(check_name, severity, evidence, affected_columns, suggested_fix, confidence)`.
5. **Proof, not opinions.** For example, a leakage claim should be backed by a baseline model result ("accuracy drops from 99% to 71% without this column").
6. **A human approves fixes.** The agent never changes the data without approval.

## 8. Planned architecture

### Graph flow (first version)

```
load_csv -> profile -> [ parallel checks ] -> router -> deep_checks (if needed)
         -> synthesize_findings -> human_approval -> final_report
```

- **load_csv / profile:** read the file, build the compact summary.
- **parallel checks:** schema, missing, duplicates, imbalance, ID-like, leakage run side by side.
- **router (conditional edge):** if a check raises a flag (for example suspected leakage), run a deeper check.
- **synthesize_findings:** merge all findings, rank by severity, ask the LLM to explain in plain language.
- **human_approval (interrupt):** the user approves or rejects suggested fixes.
- **final_report:** store the report and return it.

### Proposed stack (can change)

| Layer | Choice |
|---|---|
| Language | Python |
| Agent framework | LangGraph + LangChain |
| LLM | Local model via Ollama at first (swappable) |
| Data tools | pandas, scikit-learn |
| Backend API | FastAPI + Pydantic |
| Database | PostgreSQL (datasets, audit runs, findings, reports) + LangGraph Postgres checkpointer |
| UI | Streamlit |
| Tracing | LangSmith or Langfuse |
| Packaging | Docker |

### Proposed repo layout

```
data-detective/
  DATA_DETECTIVE_PROJECT_BRIEF.md
  README.md
  data/               # raw CSVs and planted-problem copies
  src/
    graph/            # state, nodes, edges, builder
    tools/            # pandas / sklearn check functions
    schemas/          # Pydantic models (Finding, Report)
    api/              # FastAPI app
    db/               # Postgres models and helpers
  eval/                # planted-problem test sets and scoring script
  tests/
```

## 9. Milestones (matched to LangGraph topics)

Tick each box when done and update section 1.

- [ ] **M0 — Setup:** repo, environment, load Telco CSV, plain `profile_dataset()` function (no LLM)
- [ ] **M1 — First graph (state, nodes, edges):** load -> profile -> simple text report
- [ ] **M2 — Parallel checks and reducers:** several checks run side by side and merge into one findings list
- [ ] **M3 — Conditional routing:** a flagged check triggers a deeper leakage test
- [ ] **M4 — LLM with tools (ReAct) and structured output:** the agent chooses which checks to call and returns `Finding` objects
- [ ] **M5 — Persistence and memory:** Postgres checkpointer, saved audit history, resume and replay a run
- [ ] **M7 — Backend:** FastAPI endpoints (start audit, check status, get report), Postgres tables, Docker
- [ ] **M8 — Polish:** README, CV bullets

## 10. Datasets

1. **Telco Customer Churn** (Kaggle, blastchar): about 7,043 customers, 20 features, Yes/No churn target. Use for data quality, imbalance, ID-like column, and leakage tests. Known quirk to confirm: `TotalCharges` loads as text because of blank values.
2. **Adult Census Income** (Kaggle, UCI Machine Learning upload): use for missing values written as `?`, duplicates, imbalance, and the bias-across-groups check. Confirm details after downloading.
3. Later, optional: the bank loan dataset from the earlier loan prediction project.

## 11. Evaluation (how we prove it works)

Make corrupted copies of the clean datasets, each with one known problem planted:

- Copy the target into a new column (leakage)
- Flip about 10% of labels at random (label noise)
- Shift one column's distribution between train and test (drift)
- Add duplicate rows
- Inject missing-value markers

**Score:** how many planted problems the agent finds out of the total.
**Target for the CV:** a line like "caught X of Y planted issues across Z datasets."

## 12. Out of scope for now

- Non-CSV formats (Excel, Parquet), streaming data, very large files
- Automatic data fixing without approval
- Multi-user login and permissions
- Cloud deployment (maybe later)

## 13. Phase 2 idea

**KPI Root-Cause Agent ("why did it drop?")**: a second graph in the same project. It detects an anomaly in a time series, creates hypotheses (region, product, channel, pipeline bug), tests them in parallel, and ranks likely causes with evidence. It reuses the database, API, and report code from Data Detective, and the drift check is a close relative of anomaly detection.

## 14. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-31 | Build Data Detective first, KPI root-cause later | Easier to finish as a first agent, covers more of data science, strong backend shape |
| 2026-10-31 | Start with CSV input only | Simplest format, keeps focus on the agent |
| 2026-10-31 | Start with Telco and Adult datasets | One covers quality and leakage, the other covers missing values and bias |

## 15. Session log

Add one short line per session: date, what was done, what is next.

- 2026-10-31: Project idea chosen and brief written. Next: M0.
