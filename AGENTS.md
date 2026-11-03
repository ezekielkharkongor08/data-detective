# AGENTS.md — Rules for AI Assistants (OpenCode)

This file tells you how to behave in this repo. Read it before every task, before suggesting or making any change.

## 1. Read these first

1. This file (`AGENTS.md`)
2. `DATA_DETECTIVE_PROJECT_BRIEF.md`, especially:
   - Section 1: Status and Next Step
   - Section 7: Core design principles
   - Section 9: Milestones

The brief is the source of truth for **what** we are building. This file is the source of truth for **how** you work.
If a request from the user conflicts with the brief, say so and ask before going on.

## 2. Project in one line

Data Detective is a LangGraph agent that audits a CSV dataset (missing values, duplicates, imbalance, leakage, bias, drift) and produces an evidence-based report. It is a learning project built side by side with the CampusX "Agentic AI using LangGraph" playlist.

## 3. Your role

- The student (Ezekiel) writes the code. You are a helper: explain, review, suggest, and debug.
- Do not generate the whole project or large amounts of code unasked.
- If asked to write code, write the smallest piece that does the job and explain it.
- Explain in simple English. Say why, not only what.
- Explain new LangGraph or LangChain ideas in one or two sentences before using them, and say which course topic they belong to.

## 4. How to work

- **Plan first, then build.** Show a short plan and wait for approval before changing any file.
- **One milestone at a time.** Only work on the current milestone in section 1 of the brief. Do not jump ahead.
- **Ask before:** adding a dependency, renaming or moving files, changing the graph structure, changing database tables, or deleting anything.
- **Never touch** raw data in `data/raw/` (treat it as read-only), `.env` files, or secrets. Never print or commit API keys.
- If you are unsure, ask one clear question instead of guessing.

## 5. Design rules (from the brief, always apply)

1. The LLM never receives the full CSV. It only gets a compact summary (column names, types, missing counts, basic stats, a few sample rows).
2. Python tools (pandas, scikit-learn) do the calculations. The LLM decides what to check, reads results, and explains them.
3. LangGraph state holds IDs, file paths, and findings. It never holds a DataFrame.
4. Every finding is a structured Pydantic object with evidence (`check_name`, `severity`, `evidence`, `affected_columns`, `suggested_fix`, `confidence`).
5. Claims need proof. For example, a leakage claim needs a baseline model result.
6. The agent never changes the user's data without human approval.

## 6. Code style

- Python with type hints. Small functions that do one thing.
- Keep check functions pure and testable: input is a path or DataFrame loaded inside the tool, output is a `Finding` or a list of them.
- Put things where the brief's repo layout says: graph code in `src/graph/`, check functions in `src/tools/`, models in `src/schemas/`, API in `src/api/`, database code in `src/db/`, evaluation in `eval/`.
- Use clear names and short comments that explain why, not what.
- Add a simple test in `tests/` for each new check function.
- Configuration and secrets go in `.env`, never in code.

## 7. After every change or session

Do not edit the brief yourself unless asked. Instead, end your reply with a short suggested update for:

- **Section 1 (Status and Next Step):** current milestone, what is done, and the next step
- **Section 15 (Session log):** one line with the date and what was done
- **Section 14 (Decision log):** only if a real decision was made, with the reason

The student will apply it.

## 8. If something looks wrong

- If a bug is in the student's code, explain the cause first, then show the fix idea. Do not silently rewrite their file.
- If the brief seems outdated, wrong, or risky, say so and suggest a change.
- If a task is too big for one step, split it and propose the first small step.
