"""Guard against dataset-specific literals leaking into src/.

Scans src/ for distinctive dataset/column names that would indicate the
classifier (or any other source code) is tuned to a particular dataset.
Generic words like "id", "code", "date", "target" are NOT flagged — only
distinctive proper names.

This keeps the classifier general: it must work on any CSV, not just
Telco, Adult, House, or Titanic.
"""

import re
from pathlib import Path

# Distinctive names that must never appear in src/ source code.
# These are proper names tied to specific datasets, not generic concepts.
FORBIDDEN = (
    "Churn",
    "SeniorCitizen",
    "education.num",
    "PassengerId",
    "statezip",
    "customerID",
    "Telco",
    "Adult",
    "Titanic",
    "House.csv",
)


def _strip_comments_and_docstrings(text: str) -> str:
    """Remove Python comments and docstrings so only executable code is
    scanned. Dataset names inside comments (for example, examples in a
    docstring) should not trip this guard."""
    # Remove docstrings (triple-quoted strings).
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    # Remove inline comments (everything from # to end of line, outside strings).
    # A simple approach: split on # not inside quotes is hard with regex, so
    # we strip full-line comments and inline comments on code lines.
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # full-line comment
        # Inline comment: find # not inside a string. Approximate by splitting
        # on # and keeping the part before it (good enough for this guard).
        hash_idx = line.find("#")
        if hash_idx != -1:
            # Heuristic: if there's an even number of double-quotes before #,
            # the # is outside a string.
            before = line[:hash_idx]
            if before.count('"') % 2 == 0:
                line = before
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _iter_source_files() -> list[Path]:
    src = Path("src")
    return sorted(src.rglob("*.py"))


def test_no_dataset_literals_in_src():
    """No distinctive dataset or column names may appear in executable src/ code.

    Comments and docstrings are stripped first, because examples in docstrings
    (for example leakage_check.py mentioning Adult/Telco) are not classifier
    logic and must not trip this guard.
    """
    hits: list[tuple[str, int, str]] = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        code = _strip_comments_and_docstrings(text)
        for line_no, line in enumerate(code.splitlines(), start=1):
            for name in FORBIDDEN:
                if name in line:
                    hits.append((str(path), line_no, name))
    assert not hits, (
        "Dataset-specific literals found in src/. The classifier must be "
        "general, not tuned to one dataset.\n" + "\n".join(
            f"  {f}:{l} -> {n!r}" for f, l, n in hits
        )
    )


def test_forbidden_list_is_non_empty():
    """Sanity: the guard actually checks something."""
    assert len(FORBIDDEN) > 0
