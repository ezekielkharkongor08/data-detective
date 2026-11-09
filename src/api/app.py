"""FastAPI layer over the Data Detective audit (M7).

A thin wrapper: every audit is executed by :func:`run_audit` from
``src.graph.audit_runner``. The API does not duplicate audit logic. Uploads are
stored under ``uploads/<run_id>/<file>.csv`` and plots under ``plots/<run_id>/``.
Runs are tracked in memory during a server lifetime and persisted to the
database by ``run_audit`` using the same tables as the CLI.
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import text

import src.db.connection as _db_connection
from src.config import get_settings
from src.db.repository import ensure_schema, get_run, list_runs
from src.graph.audit_runner import AuditResult, build_check_params, run_audit
from src.workbench.visualizer.visualizer_tools import PLOTS_DIR, generate_plots, profile_dataset

# Ensure the database and tables exist on startup so a deleted or missing
# database is recreated automatically (same behavior as the CLI).
ensure_schema()

app = FastAPI(title="Data Detective API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- constants

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
UPLOADS_DIR_NAME = "uploads"


# --------------------------------------------------------------------------- paths

def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


def _uploads_dir() -> Path:
    return _project_root() / UPLOADS_DIR_NAME


def _plots_dir(run_id: str) -> Path:
    return _project_root() / "plots" / run_id


# --------------------------------------------------------------------------- in-memory run store

_RUNS: dict[str, dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()


# --------------------------------------------------------------------------- helpers

def _safe_name(filename: str) -> str:
    """Sanitize an uploaded file name: strip directories and unsafe characters."""
    name = Path(filename).name
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return cleaned or "upload.csv"


def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Read an upload stream up to ``limit`` bytes, raising 413 if exceeded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = file.file.read(8192)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="file too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_csv(csv_path: str, target: str) -> None:
    """Check the CSV parses and contains ``target`` in its header.

    Reads only the header for the column check. Raises 400 on failure.
    """
    try:
        header_df = pd.read_csv(csv_path, nrows=0)
    except Exception:  # noqa: BLE001 - an unparseable upload is a 400
        raise HTTPException(status_code=400, detail="could not parse CSV file")
    if target not in [str(c) for c in header_df.columns]:
        raise HTTPException(status_code=400, detail=f"target column {target!r} not found in CSV")


def _parse_semantic_overrides(value: str) -> dict[str, str]:
    """Parse ``COLUMN=TYPE,COLUMN2=TYPE2`` into a dict.

    Invalid entries (missing '=', unknown type) are skipped silently.
    Mirrors the CLI's ``--semantic-override`` translation.
    """
    from src.tools.profiling import SEMANTIC_TYPES

    overrides: dict[str, str] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        name, new_type = part.split("=", 1)
        name = name.strip()
        new_type = new_type.strip()
        if new_type not in SEMANTIC_TYPES:
            continue
        overrides[name] = new_type
    return overrides


def _ollama_reachable() -> bool:
    """Return True when the configured Ollama server answers /api/tags."""
    settings = get_settings()
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except (URLError, HTTPError, OSError):  # noqa: BLE001 - Ollama may be down
        return False


# --------------------------------------------------------------------------- background audit

def _run_audit_background(
    run_id: str,
    csv_path: str,
    target_column: str | None,
    check_params: dict[str, dict],
    semantic_overrides: dict[str, str],
    mode: str,
) -> None:
    """Run one audit in the background and update the in-memory run store.

    The lock is held for the whole audit so audits run one at a time and a run
    stays ``queued`` until the lock is acquired, then ``running``.
    """
    with _RUNS_LOCK:
        _RUNS[run_id]["status"] = "running"
        try:
            result: AuditResult = run_audit(
                run_id=run_id,
                csv_path=csv_path,
                target_column=target_column,
                check_params=check_params,
                semantic_overrides=semantic_overrides,
                mode=mode,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a failed run
            _RUNS[run_id]["status"] = "failed"
            _RUNS[run_id]["error_message"] = str(exc)
            return

        if result.error:
            _RUNS[run_id]["status"] = "failed"
            _RUNS[run_id]["error_message"] = result.error
        else:
            _RUNS[run_id]["status"] = "completed"
            _RUNS[run_id]["mode"] = mode
            _RUNS[run_id]["result"] = {
                "overall_severity": result.overall_severity,
                "findings": result.findings,
                "report_json": result.report_json,
                "report_text": result.report_text,
                "mode": mode,
            }


# --------------------------------------------------------------------------- run store view

def _get_run_anywhere(run_id: str) -> dict[str, Any] | None:
    """Return a normalized run dict from memory or the database."""
    if run_id in _RUNS:
        entry = _RUNS[run_id]
        status_value = entry["status"]
        base = {
            "run_id": run_id,
            "status": status_value,
            "csv_path": entry.get("csv_path"),
            "target_column": entry.get("target_column"),
            "mode": entry.get("mode") or "deterministic",
        }
        if status_value == "completed":
            data = entry.get("result", {})
            base.update(
                {
                    "overall_severity": data.get("overall_severity"),
                    "findings": data.get("findings"),
                    "report_json": data.get("report_json"),
                    "report_text": data.get("report_text"),
                    "mode": data.get("mode") or entry.get("mode") or "deterministic",
                }
            )
        elif status_value == "failed":
            base["error_message"] = entry.get("error_message")
        return base

    run = get_run(run_id)
    if run is None:
        return None
    return {
        "run_id": run["run_id"],
        "status": run.get("status", "completed"),
        "csv_path": run.get("csv_path"),
        "target_column": run.get("target_column"),
        "overall_severity": run.get("overall_severity"),
        "findings": (run.get("report_json") or {}).get("findings"),
        "report_json": run.get("report_json"),
        "report_text": run.get("report_text"),
        "mode": run.get("mode") or (run.get("report_json") or {}).get("mode") or "deterministic",
    }


# --------------------------------------------------------------------------- request/response schemas

class AuditStart(BaseModel):
    run_id: str
    status: str


class AuditStatus(BaseModel):
    run_id: str
    status: str
    csv_path: str | None = None
    target_column: str | None = None
    mode: str | None = None
    error_message: str | None = None
    overall_severity: str | None = None
    findings: list[dict[str, Any]] | None = None
    report_json: dict[str, Any] | None = None
    report_text: str | None = None


class RunSummary(BaseModel):
    run_id: str
    status: str
    csv_path: str | None = None
    target_column: str | None = None
    mode: str | None = None
    overall_severity: str | None = None
    findings_count: int = 0
    created_at: str | None = None


class Health(BaseModel):
    api: str
    database: str
    backend: str
    ollama: str


class PlotsResult(BaseModel):
    plots: list[str]
    skipped: list[dict[str, str | None]]


# --------------------------------------------------------------------------- endpoints

@app.get("/health", response_model=Health)
def health() -> Health:
    """Report API, database, backend and Ollama reachability."""
    settings = get_settings()
    database = "ok"
    try:
        with _db_connection.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - database may be unreachable
        database = "unreachable"
    return Health(
        api="ok",
        database=database,
        backend=settings.backend_name,
        ollama="ok" if _ollama_reachable() else "unreachable",
    )


@app.post("/audits", status_code=status.HTTP_202_ACCEPTED, response_model=AuditStart)
def start_audit(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target: str = Form(...),
    mode: str = Form("agent"),
    sensitive: str = Form(""),
    favorable: str | None = Form(None),
    semantic_override: str = Form(""),
) -> AuditStart:
    """Upload a CSV and start an audit. Returns immediately with a run id."""
    if mode not in ("agent", "deterministic"):
        raise HTTPException(status_code=400, detail="mode must be 'agent' or 'deterministic'")

    if mode == "agent" and not _ollama_reachable():
        raise HTTPException(status_code=503, detail="Ollama is unreachable for agent mode")

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="only .csv files are accepted")

    run_id = str(uuid.uuid4())
    safe = _safe_name(file.filename)
    run_upload_dir = _uploads_dir() / run_id
    run_upload_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(run_upload_dir / safe)

    content = _read_limited(file, MAX_UPLOAD_BYTES)
    with open(csv_path, "wb") as fh:
        fh.write(content)

    _validate_csv(csv_path, target)

    check_params = build_check_params(
        sensitive=[s for s in sensitive.split(",") if s.strip()],
        favorable=favorable or None,
    )
    semantic_overrides = _parse_semantic_overrides(semantic_override)

    with _RUNS_LOCK:
        _RUNS[run_id] = {
            "status": "queued",
            "csv_path": csv_path,
            "target_column": target,
            "mode": mode,
        }

    background_tasks.add_task(
        _run_audit_background,
        run_id=run_id,
        csv_path=csv_path,
        target_column=target,
        check_params=check_params,
        semantic_overrides=semantic_overrides,
        mode=mode,
    )
    return AuditStart(run_id=run_id, status="queued")


@app.get("/audits/{run_id}", response_model=AuditStatus)
def get_audit(run_id: str) -> AuditStatus:
    """Return the status/report of a run, from memory or the database."""
    run = _get_run_anywhere(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run id")
    return AuditStatus(**run)


@app.get("/audits", response_model=list[RunSummary])
def list_audits() -> list[RunSummary]:
    """List past runs: database runs plus in-memory runs not yet saved."""
    db_runs = list_runs()
    known_ids = {r["run_id"] for r in db_runs}
    summaries: list[RunSummary] = []
    for r in db_runs:
        summaries.append(
            RunSummary(
                run_id=r["run_id"],
                status=r["status"],
                csv_path=r.get("csv_path"),
                target_column=r.get("target_column"),
                mode=r.get("mode") or "deterministic",
                overall_severity=r.get("overall_severity"),
                findings_count=r.get("findings_count", 0),
                created_at=r.get("created_at"),
            )
        )
    with _RUNS_LOCK:
        for run_id, entry in _RUNS.items():
            if run_id in known_ids:
                continue
            summaries.append(
                RunSummary(
                    run_id=run_id,
                    status=entry["status"],
                    csv_path=entry.get("csv_path"),
                    target_column=entry.get("target_column"),
                    mode=entry.get("mode") or "deterministic",
                )
            )
    summaries.sort(key=lambda s: s.created_at or "", reverse=True)
    return summaries


@app.post("/audits/{run_id}/plots", response_model=PlotsResult)
def generate_audit_plots(run_id: str) -> PlotsResult:
    """Generate plots for a completed run."""
    run = _get_run_anywhere(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run id")
    if run["status"] != "completed":
        raise HTTPException(status_code=409, detail="audit is not completed")

    csv_path = run.get("csv_path")
    target_column = run.get("target_column")
    findings = run.get("findings") or []
    if not csv_path or not Path(csv_path).exists():
        return PlotsResult(
            plots=[],
            skipped=[{"plot": None, "reason": "CSV file not found"}],
        )

    profile = profile_dataset(csv_path, target_column=target_column)
    output_dir = str(_plots_dir(run_id))
    report = generate_plots(
        findings=findings,
        profile=profile,
        csv_path=csv_path,
        target_column=target_column,
        output_dir=output_dir,
    )
    return PlotsResult(
        plots=[Path(p).name for p in report.plots],
        skipped=[{"plot": s.get("plot"), "reason": s.get("reason")} for s in report.skipped],
    )


@app.get("/audits/{run_id}/plots/{filename}")
def get_plot(run_id: str, filename: str) -> FileResponse:
    """Serve a generated plot PNG, rejecting path traversal."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid plot filename")
    run_dir = _plots_dir(run_id).resolve()
    target = (run_dir / filename).resolve()
    try:
        target.relative_to(run_dir)
    except ValueError:  # noqa: BLE001 - path escapes the run folder
        raise HTTPException(status_code=400, detail="invalid plot filename")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="plot not found")
    return FileResponse(target, media_type="image/png")


__all__ = ["app"]
