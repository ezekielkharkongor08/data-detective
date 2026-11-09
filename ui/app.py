"""Streamlit demo UI for the Data Detective FastAPI backend.

UI only: no changes to the API or the audit code. Every API call goes through
``api_get`` / ``api_post`` with timeouts, so the page never crashes when the
API is down — it shows a clear message instead.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 10
POLL_INTERVAL = 2
MAX_WAIT_SECONDS = 15 * 60
SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]

SEVERITY_META = {
    "critical": {"icon": "🔴", "badge": "[CRITICAL]"},
    "high": {"icon": "🟠", "badge": "[HIGH]"},
    "medium": {"icon": "🟡", "badge": "[MEDIUM]"},
    "low": {"icon": "🔵", "badge": "[LOW]"},
    "info": {"icon": "ℹ️", "badge": "[INFO]"},
}


# --------------------------------------------------------------------------- API helpers

def api_get(path: str) -> tuple[dict[str, Any] | None, str | None]:
    """GET from the API.

    Returns ``(json, None)`` on success or ``(None, error_message)`` on failure.
    Never raises on a connection error.
    """
    try:
        resp = requests.get(f"{API}{path}", timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:  # noqa: BLE001 - connection may be down
        return None, (
            f"API not reachable at {API}; start it with: "
            f"python -m uvicorn src.api.app:app --port 8000 ({exc})"
        )
    if resp.status_code >= 400:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text
        return None, f"API error {resp.status_code}: {detail}"
    try:
        return resp.json(), None
    except Exception:
        return None, "API returned an invalid response"


def api_get_bytes(path: str) -> tuple[bytes | None, str | None]:
    """GET a binary payload (for example a PNG plot) without parsing JSON.

    Returns ``(bytes, None)`` on success or ``(None, error_message)`` on failure.
    Never raises on a connection error.
    """
    try:
        resp = requests.get(f"{API}{path}", timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:  # noqa: BLE001 - connection may be down
        return None, (
            f"API not reachable at {API}; start it with: "
            f"python -m uvicorn src.api.app:app --port 8000 ({exc})"
        )
    if resp.status_code >= 400:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = f"status {resp.status_code}"
        return None, f"API error {resp.status_code}: {detail}"
    content_type = resp.headers.get("Content-Type", "")
    if "image/png" not in content_type:
        return None, f"API returned {content_type}, expected image/png"
    return resp.content, None


def api_post(
    path: str, data: dict[str, Any] | None = None, files: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """POST to the API.

    Returns ``(json, None)`` on success or ``(None, error_message)`` on failure.
    Never raises on a connection error.
    """
    try:
        resp = requests.post(f"{API}{path}", data=data, files=files, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:  # noqa: BLE001 - connection may be down
        return None, (
            f"API not reachable at {API}; start it with: "
            f"python -m uvicorn src.api.app:app --port 8000 ({exc})"
        )
    if resp.status_code >= 400:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text
        return None, f"API error {resp.status_code}: {detail}"
    try:
        return resp.json(), None
    except Exception:
        return None, "API returned an invalid response"


def check_health() -> tuple[dict[str, Any] | None, str | None]:
    return api_get("/health")


# --------------------------------------------------------------------------- sidebar

def render_sidebar_health() -> None:
    """Show API/database/ollama status in the sidebar with a refresh button."""
    with st.sidebar:
        st.title("🔍 Data Detective")
        st.caption("Autonomous Dataset Audit & Fairness Agent")
        st.markdown("---")
        st.subheader("System Status")
        if st.button("🔄 Refresh Status", key="refresh_health", use_container_width=True):
            st.session_state["health"] = check_health()
        if "health" not in st.session_state:
            st.session_state["health"] = check_health()
        health, error = st.session_state["health"]
        if error:
            st.error(error)
            return

        def _badge(val: str | None) -> str:
            return "🟢 Connected" if val == "ok" else f"🔴 {val}"

        st.write(f"**API:** {_badge(health.get('api'))}")
        st.write(f"**Database:** {_badge(health.get('database'))} `({health.get('backend', 'sqlite')})`")
        st.write(f"**Ollama:** {_badge(health.get('ollama'))}")
        if health.get("ollama") != "ok":
            st.warning(
                "⚠️ Ollama unreachable: agent mode needs it; deterministic mode works without it."
            )
        st.markdown("---")


# --------------------------------------------------------------------------- run view

def render_report(
    report_text: str | None,
    report_json: dict[str, Any] | None = None,
    run_id: str = "run",
) -> None:
    """Show the report text with instant download options."""
    col1, col2 = st.columns([1, 1])
    with col1:
        if report_text:
            st.download_button(
                label="📥 Download Text Report",
                data=report_text,
                file_name=f"audit_report_{run_id[:8]}.txt",
                mime="text/plain",
                key=f"dl_txt_{run_id}",
            )
    with col2:
        if report_json:
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(report_json, indent=2),
                file_name=f"audit_report_{run_id[:8]}.json",
                mime="application/json",
                key=f"dl_json_{run_id}",
            )

    st.code(report_text or "(no report text)", wrap_lines=True)


def render_findings(findings: list[dict[str, Any]]) -> None:
    """Group findings by severity with detailed structured cards."""
    if not findings:
        st.success("✅ No findings or data quality issues detected.")
        return
    grouped: dict[str, list[dict[str, Any]]] = {}
    for f in findings:
        sev = str(f.get("severity", "info")).lower()
        grouped.setdefault(sev, []).append(f)
    for sev in SEVERITY_ORDER:
        items = grouped.get(sev)
        if not items:
            continue
        meta = SEVERITY_META.get(sev, {"icon": "⚪", "badge": f"[{sev.upper()}]"})
        st.markdown(f"### {meta['icon']} {meta['badge']} ({len(items)})")
        for f in items:
            check_label = str(f.get("check_name", "check")).replace("_", " ").title()
            affected = f.get("affected_columns") or []
            cols_str = ", ".join(f"`{c}`" for c in affected) if affected else "*(none)*"
            with st.expander(f"{meta['icon']} **{check_label}** — {f.get('check_name')}"):
                st.markdown(f"**Evidence:** {f.get('evidence')}")
                st.markdown(f"**Affected columns:** {cols_str}")
                st.markdown(f"**Confidence:** `{f.get('confidence', 'N/A')}`")
                if f.get("suggested_fix"):
                    st.info(f"💡 **Suggested Fix:** {f.get('suggested_fix')}")


def render_plots(run_id: str) -> None:
    """Show cached plot PNGs for a run, generating them on first click."""
    cache_key = f"plot_data_{run_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = None
    data = st.session_state[cache_key]
    if data is None:
        if st.button("Generate plots", key=f"gen_plots_{run_id}"):
            resp, error = api_post(f"/audits/{run_id}/plots")
            if error:
                if "409" in error or "not completed" in error.lower():
                    st.warning("Audit is not completed yet; plots cannot be generated.")
                else:
                    st.error(error)
                return
            plots: list[tuple[str, bytes]] = []
            for name in resp.get("plots", []):
                img_bytes, err = api_get_bytes(f"/audits/{run_id}/plots/{name}")
                if err:
                    st.error(f"Failed to fetch {name}: {err}")
                    continue
                plots.append((name, img_bytes))
            st.session_state[cache_key] = {
                "plots": plots,
                "skipped": resp.get("skipped", []),
            }
            data = st.session_state[cache_key]
    if data:
        for name, img_bytes in data["plots"]:
            st.image(img_bytes, caption=name)
        for s in data.get("skipped", []):
            reason = s.get("reason") or "skipped"
            plot_name = s.get("plot")
            if plot_name:
                st.caption(f"Skipped: {reason} (plot: {plot_name})")
            else:
                st.caption(f"Skipped: {reason}")


def render_run_view(run_id: str, data: dict[str, Any]) -> None:
    """Render metrics dashboard, Report, Findings, and Plots for a run."""
    status = data.get("status")
    st.markdown("---")
    st.subheader(f"📊 Audit Run `{run_id}` — {status.upper() if status else 'UNKNOWN'}")
    if status == "failed":
        st.error(f"❌ Audit failed: {data.get('error_message', 'Audit encountered an error.')}")
        return
    if status != "completed":
        st.info(f"⏳ Status: {status}")
        return

    # Extract summary metadata
    report_json = data.get("report_json") or {}
    findings = data.get("findings") or report_json.get("findings") or []
    overall_sev = (data.get("overall_severity") or report_json.get("overall_severity") or "info").lower()
    total_findings = len(findings)
    crit_high = sum(
        1 for f in findings if str(f.get("severity", "")).lower() in ("critical", "high")
    )

    # Calculate dataset dimensions
    rows = report_json.get("rows")
    cols = report_json.get("cols")
    if rows is not None and cols is not None:
        dim_str = f"{rows:,} rows × {cols:,} cols"
    else:
        csv_p = data.get("csv_path") or ""
        dim_str = Path(csv_p).name if csv_p else "N/A"

    # Metric tiles
    col1, col2, col3, col4 = st.columns(4)
    sev_icon = SEVERITY_META.get(overall_sev, {}).get("icon", "⚪")
    with col1:
        st.metric("Overall Severity", f"{sev_icon} {overall_sev.upper()}")
    with col2:
        st.metric("Total Findings", total_findings)
    with col3:
        st.metric("Critical & High", crit_high)
    with col4:
        st.metric("Dataset Dimensions", dim_str)

    st.write("")

    tab_report, tab_findings, tab_plots = st.tabs(["📝 Report", "🔍 Findings", "📈 Diagnostic Plots"])
    with tab_report:
        render_report(data.get("report_text"), report_json, run_id)
    with tab_findings:
        render_findings(findings)
    with tab_plots:
        render_plots(run_id)


# --------------------------------------------------------------------------- new audit

def _poll_and_store(run_id: str) -> None:
    """Poll GET /audits/{run_id} every 2 seconds until completed/failed or timeout."""
    placeholder = st.empty()
    deadline = time.time() + MAX_WAIT_SECONDS
    final: dict[str, Any] | None = None
    while time.time() < deadline:
        data, error = api_get(f"/audits/{run_id}")
        if error:
            placeholder.error(error)
            return
        status = data.get("status")
        placeholder.info(f"Status: {status}")
        if status in ("completed", "failed"):
            final = data
            break
        time.sleep(POLL_INTERVAL)
    placeholder.empty()
    if final is None:
        st.warning(
            "Audit is still running. You can find it under Past audits "
            "and open it later."
        )
    else:
        st.session_state["view_run_id"] = run_id
        st.session_state["view_run_data"] = final


def render_new_audit() -> None:
    st.subheader("New audit")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], key="new_audit_file")
    if uploaded is None:
        return
    # Read only the header to list columns, then reset so the full file is sent.
    uploaded.seek(0)
    try:
        header_df = pd.read_csv(uploaded, nrows=0)
        columns = [str(c) for c in header_df.columns]
    except Exception as exc:  # noqa: BLE001 - unparseable upload
        st.error(f"Could not read CSV header: {exc}")
        return
    uploaded.seek(0)
    if not columns:
        st.error("CSV has no columns.")
        return

    target = st.selectbox("Target column", columns, index=0, key="new_audit_target")
    mode = st.radio("Mode", ["agent", "deterministic"], index=0, key="new_audit_mode")
    with st.expander("Fairness options"):
        sensitive = st.multiselect("Sensitive columns", columns, key="new_audit_sensitive")
        favorable = st.text_input("Favorable value", key="new_audit_favorable")
        semantic_override = st.text_input(
            "Semantic overrides (COLUMN=TYPE, ...)",
            key="new_audit_semantic_override",
        )

    if st.button("Run audit", key="new_audit_run"):
        uploaded.seek(0)
        file_bytes = uploaded.read()
        files = {"file": (uploaded.name, file_bytes, "text/csv")}
        post_data: dict[str, Any] = {"target": target, "mode": mode}
        if sensitive:
            post_data["sensitive"] = ",".join(sensitive)
        if favorable:
            post_data["favorable"] = favorable
        if semantic_override:
            post_data["semantic_override"] = semantic_override
        resp, error = api_post("/audits", data=post_data, files=files)
        if error:
            st.error(error)
            return
        run_id = resp.get("run_id")
        if not run_id:
            st.error("API did not return a run id.")
            return
        st.session_state["view_run_id"] = run_id
        st.session_state["view_run_data"] = {"run_id": run_id, "status": "queued"}
        _poll_and_store(run_id)


# --------------------------------------------------------------------------- past audits

def render_past_audits() -> None:
    st.subheader("Past audits")
    resp, error = api_get("/audits")
    if error:
        st.error(error)
        return
    runs = resp if isinstance(resp, list) else []
    if not runs:
        st.info("No past audits.")
        return
    rows: list[dict[str, Any]] = []
    for r in runs:
        csv_path = r.get("csv_path") or ""
        file_name = csv_path.split("/")[-1] if csv_path else ""
        rows.append(
            {
                "Run ID": r.get("run_id"),
                "File": file_name,
                "Target": r.get("target_column") or "",
                "Status": r.get("status"),
                "Mode": r.get("mode") or "",
                "Created": r.get("created_at") or "",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    options = {
        f"{r.get('run_id')} — {r.get('status')}": r.get("run_id") for r in runs
    }
    choice = st.selectbox("Open a past run", options=list(options.keys()), key="past_run_select")
    if choice and st.button("View run", key="view_past_run"):
        run_id = options[choice]
        data, error = api_get(f"/audits/{run_id}")
        if error:
            st.error(error)
            return
        st.session_state["view_run_id"] = run_id
        st.session_state["view_run_data"] = data


# --------------------------------------------------------------------------- main

def main() -> None:
    st.set_page_config(
        page_title="Data Detective",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🔍 Data Detective")
    st.caption("Evidence-based dataset quality, leakage, and fairness auditing for machine learning.")

    render_sidebar_health()

    health, error = st.session_state.get("health", (None, None))
    if error:
        st.error(error)

    tab_new, tab_past = st.tabs(["🚀 New audit", "📜 Past audits"])
    with tab_new:
        render_new_audit()
    with tab_past:
        render_past_audits()

    run_id = st.session_state.get("view_run_id")
    data = st.session_state.get("view_run_data")
    if run_id and data:
        render_run_view(run_id, data)


if __name__ == "__main__":
    main()
