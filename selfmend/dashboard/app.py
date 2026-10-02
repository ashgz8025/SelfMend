"""
SelfMend Streamlit dashboard.

Pages / sections:
  1. Live run table        — recent pipeline runs with status
  2. Incident trace        — full trace for a selected run
  3. Faithfulness chart    — faithfulness scores over runs
  4. Manual run trigger    — inject a fault and run from the UI

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use offline thresholds (no Groq key needed for dashboard)
os.environ.setdefault("FAITHFULNESS_THRESHOLD", "0.50")
os.environ.setdefault("CONFIDENCE_THRESHOLD",   "0.60")

st.set_page_config(
    page_title="SelfMend Dashboard",
    page_icon="🔧",
    layout="wide",
)

# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_api_runs():
    """Fetch runs from the API (tries localhost:8000, falls back to DB/in-memory)."""
    try:
        import httpx
        resp = httpx.get("http://localhost:8000/runs", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def _model_to_dict(model):
    """Convert a SQLAlchemy instance into a plain dict before the session closes."""
    if model is None:
        return None
    if isinstance(model, dict):
        return model
    data = {}
    for column in getattr(model, "__table__", {}).columns:
        key = column.name
        value = getattr(model, key, None)
        data[key] = value
    return data


def get_runs_from_db() -> list[dict]:
    try:
        from db.session import get_db_session
        from db.models import PipelineRun, Diagnosis
        from sqlalchemy import desc
        with get_db_session() as db:
            rows = db.query(PipelineRun).order_by(desc(PipelineRun.created_at)).limit(100).all()
            results = []
            for r in rows:
                diag = db.query(Diagnosis).filter_by(run_id=r.run_id).first()
                results.append({
                    "run_id": r.run_id,
                    "batch_id": r.batch_id,
                    "has_failures": r.has_failures,
                    "gx_passed": r.gx_passed,
                    "drift_detected": r.drift_detected,
                    "decision": r.decision,
                    "faithfulness_score": diag.faithfulness_score if diag else None,
                    "created_at": str(r.created_at),
                })
            return results
    except Exception:
        return []


def get_incident_trace(run_id: str) -> dict:
    try:
        from db.session import get_db_session
        from db.models import PipelineRun, CheckResult, Diagnosis, Decision, Remediation
        with get_db_session() as db:
            run   = db.query(PipelineRun).filter_by(run_id=run_id).first()
            checks= db.query(CheckResult).filter_by(run_id=run_id).all()
            diag  = db.query(Diagnosis).filter_by(run_id=run_id).first()
            dec   = db.query(Decision).filter_by(run_id=run_id).first()
            rem   = db.query(Remediation).filter_by(run_id=run_id).first()
            return {
                "run": _model_to_dict(run),
                "checks": [_model_to_dict(c) for c in checks],
                "diag": _model_to_dict(diag),
                "dec": _model_to_dict(dec),
                "rem": _model_to_dict(rem),
            }
    except Exception as exc:
        return {"error": str(exc)}


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("SelfMend")
st.sidebar.caption("Self-healing data pipeline")
page = st.sidebar.radio("Navigate", ["Live Runs", "Incident Trace", "Faithfulness Chart", "Run Pipeline"])

# ── Page: Live Runs ────────────────────────────────────────────────────────────

if page == "Live Runs":
    st.title("Pipeline Runs")
    st.caption("Showing the most recent pipeline runs from the audit log.")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh"):
            st.cache_resource.clear()

    runs = get_runs_from_db()
    if not runs:
        st.info("No runs recorded yet. Use the 'Run Pipeline' page to run the pipeline.")
    else:
        df = pd.DataFrame(runs)
        cols_display = [c for c in ["run_id","batch_id","decision","has_failures",
                                     "gx_passed","drift_detected","faithfulness_score","created_at"]
                        if c in df.columns]
        def _color_decision(val):
            if val == "auto_fix":   return "background-color: #d4edda"
            if val == "escalate":   return "background-color: #f8d7da"
            return ""

        st.dataframe(
            df[cols_display].style.map(_color_decision, subset=["decision"]),
            use_container_width=True,
        )
        st.caption(f"Total runs: {len(runs)}")


# ── Page: Incident Trace ───────────────────────────────────────────────────────

elif page == "Incident Trace":
    st.title("Incident Trace")
    st.caption("Full pipeline trace: failure -> retrieved incidents -> diagnosis -> faithfulness -> decision -> fix result")

    runs = get_runs_from_db()
    if not runs:
        st.info("No runs available yet.")
    else:
        run_ids = [r["run_id"] for r in runs]
        selected_id = st.selectbox("Select run ID", run_ids)
        if selected_id:
            trace = get_incident_trace(selected_id)
            if "error" in trace:
                st.error(f"DB error: {trace['error']}")
            else:
                run  = trace.get("run")
                diag = trace.get("diag")
                dec  = trace.get("dec")
                rem  = trace.get("rem")
                checks = trace.get("checks", [])

                # Step 1: Detection
                st.subheader("1. Detection")
                col1, col2, col3 = st.columns(3)
                col1.metric("GX Passed",  "Yes" if (run and run.get("gx_passed")) else "No")
                col2.metric("Drift",       "Detected" if (run and run.get("drift_detected")) else "None")
                col3.metric("Has Failures","Yes" if (run and run.get("has_failures")) else "No")
                if checks:
                    failed = [c for c in checks if not c.get("success")]
                    if failed:
                        st.error(f"{len(failed)} failed check(s):")
                        for fc in failed:
                            st.write(f"  [{(fc.get('severity') or '').upper()}] {fc.get('layer')}:{fc.get('check_type')} on `{fc.get('column_name')}`")

                # Step 2: Retrieved incidents
                st.subheader("2. Retrieved Incidents")
                if diag and diag.get("retrieved_incidents"):
                    for inc in diag["retrieved_incidents"][:3]:
                        with st.expander(f"[{inc.get('similarity_score','?'):.3f}] {inc.get('incident_id','?')} — {inc.get('failure_description','')[:60]}..."):
                            st.write("**Root cause:**", inc.get("root_cause", ""))
                            st.write("**Fix applied:**", inc.get("fix_applied", ""))
                else:
                    st.write("No incidents retrieved.")

                # Step 3: Diagnosis
                st.subheader("3. Diagnosis")
                if diag:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Proposed Action", diag.get("proposed_action") or "N/A")
                    col2.metric("Confidence",      f"{diag.get('confidence', 0):.0%}" if diag.get("confidence") else "N/A")
                    col3.metric("Risk Level",      diag.get("risk_level") or "N/A")
                    st.write("**Root cause:**", diag.get("root_cause"))

                # Step 4: Faithfulness
                st.subheader("4. Faithfulness Score")
                if diag:
                    score = diag.get("faithfulness_score") or 0
                    threshold = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.50"))
                    color = "normal" if score >= threshold else "inverse"
                    st.metric("Faithfulness", f"{score:.3f}", delta=f"threshold={threshold}", delta_color=color)

                # Step 5: Decision
                st.subheader("5. Policy Decision")
                if run:
                    decision = run.get("decision") or ""
                    decision_color = "success" if decision == "auto_fix" else "error"
                    st.markdown(f"**Decision:** :{decision_color}[{decision.upper()}]")
                if dec:
                    st.write("Reason:", dec.get("reason"))

                # Step 6: Remediation / Escalation
                st.subheader("6. Outcome")
                if rem:
                    col1, col2 = st.columns(2)
                    col1.metric("Action",   rem.get("action") or "N/A")
                    col2.metric("Success",  "Yes" if rem.get("success") else "No")
                    st.write("Promoted to clean:", "Yes" if rem.get("promoted") else "No")
                elif run and run.get("decision") == "escalate":
                    st.warning("Escalated for human review.")


# ── Page: Faithfulness Chart ──────────────────────────────────────────────────

elif page == "Faithfulness Chart":
    st.title("Faithfulness Score Over Runs")

    runs = get_runs_from_db()
    if not runs:
        st.info("No runs yet.")
    else:
        df = pd.DataFrame(runs)
        if "faithfulness_score" in df.columns and df["faithfulness_score"].notna().any():
            chart_df = df[df["faithfulness_score"].notna()][["batch_id","faithfulness_score"]].reset_index(drop=True)
            st.line_chart(chart_df.set_index("batch_id")["faithfulness_score"])
            threshold = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.50"))
            st.caption(f"Dashed line = faithfulness threshold ({threshold})")
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Faithfulness", f"{chart_df['faithfulness_score'].mean():.3f}")
            col2.metric("Min Faithfulness", f"{chart_df['faithfulness_score'].min():.3f}")
            col3.metric("Max Faithfulness", f"{chart_df['faithfulness_score'].max():.3f}")
        else:
            st.info("No faithfulness scores recorded yet.")


# ── Page: Run Pipeline ────────────────────────────────────────────────────────

elif page == "Run Pipeline":
    st.title("Run Pipeline")
    st.caption("Inject a fault and run the self-healing pipeline from the dashboard.")

    from faults.injector import FAULT_REGISTRY, FAULT_CLASSES

    fault_class = st.selectbox("Fault class", ["(none)", "schema", "data_quality", "distribution"])
    if fault_class == "(none)":
        candidates = list(FAULT_REGISTRY.keys())
    else:
        candidates = FAULT_CLASSES.get(fault_class, [])

    fault_type  = st.selectbox("Fault type", candidates)
    batch_index = st.slider("Batch index", 0, 19, 0)
    rng_seed    = st.number_input("RNG seed", value=42, step=1)

    if st.button("Run pipeline"):
        with st.spinner("Running pipeline..."):
            batches = sorted((ROOT / "data/batches").glob("batch_*.parquet"))
            batch_df = pd.read_parquet(batches[batch_index % len(batches)])

            from faults.injector import inject_fault
            corrupted, label = inject_fault(batch_df, fault_type, rng_seed=int(rng_seed))

            from agents.graph import run_pipeline
            state = run_pipeline(corrupted, batch_id=f"dash_{fault_type}_b{batch_index}")

            # Save to DB
            try:
                from api.main import _save_run
                _save_run(state)
            except Exception:
                pass

        st.success("Pipeline complete!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Decision", state.get("decision", "?").upper())
        col2.metric("Faithfulness", f"{state.get('faithfulness_score', 0):.3f}")
        if state.get("decision") == "auto_fix":
            col3.metric("Remediation",
                        state.get("remediation_action", "N/A"),
                        delta="SUCCESS" if state.get("remediation_success") else "FAILED")
        else:
            col3.metric("Escalation Reason",
                        (state.get("escalation_reason") or "")[:50])

        with st.expander("Full state"):
            safe_state = {k: v for k, v in state.items()
                          if k not in ("batch_df", "reference_df")}
            st.json(safe_state)
