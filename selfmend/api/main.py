"""
SelfMend FastAPI application.

Endpoints:
  POST /run              : Run pipeline on next available batch (or injected fault)
  POST /inject-fault     : Inject a specific fault into a batch and run pipeline
  GET  /runs             : List recent pipeline runs
  GET  /incidents/{id}   : Full incident trace (failure -> diagnosis -> decision -> fix)
  GET  /metrics          : Aggregated metrics (detection recall, faithfulness trend, etc.)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

ROOT = Path(__file__).resolve().parents[1]
BATCHES_DIR = ROOT / "data" / "batches"
STATS_JSON  = ROOT / "data" / "clean" / "baseline_stats.json"
REF_PQ      = ROOT / "data" / "clean" / "reference.parquet"

app = FastAPI(
    title="SelfMend API",
    description="Self-healing data pipeline — audit and control API",
    version="1.0.0",
)

# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    batch_index: Optional[int] = Field(None, description="Which batch to run (0-indexed). Defaults to next unprocessed.")
    fault_type:  Optional[str] = Field(None, description="Inject this fault type before running.")
    rng_seed:    int            = Field(42,  description="RNG seed for fault injection.")


class InjectFaultRequest(BaseModel):
    fault_type:  str = Field(..., description="Fault type from FAULT_REGISTRY")
    batch_index: int = Field(0,   description="Which batch to inject into")
    rng_seed:    int = Field(42)


class RunResponse(BaseModel):
    run_id:           str
    batch_id:         str
    decision:         str
    has_failures:     bool
    faithfulness_score: float
    remediation_success: Optional[bool]
    message:          str


# ── In-memory run store (fallback when DB not connected) ──────────────────────

_run_store: dict[str, dict] = {}


def _save_run(state: dict) -> None:
    """Persist run state to DB and in-memory store."""
    run_id = state.get("run_id", str(uuid.uuid4()))
    _run_store[run_id] = state
    try:
        from db.session import get_db_session  # noqa: PLC0415
        from db.models import (  # noqa: PLC0415
            PipelineRun, CheckResult, Diagnosis, Decision, Remediation
        )
        with get_db_session() as db:
            det = state.get("detection_report") or {}
            run = PipelineRun(
                run_id=run_id,
                batch_id=state.get("batch_id", ""),
                has_failures=det.get("has_failures", False),
                gx_passed=det.get("gx_passed", True),
                drift_detected=det.get("drift_detected", False),
                decision=state.get("decision", ""),
                status="done",
            )
            db.add(run)

            for fc in det.get("failed_checks", []):
                db.add(CheckResult(
                    run_id=run_id,
                    layer=fc.get("layer", ""),
                    check_type=fc.get("check_type", ""),
                    column_name=fc.get("column", ""),
                    severity=fc.get("severity", ""),
                    success=False,
                    detail=fc.get("detail", {}),
                ))

            diag = state.get("diagnosis") or {}
            if diag:
                db.add(Diagnosis(
                    run_id=run_id,
                    root_cause=diag.get("root_cause", ""),
                    proposed_action=diag.get("proposed_action", ""),
                    confidence=diag.get("confidence", 0.0),
                    risk_level=diag.get("risk_level", ""),
                    faithfulness_score=state.get("faithfulness_score", 0.0),
                    faithfulness_passed=state.get("faithfulness_passed", False),
                    evidence_used=diag.get("evidence_used", []),
                    retrieved_incidents=state.get("retrieved_incidents", []),
                    diagnosis_raw=state.get("diagnosis_raw", ""),
                ))

            db.add(Decision(
                run_id=run_id,
                decision=state.get("decision", ""),
                reason=state.get("decision_reason", "") or state.get("escalation_reason", ""),
                faithfulness_score=state.get("faithfulness_score", 0.0),
            ))

            if state.get("remediation_action"):
                det_post = (state.get("remediation_detail") or {}).get("post_detection", {})
                db.add(Remediation(
                    run_id=run_id,
                    action=state["remediation_action"],
                    success=bool(state.get("remediation_success", False)),
                    promoted=bool((state.get("remediation_detail") or {}).get("promoted", False)),
                    pre_check=det,
                    post_check=det_post,
                    reason=(state.get("remediation_detail") or {}).get("reason", ""),
                ))

    except Exception as exc:
        log.warning("DB write failed (non-fatal): %s", exc)


def _run_pipeline_task(batch_df: pd.DataFrame, batch_id: str) -> dict:
    """Run the full pipeline and save results."""
    import sys; sys.path.insert(0, str(ROOT))
    os.environ.setdefault("FAITHFULNESS_THRESHOLD", "0.50")
    os.environ.setdefault("CONFIDENCE_THRESHOLD",   "0.60")
    from agents.graph import run_pipeline  # noqa: PLC0415
    state = run_pipeline(batch_df, batch_id=batch_id)
    _save_run(state)
    return state


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run_pipeline_endpoint(req: RunRequest):
    """Run the pipeline on a batch (optionally with an injected fault)."""
    batches = sorted(BATCHES_DIR.glob("batch_*.parquet"))
    if not batches:
        raise HTTPException(status_code=503, detail="No batches available. Run ETL first.")

    idx = req.batch_index if req.batch_index is not None else 0
    if idx >= len(batches):
        idx = idx % len(batches)

    batch_df  = pd.read_parquet(batches[idx])
    batch_id  = batches[idx].stem

    if req.fault_type:
        from faults.injector import inject_fault, FAULT_REGISTRY  # noqa: PLC0415
        if req.fault_type not in FAULT_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unknown fault_type: {req.fault_type}")
        batch_df, _ = inject_fault(batch_df, req.fault_type, rng_seed=req.rng_seed)
        batch_id = f"{batch_id}__{req.fault_type}"

    state = _run_pipeline_task(batch_df, batch_id)

    return RunResponse(
        run_id=state.get("run_id", ""),
        batch_id=batch_id,
        decision=state.get("decision", ""),
        has_failures=bool((state.get("detection_report") or {}).get("has_failures", False)),
        faithfulness_score=state.get("faithfulness_score", 0.0),
        remediation_success=state.get("remediation_success"),
        message=(
            f"Remediation: {state.get('remediation_action')} "
            f"(success={state.get('remediation_success')})"
            if state.get("decision") == "auto_fix"
            else f"Escalated: {state.get('escalation_reason', '')[:100]}"
        ),
    )


@app.post("/inject-fault", response_model=RunResponse)
def inject_fault_endpoint(req: InjectFaultRequest):
    """Inject a specific fault and run the full pipeline."""
    return run_pipeline_endpoint(RunRequest(
        batch_index=req.batch_index,
        fault_type=req.fault_type,
        rng_seed=req.rng_seed,
    ))


@app.get("/runs")
def list_runs(limit: int = 20):
    """Return recent pipeline runs (from DB or in-memory fallback)."""
    try:
        from db.session import get_db_session  # noqa: PLC0415
        from db.models import PipelineRun  # noqa: PLC0415
        from sqlalchemy import desc  # noqa: PLC0415
        with get_db_session() as db:
            rows = db.query(PipelineRun).order_by(desc(PipelineRun.created_at)).limit(limit).all()
            return [
                {
                    "run_id": r.run_id, "batch_id": r.batch_id,
                    "has_failures": r.has_failures, "gx_passed": r.gx_passed,
                    "drift_detected": r.drift_detected, "decision": r.decision,
                    "status": r.status, "created_at": str(r.created_at),
                }
                for r in rows
            ]
    except Exception:
        # Fallback: in-memory store
        runs = list(_run_store.values())[-limit:]
        return [
            {
                "run_id": r.get("run_id"), "batch_id": r.get("batch_id"),
                "has_failures": (r.get("detection_report") or {}).get("has_failures"),
                "decision": r.get("decision"),
                "faithfulness_score": r.get("faithfulness_score"),
            }
            for r in reversed(runs)
        ]


@app.get("/incidents/{run_id}")
def get_incident(run_id: str):
    """Full incident trace for a run."""
    if run_id in _run_store:
        state = _run_store[run_id]
        return {
            "run_id": run_id,
            "batch_id": state.get("batch_id"),
            "detection": state.get("detection_report"),
            "retrieved_incidents": state.get("retrieved_incidents"),
            "diagnosis": state.get("diagnosis"),
            "faithfulness_score": state.get("faithfulness_score"),
            "faithfulness_passed": state.get("faithfulness_passed"),
            "decision": state.get("decision"),
            "decision_reason": state.get("decision_reason"),
            "remediation_action": state.get("remediation_action"),
            "remediation_success": state.get("remediation_success"),
            "escalation_reason": state.get("escalation_reason"),
        }
    try:
        from db.session import get_db_session  # noqa: PLC0415
        from db.models import PipelineRun, Diagnosis, Decision, Remediation  # noqa: PLC0415
        with get_db_session() as db:
            run   = db.query(PipelineRun).filter_by(run_id=run_id).first()
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            diag  = db.query(Diagnosis).filter_by(run_id=run_id).first()
            dec   = db.query(Decision).filter_by(run_id=run_id).first()
            rem   = db.query(Remediation).filter_by(run_id=run_id).first()
            return {
                "run_id": run_id,
                "batch_id": run.batch_id,
                "decision": run.decision,
                "diagnosis": {
                    "root_cause": diag.root_cause if diag else None,
                    "proposed_action": diag.proposed_action if diag else None,
                    "faithfulness_score": diag.faithfulness_score if diag else None,
                    "retrieved_incidents": diag.retrieved_incidents if diag else None,
                } if diag else None,
                "decision_reason": dec.reason if dec else None,
                "remediation": {
                    "action": rem.action, "success": rem.success, "promoted": rem.promoted
                } if rem else None,
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
def get_metrics():
    """Aggregated pipeline metrics."""
    runs = list(_run_store.values())
    if not runs:
        try:
            from db.session import get_db_session  # noqa: PLC0415
            from db.models import PipelineRun, Diagnosis, Remediation  # noqa: PLC0415
            with get_db_session() as db:
                total = db.query(PipelineRun).count()
                n_failures = db.query(PipelineRun).filter_by(has_failures=True).count()
                n_autofix  = db.query(PipelineRun).filter_by(decision="auto_fix").count()
                n_escalate = db.query(PipelineRun).filter_by(decision="escalate").count()
                faithfulness_scores = [
                    d.faithfulness_score for d in db.query(Diagnosis).all()
                    if d.faithfulness_score is not None
                ]
                return {
                    "total_runs": total,
                    "failure_rate": n_failures / total if total else 0,
                    "auto_fix_rate": n_autofix / total if total else 0,
                    "escalation_rate": n_escalate / total if total else 0,
                    "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
                }
        except Exception:
            return {"total_runs": 0}

    n_failures = sum(1 for r in runs if (r.get("detection_report") or {}).get("has_failures"))
    n_autofix  = sum(1 for r in runs if r.get("decision") == "auto_fix")
    n_escalate = sum(1 for r in runs if r.get("decision") == "escalate")
    faith_scores = [r.get("faithfulness_score", 0) for r in runs]
    return {
        "total_runs": len(runs),
        "failure_rate": n_failures / len(runs),
        "auto_fix_rate": n_autofix / len(runs),
        "escalation_rate": n_escalate / len(runs),
        "avg_faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else 0,
        "faithfulness_trend": faith_scores[-10:],
    }
