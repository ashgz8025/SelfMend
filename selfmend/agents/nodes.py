"""
LangGraph node implementations for SelfMend.

Nodes:
  detector_node    : runs GX + Evidently, populates detection_report
  retriever_node   : fetches top-k incidents from ChromaDB
  diagnoser_node   : calls Groq LLM with failure report + retrieved evidence
  verifier_node    : scores diagnosis faithfulness
  policy_node      : applies policy gate → "auto_fix" | "escalate"
  remediator_node  : applies whitelisted fix to a COPY, re-validates
  escalate_node    : records escalation reason

Each node receives the full PipelineState dict and returns a partial update.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from agents.state import PipelineState, DiagnosisOutput
from detection.detector import detect, DetectionReport
from memory.incident_store import get_store
from trust.faithfulness import score_faithfulness

log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
STATS_JSON = ROOT / "data" / "clean" / "baseline_stats.json"
REF_PQ     = ROOT / "data" / "clean" / "reference.parquet"

# ── LLM setup ─────────────────────────────────────────────────────────────────

def _get_llm():
    """Build Groq LLM, or return None for offline mode."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("gsk_REPLACE"):
        return None
    try:
        from langchain_groq import ChatGroq  # noqa: PLC0415
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            api_key=api_key,
            max_retries=2,
        )
    except Exception as exc:
        log.error("LLM init failed: %s", exc)
        return None


def _offline_diagnosis(failure_summary: str, incidents: list[dict]) -> DiagnosisOutput:
    """
    Deterministic offline fallback diagnosis.
    Maps known failure patterns to the correct whitelisted fix.
    """
    from policy.gate import WHITELISTED_ACTIONS  # noqa: PLC0415

    summary_lower = failure_summary.lower()

    # Check for schema failures first (missing columns → must escalate)
    if "expectcolumntoexist" in summary_lower or "column" in summary_lower and ("missing" in summary_lower or "not found" in summary_lower):
        fix        = "escalate"
        root_cause = "A required column is missing from the batch. Schema validation failed."
        risk       = "high"
        confidence = 0.80
    # Check for dtype failures
    elif "dtype" in summary_lower and ("change" in summary_lower or "string" in summary_lower or "object" in summary_lower):
        fix        = "escalate"
        root_cause = "Column dtype changed unexpectedly. Schema regression detected."
        risk       = "high"
        confidence = 0.80
    elif "total" in summary_lower and ("mismatch" in summary_lower or "integrity" in summary_lower):
        fix        = "restore_total_from_quantity_price"
        root_cause = "Total Spent does not match Quantity x Price Per Unit. Derivation mismatch detected."
        risk       = "low"
        confidence = 0.75
    elif "duplicate" in summary_lower:
        fix        = "drop_duplicates"
        root_cause = "Duplicate rows detected in the batch. Likely a double-ingestion or retry event."
        risk       = "low"
        confidence = 0.75
    elif ("sentinel" in summary_lower or "inset" in summary_lower or "category" in summary_lower
          or "allowedvalue" in summary_lower):
        fix        = "normalize_sentinels_to_null"
        root_cause = "Sentinel strings (ERROR/UNKNOWN) re-appeared in categorical columns."
        risk       = "low"
        confidence = 0.72
    elif ("null" in summary_lower or "notnull" in summary_lower or "missing" in summary_lower
          or "notbenull" in summary_lower):
        fix        = "impute_nulls"
        root_cause = "Null rate spike detected. Missing values exceed baseline by more than the allowed margin."
        risk       = "low"
        confidence = 0.72
    elif "range" in summary_lower or "between" in summary_lower or "outlier" in summary_lower or "clip" in summary_lower:
        fix        = "clip_outliers"
        root_cause = "Out-of-range values detected. Values exceed expected min/max range."
        risk       = "low"
        confidence = 0.72
    elif "drift" in summary_lower:
        fix        = "escalate"
        root_cause = "Statistical drift detected. Distribution has shifted from baseline."
        risk       = "medium"
        confidence = 0.70
    elif incidents:
        best = incidents[0]
        fix  = best.get("fix_applied", "escalate")
        root_cause = best.get("root_cause", "Unknown — matched from incident store.")
        risk       = "low" if fix != "escalated" else "high"
        confidence = 0.65
    else:
        fix        = "escalate"
        root_cause = "Unknown failure pattern — no matching incidents and no LLM."
        risk       = "high"
        confidence = 0.40

    # Normalise "escalated" → "escalate"
    if fix == "escalated" or fix not in WHITELISTED_ACTIONS:
        fix        = "escalate"
        risk       = "high"
        confidence = 0.40

    return DiagnosisOutput(
        root_cause=root_cause,
        evidence_used=[i.get("incident_id", "") for i in incidents[:3]],
        proposed_action=fix,
        confidence=confidence,
        risk_level=risk,
    )


# ── Node: detector ─────────────────────────────────────────────────────────────

def detector_node(state: PipelineState) -> dict:
    """Run GX + Evidently detection. Populates detection_report + failure_summary."""
    batch_df      = state["batch_df"]
    reference_df  = state.get("reference_df")
    stats         = state.get("stats")
    batch_id      = state.get("batch_id", "unknown")

    if stats is None:
        stats = json.loads(STATS_JSON.read_text())
    if reference_df is None:
        reference_df = pd.read_parquet(REF_PQ)

    report = detect(batch_df, reference_df=reference_df, stats=stats, batch_id=batch_id)
    summary = report.summary_text()

    log.info("[detector] %s", summary.split("\n")[0])
    return {
        "detection_report": report.to_dict(),
        "failure_summary": summary,
        "stats": stats,
        "reference_df": reference_df,
    }


# ── Node: retriever ────────────────────────────────────────────────────────────

def retriever_node(state: PipelineState) -> dict:
    """Retrieve top-3 similar incidents from ChromaDB."""
    query = state.get("failure_summary", "data quality failure")
    store = get_store()
    hits  = store.retrieve(query, top_k=3)

    incidents = []
    for h in hits:
        incidents.append({
            "incident_id": h.incident_id,
            "failure_description": h.failure_description,
            "root_cause": h.root_cause,
            "fix_applied": h.fix_applied,
            "outcome": h.outcome,
            "fault_class": h.fault_class,
            "similarity_score": h.similarity_score,
        })

    log.info("[retriever] Found %d incidents (scores: %s)",
             len(incidents),
             [round(i["similarity_score"], 3) for i in incidents])
    return {"retrieved_incidents": incidents}


# ── Node: diagnoser ────────────────────────────────────────────────────────────

def diagnoser_node(state: PipelineState) -> dict:
    """
    Call Groq LLM to produce a structured diagnosis.
    Falls back to offline diagnosis if LLM is unavailable.
    """
    failure_summary = state.get("failure_summary", "")
    incidents       = state.get("retrieved_incidents", [])

    llm = _get_llm()

    if llm is None:
        diagnosis = _offline_diagnosis(failure_summary, incidents)
        raw_text  = (
            f"Root cause: {diagnosis['root_cause']}. "
            f"Proposed action: {diagnosis['proposed_action']}. "
            f"Confidence: {diagnosis['confidence']}. "
            f"Risk: {diagnosis['risk_level']}."
        )
        log.info("[diagnoser] Offline mode — using fallback diagnosis.")
        return {"diagnosis": diagnosis, "diagnosis_raw": raw_text}

    # Build evidence block for the prompt
    evidence_block = "\n\n".join(
        f"Incident {i+1} (id={inc['incident_id']}, similarity={inc['similarity_score']:.3f}):\n"
        f"  Description: {inc['failure_description']}\n"
        f"  Root cause: {inc['root_cause']}\n"
        f"  Fix applied: {inc['fix_applied']}\n"
        f"  Outcome: {inc['outcome']}"
        for i, inc in enumerate(incidents)
    )

    from policy.gate import WHITELISTED_ACTIONS  # noqa: PLC0415
    whitelist_str = ", ".join(WHITELISTED_ACTIONS)

    prompt = f"""You are a data quality expert diagnosing a pipeline failure.

FAILURE REPORT:
{failure_summary}

RETRIEVED SIMILAR INCIDENTS:
{evidence_block}

WHITELISTED FIX ACTIONS: {whitelist_str}

Based on the failure report and the retrieved incidents, provide a diagnosis.
Your response MUST be valid JSON with exactly these fields:
{{
  "root_cause": "<one sentence explaining why this failure occurred>",
  "evidence_used": ["<incident_id_1>", "<incident_id_2>"],
  "proposed_action": "<one of the whitelisted actions, or 'escalate' if none apply>",
  "confidence": <float 0.0-1.0>,
  "risk_level": "<low|medium|high>"
}}

Rules:
- proposed_action MUST be one of: {whitelist_str}, escalate
- confidence reflects how certain you are about the diagnosis
- risk_level: low = safe auto-fix, medium = needs monitoring, high = requires human review
- Only cite evidence that directly supports your diagnosis
"""

    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    try:
        response = llm.invoke([
            SystemMessage(content="You are a data quality expert. Respond with valid JSON only."),
            HumanMessage(content=prompt),
        ])
        raw_text = response.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            raise ValueError(f"No JSON found in LLM response: {raw_text[:200]}")

        # Validate and coerce
        diagnosis = DiagnosisOutput(
            root_cause=str(parsed.get("root_cause", "Unknown")),
            evidence_used=list(parsed.get("evidence_used", [])),
            proposed_action=str(parsed.get("proposed_action", "escalate")),
            confidence=float(parsed.get("confidence", 0.5)),
            risk_level=str(parsed.get("risk_level", "medium")),
        )
        log.info("[diagnoser] action=%s  confidence=%.2f  risk=%s",
                 diagnosis["proposed_action"], diagnosis["confidence"], diagnosis["risk_level"])

    except Exception as exc:
        log.error("[diagnoser] LLM call failed: %s", exc)
        diagnosis = _offline_diagnosis(failure_summary, incidents)
        raw_text  = str(exc)

    return {"diagnosis": diagnosis, "diagnosis_raw": raw_text}


# ── Node: verifier ─────────────────────────────────────────────────────────────

def verifier_node(state: PipelineState) -> dict:
    """
    Score faithfulness of the diagnosis against retrieved incident evidence.
    Sets faithfulness_score and faithfulness_passed.

    If state["faithfulness_override"] is set (for testing), uses that value directly.
    """
    # ── Test override path ────────────────────────────────────────────────
    override = state.get("faithfulness_override")
    if override is not None:
        threshold = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.8"))
        log.info("[verifier] faithfulness_override=%.3f  threshold=%.2f  passed=%s",
                 override, threshold, override >= threshold)
        return {
            "faithfulness_score": override,
            "faithfulness_passed": override >= threshold,
        }

    # ── Normal scoring path ───────────────────────────────────────────────
    diagnosis_raw = state.get("diagnosis_raw", "")
    incidents     = state.get("retrieved_incidents", [])

    evidence_texts = [
        f"{i.get('failure_description', '')} {i.get('root_cause', '')}"
        for i in incidents
    ]

    from trust.faithfulness import score_faithfulness  # noqa: PLC0415

    llm = _get_llm()
    score = score_faithfulness(diagnosis_raw, evidence_texts, llm=llm)

    # Threshold from env/config (default 0.8)
    threshold = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.8"))
    passed    = score >= threshold

    log.info("[verifier] faithfulness=%.3f  threshold=%.2f  passed=%s", score, threshold, passed)
    return {
        "faithfulness_score": score,
        "faithfulness_passed": passed,
    }


# ── Node: policy ───────────────────────────────────────────────────────────────

def policy_node(state: PipelineState) -> dict:
    """
    Apply policy gate: decides auto_fix or escalate.
    Logic is in policy/gate.py — this node just calls it.
    """
    from policy.gate import evaluate_policy  # noqa: PLC0415

    decision, reason = evaluate_policy(state)
    log.info("[policy] decision=%s  reason=%s", decision, reason[:80])
    return {"decision": decision, "decision_reason": reason}


# ── Node: remediator ───────────────────────────────────────────────────────────

def remediator_node(state: PipelineState) -> dict:
    """
    Apply the whitelisted fix to a COPY of the batch.
    Re-runs detection. Promotes to clean if detection passes.
    """
    from remediation.executor import execute_remediation  # noqa: PLC0415

    diagnosis = state.get("diagnosis", {})
    action    = diagnosis.get("proposed_action", "escalate") if diagnosis else "escalate"
    batch_df  = state["batch_df"]
    batch_id  = state.get("batch_id", "unknown")

    result = execute_remediation(
        batch_df=batch_df,
        action=action,
        batch_id=batch_id,
        stats=state.get("stats"),
        reference_df=state.get("reference_df"),
    )

    log.info("[remediator] action=%s  success=%s", action, result["success"])
    return {
        "remediation_action": action,
        "remediation_success": result["success"],
        "remediation_detail": result,
    }


# ── Node: escalate ─────────────────────────────────────────────────────────────

def escalate_node(state: PipelineState) -> dict:
    """Record escalation reason."""
    reason_parts = []

    if not state.get("faithfulness_passed", True):
        reason_parts.append(
            f"Low faithfulness score ({state.get('faithfulness_score', 0):.2f} < threshold)"
        )
    decision_reason = state.get("decision_reason", "")
    if decision_reason:
        reason_parts.append(decision_reason)

    reason = "; ".join(reason_parts) or "Policy gate: manual review required."
    log.info("[escalate] %s", reason)
    return {"escalation_reason": reason, "decision": "escalate"}
