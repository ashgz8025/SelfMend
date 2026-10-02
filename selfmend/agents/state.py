"""
LangGraph pipeline state for SelfMend.

All nodes read from and write to PipelineState, which is a TypedDict
passed through the StateGraph.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

import pandas as pd


class DiagnosisOutput(TypedDict):
    root_cause: str
    evidence_used: list[str]        # list of incident_ids cited
    proposed_action: str            # must be a whitelisted action
    confidence: float               # 0.0 – 1.0
    risk_level: str                 # "low" | "medium" | "high"


class PipelineState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────
    batch_id: str
    batch_df: Any                   # pd.DataFrame  (not serialisable by TypedDict, use Any)
    reference_df: Optional[Any]
    stats: Optional[dict]

    # ── Detection ─────────────────────────────────────────────────────────
    detection_report: Optional[dict]        # DetectionReport.to_dict()
    failure_summary: str                    # human-readable summary for LLM

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_incidents: list[dict]         # list of IncidentRecord dicts

    # ── Diagnosis ─────────────────────────────────────────────────────────
    diagnosis: Optional[DiagnosisOutput]
    diagnosis_raw: str                      # raw LLM text (for RAGAS)

    # ── Faithfulness / Trust ──────────────────────────────────────────────
    faithfulness_score: float               # 0.0 – 1.0
    faithfulness_passed: bool
    faithfulness_override: Optional[float]  # if set, bypass scoring (tests only)

    # ── Policy decision ───────────────────────────────────────────────────
    decision: str                           # "auto_fix" | "escalate"
    decision_reason: str

    # ── Remediation ───────────────────────────────────────────────────────
    remediation_action: Optional[str]
    remediation_success: Optional[bool]
    remediation_detail: Optional[dict]

    # ── Escalation ────────────────────────────────────────────────────────
    escalation_reason: Optional[str]

    # ── Audit metadata ────────────────────────────────────────────────────
    run_id: Optional[str]
    errors: list[str]                       # non-fatal errors accumulated during run
