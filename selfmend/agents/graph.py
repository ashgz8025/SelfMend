"""
LangGraph StateGraph for SelfMend.

Graph topology:
  START
    └─► detector_node
          └─► retriever_node
                └─► diagnoser_node
                      └─► verifier_node
                            └─► policy_node
                                  ├─ [auto_fix]  ──► remediator_node ──► END
                                  └─ [escalate] ──► escalate_node  ──► END

Conditional edge after policy_node:
  - if state["decision"] == "auto_fix"  → remediator_node
  - else                                → escalate_node

Low faithfulness is routed to escalate_node by the policy_node
(which checks faithfulness_passed), not by a separate condition.
This is enforced in policy/gate.py, not in the LLM prompt.

Public API:
    build_graph() -> CompiledGraph
    run_pipeline(batch_df, batch_id, reference_df, stats) -> PipelineState
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.graph import StateGraph, START, END

from agents.state import PipelineState
from agents.nodes import (
    detector_node,
    retriever_node,
    diagnoser_node,
    verifier_node,
    policy_node,
    remediator_node,
    escalate_node,
)

log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
STATS_JSON = ROOT / "data" / "clean" / "baseline_stats.json"
REF_PQ     = ROOT / "data" / "clean" / "reference.parquet"


def _route_after_policy(state: PipelineState) -> str:
    """
    Conditional edge: route to remediator or escalate based on policy decision.
    This is a pure function of state — it never calls the LLM or prompt.
    """
    decision = state.get("decision", "escalate")
    return "remediator_node" if decision == "auto_fix" else "escalate_node"


def build_graph() -> Any:
    """
    Build and compile the SelfMend StateGraph.

    Returns a compiled LangGraph that can be invoked with:
        graph.invoke(initial_state)
    """
    builder = StateGraph(PipelineState)

    # Add nodes
    builder.add_node("detector_node",   detector_node)
    builder.add_node("retriever_node",  retriever_node)
    builder.add_node("diagnoser_node",  diagnoser_node)
    builder.add_node("verifier_node",   verifier_node)
    builder.add_node("policy_node",     policy_node)
    builder.add_node("remediator_node", remediator_node)
    builder.add_node("escalate_node",   escalate_node)

    # Linear edges up to policy
    builder.add_edge(START,              "detector_node")
    builder.add_edge("detector_node",    "retriever_node")
    builder.add_edge("retriever_node",   "diagnoser_node")
    builder.add_edge("diagnoser_node",   "verifier_node")
    builder.add_edge("verifier_node",    "policy_node")

    # Conditional edge after policy
    builder.add_conditional_edges(
        "policy_node",
        _route_after_policy,
        {
            "remediator_node": "remediator_node",
            "escalate_node":   "escalate_node",
        },
    )

    # Both terminal nodes go to END
    builder.add_edge("remediator_node", END)
    builder.add_edge("escalate_node",   END)

    return builder.compile()


# Singleton compiled graph (lazy)
_compiled_graph: Any = None


def get_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(
    batch_df: pd.DataFrame,
    batch_id: str | None = None,
    reference_df: pd.DataFrame | None = None,
    stats: dict | None = None,
    faithfulness_override: float | None = None,   # for testing
) -> PipelineState:
    """
    Run the full SelfMend pipeline on a batch.

    Args:
        batch_df             : The batch to process.
        batch_id             : Human-readable ID (auto-generated if None).
        reference_df         : Reference window (loaded from disk if None).
        stats                : Baseline stats (loaded from disk if None).
        faithfulness_override: If set, injects this faithfulness score directly
                               (used in pytest to force low-faithfulness escalation).

    Returns:
        Final PipelineState after all nodes have run.
    """
    if batch_id is None:
        batch_id = str(uuid.uuid4())[:8]

    if stats is None:
        stats = json.loads(STATS_JSON.read_text())
    if reference_df is None:
        reference_df = pd.read_parquet(REF_PQ)

    initial_state: PipelineState = {
        "batch_id":            batch_id,
        "batch_df":            batch_df,
        "reference_df":        reference_df,
        "stats":               stats,
        "detection_report":    None,
        "failure_summary":     "",
        "retrieved_incidents": [],
        "diagnosis":           None,
        "diagnosis_raw":       "",
        "faithfulness_score":  0.0,
        "faithfulness_passed": False,
        "faithfulness_override": faithfulness_override,
        "decision":            "escalate",
        "decision_reason":     "",
        "remediation_action":  None,
        "remediation_success": None,
        "remediation_detail":  None,
        "escalation_reason":   None,
        "run_id":              str(uuid.uuid4()),
        "errors":              [],
    }

    graph = get_graph()

    if faithfulness_override is not None:
        log.info("[graph] faithfulness_override=%.3f — will be injected at verifier node", faithfulness_override)

    final_state = graph.invoke(initial_state)
    log.info(
        "[graph] Pipeline complete: batch=%s  decision=%s  faith=%.3f",
        batch_id,
        final_state.get("decision"),
        final_state.get("faithfulness_score", 0),
    )
    return final_state
