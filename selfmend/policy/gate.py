"""
Policy gate for SelfMend.

Decides whether to auto-fix or escalate based on:
  - risk_level from diagnosis (must be "low")
  - faithfulness_score (must be >= threshold, default 0.8)
  - confidence (must be >= threshold, default 0.7)
  - proposed_action is in the whitelist

All thresholds are read from config.yaml (with env-var overrides).

Public API:
    evaluate_policy(state) -> (decision: str, reason: str)
    WHITELISTED_ACTIONS: list[str]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

# ── Whitelist ──────────────────────────────────────────────────────────────────
WHITELISTED_ACTIONS: list[str] = [
    "normalize_sentinels_to_null",
    "restore_total_from_quantity_price",
    "impute_nulls",
    "drop_duplicates",
    "clip_outliers",
]


def _load_policy_config() -> dict:
    """Load policy thresholds from config.yaml."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("policy", {}).get("auto_fix_conditions", {})
    except Exception:
        return {}


def evaluate_policy(state: dict) -> tuple[str, str]:
    """
    Evaluate whether the pipeline can auto-fix or must escalate.

    Returns:
        ("auto_fix", reason) or ("escalate", reason)
    """
    cfg = _load_policy_config()

    # Thresholds (config → env override → defaults)
    faithfulness_min = float(
        os.getenv("FAITHFULNESS_THRESHOLD",
                  str(cfg.get("faithfulness_min", 0.8)))
    )
    confidence_min = float(
        os.getenv("CONFIDENCE_THRESHOLD",
                  str(cfg.get("confidence_min", 0.7)))
    )
    allowed_risk = cfg.get("risk_levels_allowed", ["low"])

    faithfulness  = state.get("faithfulness_score", 0.0)
    faith_passed  = state.get("faithfulness_passed", False)
    diagnosis     = state.get("diagnosis") or {}
    confidence    = float(diagnosis.get("confidence", 0.0))
    risk_level    = str(diagnosis.get("risk_level", "high"))
    action        = str(diagnosis.get("proposed_action", "escalate"))

    # ── Evaluate each gate ─────────────────────────────────────────────────
    if not faith_passed:
        return (
            "escalate",
            f"Faithfulness {faithfulness:.3f} below threshold {faithfulness_min:.2f}.",
        )

    if risk_level not in allowed_risk:
        return (
            "escalate",
            f"Risk level '{risk_level}' not in allowed: {allowed_risk}.",
        )

    if confidence < confidence_min:
        return (
            "escalate",
            f"Diagnosis confidence {confidence:.2f} below threshold {confidence_min:.2f}.",
        )

    if action not in WHITELISTED_ACTIONS:
        return (
            "escalate",
            f"Proposed action '{action}' is not in the whitelist: {WHITELISTED_ACTIONS}.",
        )

    return (
        "auto_fix",
        f"All gates passed: risk={risk_level}, faith={faithfulness:.3f}, "
        f"conf={confidence:.2f}, action={action}.",
    )
