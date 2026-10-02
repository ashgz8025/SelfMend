"""
Unified detector: merges GX rule-checks + Evidently drift into one structured
failure report.

Public API:
    detect(batch_df, reference_df, stats, config) -> DetectionReport
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from detection.gx_checks import GXResult, run_gx_checks
from detection.drift_check import DriftResult, run_drift_checks

log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parents[1]
STATS_JSON   = ROOT / "data" / "clean" / "baseline_stats.json"
REFERENCE_PQ = ROOT / "data" / "clean" / "reference.parquet"


def _load_stats() -> dict:
    with open(STATS_JSON) as f:
        return json.load(f)


def _load_reference() -> pd.DataFrame:
    return pd.read_parquet(REFERENCE_PQ)


# ── Severity classification ────────────────────────────────────────────────────

def _classify_severity(check: dict) -> str:
    """
    Assign severity to a failed check.
      high   : schema failures (missing column, wrong dtype)
      medium : null spikes, out-of-range, Total mismatch
      low    : category violations, drift-only
    """
    t = check.get("type", "")
    if t == "ExpectColumnToExist":
        return "high"
    if t in ("total_integrity",):
        return "medium"
    if "NotBeNull" in t or "Between" in t:
        return "medium"
    if "InSet" in t:
        return "low"
    if t == "drift":
        return "low"
    return "medium"


# ── Failure report dataclass ───────────────────────────────────────────────────

@dataclass
class FailedCheck:
    check_type: str
    column: str
    severity: str
    layer: str                  # "gx" | "drift"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionReport:
    """
    Unified failure report combining GX and drift results.
    """
    batch_id: str
    has_failures: bool
    gx_passed: bool
    drift_detected: bool

    failed_checks: list[FailedCheck] = field(default_factory=list)
    drift_scores: dict[str, float]   = field(default_factory=dict)
    drifted_columns: list[str]       = field(default_factory=list)
    drift_share: float               = 0.0

    gx_result:    GXResult | None    = None
    drift_result: DriftResult | None = None

    def severity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for fc in self.failed_checks:
            counts[fc.severity] = counts.get(fc.severity, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "has_failures": self.has_failures,
            "gx_passed": self.gx_passed,
            "drift_detected": self.drift_detected,
            "severity_counts": self.severity_counts(),
            "failed_checks": [
                {
                    "check_type": fc.check_type,
                    "column": fc.column,
                    "severity": fc.severity,
                    "layer": fc.layer,
                    "detail": fc.detail,
                }
                for fc in self.failed_checks
            ],
            "drift_scores": self.drift_scores,
            "drifted_columns": self.drifted_columns,
            "drift_share": self.drift_share,
        }

    def summary_text(self) -> str:
        if not self.has_failures:
            return f"[{self.batch_id}] No failures detected."
        lines = [f"[{self.batch_id}] {len(self.failed_checks)} failure(s):"]
        for fc in self.failed_checks:
            lines.append(f"  [{fc.severity.upper()}] {fc.layer}:{fc.check_type} on '{fc.column}'")
        if self.drift_detected:
            lines.append(f"  Drift in: {', '.join(self.drifted_columns)}")
        return "\n".join(lines)


# ── Main detection function ────────────────────────────────────────────────────

def detect(
    batch_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    stats: dict | None = None,
    batch_id: str = "unknown",
) -> DetectionReport:
    """
    Run both detection layers and return a merged DetectionReport.

    Args:
        batch_df    : The incoming batch to check.
        reference_df: Reference window (loaded from disk if None).
        stats       : Baseline stats dict (loaded from disk if None).
        batch_id    : Human-readable ID for this batch.
    """
    if stats is None:
        stats = _load_stats()
    if reference_df is None:
        reference_df = _load_reference()

    # Layer 1: GX rule checks
    log.info("Running GX checks on batch '%s' (%d rows)…", batch_id, len(batch_df))
    gx_result = run_gx_checks(batch_df, stats)

    # Layer 2: Evidently drift
    log.info("Running drift checks on batch '%s'…", batch_id)
    drift_result = run_drift_checks(batch_df, reference_df)

    # ── Build failed check list ───────────────────────────────────────────────
    failed_checks: list[FailedCheck] = []

    for fc in gx_result.failed_checks:
        sev = _classify_severity(fc)
        failed_checks.append(FailedCheck(
            check_type=fc["type"],
            column=fc["column"],
            severity=sev,
            layer="gx",
            detail=fc.get("observed", {}),
        ))

    if drift_result.drift_detected:
        for col in drift_result.drifted_columns:
            failed_checks.append(FailedCheck(
                check_type="drift",
                column=col,
                severity="low",
                layer="drift",
                detail={"p_value": drift_result.drift_scores.get(col, -1.0)},
            ))

    has_failures = len(failed_checks) > 0

    report = DetectionReport(
        batch_id=batch_id,
        has_failures=has_failures,
        gx_passed=gx_result.success,
        drift_detected=drift_result.drift_detected,
        failed_checks=failed_checks,
        drift_scores=drift_result.drift_scores,
        drifted_columns=drift_result.drifted_columns,
        drift_share=drift_result.drift_share,
        gx_result=gx_result,
        drift_result=drift_result,
    )

    log.info(
        "Detection complete: has_failures=%s  gx_passed=%s  drift=%s  severity=%s",
        has_failures, gx_result.success, drift_result.drift_detected,
        report.severity_counts(),
    )
    return report
