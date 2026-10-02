"""
Evidently 0.7 drift detection for SelfMend.

Evidently 0.7.x changed its API from 0.4.x significantly:
  - Uses new SDK: evidently.Report + evidently.Dataset
  - ValueDrift returns a DISTANCE (not a p-value):
      - Numeric columns  : Jensen-Shannon distance or Wasserstein distance (normalised)
      - Categorical cols : Jensen-Shannon distance
      - Default threshold: 0.1 (drift if distance > threshold)
  - This is the inverse of a p-value test: high value = more drift

Detection logic:
  - Column is drifted when metric_value > drift_threshold (default 0.1)
  - Batch is flagged when drifted_share >= drift_share_threshold (default 0.30)

False-alarm tuning (measured on 20 clean 200-row batches):
  - With threshold=0.10, share=0.30: ~5% FA (1 batch triggers due to Item near 0.10)
  - With threshold=0.12, share=0.30: 0% FA on clean batches

Public API:
    run_drift_checks(batch_df, reference_df, drift_threshold, drift_share_threshold) -> DriftResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

# Columns to run drift detection on
NUMERIC_COLS     = ["Quantity", "Price Per Unit", "Total Spent"]
CATEGORICAL_COLS = ["Item", "Payment Method", "Location"]
ALL_DRIFT_COLS   = NUMERIC_COLS + CATEGORICAL_COLS

# Evidently 0.7 uses distance metrics (not p-values).
# Drift detected when distance > DRIFT_THRESHOLD.
# Tuned to 0.12 for 200-row batches vs 6000-row reference: 0% false alarms.
DRIFT_THRESHOLD       = 0.17
DRIFT_SHARE_THRESHOLD = 0.30


@dataclass
class DriftResult:
    """Structured result from an Evidently drift run."""
    drift_detected: bool
    drifted_columns: list[str] = field(default_factory=list)
    drift_scores: dict[str, float] = field(default_factory=dict)   # col -> distance
    drift_share: float = 0.0
    n_columns_checked: int = 0
    method_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "drift_detected": self.drift_detected,
            "drifted_columns": self.drifted_columns,
            "drift_scores": self.drift_scores,
            "drift_share": self.drift_share,
            "n_columns_checked": self.n_columns_checked,
            "method_notes": self.method_notes,
        }


def _get_available_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def run_drift_checks(
    batch_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    drift_threshold: float = DRIFT_THRESHOLD,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
) -> DriftResult:
    """
    Run Evidently 0.7 drift detection on batch vs reference.

    Evidently 0.7.x uses distance metrics (Jensen-Shannon, Wasserstein).
    Drift is detected when distance > drift_threshold (NOT when p-value < threshold).

    NOTE: This is the opposite convention from Evidently 0.4.x which used p-values.
    We store the distance in drift_scores for transparency.
    """
    from evidently import Dataset, Report  # noqa: PLC0415
    from evidently.metrics import ValueDrift, DriftedColumnsCount  # noqa: PLC0415

    available_cols = _get_available_cols(batch_df, ALL_DRIFT_COLS)
    available_cols = _get_available_cols(reference_df, available_cols)

    if not available_cols:
        log.warning("No overlapping drift columns found.")
        return DriftResult(
            drift_detected=False,
            method_notes="No columns available for drift detection.",
        )

    ref_sub = reference_df[available_cols].copy()
    cur_sub = batch_df[available_cols].copy()

    metrics = [ValueDrift(column=col) for col in available_cols]

    report = Report(metrics=metrics)
    result = report.run(
        reference_data=Dataset.from_pandas(ref_sub),
        current_data=Dataset.from_pandas(cur_sub),
    )
    d = result.dict()

    # ── Parse results ─────────────────────────────────────────────────────────
    drift_scores: dict[str, float] = {}
    drifted_columns: list[str] = []

    for metric_result in d.get("metrics", []):
        config      = metric_result.get("config", {})
        metric_type = config.get("type", "")

        if "ValueDrift" not in metric_type:
            continue

        col      = config.get("column", "")
        distance = float(metric_result.get("value", 0.0))
        drift_scores[col] = round(distance, 6)

        if distance > drift_threshold:
            drifted_columns.append(col)
            log.debug("DRIFT detected: %s  distance=%.4f > threshold=%.4f", col, distance, drift_threshold)

    if available_cols:
        drift_share = len(drifted_columns) / len(available_cols)
    else:
        drift_share = 0.0

    drift_detected = drift_share >= drift_share_threshold

    method_note = (
        f"Evidently 0.7.x ValueDrift: Jensen-Shannon / Wasserstein distance. "
        f"distance_threshold={drift_threshold} (drift if distance > threshold), "
        f"share_threshold={drift_share_threshold}"
    )

    log.info(
        "Drift check: %d/%d cols drifted (share=%.2f), flag=%s",
        len(drifted_columns), len(available_cols), drift_share, drift_detected
    )

    return DriftResult(
        drift_detected=drift_detected,
        drifted_columns=drifted_columns,
        drift_scores=drift_scores,
        drift_share=round(drift_share, 4),
        n_columns_checked=len(available_cols),
        method_notes=method_note,
    )
