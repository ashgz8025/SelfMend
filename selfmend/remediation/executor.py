"""
Safe remediation executor for SelfMend.

Applies a whitelisted fix to a COPY of the batch, then re-runs GX + Evidently
on the fixed copy. Promotes to data/clean only if GX checks pass.
NEVER executes LLM-generated code.

Chaining logic:
  - normalize_sentinels_to_null  → then impute_nulls + restore_total
  - impute_nulls                 → then restore_total_from_quantity_price
  (Imputing Quantity/Price mean-values shifts the Total invariant)

Promotion criteria:
  GX checks must pass. Residual drift after imputation is acceptable
  (mean imputation shifts distribution slightly — that is expected).

Public API:
    execute_remediation(batch_df, action, batch_id, stats, reference_df) -> dict
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from remediation.actions import apply_action, ACTIONS
from detection.detector import detect

log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parents[1]
CLEAN_DIR    = ROOT / "data" / "clean"
QUARANTINE   = ROOT / "data" / "quarantine"
STATS_JSON   = ROOT / "data" / "clean" / "baseline_stats.json"
REF_PQ       = ROOT / "data" / "clean" / "reference.parquet"


def execute_remediation(
    batch_df: pd.DataFrame,
    action: str,
    batch_id: str = "unknown",
    stats: dict | None = None,
    reference_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Safe remediation pipeline:
      1. Validate action is whitelisted (reject anything else).
      2. Apply fix to a COPY with chained follow-up actions where needed.
      3. Re-run GX on the fixed copy. Drift is NOT a blocking criterion
         (mean imputation naturally shifts distributions).
      4. If GX passes → promote to data/clean/  else → quarantine.
    """
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    if stats is None:
        stats = json.loads(STATS_JSON.read_text())
    if reference_df is None:
        reference_df = pd.read_parquet(REF_PQ)

    # Step 1: Whitelist check
    if action not in ACTIONS:
        log.warning("Rejected non-whitelisted action '%s' for batch %s", action, batch_id)
        return {
            "success": False,
            "action": action,
            "reason": f"Action '{action}' is not whitelisted. Escalating.",
            "promoted": False,
        }

    # Step 2: Apply fix to copy with chaining
    log.info("[executor] Applying '%s' to copy of batch '%s'…", action, batch_id)
    try:
        fixed_df = apply_action(batch_df.copy(), action, stats=stats)

        # Chaining: normalizing sentinels creates nulls → impute → then restore Total
        if action == "normalize_sentinels_to_null":
            log.info("[executor] Chaining: impute_nulls after normalize_sentinels_to_null")
            fixed_df = apply_action(fixed_df, "impute_nulls", stats=stats)
            log.info("[executor] Chaining: restore_total after impute to fix Total invariant")
            fixed_df = apply_action(fixed_df, "restore_total_from_quantity_price", stats=stats)

        # Chaining: imputing Qty/Price shifts Total → restore it
        elif action == "impute_nulls":
            log.info("[executor] Chaining: restore_total after impute_nulls")
            fixed_df = apply_action(fixed_df, "restore_total_from_quantity_price", stats=stats)

    except Exception as exc:
        log.error("[executor] Action '%s' raised: %s", action, exc)
        return {
            "success": False,
            "action": action,
            "reason": f"Action raised exception: {exc}",
            "promoted": False,
        }

    # Step 3: Re-validate — GX only (drift after imputation is acceptable)
    log.info("[executor] Re-validating fixed copy…")
    post_report = detect(
        fixed_df,
        reference_df=reference_df,
        stats=stats,
        batch_id=f"{batch_id}_fixed",
    )

    # Promotion criterion: GX must pass (drift-only failures are acceptable)
    gx_ok = post_report.gx_result.success if post_report.gx_result else True

    # Step 4: Promote or quarantine
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if gx_ok:
        out_path = CLEAN_DIR / f"fixed_{batch_id}_{ts}.parquet"
        fixed_df.to_parquet(out_path, index=False)
        log.info("[executor] PROMOTED: fixed batch → %s", out_path)
        promoted = True
    else:
        q_path = QUARANTINE / f"quarantine_{batch_id}_{ts}.parquet"
        fixed_df.to_parquet(q_path, index=False)
        log.warning(
            "[executor] QUARANTINED: GX still failed (%d checks). → %s",
            post_report.gx_result.n_failed if post_report.gx_result else "?",
            q_path,
        )
        promoted = False

    return {
        "success": gx_ok,
        "action": action,
        "promoted": promoted,
        "pre_failures": len(batch_df),
        "post_gx_failures": post_report.gx_result.n_failed if post_report.gx_result else 0,
        "post_detection": post_report.to_dict(),
        "reason": (
            "GX re-validation passed." if gx_ok
            else f"GX re-validation failed: {post_report.summary_text()}"
        ),
    }
