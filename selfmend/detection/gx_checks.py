"""
GX-based rule checks for SelfMend.

Uses Great Expectations 1.23 ephemeral context (no persistent GX store needed).
Expectations are built dynamically from baseline_stats.json so thresholds are
always data-driven, never guessed.

Public API:
    build_suite(stats)          -> gx.ExpectationSuite
    run_gx_checks(batch_df, stats) -> GXResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

NULL_RATE_MARGIN = 0.05          # fail if null_rate > baseline + margin
TOTAL_TOLERANCE  = 0.01          # Total == Qty * Price within 1 %


@dataclass
class GXResult:
    """Structured result from a GX validation run."""
    success: bool
    failed_checks: list[dict[str, Any]] = field(default_factory=list)
    passed_checks: list[dict[str, Any]] = field(default_factory=list)
    n_total: int = 0
    n_failed: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "n_total": self.n_total,
            "n_failed": self.n_failed,
            "failed_checks": self.failed_checks,
            "passed_checks": self.passed_checks,
        }


def _import_gx():
    """Lazy import so other modules can import this file without GX being slow to load."""
    import great_expectations as gx  # noqa: PLC0415
    from great_expectations.expectations import (  # noqa: PLC0415
        ExpectColumnToExist,
        ExpectColumnValuesToBeBetween,
        ExpectColumnValuesToBeInSet,
        ExpectColumnValuesToNotBeNull,
        ExpectColumnProportionOfUniqueValuesToBeBetween,
    )
    return gx, ExpectColumnToExist, ExpectColumnValuesToBeBetween, \
           ExpectColumnValuesToBeInSet, ExpectColumnValuesToNotBeNull, \
           ExpectColumnProportionOfUniqueValuesToBeBetween


def build_suite(stats: dict) -> Any:
    """
    Build a GX ExpectationSuite from baseline stats.

    Thresholds derived from observed baseline null rates + a fixed margin.
    """
    gx, ExpectColumnToExist, ExpectColumnValuesToBeBetween, \
        ExpectColumnValuesToBeInSet, ExpectColumnValuesToNotBeNull, \
        ExpectColumnProportionOfUniqueValuesToBeBetween = _import_gx()

    suite = gx.ExpectationSuite("selfmend_suite")

    expected_cols: list[str] = stats.get("columns", [])
    null_rates: dict[str, float] = stats.get("null_rates", {})
    numeric_ranges: dict[str, dict] = stats.get("numeric_ranges", {})
    category_sets: dict[str, list] = stats.get("category_sets", {})

    # 1. Schema: all expected columns must be present
    for col in expected_cols:
        suite.add_expectation(ExpectColumnToExist(column=col))

    # 2. Null rates: batch null_rate must not exceed baseline + margin
    #    We express this as a "mostly" constraint: mostly >= (1 - max_allowed_null_rate)
    for col, baseline_rate in null_rates.items():
        max_null = min(baseline_rate + NULL_RATE_MARGIN, 1.0)
        mostly   = round(1.0 - max_null, 4)
        if mostly > 0:
            suite.add_expectation(
                ExpectColumnValuesToNotBeNull(column=col, mostly=mostly)
            )

    # 3. Numeric ranges: values must be within [p01 - slack, p99 + slack]
    for col, r in numeric_ranges.items():
        slack = max(abs(r["std"]) * 2, 1.0)       # 2-sigma slack
        suite.add_expectation(
            ExpectColumnValuesToBeBetween(
                column=col,
                min_value=max(0, r["p01"] - slack),
                max_value=r["p99"] + slack,
                mostly=0.95,                       # allow 5 % outliers
            )
        )

    # 4. Allowed categories
    for col, allowed in category_sets.items():
        if allowed:
            suite.add_expectation(
                ExpectColumnValuesToBeInSet(
                    column=col,
                    value_set=allowed,
                    mostly=0.99,           # 1 % tolerance for unseen cats in clean data
                )
            )

    # Total == Quantity × Price is checked separately in run_gx_checks (computed column check)

    return suite


def run_gx_checks(batch_df: pd.DataFrame, stats: dict) -> GXResult:
    """
    Run GX checks on a batch DataFrame.

    Steps:
      1. Standard expectation suite (schema, nulls, ranges, categories)
      2. Custom: Total Spent == Quantity × Price Per Unit (tolerance 1 %)

    Returns a GXResult with full detail on failures.
    """
    gx, *_ = _import_gx()

    ctx   = gx.get_context(mode="ephemeral")
    ds    = ctx.data_sources.add_pandas("batch_ds")
    da    = ds.add_dataframe_asset("batch_asset")
    bd    = da.add_batch_definition_whole_dataframe("bd")

    suite = build_suite(stats)
    suite = ctx.suites.add(suite)

    vd = ctx.validation_definitions.add(
        gx.ValidationDefinition(name="selfmend_vd", data=bd, suite=suite)
    )

    raw_result = vd.run(batch_parameters={"dataframe": batch_df})

    # ── parse results ────────────────────────────────────────────────────────
    failed_checks: list[dict] = []
    passed_checks: list[dict] = []

    for r in raw_result.results:
        exp_type = r.expectation_config.type
        col      = r.expectation_config.kwargs.get("column", "N/A")
        check    = {
            "type": exp_type,
            "column": col,
            "success": r.success,
            "kwargs": dict(r.expectation_config.kwargs),
            "observed": dict(r.result) if r.result else {},
        }
        if r.success:
            passed_checks.append(check)
        else:
            failed_checks.append(check)
            log.debug("GX FAIL: %s on %s", exp_type, col)

    # ── custom Total check ───────────────────────────────────────────────────
    total_check = _check_total_integrity(batch_df)
    if total_check["success"]:
        passed_checks.append(total_check)
    else:
        failed_checks.append(total_check)
        log.debug("GX FAIL: Total integrity check")

    n_total  = len(failed_checks) + len(passed_checks)
    n_failed = len(failed_checks)

    result = GXResult(
        success=n_failed == 0,
        failed_checks=failed_checks,
        passed_checks=passed_checks,
        n_total=n_total,
        n_failed=n_failed,
    )

    log.info(
        "GX checks: %d total, %d failed, success=%s",
        n_total, n_failed, result.success
    )
    return result


def _check_total_integrity(df: pd.DataFrame) -> dict:
    """
    Custom check: Total Spent must equal Quantity × Price Per Unit
    within TOTAL_TOLERANCE for at least 95 % of derivable rows.
    """
    # Guard: all three columns must exist
    for col in ("Quantity", "Price Per Unit", "Total Spent"):
        if col not in df.columns:
            return {
                "type": "total_integrity",
                "column": "Total Spent",
                "success": False,
                "kwargs": {},
                "observed": {"missing_column": col},
            }

    qty   = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["Price Per Unit"], errors="coerce")
    total = pd.to_numeric(df["Total Spent"], errors="coerce")

    can_derive = qty.notna() & price.notna() & total.notna()
    n_derivable = int(can_derive.sum())

    if n_derivable == 0:
        # No derivable rows — can't check, pass
        return {
            "type": "total_integrity",
            "column": "Total Spent",
            "success": True,
            "kwargs": {},
            "observed": {"n_derivable": 0, "mismatch_rate": 0.0},
        }

    computed = qty * price
    rel_err  = ((total - computed).abs() / computed.abs().clip(lower=1e-9))
    mismatch = (rel_err > TOTAL_TOLERANCE) & can_derive
    mismatch_rate = float(mismatch.sum() / n_derivable)
    success  = mismatch_rate <= 0.05   # fail if > 5 % mismatch

    return {
        "type": "total_integrity",
        "column": "Total Spent",
        "success": success,
        "kwargs": {"tolerance": TOTAL_TOLERANCE, "mostly": 0.95},
        "observed": {
            "n_derivable": n_derivable,
            "n_mismatch": int(mismatch.sum()),
            "mismatch_rate": round(mismatch_rate, 4),
        },
    }
