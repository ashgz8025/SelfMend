"""
Whitelisted remediation actions for SelfMend.

Each action takes a DataFrame and returns a fixed copy.
LLM-generated code is NEVER executed — the LLM can only name an action
from this whitelist.

Public API:
    apply_action(df, action_name, stats) -> pd.DataFrame
    ACTIONS: dict[str, callable]
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SENTINEL_VALUES = {"ERROR", "UNKNOWN", "error", "unknown", "N/A", "n/a", "NULL", "null", ""}


# ── Action implementations ─────────────────────────────────────────────────────

def normalize_sentinels_to_null(df: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
    """
    Replace sentinel strings (ERROR, UNKNOWN, etc.) with NaN in all object columns.
    """
    df = df.copy()
    changed = 0
    for col in df.select_dtypes(include=["object", "str"]).columns:
        mask = df[col].isin(SENTINEL_VALUES)
        changed += int(mask.sum())
        df.loc[mask, col] = np.nan
    log.info("normalize_sentinels_to_null: replaced %d sentinel values", changed)
    return df


def restore_total_from_quantity_price(df: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
    """
    Recompute Total Spent = Quantity × Price Per Unit for:
      - rows where Total Spent is NaN but Qty and Price are available
      - rows where Total differs from Qty × Price by more than 1 %
    """
    df = df.copy()
    qty   = pd.to_numeric(df.get("Quantity"), errors="coerce")
    price = pd.to_numeric(df.get("Price Per Unit"), errors="coerce")
    total = pd.to_numeric(df.get("Total Spent"), errors="coerce")

    can_derive = qty.notna() & price.notna()
    computed   = (qty * price).round(2)

    # Fix nulls
    null_total = total.isna() & can_derive
    df.loc[null_total, "Total Spent"] = computed[null_total]

    # Fix mismatches (> 1 % relative error)
    rel_err  = ((total - computed).abs() / computed.abs().clip(lower=1e-9))
    mismatch = can_derive & total.notna() & (rel_err > 0.01)
    df.loc[mismatch, "Total Spent"] = computed[mismatch]

    n_fixed = int(null_total.sum()) + int(mismatch.sum())
    log.info("restore_total_from_quantity_price: fixed %d rows", n_fixed)
    return df


def impute_nulls(df: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
    """
    Impute null values:
      - Numeric columns: median imputation
      - Categorical columns: mode imputation
    Uses baseline stats if available; otherwise computes from the batch itself.
    """
    df = df.copy()
    numeric_cols     = ["Quantity", "Price Per Unit", "Total Spent"]
    categorical_cols = ["Item", "Payment Method", "Location"]

    for col in numeric_cols:
        if col not in df.columns:
            continue
        null_mask = df[col].isna()
        if null_mask.sum() == 0:
            continue
        # Use baseline median if available, else batch median
        if stats and "numeric_ranges" in stats and col in stats["numeric_ranges"]:
            fill_val = stats["numeric_ranges"][col]["mean"]   # mean as proxy for median
        else:
            fill_val = pd.to_numeric(df[col], errors="coerce").median()
        df.loc[null_mask, col] = round(fill_val, 2)
        log.info("impute_nulls: %s  %d nulls → %.2f", col, int(null_mask.sum()), fill_val)

    for col in categorical_cols:
        if col not in df.columns:
            continue
        null_mask = df[col].isna()
        if null_mask.sum() == 0:
            continue
        if stats and "category_sets" in stats and col in stats["category_sets"]:
            # Use most common category from baseline
            cats  = stats["category_sets"][col]
            fill_val = cats[0] if cats else df[col].mode().iloc[0]
        else:
            mode = df[col].mode()
            fill_val = mode.iloc[0] if not mode.empty else "Unknown"
        df.loc[null_mask, col] = fill_val
        log.info("impute_nulls: %s  %d nulls → %s", col, int(null_mask.sum()), fill_val)

    return df


def drop_duplicates(df: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    log.info("drop_duplicates: removed %d duplicate rows", removed)
    return df


def clip_outliers(df: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
    """
    Clip numeric columns to [p01 - 2*std, p99 + 2*std] range from baseline stats.
    Falls back to IQR-based clipping if stats are unavailable.
    """
    df = df.copy()
    numeric_cols = ["Quantity", "Price Per Unit", "Total Spent"]

    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        null_mask = series.isna()

        if stats and "numeric_ranges" in stats and col in stats["numeric_ranges"]:
            r     = stats["numeric_ranges"][col]
            slack = max(abs(r["std"]) * 2, 1.0)
            lo    = max(0, r["p01"] - slack)
            hi    = r["p99"] + slack
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lo  = max(0, q1 - 1.5 * iqr)
            hi  = q3 + 1.5 * iqr

        n_clipped = int(((series < lo) | (series > hi)).sum())
        df[col] = series.clip(lower=lo, upper=hi)
        df.loc[null_mask, col] = np.nan   # restore nulls
        log.info("clip_outliers: %s  clipped=%d  [%.2f, %.2f]", col, n_clipped, lo, hi)

    return df


# ── Registry ───────────────────────────────────────────────────────────────────

ACTIONS: dict[str, callable] = {
    "normalize_sentinels_to_null":        normalize_sentinels_to_null,
    "restore_total_from_quantity_price":  restore_total_from_quantity_price,
    "impute_nulls":                       impute_nulls,
    "drop_duplicates":                    drop_duplicates,
    "clip_outliers":                      clip_outliers,
}


def apply_action(df: pd.DataFrame, action_name: str, stats: dict | None = None) -> pd.DataFrame:
    """
    Apply a whitelisted action to a DataFrame copy.

    Raises:
        ValueError if action_name is not in ACTIONS (LLM-generated code is rejected here).
    """
    if action_name not in ACTIONS:
        raise ValueError(
            f"Action '{action_name}' is not in the whitelist. "
            f"Allowed: {list(ACTIONS.keys())}"
        )
    return ACTIONS[action_name](df.copy(), stats)
