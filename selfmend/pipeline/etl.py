"""
ETL pipeline for the Dirty Cafe Sales dataset.

Steps (deterministic, logged):
  1. Load raw CSV
  2. Sentinels → NaN  (ERROR, UNKNOWN, empty string)
  3. Cast / coerce types
  4. Recompute Total Spent = Quantity × Price Per Unit where both are available
  5. Parse dates → datetime, sort by date
  6. Standardise categories (strip whitespace, title-case)
  7. Drop exact duplicate rows
  8. Save clean parquet + baseline stats JSON
  9. Split by date: first 60 % = reference window, rest = sequential batches

Usage:
    python -m pipeline.etl
    python pipeline/etl.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_CSV      = ROOT / "data" / "raw"  / "dirty_cafe_sales.csv"
CLEAN_DIR    = ROOT / "data" / "clean"
BATCHES_DIR  = ROOT / "data" / "batches"
CLEAN_PQ     = CLEAN_DIR / "clean.parquet"
REFERENCE_PQ = CLEAN_DIR / "reference.parquet"
STATS_JSON   = CLEAN_DIR / "baseline_stats.json"

SENTINEL_VALUES: set[str] = {"ERROR", "UNKNOWN", "error", "unknown",
                              "N/A", "n/a", "NULL", "null", ""}

EXPECTED_COLS = [
    "Transaction ID", "Date", "Item", "Quantity",
    "Price Per Unit", "Total Spent", "Payment Method", "Location",
]

REFERENCE_SPLIT = 0.60          # first 60 % → reference window
BATCH_SIZE      = 200           # rows per sequential batch file


# ── helpers ───────────────────────────────────────────────────────────────────

def _log_step(step: str, before: int, after: int) -> None:
    removed = before - after
    log.info("  [%s]  rows before=%d  removed=%d  rows after=%d", step, before, removed, after)


def _nullify_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace sentinel strings with NaN across all object columns."""
    df = df.copy()
    total_replaced = 0
    for col in df.select_dtypes(include=["object", "str"]).columns:
        mask = df[col].isin(SENTINEL_VALUES)
        total_replaced += int(mask.sum())
        df.loc[mask, col] = np.nan
    log.info("  [sentinels→NaN]  cells replaced=%d", total_replaced)
    return df, total_replaced


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric and date columns to their proper types."""
    df = df.copy()
    for col in ("Quantity", "Price Per Unit", "Total Spent"):
        before_null = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_null = df[col].isna().sum()
        new_nulls = after_null - before_null
        if new_nulls:
            log.info("  [cast] %s: %d new NaN from coerce", col, new_nulls)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    n_bad_dates = df["Date"].isna().sum()
    if n_bad_dates:
        log.warning("  [cast] Date: %d rows with un-parseable dates set to NaT", n_bad_dates)
    return df


def _recompute_total(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Where Total Spent is NaN but Quantity and Price Per Unit are both available,
    recompute Total = Quantity × Price.
    Where Total ≠ Quantity × Price (tolerance 1 %), also recompute.
    """
    df = df.copy()
    can_derive = df["Quantity"].notna() & df["Price Per Unit"].notna()
    computed   = df["Quantity"] * df["Price Per Unit"]

    # Fix NaN totals that are derivable
    null_total = df["Total Spent"].isna() & can_derive
    df.loc[null_total, "Total Spent"] = computed[null_total].round(2)

    # Fix mismatched totals (> 1 % relative error)
    rel_err = ((df["Total Spent"] - computed).abs() / computed.abs().clip(lower=1e-9))
    mismatch = can_derive & df["Total Spent"].notna() & (rel_err > 0.01)
    df.loc[mismatch, "Total Spent"] = computed[mismatch].round(2)

    n_fixed = int(null_total.sum()) + int(mismatch.sum())
    log.info("  [recompute_total]  fixed=%d (NaN=%d  mismatch=%d)",
             n_fixed, int(null_total.sum()), int(mismatch.sum()))
    return df, n_fixed


def _standardise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and title-case categorical text columns."""
    df = df.copy()
    for col in ("Item", "Payment Method", "Location"):
        df[col] = df[col].str.strip().str.title()
    return df


def _drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    log.info("  [drop_duplicates]  removed=%d", removed)
    return df, removed


def _compute_baseline_stats(df: pd.DataFrame) -> dict:
    """Compute stats used to calibrate GX expectations and Evidently reference."""
    stats: dict = {
        "n_rows": len(df),
        "columns": EXPECTED_COLS,
        "null_rates": {},
        "numeric_ranges": {},
        "category_sets": {},
        "total_mismatch_rate": 0.0,
    }

    for col in EXPECTED_COLS:
        if col not in df.columns:
            continue
        null_rate = float(df[col].isna().mean())
        stats["null_rates"][col] = round(null_rate, 4)

    for col in ("Quantity", "Price Per Unit", "Total Spent"):
        s = df[col].dropna()
        stats["numeric_ranges"][col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p01": float(s.quantile(0.01)),
            "p99": float(s.quantile(0.99)),
        }

    for col in ("Item", "Payment Method", "Location"):
        stats["category_sets"][col] = sorted(df[col].dropna().unique().tolist())

    # Total mismatch rate in clean data (should be ~0 after ETL)
    can_derive = df["Quantity"].notna() & df["Price Per Unit"].notna() & df["Total Spent"].notna()
    if can_derive.sum() > 0:
        computed = df["Quantity"] * df["Price Per Unit"]
        rel_err = ((df["Total Spent"] - computed).abs() / computed.abs().clip(lower=1e-9))
        mismatch_rate = float((rel_err > 0.01)[can_derive].mean())
        stats["total_mismatch_rate"] = round(mismatch_rate, 4)

    return stats


def _split_batches(df: pd.DataFrame, ref_pct: float, batch_size: int) -> int:
    """
    Split sorted-by-date data:
      - first ref_pct rows → reference.parquet
      - remainder  → sequential batches of `batch_size` rows each

    Returns number of batches written.
    """
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    # Remove stale batches
    for f in BATCHES_DIR.glob("batch_*.parquet"):
        f.unlink()

    split_idx = int(len(df) * ref_pct)
    ref_df    = df.iloc[:split_idx].reset_index(drop=True)
    rest_df   = df.iloc[split_idx:].reset_index(drop=True)

    ref_df.to_parquet(REFERENCE_PQ, index=False)
    log.info("  Reference window: %d rows → %s", len(ref_df), REFERENCE_PQ)

    n_batches = 0
    for i in range(0, len(rest_df), batch_size):
        chunk = rest_df.iloc[i : i + batch_size]
        out   = BATCHES_DIR / f"batch_{n_batches:03d}.parquet"
        chunk.to_parquet(out, index=False)
        n_batches += 1

    log.info("  %d batches of ≤%d rows → %s", n_batches, batch_size, BATCHES_DIR)
    return n_batches


# ── main ETL function ─────────────────────────────────────────────────────────

def run_etl(
    raw_csv: Path = RAW_CSV,
    clean_pq: Path = CLEAN_PQ,
    ref_pct: float = REFERENCE_SPLIT,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Run the full ETL pipeline. Returns the cleaned DataFrame.
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    log.info("═══ ETL START ════════════════════════════════════════")
    log.info("Input : %s", raw_csv)

    # 1. Load
    df = pd.read_csv(raw_csv, low_memory=False)
    log.info("Loaded %d rows × %d cols", *df.shape)
    original_rows = len(df)

    # 2. Sentinels → NaN
    df, _ = _nullify_sentinels(df)

    # 3. Cast types
    df = _cast_types(df)

    # 4. Recompute Total
    df, _ = _recompute_total(df)

    # 5. Standardise categories
    df = _standardise_categories(df)
    log.info("  [standardise_categories]  done")

    # 6. Sort by date (deterministic ordering for splits)
    df = df.sort_values("Date", na_position="last").reset_index(drop=True)
    log.info("  [sort_by_date]  done")

    # 7. Drop exact duplicates
    before_dup = len(df)
    df, n_dups = _drop_duplicates(df)
    _log_step("after_dedup", original_rows, len(df))

    # 8. Save clean parquet
    df.to_parquet(clean_pq, index=False)
    log.info("  Saved clean data: %d rows → %s", len(df), clean_pq)

    # 9. Baseline stats
    stats = _compute_baseline_stats(df)
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info("  Saved baseline stats → %s", STATS_JSON)

    # 10. Split reference + batches
    n_batches = _split_batches(df, ref_pct, batch_size)

    log.info("═══ ETL COMPLETE ═════════════════════════════════════")
    log.info("Summary:")
    log.info("  Raw rows      : %d", original_rows)
    log.info("  Clean rows    : %d", len(df))
    log.info("  Dups removed  : %d", n_dups)
    log.info("  Null rates    :")
    for col, rate in stats["null_rates"].items():
        log.info("    %-22s %.1f%%", col, rate * 100)
    log.info("  Total mismatch rate (post-ETL): %.2f%%", stats["total_mismatch_rate"] * 100)
    log.info("  Batches created : %d", n_batches)

    return df


if __name__ == "__main__":
    run_etl()
