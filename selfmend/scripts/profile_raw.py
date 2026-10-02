"""
Profile the raw Kaggle Dirty Cafe Sales dataset.

Prints per-column:
  - dtype
  - null %
  - sentinel values  ("ERROR", "UNKNOWN", "")
  - duplicate row count
  - out-of-range / negative values (for numeric columns)
  - unique value counts for low-cardinality columns
  - sample values

Usage:
    python scripts/profile_raw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "dirty_cafe_sales.csv"

console = Console()
SENTINELS = {"ERROR", "UNKNOWN", "error", "unknown", "N/A", "n/a", "NULL", "null", ""}


def _sentinel_pct(series: pd.Series) -> tuple[float, list[str]]:
    """Return (pct_sentinel, list_of_found_sentinels) for a string-ish column."""
    s = series.astype(str)
    found = [v for v in SENTINELS if (s == v).any()]
    mask = s.isin(SENTINELS)
    pct = mask.mean() * 100
    return pct, found


def _out_of_range(series: pd.Series) -> tuple[int, float, float]:
    """Return (n_negative, min_val, max_val) for numeric columns."""
    numeric = pd.to_numeric(series, errors="coerce")
    n_neg = int((numeric < 0).sum())
    return n_neg, float(numeric.min()), float(numeric.max())


def profile(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        console.print(
            f"[bold red]Raw CSV not found:[/] {csv_path}\n"
            "Run  python scripts/download_data.py  first, or see the manual"
            " fallback instructions in that script.",
            style="red",
        )
        sys.exit(1)

    df = pd.read_csv(csv_path, low_memory=False)
    n_rows, n_cols = df.shape
    n_dups = int(df.duplicated().sum())

    console.rule("[bold cyan]RAW DATA PROFILE — Dirty Cafe Sales[/]")
    console.print(f"  Rows     : [bold]{n_rows:,}[/]")
    console.print(f"  Columns  : [bold]{n_cols}[/]")
    console.print(f"  Exact dup rows : [bold yellow]{n_dups:,}[/] ({n_dups/n_rows*100:.1f} %)")
    console.print()

    # ------------------------------------------------------------------ #
    # Per-column table
    # ------------------------------------------------------------------ #
    tbl = Table(
        "Column",
        "dtype",
        "Null %",
        "Sentinel %",
        "Sentinels found",
        "Neg vals",
        "Min",
        "Max",
        "Unique",
        "Sample values",
        box=box.SIMPLE_HEAD,
        show_lines=False,
        header_style="bold magenta",
    )

    summary_rows: list[dict] = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        null_pct = series.isna().mean() * 100

        sent_pct, sentinels_found = _sentinel_pct(series)
        neg_count, vmin, vmax = _out_of_range(series)

        n_unique = series.nunique(dropna=False)
        # Show sample non-null values (up to 3)
        sample_vals = series.dropna().unique()[:3].tolist()
        sample_str = ", ".join(str(v) for v in sample_vals)

        null_str = f"[red]{null_pct:.1f}%[/]" if null_pct > 0 else "0.0%"
        sent_str = f"[yellow]{sent_pct:.1f}%[/]" if sent_pct > 0 else "0.0%"
        neg_str = f"[red]{neg_count}[/]" if neg_count > 0 else "0"
        min_str = f"{vmin:.2f}" if not np.isnan(vmin) else "-"
        max_str = f"{vmax:.2f}" if not np.isnan(vmax) else "-"

        tbl.add_row(
            col,
            dtype,
            null_str,
            sent_str,
            ", ".join(sentinels_found) if sentinels_found else "-",
            neg_str,
            min_str,
            max_str,
            str(n_unique),
            sample_str[:60],
        )

        summary_rows.append(
            {
                "column": col,
                "dtype": dtype,
                "null_pct": round(null_pct, 3),
                "sentinel_pct": round(sent_pct, 3),
                "sentinels_found": sentinels_found,
                "n_negative": neg_count,
                "min": vmin,
                "max": vmax,
                "n_unique": n_unique,
            }
        )

    console.print(tbl)

    # ------------------------------------------------------------------ #
    # Low-cardinality columns — full value counts
    # ------------------------------------------------------------------ #
    console.rule("[bold cyan]Low-Cardinality Columns (unique ≤ 20)[/]")
    for row in summary_rows:
        col = row["column"]
        if row["n_unique"] <= 20:
            vc = df[col].value_counts(dropna=False).head(20)
            console.print(f"\n  [bold]{col}[/] (unique={row['n_unique']})")
            for val, cnt in vc.items():
                pct = cnt / n_rows * 100
                console.print(f"    {str(val):<25}  {cnt:>5}  ({pct:.1f}%)")

    # ------------------------------------------------------------------ #
    # Derived column check: does Total == Quantity * Price Per Unit?
    # ------------------------------------------------------------------ #
    console.rule("[bold cyan]Derived Column Integrity: Total Spent == Quantity × Price Per Unit[/]")
    try:
        qty_col = "Quantity"
        price_col = "Price Per Unit"
        total_col = "Total Spent"
        qty = pd.to_numeric(df[qty_col], errors="coerce")
        price = pd.to_numeric(df[price_col], errors="coerce")
        total = pd.to_numeric(df[total_col], errors="coerce")
        computed = qty * price
        diff = (total - computed).abs()
        mismatch = (diff > 0.01).sum()
        derivable = (~qty.isna() & ~price.isna() & ~total.isna()).sum()
        console.print(f"  Rows with all three columns non-null : {derivable:,}")
        console.print(f"  Total ≠ Qty × Price (tol 0.01)       : [{'red' if mismatch > 0 else 'green'}]{mismatch:,}[/]")
        if derivable > 0:
            pct_bad = mismatch / derivable * 100
            console.print(f"  Mismatch rate                        : {pct_bad:.1f}%")
    except KeyError as e:
        console.print(f"  [yellow]Column not found: {e}[/]")

    # ------------------------------------------------------------------ #
    # Date column check
    # ------------------------------------------------------------------ #
    console.rule("[bold cyan]Date Column Check[/]")
    date_cols = [c for c in df.columns if "date" in c.lower() or "Date" in c]
    for dc in date_cols:
        parsed = pd.to_datetime(df[dc], errors="coerce")
        n_parsed = parsed.notna().sum()
        n_failed = parsed.isna().sum()
        date_range = f"{parsed.min().date()} → {parsed.max().date()}" if n_parsed > 0 else "N/A"
        console.print(f"  {dc}: parsed={n_parsed:,}  failed={n_failed:,}  range={date_range}")

    console.rule()
    console.print("[bold green]Profile complete.[/]")

    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    profile(RAW_CSV)
