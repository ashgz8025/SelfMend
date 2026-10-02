"""
Seed script: generate a realistic replica of the Kaggle
"Dirty Cafe Sales" dataset for development/testing.

This script is used ONLY when Kaggle credentials are unavailable.
It replicates the known dirty patterns of the real dataset:
  - Same 8-column schema
  - Sentinel strings: "ERROR", "UNKNOWN" in Item, Payment Method, Location
  - ~10 % null Price Per Unit  (known pattern in the real dataset)
  - ~8  % null Quantity
  - ~12 % null Total Spent
  - Total ≠ Quantity × Price for ~6 % of rows (real dataset corruption)
  - Duplicate rows (~2 %)
  - Date range: 2023-01-01 to 2023-12-31
  - Item mix: Coffee, Tea, Sandwich, Salad, Cake, Juice, Cookie, Smoothie
  - Location: Downtown, Uptown, Airport, Mall
  - Payment: Cash, Credit Card, Digital Wallet

Usage:
    python scripts/seed_raw_data.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 10_000
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dirty_cafe_sales.csv"

rng = np.random.default_rng(SEED)
random.seed(SEED)

ITEMS = ["Coffee", "Tea", "Sandwich", "Salad", "Cake", "Juice", "Cookie", "Smoothie"]
LOCATIONS = ["Downtown", "Uptown", "Airport", "Mall"]
PAYMENTS = ["Cash", "Credit Card", "Digital Wallet"]
ITEM_PRICE: dict[str, float] = {
    "Coffee": 3.50, "Tea": 2.50, "Sandwich": 6.00, "Salad": 7.50,
    "Cake": 4.00, "Juice": 3.00, "Cookie": 1.50, "Smoothie": 5.00,
}


def _make_clean_row(idx: int) -> dict:
    item = random.choice(ITEMS)
    qty = int(rng.integers(1, 6))
    price = ITEM_PRICE[item]
    total = round(qty * price, 2)
    date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
    return {
        "Transaction ID": f"TXN-{idx:05d}",
        "Date": date.strftime("%Y-%m-%d"),
        "Item": item,
        "Quantity": qty,
        "Price Per Unit": price,
        "Total Spent": total,
        "Payment Method": random.choice(PAYMENTS),
        "Location": random.choice(LOCATIONS),
    }


def generate_dirty_dataset() -> pd.DataFrame:
    rows = [_make_clean_row(i) for i in range(1, N_ROWS + 1)]
    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Inject dirtiness matching the real Kaggle dataset patterns
    # ------------------------------------------------------------------ #

    # 1. Sentinel strings in categorical columns
    sentinel_item_idx = rng.choice(N_ROWS, size=int(N_ROWS * 0.05), replace=False)
    df.loc[sentinel_item_idx, "Item"] = rng.choice(["ERROR", "UNKNOWN"], size=len(sentinel_item_idx))

    sentinel_pay_idx = rng.choice(N_ROWS, size=int(N_ROWS * 0.04), replace=False)
    df.loc[sentinel_pay_idx, "Payment Method"] = rng.choice(
        ["ERROR", "UNKNOWN"], size=len(sentinel_pay_idx)
    )

    sentinel_loc_idx = rng.choice(N_ROWS, size=int(N_ROWS * 0.03), replace=False)
    df.loc[sentinel_loc_idx, "Location"] = "ERROR"

    # 2. Null spikes
    null_price_idx = rng.choice(N_ROWS, size=int(N_ROWS * 0.10), replace=False)
    df.loc[null_price_idx, "Price Per Unit"] = np.nan

    null_qty_idx = rng.choice(N_ROWS, size=int(N_ROWS * 0.08), replace=False)
    df.loc[null_qty_idx, "Quantity"] = np.nan

    null_total_idx = rng.choice(N_ROWS, size=int(N_ROWS * 0.12), replace=False)
    df.loc[null_total_idx, "Total Spent"] = np.nan

    # 3. Total ≠ Quantity × Price (corruption)
    mismatch_idx = rng.choice(N_ROWS, size=int(N_ROWS * 0.06), replace=False)
    df.loc[mismatch_idx, "Total Spent"] = df.loc[mismatch_idx, "Total Spent"].apply(
        lambda x: round(x * rng.uniform(0.5, 2.0), 2) if not pd.isna(x) else x
    )

    # 4. Duplicate rows (~2 %)
    dup_source_idx = rng.choice(N_ROWS, size=int(N_ROWS * 0.02), replace=False)
    dup_rows = df.iloc[dup_source_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Shuffle so duplicates are not all at the end
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dirty_dataset()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape  : {df.shape}")
