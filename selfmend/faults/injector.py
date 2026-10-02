"""
Fault injector for SelfMend.

Injects labeled faults into clean batches and returns (corrupted_df, label).

Three fault classes:
  1. schema    : dropped/renamed column, dtype change
  2. data_quality : null spikes, out-of-range values, duplicates,
                    sentinel strings ("ERROR"/"UNKNOWN" reappearing),
                    Total not matching Quantity × Price
  3. distribution : shifted mean (numeric), skewed item mix
                    — should PASS GX rules, TRIGGER Evidently drift

Label schema:
  {
      "fault_type": str,         # e.g. "null_spike"
      "fault_class": str,        # "schema" | "data_quality" | "distribution"
      "affected_column": str,
      "correct_fix": str,        # whitelisted action name
      "origin": "injected",
      "params": dict             # fault-specific parameters
  }

Public API:
    inject_fault(df, fault_type, rng_seed) -> (pd.DataFrame, dict)
    random_fault(df, fault_class, rng_seed) -> (pd.DataFrame, dict)
    FAULT_REGISTRY: dict[str, callable]
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd

# ── Sentinel values that should not appear in clean data ─────────────────────
SENTINELS = ["ERROR", "UNKNOWN"]

# ── Fault registry ────────────────────────────────────────────────────────────
# Maps fault_type -> (injector_fn, fault_class, affected_column, correct_fix)
_REGISTRY: dict[str, tuple[Callable, str, str, str]] = {}


def _register(fault_type: str, fault_class: str, affected_column: str, correct_fix: str):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[fault_type] = (fn, fault_class, affected_column, correct_fix)
        return fn
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SCHEMA FAULTS
# ═══════════════════════════════════════════════════════════════════════════════

@_register("drop_column", "schema", "Price Per Unit", "escalate")
def _drop_column(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Drop a critical column from the batch."""
    col = "Price Per Unit"
    return df.drop(columns=[col]), {"dropped_column": col}


@_register("rename_column", "schema", "Total Spent", "escalate")
def _rename_column(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Rename a column to an unexpected name."""
    return df.rename(columns={"Total Spent": "Total_Spent_usd"}), \
           {"original": "Total Spent", "new_name": "Total_Spent_usd"}


@_register("dtype_change", "schema", "Quantity", "escalate")
def _dtype_change(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Cast a numeric column to string (simulates a schema regression)."""
    df = df.copy()
    df["Quantity"] = df["Quantity"].astype(str)
    return df, {"column": "Quantity", "new_dtype": "str"}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY FAULTS
# ═══════════════════════════════════════════════════════════════════════════════

@_register("null_spike_price", "data_quality", "Price Per Unit", "impute_nulls")
def _null_spike_price(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Spike null rate in Price Per Unit to ~50 % (baseline ~10 %)."""
    df = df.copy()
    idx = rng.choice(len(df), size=int(len(df) * 0.50), replace=False)
    df.loc[idx, "Price Per Unit"] = np.nan
    return df, {"null_rate_injected": 0.50, "baseline_null_rate": 0.10}


@_register("null_spike_quantity", "data_quality", "Quantity", "impute_nulls")
def _null_spike_quantity(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Spike null rate in Quantity to ~50 % (baseline ~8 %)."""
    df = df.copy()
    idx = rng.choice(len(df), size=int(len(df) * 0.50), replace=False)
    df.loc[idx, "Quantity"] = np.nan
    return df, {"null_rate_injected": 0.50, "baseline_null_rate": 0.08}


@_register("sentinel_reappear", "data_quality", "Item", "normalize_sentinels_to_null")
def _sentinel_reappear(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Re-introduce ERROR/UNKNOWN sentinel strings into Item column (~20 % of rows)."""
    df = df.copy()
    idx = rng.choice(len(df), size=int(len(df) * 0.20), replace=False)
    df.loc[idx, "Item"] = rng.choice(SENTINELS, size=len(idx))
    return df, {"sentinel_rate_injected": 0.20, "sentinels": SENTINELS}


@_register("sentinel_payment", "data_quality", "Payment Method", "normalize_sentinels_to_null")
def _sentinel_payment(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Re-introduce ERROR sentinel into Payment Method column (~20 % of rows)."""
    df = df.copy()
    idx = rng.choice(len(df), size=int(len(df) * 0.20), replace=False)
    df.loc[idx, "Payment Method"] = "ERROR"
    return df, {"sentinel_rate_injected": 0.20, "sentinel": "ERROR"}


@_register("total_mismatch", "data_quality", "Total Spent", "restore_total_from_quantity_price")
def _total_mismatch(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Corrupt Total Spent so it no longer matches Quantity × Price Per Unit (~30 % of rows)."""
    df = df.copy()
    valid = df["Quantity"].notna() & df["Price Per Unit"].notna() & df["Total Spent"].notna()
    valid_idx = df.index[valid].tolist()
    n_corrupt = int(len(valid_idx) * 0.30)
    corrupt_idx = rng.choice(valid_idx, size=n_corrupt, replace=False)
    noise = rng.uniform(2.0, 5.0, size=n_corrupt)
    df.loc[corrupt_idx, "Total Spent"] = (df.loc[corrupt_idx, "Total Spent"] + noise).round(2)
    return df, {"mismatch_rate_injected": 0.30}


@_register("out_of_range_quantity", "data_quality", "Quantity", "clip_outliers")
def _out_of_range_quantity(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Inject out-of-range Quantity values (e.g., 50-200) in ~15 % of rows."""
    df = df.copy()
    idx = rng.choice(len(df), size=int(len(df) * 0.15), replace=False)
    df.loc[idx, "Quantity"] = rng.integers(50, 200, size=len(idx)).astype(float)
    return df, {"out_of_range_rate": 0.15, "injected_range": [50, 200]}


@_register("duplicate_rows", "data_quality", "Transaction ID", "drop_duplicates")
def _duplicate_rows(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Duplicate ~20 % of rows."""
    df = df.copy()
    n_dup = int(len(df) * 0.20)
    dup_idx = rng.choice(len(df), size=n_dup, replace=False)
    dup_rows = df.iloc[dup_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df, {"duplicate_rate_injected": 0.20, "n_duplicates": n_dup}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUTION FAULTS (pass GX, trigger Evidently)
# ═══════════════════════════════════════════════════════════════════════════════

@_register("mean_shift_quantity", "distribution", "Quantity", "escalate")
def _mean_shift_quantity(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """
    Shift Quantity mean by +2.5 (from ~3.0 to ~5.5).
    Values stay in [1,5] range so GX range check passes,
    but KS-test should detect the shift.
    """
    df = df.copy()
    # Fill nulls temporarily to apply shift
    qty = df["Quantity"].copy()
    non_null = qty.notna()
    shifted = (qty[non_null] + 2.5).clip(upper=5.0)
    df.loc[non_null, "Quantity"] = shifted
    original_mean = float(qty[non_null].mean())
    return df, {
        "shift": 2.5,
        "original_mean": round(original_mean, 2),
        "new_mean": round(float(shifted.mean()), 2),
    }


@_register("mean_shift_price", "distribution", "Price Per Unit", "escalate")
def _mean_shift_price(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """
    Shift Price Per Unit by replacing all values with only high-price items.
    """
    df = df.copy()
    high_prices = [5.0, 6.0, 7.5]
    non_null = df["Price Per Unit"].notna()
    n = int(non_null.sum())
    df.loc[non_null, "Price Per Unit"] = rng.choice(high_prices, size=n)
    return df, {"price_distribution": "only_high_prices", "values": high_prices}


@_register("skewed_item_mix", "distribution", "Item", "escalate")
def _skewed_item_mix(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """
    Replace all Item values with 'Coffee' (extreme skew).
    Category is valid (passes GX InSet), but chi-square test detects the shift.
    """
    df = df.copy()
    non_null = df["Item"].notna()
    df.loc[non_null, "Item"] = "Coffee"
    return df, {"skew": "100% Coffee", "original_uniform": True}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

FAULT_REGISTRY: dict[str, tuple[Callable, str, str, str]] = _REGISTRY

FAULT_CLASSES: dict[str, list[str]] = {
    "schema": [k for k, (_, cls, _, _) in _REGISTRY.items() if cls == "schema"],
    "data_quality": [k for k, (_, cls, _, _) in _REGISTRY.items() if cls == "data_quality"],
    "distribution": [k for k, (_, cls, _, _) in _REGISTRY.items() if cls == "distribution"],
}


def inject_fault(
    df: pd.DataFrame,
    fault_type: str,
    rng_seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Inject a specific fault into a clean batch.

    Args:
        df         : Clean batch DataFrame (will be copied; not modified in place).
        fault_type : One of the keys in FAULT_REGISTRY.
        rng_seed   : Optional seed for reproducibility.

    Returns:
        (corrupted_df, label) where label conforms to:
        {
            "fault_type": str,
            "fault_class": str,
            "affected_column": str,
            "correct_fix": str,
            "origin": "injected",
            "params": dict,
        }

    Raises:
        ValueError if fault_type is not in the registry.
    """
    if fault_type not in _REGISTRY:
        raise ValueError(
            f"Unknown fault_type '{fault_type}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )

    rng = np.random.default_rng(rng_seed)
    fn, fault_class, affected_column, correct_fix = _REGISTRY[fault_type]

    corrupted_df, params = fn(df.copy(), rng)

    label = {
        "fault_type": fault_type,
        "fault_class": fault_class,
        "affected_column": affected_column,
        "correct_fix": correct_fix,
        "origin": "injected",
        "params": params,
    }
    return corrupted_df, label


def random_fault(
    df: pd.DataFrame,
    fault_class: str | None = None,
    rng_seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Inject a random fault, optionally restricted to a fault class.

    Args:
        df          : Clean batch DataFrame.
        fault_class : "schema" | "data_quality" | "distribution" | None (any).
        rng_seed    : Optional seed.

    Returns:
        (corrupted_df, label)
    """
    rnd = random.Random(rng_seed)

    if fault_class is not None:
        candidates = FAULT_CLASSES.get(fault_class, [])
        if not candidates:
            raise ValueError(f"Unknown fault_class '{fault_class}'")
    else:
        candidates = list(_REGISTRY.keys())

    chosen = rnd.choice(candidates)
    return inject_fault(df, chosen, rng_seed=rng_seed)
