"""
Run the detector on all clean natural batches and report the false-alarm rate.

A 'false alarm' is when has_failures=True on a clean (uninjected) batch.
We distinguish:
  - GX false alarm: a GX check fails on clean data
  - Drift false alarm: Evidently flags drift on a naturally clean batch

Usage:
    python scripts/run_false_alarm_check.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)   # suppress per-batch INFO noise

from detection.detector import detect

BATCHES_DIR = ROOT / "data" / "batches"
STATS_JSON  = ROOT / "data" / "clean" / "baseline_stats.json"
REF_PQ      = ROOT / "data" / "clean" / "reference.parquet"


def main() -> None:
    stats     = json.loads(STATS_JSON.read_text())
    reference = pd.read_parquet(REF_PQ)
    batches   = sorted(BATCHES_DIR.glob("batch_*.parquet"))

    print(f"Running detector on {len(batches)} clean batches…\n")

    results = []
    for bp in batches:
        batch_df = pd.read_parquet(bp)
        report   = detect(batch_df, reference_df=reference, stats=stats, batch_id=bp.name)
        results.append({
            "batch": bp.name,
            "has_failures": report.has_failures,
            "gx_passed": report.gx_passed,
            "drift_detected": report.drift_detected,
            "n_gx_failures": len([f for f in report.failed_checks if f.layer == "gx"]),
            "n_drift_failures": len([f for f in report.failed_checks if f.layer == "drift"]),
            "drifted_cols": ", ".join(report.drifted_columns),
        })

    total          = len(results)
    any_failure    = sum(1 for r in results if r["has_failures"])
    gx_false_alarm = sum(1 for r in results if not r["gx_passed"])
    drift_alarm    = sum(1 for r in results if r["drift_detected"])

    print(f"{'Batch':<20} {'Failures':>8} {'GX OK':>6} {'Drift':>6} {'GX fails':>9} {'Drift fails':>12} {'Drifted cols'}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['batch']:<20} "
            f"{'YES' if r['has_failures'] else 'NO':>8} "
            f"{'OK' if r['gx_passed'] else 'FAIL':>6} "
            f"{'YES' if r['drift_detected'] else 'no':>6} "
            f"{r['n_gx_failures']:>9} "
            f"{r['n_drift_failures']:>12} "
            f"{r['drifted_cols']}"
        )

    print("\n" + "=" * 60)
    print(f"Total batches          : {total}")
    print(f"Any failure (false alarm): {any_failure}/{total}  ({any_failure/total*100:.1f}%)")
    print(f"GX false alarms        : {gx_false_alarm}/{total}  ({gx_false_alarm/total*100:.1f}%)")
    print(f"Drift false alarms     : {drift_alarm}/{total}  ({drift_alarm/total*100:.1f}%)")
    print(f"\nFALSE-ALARM RATE: {any_failure/total*100:.1f}%")


if __name__ == "__main__":
    main()
