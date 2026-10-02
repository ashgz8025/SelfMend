"""
Tune drift thresholds to achieve < 10 % false-alarm rate on clean batches.
Tests combinations of (p_threshold, drift_share_threshold).
"""
from __future__ import annotations
import json, sys, logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING)

from detection.drift_check import run_drift_checks

BATCHES_DIR = ROOT / "data" / "batches"
REF_PQ      = ROOT / "data" / "clean" / "reference.parquet"

reference = pd.read_parquet(REF_PQ)
batches   = [pd.read_parquet(p) for p in sorted(BATCHES_DIR.glob("batch_*.parquet"))]

p_vals       = [0.05, 0.01, 0.005, 0.001, 0.0001]
share_vals   = [0.30, 0.40, 0.50, 0.60]

print(f"{'p_thresh':>10} {'share_thresh':>12} {'FA_rate':>8} {'avg_share':>10}")
print("-" * 44)

best = None
for p in p_vals:
    for s in share_vals:
        alarms, shares = 0, []
        for b in batches:
            r = run_drift_checks(b, reference, drift_threshold=p, drift_share_threshold=s)
            if r.drift_detected:
                alarms += 1
            shares.append(r.drift_share)
        fa = alarms / len(batches) * 100
        avg_s = sum(shares) / len(shares)
        flag = " <-- target" if fa <= 10 and best is None else ""
        print(f"{p:>10.4f} {s:>12.2f} {fa:>7.1f}% {avg_s:>10.3f}{flag}")
        if fa <= 10 and best is None:
            best = (p, s)

print()
if best:
    print(f"Best thresholds: p={best[0]}, drift_share={best[1]}")
else:
    print("No combination achieved FA<=10% — need larger batches or different test.")
