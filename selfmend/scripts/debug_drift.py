"""Debug why skewed_item_mix doesn't drift."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from faults.injector import inject_fault
from detection.drift_check import run_drift_checks

batch = pd.read_parquet(ROOT / "data/batches/batch_000.parquet")
corrupted, label = inject_fault(batch.copy(), "skewed_item_mix", rng_seed=11)
ref = pd.read_parquet(ROOT / "data/clean/reference.parquet")

print("Corrupted Item value counts:")
print(corrupted["Item"].value_counts())
print()
print("Reference Item value counts (first 10):")
print(ref["Item"].value_counts().head(10))
print()

result = run_drift_checks(corrupted, ref, drift_threshold=0.01, drift_share_threshold=0.10)
print("Drift scores:", result.drift_scores)
print("Drifted cols:", result.drifted_columns)
print("Drift detected:", result.drift_detected)
