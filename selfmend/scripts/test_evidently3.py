"""Drill into Evidently 0.7 result structure."""
import pandas as pd
import numpy as np
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift, DriftedColumnsCount

np.random.seed(42)
ref = pd.DataFrame({
    "Quantity": np.random.normal(3, 1, 200).clip(1, 5),
    "Item": np.random.choice(["Coffee", "Tea", "Sandwich", "Salad"], 200),
})
cur = pd.DataFrame({
    "Quantity": np.random.normal(5, 0.5, 50).clip(1, 10),
    "Item": np.random.choice(["Coffee", "Coffee", "Coffee", "Salad"], 50),
})

# Test 1: per-column ValueDrift
report = Report([ValueDrift(column="Quantity"), ValueDrift(column="Item"), DriftedColumnsCount()])
result = report.run(reference_data=Dataset.from_pandas(ref), current_data=Dataset.from_pandas(cur))
d = result.dict()

print("=== ValueDrift results ===")
for m in d["metrics"]:
    print(repr(m)[:300])
    print()

# Test 2: try snapshot() for legacy-style access
print("\n=== snapshot ===")
snap = result.snapshot()
print(type(snap))
print(list(snap.keys()) if isinstance(snap, dict) else dir(snap)[:20])
