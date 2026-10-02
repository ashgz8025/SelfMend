"""Test Evidently drift with larger batch and direct PSI/chi-square."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from evidently import Dataset, Report
from evidently.metrics import ValueDrift

# Test 1: small batch (200 rows)
batch = pd.read_parquet(ROOT / "data/batches/batch_000.parquet")
ref = pd.read_parquet(ROOT / "data/clean/reference.parquet")

# All Coffee
cur_skewed = batch.copy()
non_null = cur_skewed["Item"].notna()
cur_skewed.loc[non_null, "Item"] = "Coffee"

report = Report([ValueDrift(column="Item")])
result = report.run(
    reference_data=Dataset.from_pandas(ref[["Item"]]),
    current_data=Dataset.from_pandas(cur_skewed[["Item"]]),
)
d = result.dict()
for m in d["metrics"]:
    print("ValueDrift Item (200 rows, all-Coffee):", m.get("value"))
    print("  config:", m.get("config"))

# Test 2: try with a much larger "current" sample (merge multiple batches)
import glob
all_batches = [pd.read_parquet(p) for p in sorted((ROOT / "data/batches").glob("batch_*.parquet"))]
cur_large = pd.concat(all_batches, ignore_index=True)
non_null2 = cur_large["Item"].notna()
cur_large.loc[non_null2, "Item"] = "Coffee"
print(f"\nLarge skewed batch: {len(cur_large)} rows, all Coffee")

report2 = Report([ValueDrift(column="Item")])
result2 = report2.run(
    reference_data=Dataset.from_pandas(ref[["Item"]]),
    current_data=Dataset.from_pandas(cur_large[["Item"]]),
)
d2 = result2.dict()
for m in d2["metrics"]:
    print("ValueDrift Item (4000 rows, all-Coffee):", m.get("value"))

# Test 3: try PSI metric
try:
    from evidently.metrics import ColumnDriftMetric
    print("\nColumnDriftMetric available")
except ImportError:
    print("\nColumnDriftMetric not available")
