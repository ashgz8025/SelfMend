import sys; sys.path.insert(0, ".")
import pandas as pd
from detection.drift_check import run_drift_checks

ref = pd.read_parquet("data/clean/reference.parquet")
b = pd.read_parquet("data/batches/batch_017.parquet")
r = run_drift_checks(b, ref)
print("Drift scores:", r.drift_scores)
print("Drifted:", r.drifted_columns)
print("Share:", r.drift_share)
