"""Understand the ValueDrift metric method per column type."""
import sys, pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidently import Dataset, Report
from evidently.metrics import ValueDrift

ref = pd.read_parquet(ROOT / "data/clean/reference.parquet")
batches = sorted((ROOT / "data/batches").glob("batch_*.parquet"))
batch = pd.read_parquet(batches[0])

# Check default methods per column type
cols_to_check = ["Quantity", "Price Per Unit", "Item", "Payment Method"]
for col in cols_to_check:
    report = Report([ValueDrift(column=col)])
    result = report.run(
        reference_data=Dataset.from_pandas(ref[[col]]),
        current_data=Dataset.from_pandas(batch[[col]]),
    )
    for m in result.dict()["metrics"]:
        method = m.get("config", {}).get("method", "?")
        threshold = m.get("config", {}).get("threshold", "?")
        value = m.get("value")
        # Drift when value > threshold (distance metrics) OR value < threshold (p-values)
        is_pvalue = "p_value" in method.lower() or "p-value" in method.lower() or "ks" in method.lower().replace("-","")
        if is_pvalue:
            drifted = float(value) < float(threshold)
        else:
            drifted = float(value) > float(threshold)
        print(f"  {col:<22} method={method:<30} threshold={threshold}  value={value:.4f}  drifted={drifted}")
