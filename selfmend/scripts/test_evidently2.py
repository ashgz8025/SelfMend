"""Test Evidently 0.7 new SDK drift check."""
import pandas as pd
import numpy as np
import evidently
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset

print("Evidently:", evidently.__version__)

# Simple numeric + categorical reference vs current
ref = pd.DataFrame({
    "Quantity": np.random.normal(3, 1, 200).clip(1, 5),
    "Price Per Unit": np.random.choice([1.5, 2.5, 3.5, 4.0, 5.0, 6.0, 7.5], 200),
    "Item": np.random.choice(["Coffee", "Tea", "Sandwich", "Salad"], 200),
    "Location": np.random.choice(["Downtown", "Uptown", "Airport", "Mall"], 200),
})
# Drifted current: shift Quantity mean significantly
cur = pd.DataFrame({
    "Quantity": np.random.normal(5, 0.5, 50).clip(1, 10),   # shifted
    "Price Per Unit": np.random.choice([1.5, 2.5, 3.5, 4.0, 5.0, 6.0, 7.5], 50),
    "Item": np.random.choice(["Coffee", "Coffee", "Coffee", "Salad"], 50),   # skewed
    "Location": np.random.choice(["Downtown", "Uptown", "Airport", "Mall"], 50),
})

# New Evidently 0.7 SDK: Dataset wraps the dataframe
ref_ds = Dataset.from_pandas(ref)
cur_ds = Dataset.from_pandas(cur)

report = Report([DataDriftPreset()])
result = report.run(reference_data=ref_ds, current_data=cur_ds)
d = result.dict()
print("Keys:", list(d.keys()))

# Explore structure
for i, metric in enumerate(d.get("metrics", [])[:5]):
    print(f"\nMetric[{i}]: {metric.get('metric_id', metric.get('type', '?'))}")
    res = metric.get("value", metric.get("result", {}))
    if isinstance(res, dict):
        print(f"  Keys: {list(res.keys())[:10]}")
