"""Test all fault injectors."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from faults.injector import FAULT_REGISTRY, FAULT_CLASSES, inject_fault
import pandas as pd

df = pd.read_parquet(ROOT / "data/batches/batch_000.parquet")
print(f"Base batch: {len(df)} rows")
print()
print("FAULT_CLASSES:")
for cls, faults in FAULT_CLASSES.items():
    print(f"  {cls}: {faults}")
print()

for ft in sorted(FAULT_REGISTRY.keys()):
    try:
        corrupted, label = inject_fault(df.copy(), ft, rng_seed=42)
        col = label["affected_column"]
        fix = label["correct_fix"]
        print(f"  OK  {ft:<35} rows={len(corrupted):<5}  col={col}  fix={fix}")
    except Exception as e:
        print(f"  ERR {ft}: {e}")
