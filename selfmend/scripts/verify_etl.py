"""Quick verification of ETL outputs."""
import pandas as pd
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
clean  = pd.read_parquet(root / "data/clean/clean.parquet")
ref    = pd.read_parquet(root / "data/clean/reference.parquet")
stats  = json.loads((root / "data/clean/baseline_stats.json").read_text())
batches = sorted((root / "data/batches").glob("batch_*.parquet"))

print(f"Clean    : {len(clean):,} rows x {clean.shape[1]} cols")
print(f"Reference: {len(ref):,} rows  ({len(ref)/len(clean)*100:.1f}%)")
print(f"Batches  : {len(batches)} files")
print()
print("Null rates (baseline):")
for col, rate in stats["null_rates"].items():
    print(f"  {col:<22}  {rate*100:.1f}%")
print()
print(f"Total mismatch rate (post-ETL): {stats['total_mismatch_rate']}")
print()
print("Category sets:")
for col, vals in stats["category_sets"].items():
    print(f"  {col}: {vals}")
print()
print("Numeric ranges:")
for col, r in stats["numeric_ranges"].items():
    print(f"  {col}: min={r['min']:.2f}  max={r['max']:.2f}  mean={r['mean']:.2f}  std={r['std']:.2f}")
print()
print("Sample batch sizes:")
for b in batches[:5]:
    df_b = pd.read_parquet(b)
    print(f"  {b.name}: {len(df_b)} rows")
