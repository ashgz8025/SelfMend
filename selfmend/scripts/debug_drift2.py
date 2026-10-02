"""More detailed drift diagnosis."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from faults.injector import inject_fault

batch = pd.read_parquet(ROOT / "data/batches/batch_000.parquet")
corrupted, label = inject_fault(batch.copy(), "skewed_item_mix", rng_seed=11)
ref = pd.read_parquet(ROOT / "data/clean/reference.parquet")

# Manually compute chi-square
cats = ['Coffee','Tea','Sandwich','Salad','Cake','Juice','Cookie','Smoothie']
ref_counts = ref['Item'].value_counts().reindex(cats, fill_value=0)
cur_counts = corrupted['Item'].value_counts().reindex(cats, fill_value=0)
print("Ref counts:", ref_counts.to_dict())
print("Cur counts:", cur_counts.to_dict())

# Combine into contingency table
contingency = pd.DataFrame({'ref': ref_counts, 'cur': cur_counts}).T
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nManual chi2={chi2:.2f}  p={p:.6e}  dof={dof}")

# Try mean_shift_quantity which should work
corrupted2, label2 = inject_fault(batch.copy(), "mean_shift_quantity", rng_seed=11)
print("\nmean_shift_quantity Quantity stats:")
print(f"  ref mean: {ref['Quantity'].mean():.2f}")
print(f"  corrupted mean: {corrupted2['Quantity'].mean():.2f}")

from scipy.stats import ks_2samp
ref_qty = ref['Quantity'].dropna()
cur_qty = corrupted2['Quantity'].dropna()
stat, pval = ks_2samp(ref_qty, cur_qty)
print(f"  KS stat={stat:.4f}  p={pval:.6e}")
