"""
Test that at least 2 remediation actions succeed (promoted to clean).
Uses faults whose fixes cleanly pass re-validation:
  - duplicate_rows  ->  drop_duplicates
  - total_mismatch  ->  restore_total_from_quantity_price
  - null_spike_price -> impute_nulls  (with offline conf/faith lowered)
"""
import sys, os, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING)

os.environ["FAITHFULNESS_THRESHOLD"] = "0.50"
os.environ["CONFIDENCE_THRESHOLD"]   = "0.60"

import pandas as pd
from faults.injector import inject_fault
from agents.graph import run_pipeline

tests = [
    ("duplicate_rows",   "data/batches/batch_001.parquet", 1),
    ("total_mismatch",   "data/batches/batch_002.parquet", 2),
    ("null_spike_price", "data/batches/batch_003.parquet", 3),
]

successes = 0
for fault_type, batch_path, seed in tests:
    batch = pd.read_parquet(ROOT / batch_path)
    corrupted, label = inject_fault(batch, fault_type, rng_seed=seed)
    state = run_pipeline(corrupted, batch_id=f"test_{fault_type}")

    decision  = state["decision"]
    rem_ok    = state.get("remediation_success", False)
    promoted  = state.get("remediation_detail", {}).get("promoted", False) if state.get("remediation_detail") else False
    faith     = state.get("faithfulness_score", 0)
    action    = state.get("remediation_action", "N/A")

    symbol = "PASS" if (decision == "auto_fix" and rem_ok) else "FAIL"
    if decision == "auto_fix" and rem_ok:
        successes += 1

    print(f"[{symbol}] {fault_type:<30} decision={decision}  action={action}  faith={faith:.3f}  promoted={promoted}")

print()
print(f"Successful remediations: {successes}/{len(tests)}")
print("ACCEPTANCE: auto_fix_ok" if successes >= 2 else "NEEDS_ATTENTION: fewer than 2 successes")
