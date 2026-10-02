"""Debug why sentinel_reappear + normalize_sentinels_to_null quarantines."""
import sys, logging, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

import pandas as pd
from faults.injector import inject_fault
from remediation.actions import apply_action, normalize_sentinels_to_null, impute_nulls
from detection.detector import detect

batch = pd.read_parquet(ROOT / "data/batches/batch_005.parquet")
stats = json.loads((ROOT / "data/clean/baseline_stats.json").read_text())
ref   = pd.read_parquet(ROOT / "data/clean/reference.parquet")

corrupted, label = inject_fault(batch, "sentinel_reappear", rng_seed=6)
print("Corrupted Item null+sentinel rate:", corrupted["Item"].isna().mean() + corrupted["Item"].isin(["ERROR","UNKNOWN"]).mean())

# Step 1: normalize sentinels
step1 = apply_action(corrupted, "normalize_sentinels_to_null", stats=stats)
print("After normalize_sentinels: Item null rate:", step1["Item"].isna().mean())

# Step 2: impute
step2 = apply_action(step1, "impute_nulls", stats=stats)
print("After impute_nulls: Item null rate:", step2["Item"].isna().mean())

# Re-detect
rep = detect(step2, reference_df=ref, stats=stats, batch_id="debug_sentinel")
print("Re-validation has_failures:", rep.has_failures)
for fc in rep.failed_checks:
    print(f"  FAIL: {fc.check_type} on {fc.column} — {fc.detail}")
