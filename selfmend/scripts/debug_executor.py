"""Test executor chaining."""
import sys, logging, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
os.environ["FAITHFULNESS_THRESHOLD"] = "0.50"
os.environ["CONFIDENCE_THRESHOLD"] = "0.60"

import pandas as pd
from faults.injector import inject_fault
from remediation.executor import execute_remediation

stats = json.loads((ROOT / "data/clean/baseline_stats.json").read_text())
ref   = pd.read_parquet(ROOT / "data/clean/reference.parquet")
batch = pd.read_parquet(ROOT / "data/batches/batch_005.parquet")

# Test 1: sentinel_reappear -> normalize_sentinels_to_null (executor chains impute + restore_total)
corrupted, _ = inject_fault(batch, "sentinel_reappear", rng_seed=6)
result = execute_remediation(corrupted, "normalize_sentinels_to_null", "test_sentinel", stats=stats, reference_df=ref)
print(f"sentinel_reappear: success={result['success']}  promoted={result['promoted']}")
if not result['success']:
    det = result.get('post_detection', {})
    for fc in det.get('failed_checks', []):
        print(f"  FAIL: {fc['check_type']} on {fc['column']} — {fc['detail']}")

# Test 2: null_spike_price -> impute_nulls (executor chains restore_total)
batch2 = pd.read_parquet(ROOT / "data/batches/batch_003.parquet")
corrupted2, _ = inject_fault(batch2, "null_spike_price", rng_seed=4)
result2 = execute_remediation(corrupted2, "impute_nulls", "test_null_spike", stats=stats, reference_df=ref)
print(f"null_spike_price: success={result2['success']}  promoted={result2['promoted']}")
if not result2['success']:
    det = result2.get('post_detection', {})
    for fc in det.get('failed_checks', []):
        print(f"  FAIL: {fc['check_type']} on {fc['column']} — {fc['detail']}")
