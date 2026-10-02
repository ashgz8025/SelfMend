"""
Test the auto_fix path by temporarily lowering the faithfulness threshold.
Uses sentinel_reappear fault (high evidence overlap → high mock faithfulness).
"""
import sys, os, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING)

# Lower threshold so offline mock score passes
os.environ["FAITHFULNESS_THRESHOLD"] = "0.50"
os.environ["CONFIDENCE_THRESHOLD"] = "0.60"

import pandas as pd
from faults.injector import inject_fault
from agents.graph import run_pipeline, get_graph

batch = pd.read_parquet(ROOT / "data/batches/batch_005.parquet")
corrupted, label = inject_fault(batch, "sentinel_reappear", rng_seed=42)
print(f"Fault: {label['fault_type']}  correct_fix: {label['correct_fix']}")

state = run_pipeline(corrupted, batch_id="test_sentinel_autofix")

print(f"Decision         : {state['decision']}")
print(f"Faithfulness     : {state['faithfulness_score']:.3f}")
print(f"Faith passed     : {state['faithfulness_passed']}")
print(f"Proposed action  : {state['diagnosis']['proposed_action'] if state['diagnosis'] else 'N/A'}")
print(f"Confidence       : {state['diagnosis']['confidence'] if state['diagnosis'] else 'N/A'}")
print(f"Risk             : {state['diagnosis']['risk_level'] if state['diagnosis'] else 'N/A'}")
print(f"Decision reason  : {state['decision_reason']}")

if state["decision"] == "auto_fix":
    print(f"Remediation action  : {state['remediation_action']}")
    print(f"Remediation success : {state['remediation_success']}")
    if state.get("remediation_detail"):
        det = state["remediation_detail"]
        print(f"Promoted to clean   : {det.get('promoted', False)}")
        print(f"Post-fix has_failures: {det.get('post_detection', {}).get('has_failures', '?')}")
else:
    print(f"Escalation: {state['escalation_reason']}")
