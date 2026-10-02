"""
Quick smoke-test: inject a null_spike_price fault and run the full pipeline.
No Groq key needed (offline mode).
"""
import sys, logging, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

import pandas as pd
from faults.injector import inject_fault
from agents.graph import run_pipeline

# Load a clean batch
batch = pd.read_parquet(ROOT / "data/batches/batch_000.parquet")
print(f"Clean batch: {len(batch)} rows")

# Inject a null spike in Price Per Unit
corrupted, label = inject_fault(batch, "null_spike_price", rng_seed=42)
print(f"Fault: {label['fault_type']} on {label['affected_column']}")
print(f"Correct fix: {label['correct_fix']}")
print()

# Run pipeline
state = run_pipeline(corrupted, batch_id="test_null_spike")

print("=" * 60)
print(f"Decision        : {state['decision']}")
print(f"Faithfulness    : {state['faithfulness_score']:.3f}")
print(f"Faith passed    : {state['faithfulness_passed']}")
print(f"Diagnosis action: {state['diagnosis']['proposed_action'] if state['diagnosis'] else 'N/A'}")
print(f"Confidence      : {state['diagnosis']['confidence'] if state['diagnosis'] else 'N/A'}")
print(f"Risk            : {state['diagnosis']['risk_level'] if state['diagnosis'] else 'N/A'}")
print(f"Decision reason : {state['decision_reason']}")
if state['decision'] == 'auto_fix':
    print(f"Remediation     : {state['remediation_action']}  success={state['remediation_success']}")
else:
    print(f"Escalation      : {state['escalation_reason']}")
print()

# Show retrieved incidents
print("Retrieved incidents:")
for inc in state.get("retrieved_incidents", []):
    print(f"  [{inc['similarity_score']:.3f}] {inc['incident_id']}: {inc['failure_description'][:60]}...")
