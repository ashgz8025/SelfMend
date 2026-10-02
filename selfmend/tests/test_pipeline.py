"""
pytest tests for SelfMend pipeline.

Tests:
  1. test_injector_all_classes      : all 3 fault classes produce valid labels
  2. test_detector_gx_passes_clean  : clean batch passes GX
  3. test_detector_flags_fault      : injected fault is detected
  4. test_policy_whitelist          : non-whitelisted action is rejected
  5. test_low_faithfulness_escalates: forced low faithfulness routes to escalate (not auto_fix)
  6. test_remediation_drop_duplicates : drop_duplicates succeeds and promotes
  7. test_remediation_restore_total   : restore_total_from_quantity_price succeeds
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use offline thresholds so tests don't need GROQ key
os.environ.setdefault("FAITHFULNESS_THRESHOLD", "0.50")
os.environ.setdefault("CONFIDENCE_THRESHOLD",   "0.60")

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clean_batch() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/batches/batch_000.parquet")


@pytest.fixture(scope="module")
def reference_df() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/clean/reference.parquet")


@pytest.fixture(scope="module")
def baseline_stats() -> dict:
    return json.loads((ROOT / "data/clean/baseline_stats.json").read_text())


# ── Test 1: Fault injector ─────────────────────────────────────────────────────

class TestFaultInjector:
    def test_schema_fault_produces_valid_label(self, clean_batch):
        from faults.injector import inject_fault
        corrupted, label = inject_fault(clean_batch.copy(), "drop_column", rng_seed=1)
        assert label["fault_class"] == "schema"
        assert label["origin"] == "injected"
        assert "Price Per Unit" not in corrupted.columns

    def test_data_quality_fault_produces_valid_label(self, clean_batch):
        from faults.injector import inject_fault
        corrupted, label = inject_fault(clean_batch.copy(), "null_spike_price", rng_seed=2)
        assert label["fault_class"] == "data_quality"
        assert label["affected_column"] == "Price Per Unit"
        null_rate = corrupted["Price Per Unit"].isna().mean()
        assert null_rate > 0.30, f"Expected null rate > 0.30, got {null_rate:.2f}"

    def test_distribution_fault_produces_valid_label(self, clean_batch):
        from faults.injector import inject_fault
        corrupted, label = inject_fault(clean_batch.copy(), "skewed_item_mix", rng_seed=3)
        assert label["fault_class"] == "distribution"
        assert corrupted["Item"].dropna().eq("Coffee").all()

    def test_all_fault_types_run(self, clean_batch):
        from faults.injector import FAULT_REGISTRY, inject_fault
        for ft in FAULT_REGISTRY:
            corrupted, label = inject_fault(clean_batch.copy(), ft, rng_seed=42)
            assert label["fault_type"] == ft
            assert label["origin"] == "injected"

    def test_unknown_fault_raises(self, clean_batch):
        from faults.injector import inject_fault
        with pytest.raises(ValueError, match="Unknown fault_type"):
            inject_fault(clean_batch.copy(), "nonexistent_fault")


# ── Test 2: Detector on clean data ────────────────────────────────────────────

class TestDetector:
    def test_clean_batch_passes_gx(self, clean_batch, reference_df, baseline_stats):
        from detection.gx_checks import run_gx_checks
        result = run_gx_checks(clean_batch, baseline_stats)
        assert result.success, f"GX failed on clean batch: {result.failed_checks}"

    def test_clean_batch_no_drift(self, clean_batch, reference_df):
        from detection.drift_check import run_drift_checks
        result = run_drift_checks(clean_batch, reference_df)
        assert not result.drift_detected, (
            f"Drift falsely detected on clean batch: {result.drifted_columns}"
        )

    def test_null_spike_detected_by_gx(self, clean_batch, reference_df, baseline_stats):
        from faults.injector import inject_fault
        from detection.gx_checks import run_gx_checks
        corrupted, _ = inject_fault(clean_batch.copy(), "null_spike_price", rng_seed=10)
        result = run_gx_checks(corrupted, baseline_stats)
        assert not result.success, "GX should fail on null spike"
        cols = [fc["column"] for fc in result.failed_checks]
        assert "Price Per Unit" in cols

    def test_drift_only_detected_by_evidently_not_gx(self, clean_batch, reference_df, baseline_stats):
        """
        Distribution fault: GX passes (Coffee is valid), Evidently flags drift.

        Evidently 0.7 uses Jensen-Shannon distance for categorical columns.
        Drift detected when distance > threshold (default 0.17).
        For 100% Coffee batch vs balanced reference, JS distance >> 0.17.
        """
        from faults.injector import inject_fault
        from detection.gx_checks import run_gx_checks
        from detection.drift_check import run_drift_checks
        corrupted, label = inject_fault(clean_batch.copy(), "skewed_item_mix", rng_seed=11)
        gx_result    = run_gx_checks(corrupted, baseline_stats)
        # Run with low share threshold (1 column = 1/6 = 16.7% share)
        drift_result = run_drift_checks(corrupted, reference_df,
                                        drift_threshold=0.17,
                                        drift_share_threshold=0.10)
        # GX should PASS (Coffee is a valid category, mostly rule allows it)
        assert gx_result.success, (
            f"GX should pass for skewed_item_mix, got failures: {gx_result.failed_checks}"
        )
        # Evidently drift_scores contains JS distances (NOT p-values)
        # JS distance for 100% Coffee batch is ~0.71 >> threshold 0.17
        assert "Item" in drift_result.drift_scores, "Item should be in drift scores"
        item_dist = drift_result.drift_scores.get("Item", 0.0)
        assert item_dist > 0.17, (
            f"Item JS distance should be > 0.17 for 100% Coffee distribution, got {item_dist:.4f}"
        )
        assert drift_result.drift_detected, (
            f"Drift should be detected with share_threshold=0.10: "
            f"drifted={drift_result.drifted_columns}, scores={drift_result.drift_scores}"
        )


# ── Test 3: Policy gate ────────────────────────────────────────────────────────

class TestPolicyGate:
    def test_whitelisted_action_can_auto_fix(self):
        from policy.gate import evaluate_policy
        state = {
            "faithfulness_score": 0.90,
            "faithfulness_passed": True,
            "diagnosis": {
                "proposed_action": "drop_duplicates",
                "confidence": 0.80,
                "risk_level": "low",
            },
        }
        decision, reason = evaluate_policy(state)
        assert decision == "auto_fix", f"Expected auto_fix, got {decision}: {reason}"

    def test_non_whitelisted_action_escalates(self):
        from policy.gate import evaluate_policy
        state = {
            "faithfulness_score": 0.95,
            "faithfulness_passed": True,
            "diagnosis": {
                "proposed_action": "delete_column",   # not whitelisted
                "confidence": 0.90,
                "risk_level": "low",
            },
        }
        decision, reason = evaluate_policy(state)
        assert decision == "escalate"
        assert "whitelist" in reason.lower()

    def test_high_risk_escalates(self):
        from policy.gate import evaluate_policy
        state = {
            "faithfulness_score": 0.95,
            "faithfulness_passed": True,
            "diagnosis": {
                "proposed_action": "drop_duplicates",
                "confidence": 0.90,
                "risk_level": "high",
            },
        }
        decision, _ = evaluate_policy(state)
        assert decision == "escalate"

    def test_low_confidence_escalates(self):
        from policy.gate import evaluate_policy
        state = {
            "faithfulness_score": 0.90,
            "faithfulness_passed": True,
            "diagnosis": {
                "proposed_action": "drop_duplicates",
                "confidence": 0.30,   # below 0.60
                "risk_level": "low",
            },
        }
        decision, _ = evaluate_policy(state)
        assert decision == "escalate"


# ── Test 4: Low faithfulness forces escalation ────────────────────────────────

class TestLowFaithfulnessEscalation:
    """
    SPEC requirement: "pytest: force a low faithfulness score and assert routing
    to escalation, not auto-fix."

    Tests the policy gate directly with an injected state containing
    faithfulness_score=0.10, well below the 0.80 threshold.
    Also tests via the full graph using faithfulness_override parameter.
    """

    def test_low_faithfulness_policy_escalates(self):
        """Unit test: policy gate escalates when faithfulness < threshold."""
        from policy.gate import evaluate_policy
        # Temporarily reset to standard threshold
        old = os.environ.get("FAITHFULNESS_THRESHOLD")
        os.environ["FAITHFULNESS_THRESHOLD"] = "0.80"

        state = {
            "faithfulness_score": 0.10,
            "faithfulness_passed": False,   # pre-computed in verifier_node
            "diagnosis": {
                "proposed_action": "impute_nulls",
                "confidence": 0.80,
                "risk_level": "low",
            },
        }
        decision, reason = evaluate_policy(state)

        if old:
            os.environ["FAITHFULNESS_THRESHOLD"] = old
        else:
            del os.environ["FAITHFULNESS_THRESHOLD"]

        assert decision == "escalate", (
            f"Expected escalate for faithfulness=0.10, got {decision}"
        )
        assert "faithfulness" in reason.lower(), (
            f"Reason should mention faithfulness: {reason}"
        )

    def test_escalate_node_not_remediator_for_low_faithfulness(self, clean_batch, reference_df, baseline_stats):
        """
        Integration test: inject a fault, force low faithfulness via the graph,
        verify the escalate path ran (not remediator).

        Uses the graph's faithfulness_override parameter which directly sets
        faithfulness_score and faithfulness_passed in the verifier output.
        """
        from faults.injector import inject_fault
        from agents.graph import run_pipeline

        old = os.environ.get("FAITHFULNESS_THRESHOLD")
        os.environ["FAITHFULNESS_THRESHOLD"] = "0.80"

        corrupted, _ = inject_fault(clean_batch.copy(), "null_spike_price", rng_seed=99)
        state = run_pipeline(
            corrupted,
            batch_id="test_low_faith_graph",
            reference_df=reference_df,
            stats=baseline_stats,
            faithfulness_override=0.10,
        )

        if old:
            os.environ["FAITHFULNESS_THRESHOLD"] = old
        else:
            os.environ["FAITHFULNESS_THRESHOLD"] = "0.50"   # restore offline default

        assert state.get("faithfulness_score") == pytest.approx(0.10), (
            f"faithfulness_score should be 0.10, got {state.get('faithfulness_score')}"
        )
        assert state.get("faithfulness_passed") is False
        assert state["decision"] == "escalate", (
            f"Expected escalate, got {state['decision']}"
        )
        # Remediator should NOT have run
        assert state.get("remediation_action") is None, (
            f"Remediator ran unexpectedly: {state.get('remediation_action')}"
        )


# ── Test 5: Remediation actions ───────────────────────────────────────────────

class TestRemediationActions:
    def test_drop_duplicates_removes_dupes(self, clean_batch, baseline_stats):
        from faults.injector import inject_fault
        from remediation.actions import apply_action
        corrupted, _ = inject_fault(clean_batch.copy(), "duplicate_rows", rng_seed=5)
        n_before = len(corrupted)
        fixed = apply_action(corrupted, "drop_duplicates", stats=baseline_stats)
        assert len(fixed) < n_before
        assert fixed.duplicated().sum() == 0

    def test_restore_total_fixes_mismatch(self, clean_batch, baseline_stats):
        from faults.injector import inject_fault
        from remediation.actions import apply_action
        import numpy as np
        corrupted, _ = inject_fault(clean_batch.copy(), "total_mismatch", rng_seed=6)
        fixed = apply_action(corrupted, "restore_total_from_quantity_price", stats=baseline_stats)
        qty   = pd.to_numeric(fixed["Quantity"], errors="coerce")
        price = pd.to_numeric(fixed["Price Per Unit"], errors="coerce")
        total = pd.to_numeric(fixed["Total Spent"], errors="coerce")
        can   = qty.notna() & price.notna() & total.notna()
        diff  = ((total[can] - (qty[can] * price[can])).abs() / (qty[can] * price[can]).abs().clip(lower=1e-9))
        assert float(diff.max()) < 0.02, f"Total mismatch not fixed: max rel error = {float(diff.max()):.4f}"

    def test_non_whitelisted_action_raises(self, clean_batch, baseline_stats):
        from remediation.actions import apply_action
        with pytest.raises(ValueError, match="not in the whitelist"):
            apply_action(clean_batch.copy(), "exec_python_code", stats=baseline_stats)

    def test_normalize_sentinels_removes_error_unknown(self, clean_batch, baseline_stats):
        from faults.injector import inject_fault
        from remediation.actions import apply_action
        corrupted, _ = inject_fault(clean_batch.copy(), "sentinel_reappear", rng_seed=7)
        assert corrupted["Item"].isin(["ERROR", "UNKNOWN"]).any()
        fixed = apply_action(corrupted, "normalize_sentinels_to_null", stats=baseline_stats)
        assert not fixed["Item"].isin(["ERROR", "UNKNOWN"]).any()

    def test_impute_nulls_reduces_null_rate(self, clean_batch, baseline_stats):
        from faults.injector import inject_fault
        from remediation.actions import apply_action
        corrupted, _ = inject_fault(clean_batch.copy(), "null_spike_quantity", rng_seed=8)
        null_before = corrupted["Quantity"].isna().mean()
        fixed = apply_action(corrupted, "impute_nulls", stats=baseline_stats)
        null_after = fixed["Quantity"].isna().mean()
        assert null_after < null_before, "impute_nulls should reduce null rate"
        assert null_after == 0.0, "impute_nulls should eliminate all nulls"
