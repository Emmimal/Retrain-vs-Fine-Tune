"""
tests/test_all.py
-----------------
28-test suite for Article 08: Retrain vs Fine-Tune vs Train from Scratch.

Covers:
  - MLP architecture (head expansion, freeze/unfreeze, snapshot/restore)
  - Data generators (all four scenarios)
  - All three strategy implementations
  - Evaluation metrics (accuracy, F1, ECE)
  - Decision engine (all six strategies, constraint elimination)
  - Cost model

Run with:
  python -m pytest tests/test_all.py -v
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generators import (
    NewDataScenario,
    DistributionShiftScenario,
    NewClassScenario,
    DomainTransferScenario,
)
from data.loaders import make_loader, make_eval_loader
from evaluation.decision_framework import (
    DataSignals, ModelSignals, DriftSignals, ConstraintSignals,
    DecisionEngine, Strategy,
)
from evaluation.metrics import (
    compute_accuracy, compute_macro_f1, compute_ece,
    CostTracker,
)
from experiments.cost_model import CostModel
from models.base_model import MLP
from strategies.fine_tune import FineTuner, FineTuneMode
from strategies.retrain import Retrainer, RetrainMode
from strategies.train_from_scratch import TrainFromScratch


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def small_model():
    return MLP(input_dim=10, hidden_dims=[32, 16], output_dim=3)


@pytest.fixture
def tiny_loader():
    X = torch.randn(60, 10)
    y = torch.randint(0, 3, (60,))
    return make_loader(X, y, batch_size=16, seed=42)


@pytest.fixture
def tiny_eval_loader():
    X = torch.randn(30, 10)
    y = torch.randint(0, 3, (30,))
    return make_eval_loader(X, y, batch_size=30)


# ===========================================================================
# MLP architecture tests
# ===========================================================================

class TestMLP:

    def test_forward_shape(self, small_model):
        x = torch.randn(8, 10)
        out = small_model(x)
        assert out.shape == (8, 3), f"Expected (8,3), got {out.shape}"

    def test_freeze_trunk(self, small_model):
        small_model.freeze_trunk()
        assert small_model.trunk_is_frozen()
        # Head must still be trainable
        assert any(p.requires_grad for p in small_model.head.parameters())

    def test_unfreeze_trunk(self, small_model):
        small_model.freeze_trunk()
        small_model.unfreeze_trunk()
        assert not small_model.trunk_is_frozen()

    def test_expand_head_preserves_old_weights(self, small_model):
        old_weights = small_model.head.weight.data[:3].clone()
        small_model.expand_head(n_new_classes=2)
        assert small_model.head.out_features == 5
        # First 3 rows (old classes) must be preserved exactly
        assert torch.allclose(small_model.head.weight.data[:3], old_weights), \
            "expand_head() must preserve existing class weights"

    def test_expand_head_new_weights_small(self, small_model):
        small_model.expand_head(n_new_classes=2)
        new_weights = small_model.head.weight.data[3:]
        # New weights should be small (std=0.01 init)
        assert new_weights.abs().max().item() < 0.5

    def test_snapshot_restore(self, small_model):
        state = small_model.snapshot()
        # Mutate the model
        with torch.no_grad():
            for p in small_model.parameters():
                p.fill_(999.0)
        small_model.restore(state)
        # Check restored
        for p in small_model.parameters():
            assert not (p == 999.0).all(), "restore() did not recover weights"

    def test_count_params(self, small_model):
        total     = small_model.count_total_params()
        trainable = small_model.count_trainable_params()
        assert total > 0
        assert trainable == total  # nothing frozen initially

    def test_freeze_reduces_trainable_count(self, small_model):
        total_before = small_model.count_trainable_params()
        small_model.freeze_trunk()
        trainable_after = small_model.count_trainable_params()
        assert trainable_after < total_before


# ===========================================================================
# Data generator tests
# ===========================================================================

class TestGenerators:

    def test_new_data_scenario_shapes(self):
        s = NewDataScenario(n_initial=100, n_new=50, n_test=40, n_features=8, seed=42)
        X_tr, y_tr, X_te, y_te = s.phase1()
        assert X_tr.shape == (100, 8)
        assert y_tr.shape == (100,)
        assert X_te.shape == (40, 8)

    def test_new_data_combined_size(self):
        s = NewDataScenario(n_initial=100, n_new=50, n_features=8, seed=42)
        X_comb, y_comb, _, _ = s.combined()
        assert X_comb.shape[0] == 150

    def test_distribution_shift_scenario(self):
        s = DistributionShiftScenario(n_initial=100, n_new=50, n_features=8, seed=42)
        X_orig, y_orig, _, _ = s.original_distribution()
        X_shift, y_shift, _, _ = s.shifted_distribution()
        # Shifted mean should differ from original
        assert not torch.allclose(X_orig.mean(), X_shift.mean(), atol=0.1)

    def test_new_class_scenario_label_ranges(self):
        s = NewClassScenario(
            n_per_class_train=50, n_per_class_test=20,
            n_features=8, original_classes=3, new_classes=2, seed=42
        )
        X_orig, y_orig, _, _ = s.original_data()
        X_new, y_new, _, _   = s.new_class_data()
        assert y_orig.max().item() == 2       # classes 0,1,2
        assert y_new.min().item()  == 3       # classes 3,4

    def test_domain_transfer_scenario(self):
        s = DomainTransferScenario(
            n_source_train=200, n_target_train=50, n_test=40,
            n_features=8, n_classes=3, domain_shift=2.0, seed=42
        )
        X_src, y_src, _, _ = s.source_data()
        X_tgt, y_tgt, _, _ = s.target_data()
        # n_samples may be floor(n // n_classes) * n_classes — allow small rounding
        assert X_src.shape[0] >= 198
        assert X_tgt.shape[0] >= 48
        # Domain offset should make means differ
        assert abs(X_src.mean().item() - X_tgt.mean().item()) > 0.5


# ===========================================================================
# Strategy tests
# ===========================================================================

class TestTrainFromScratch:

    def test_reinitialises_weights(self, small_model, tiny_loader):
        # Fill weights with known values
        with torch.no_grad():
            for p in small_model.parameters():
                p.fill_(5.0)
        original_sum = sum(p.sum().item() for p in small_model.parameters())

        trainer = TrainFromScratch(small_model, epochs=1)
        trainer.train(tiny_loader)

        new_sum = sum(p.sum().item() for p in small_model.parameters())
        assert abs(new_sum - original_sum) > 1.0, \
            "Scratch training should change weights from the filled-5.0 state"

    def test_returns_result(self, small_model, tiny_loader):
        trainer = TrainFromScratch(small_model, epochs=2)
        result  = trainer.train(tiny_loader)
        assert result.epochs_trained == 2
        assert result.runtime_s > 0
        assert result.params_trained > 0


class TestFineTuner:

    def test_head_only_trunk_unchanged(self, small_model, tiny_loader):
        # Train champion first
        trainer = TrainFromScratch(small_model, epochs=3)
        trainer.train(tiny_loader)
        trunk_before = copy.deepcopy(
            list(small_model.trunk.parameters())[0].data.clone()
        )

        tuner = FineTuner(small_model, mode=FineTuneMode.HEAD_ONLY, epochs=3)
        tuner.fine_tune(tiny_loader)

        trunk_after = list(small_model.trunk.parameters())[0].data
        assert torch.allclose(trunk_before, trunk_after), \
            "HEAD_ONLY mode must not update trunk weights"

    def test_full_network_trunk_changes(self, small_model, tiny_loader):
        trainer = TrainFromScratch(small_model, epochs=3)
        trainer.train(tiny_loader)
        trunk_before = list(small_model.trunk.parameters())[0].data.clone()

        tuner = FineTuner(small_model, mode=FineTuneMode.FULL_NETWORK, epochs=3, lr=0.01)
        tuner.fine_tune(tiny_loader)

        trunk_after = list(small_model.trunk.parameters())[0].data
        assert not torch.allclose(trunk_before, trunk_after), \
            "FULL_NETWORK mode must update trunk weights"

    def test_trunk_unfrozen_after_fine_tune(self, small_model, tiny_loader):
        """After fine-tuning, trunk should be in unfrozen state for clean handoff."""
        tuner = FineTuner(small_model, mode=FineTuneMode.HEAD_ONLY, epochs=2)
        tuner.fine_tune(tiny_loader)
        assert not small_model.trunk_is_frozen(), \
            "Trunk should be unfrozen after fine_tune() returns"

    def test_fine_tune_result_fields(self, small_model, tiny_loader):
        tuner  = FineTuner(small_model, mode=FineTuneMode.HEAD_ONLY, epochs=2)
        result = tuner.fine_tune(tiny_loader)
        assert result.epochs_trained == 2
        assert result.trunk_was_frozen is True
        assert result.runtime_s > 0


class TestRetrainer:

    def test_warm_new_only(self, small_model, tiny_loader):
        trainer = TrainFromScratch(small_model, epochs=3)
        trainer.train(tiny_loader)
        weights_before = list(small_model.parameters())[0].data.clone()

        rt = Retrainer(small_model, mode=RetrainMode.WARM_NEW_ONLY, epochs=3, lr=0.01)
        r  = rt.retrain(tiny_loader)

        weights_after = list(small_model.parameters())[0].data
        assert not torch.allclose(weights_before, weights_after), \
            "Warm-new-only retraining must update weights"
        assert r.warm_start is True

    def test_cold_combined_reinitialises(self, small_model, tiny_loader):
        with torch.no_grad():
            for p in small_model.parameters():
                p.fill_(7.0)

        rt = Retrainer(small_model, mode=RetrainMode.COLD_COMBINED, epochs=2, lr=0.01)
        rt.retrain(tiny_loader, old_loader=tiny_loader)

        # Weights should no longer be 7.0 after cold re-init + training
        all_sevens = all((p == 7.0).all().item() for p in small_model.parameters())
        assert not all_sevens

    def test_warm_combined_returns_result(self, small_model, tiny_loader):
        trainer = TrainFromScratch(small_model, epochs=3)
        trainer.train(tiny_loader)

        rt = Retrainer(small_model, mode=RetrainMode.WARM_COMBINED, epochs=3)
        r  = rt.retrain(tiny_loader, old_loader=tiny_loader)
        assert r.mode == "warm_combined"
        assert r.runtime_s > 0


# ===========================================================================
# Evaluation metric tests
# ===========================================================================

class TestMetrics:

    def test_accuracy_perfect(self, small_model, tiny_eval_loader):
        # Intercept predictions to force perfect accuracy
        X = torch.randn(30, 10)
        y = torch.zeros(30, dtype=torch.long)
        loader = make_eval_loader(X, y)

        # Make model always predict class 0
        with torch.no_grad():
            small_model.head.weight.fill_(0.0)
            small_model.head.bias.fill_(0.0)
            small_model.head.bias[0] = 100.0  # class 0 dominates

        acc = compute_accuracy(small_model, loader)
        assert acc == pytest.approx(1.0, abs=1e-4)

    def test_accuracy_range(self, small_model, tiny_eval_loader):
        acc = compute_accuracy(small_model, tiny_eval_loader)
        assert 0.0 <= acc <= 1.0

    def test_macro_f1_range(self, small_model, tiny_eval_loader):
        f1 = compute_macro_f1(small_model, tiny_eval_loader, n_classes=3)
        assert 0.0 <= f1 <= 1.0

    def test_ece_range(self, small_model, tiny_eval_loader):
        ece = compute_ece(small_model, tiny_eval_loader)
        assert 0.0 <= ece <= 1.0

    def test_cost_tracker(self):
        tracker = CostTracker()
        tracker.record("strategy_a", runtime_s=2.0, params_updated=1000, data_samples=500)
        tracker.record("strategy_b", runtime_s=5.0, params_updated=50000, data_samples=500)
        summary = tracker.summary()
        assert "strategy_a" in summary
        assert "strategy_b" in summary
        assert "7.00" in summary   # total runtime


# ===========================================================================
# Decision engine tests
# ===========================================================================

class TestDecisionEngine:

    def setup_method(self):
        self.engine = DecisionEngine()

    def _decide(self, **kwargs):
        data        = kwargs.get("data",        DataSignals(n_new_samples=500))
        model       = kwargs.get("model",       ModelSignals(transfer_quality=0.75))
        drift       = kwargs.get("drift",       DriftSignals())
        constraints = kwargs.get("constraints", ConstraintSignals())
        return self.engine.decide(data, model, drift, constraints)

    def test_few_shot_good_transfer_recommends_head_only(self):
        result = self._decide(
            data  = DataSignals(n_new_samples=200),
            model = ModelSignals(transfer_quality=0.80),
            drift = DriftSignals(drift_detected=False, drift_severity="none"),
        )
        assert result.primary.strategy == Strategy.FINE_TUNE_HEAD_ONLY
        assert result.primary.confidence == "high"

    def test_abundant_data_poor_transfer_recommends_scratch(self):
        result = self._decide(
            data  = DataSignals(n_new_samples=10_000),
            model = ModelSignals(transfer_quality=0.40),
            drift = DriftSignals(drift_detected=True, drift_severity="severe", new_task=True),
        )
        assert result.primary.strategy == Strategy.TRAIN_FROM_SCRATCH

    def test_drift_with_history_recommends_warm_combined(self):
        result = self._decide(
            data  = DataSignals(n_new_samples=1_000, n_historical_samples=2_000),
            model = ModelSignals(transfer_quality=0.65),
            drift = DriftSignals(drift_detected=True, drift_severity="moderate"),
        )
        assert result.primary.strategy == Strategy.RETRAIN_WARM_COMBINED

    def test_no_historical_data_eliminates_combined_strategies(self):
        result = self._decide(
            data        = DataSignals(n_new_samples=500, n_historical_samples=0),
            constraints = ConstraintSignals(can_store_historical=False),
        )
        ruled_out_strategies = {s for s, _ in result.ruled_out}
        assert Strategy.RETRAIN_WARM_COMBINED in ruled_out_strategies
        assert Strategy.RETRAIN_COLD_COMBINED in ruled_out_strategies

    def test_forgetting_forbidden_eliminates_risky_strategies(self):
        result = self._decide(
            constraints = ConstraintSignals(catastrophic_forgetting_forbidden=True),
        )
        ruled_out_strategies = {s for s, _ in result.ruled_out}
        assert Strategy.FINE_TUNE_FULL_NETWORK in ruled_out_strategies
        assert Strategy.RETRAIN_WARM_NEW_ONLY in ruled_out_strategies

    def test_result_always_has_primary(self):
        result = self._decide()
        assert result.primary is not None
        assert result.primary.strategy in list(Strategy)

    def test_result_has_rationale(self):
        result = self._decide()
        assert len(result.primary.rationale) > 10


# ===========================================================================
# Cost model tests
# ===========================================================================

class TestCostModel:

    def test_head_only_is_cheapest(self):
        model = CostModel(
            n_new_samples=500, n_historical_samples=1000,
            n_params_total=100_000, n_params_head=200, epochs=10
        )
        estimates = model.estimate_all()
        head_only = next(e for e in estimates if e.strategy == "fine_tune_head_only")
        scratch   = next(e for e in estimates if e.strategy == "train_from_scratch")
        assert head_only.compute_factor <= scratch.compute_factor

    def test_combined_requires_more_data(self):
        model = CostModel(
            n_new_samples=500, n_historical_samples=1000,
            n_params_total=100_000, n_params_head=200, epochs=10
        )
        estimates = model.estimate_all()
        head_only = next(e for e in estimates if e.strategy == "fine_tune_head_only")
        combined  = next(e for e in estimates if e.strategy == "retrain_warm_combined")
        assert combined.data_samples_required > head_only.data_samples_required

    def test_comparison_table_renders(self):
        model = CostModel(
            n_new_samples=500, n_historical_samples=1000,
            n_params_total=100_000, n_params_head=200, epochs=10
        )
        table = model.comparison_table(model.estimate_all())
        assert "fine_tune_head_only" in table
        assert "train_from_scratch"  in table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
