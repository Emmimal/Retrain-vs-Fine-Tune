"""
demo.py
-------
Single-file demo for Article 08: Retrain vs Fine-Tune vs Train from Scratch.

Runs a condensed end-to-end example covering all three strategies across
a simple new-data scenario.  Run this first to verify your installation.

Usage
-----
python demo.py

Then run the full benchmark:
python benchmarks/benchmark.py

Series: Production ML Engineering — Article 08 of 15
https://emitechlogic.com/
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.generators import NewDataScenario, DomainTransferScenario
from data.loaders import make_loader, make_eval_loader
from evaluation.decision_framework import (
    DataSignals, ModelSignals, DriftSignals, ConstraintSignals, DecisionEngine,
)
from evaluation.metrics import compute_accuracy, compute_macro_f1, compute_ece
from experiments.cost_model import CostModel
from models.base_model import MLP
from strategies.fine_tune import FineTuner, FineTuneMode
from strategies.retrain import Retrainer, RetrainMode
from strategies.train_from_scratch import TrainFromScratch
from utils.reproducibility import set_global_seed


def _row(name, acc, f1, ece, runtime):
    return f"  {name:<32} acc={acc:.4f}  f1={f1:.4f}  ece={ece:.4f}  {runtime:.2f}s"


def main():
    set_global_seed(42)

    print("=" * 68)
    print("  DEMO: Retrain vs Fine-Tune vs Train from Scratch")
    print("  Production ML Engineering — Article 08 of 15")
    print("  https://emitechlogic.com/")
    print("=" * 68)

    # ------------------------------------------------------------------
    # 1. Build dataset
    # ------------------------------------------------------------------
    print("\n[1/4] Generating dataset (Scenario A: new data, same distribution)")
    s = NewDataScenario(n_initial=800, n_new=400, n_test=300, n_features=20, seed=42)
    X_tr1, y_tr1, X_te, y_te = s.phase1()
    X_new, y_new, _, _        = s.phase2()
    X_comb, y_comb, _, _      = s.combined()

    test_loader = make_eval_loader(X_te, y_te)
    new_loader  = make_loader(X_new, y_new, batch_size=64, seed=42)
    old_loader  = make_loader(X_tr1, y_tr1, batch_size=64, seed=42)
    comb_loader = make_loader(X_comb, y_comb, batch_size=64, seed=42)

    # ------------------------------------------------------------------
    # 2. Train champion
    # ------------------------------------------------------------------
    print("[2/4] Training champion model on initial data...")
    champion = MLP(input_dim=20, hidden_dims=[128, 64], output_dim=2)
    print(f"      {champion}")

    trainer = TrainFromScratch(champion, epochs=15, lr=0.01)
    trainer.train(make_loader(X_tr1, y_tr1, batch_size=64, seed=42))

    champ_acc = compute_accuracy(champion, test_loader)
    champ_f1  = compute_macro_f1(champion, test_loader, n_classes=2)
    print(f"      Champion — acc={champ_acc:.4f}  f1={champ_f1:.4f}")

    # ------------------------------------------------------------------
    # 3. Apply all strategies and compare
    # ------------------------------------------------------------------
    print("\n[3/4] Applying update strategies...")
    print(f"\n  {'Strategy':<32} {'Acc':>8}  {'F1':>8}  {'ECE':>8}  {'Time':>6}")
    print("  " + "-" * 64)

    # Champion (no update)
    acc = compute_accuracy(champion, test_loader)
    f1  = compute_macro_f1(champion, test_loader, n_classes=2)
    ece = compute_ece(champion, test_loader)
    print(_row("Champion (no update)", acc, f1, ece, 0.0))

    # Fine-tune head only
    m = copy.deepcopy(champion)
    t = FineTuner(m, mode=FineTuneMode.HEAD_ONLY, epochs=10, lr=0.01)
    r = t.fine_tune(new_loader)
    print(_row("FineTune head-only",
               compute_accuracy(m, test_loader),
               compute_macro_f1(m, test_loader, n_classes=2),
               compute_ece(m, test_loader),
               r.runtime_s))

    # Fine-tune full network
    m = copy.deepcopy(champion)
    t = FineTuner(m, mode=FineTuneMode.FULL_NETWORK, epochs=10, lr=0.001)
    r = t.fine_tune(new_loader)
    print(_row("FineTune full network",
               compute_accuracy(m, test_loader),
               compute_macro_f1(m, test_loader, n_classes=2),
               compute_ece(m, test_loader),
               r.runtime_s))

    # Retrain warm combined
    m  = copy.deepcopy(champion)
    rt = Retrainer(m, mode=RetrainMode.WARM_COMBINED, epochs=10, lr=0.005)
    r  = rt.retrain(new_loader, old_loader=old_loader)
    print(_row("Retrain warm combined",
               compute_accuracy(m, test_loader),
               compute_macro_f1(m, test_loader, n_classes=2),
               compute_ece(m, test_loader),
               r.runtime_s))

    # Train from scratch
    m  = MLP(input_dim=20, hidden_dims=[128, 64], output_dim=2)
    tr = TrainFromScratch(m, epochs=15, lr=0.01)
    r  = tr.train(comb_loader)
    print(_row("Scratch (combined)",
               compute_accuracy(m, test_loader),
               compute_macro_f1(m, test_loader, n_classes=2),
               compute_ece(m, test_loader),
               r.runtime_s))

    # ------------------------------------------------------------------
    # 4. Decision engine demo
    # ------------------------------------------------------------------
    print("\n[4/4] Decision Engine — strategy recommendation for this scenario:")
    engine = DecisionEngine()
    result = engine.decide(
        data        = DataSignals(n_new_samples=400, n_historical_samples=800),
        model       = ModelSignals(
            current_accuracy     = champ_acc,
            accuracy_on_new_data = champ_acc,
            transfer_quality     = 0.75,
        ),
        drift       = DriftSignals(drift_detected=False, drift_severity="none"),
        constraints = ConstraintSignals(can_store_historical=True),
    )
    print(result.summary())

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------
    print("\nCost model (relative compute estimates):")
    cost = CostModel(
        n_new_samples=400, n_historical_samples=800,
        n_params_total=champion.count_total_params(),
        n_params_head=champion.head.out_features * champion._trunk_out_dim,
        epochs=10,
    )
    print(cost.comparison_table(cost.estimate_all()))

    print("\nDemo complete.")
    print("Run the full benchmark: python benchmarks/benchmark.py")
    print("Run tests:             python -m pytest tests/test_all.py -v")


if __name__ == "__main__":
    main()
