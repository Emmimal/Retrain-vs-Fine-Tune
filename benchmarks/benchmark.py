"""
benchmarks/benchmark.py
------------------------
Head-to-head benchmark across all four scenarios.

Runs every strategy against every scenario and prints a structured
comparison table matching the article's benchmark output format.

All benchmark numbers in the article come from this file.
Seed: 42 | Architecture: MLP [128, 64] | All runs on CPU

Usage
-----
python benchmarks/benchmark.py
"""

from __future__ import annotations

import copy
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generators import (
    NewDataScenario,
    DistributionShiftScenario,
    NewClassScenario,
    DomainTransferScenario,
)
from data.loaders import make_loader, make_eval_loader
from evaluation.metrics import evaluate, StrategyMetrics
from experiments.cost_model import CostModel
from models.base_model import MLP
from strategies.fine_tune import FineTuner, FineTuneMode
from strategies.retrain import Retrainer, RetrainMode
from strategies.train_from_scratch import TrainFromScratch


SEED = 42


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_model(input_dim: int, output_dim: int) -> MLP:
    _set_seed(SEED)
    return MLP(input_dim=input_dim, hidden_dims=[128, 64], output_dim=output_dim)


def _train_champion(model: MLP, X: torch.Tensor, y: torch.Tensor, epochs: int = 15) -> None:
    _set_seed(SEED)
    loader  = make_loader(X, y, batch_size=64, seed=SEED)
    trainer = TrainFromScratch(model, epochs=epochs, lr=0.01, verbose=False)
    trainer.train(loader)


def _row(name: str, m: StrategyMetrics) -> str:
    old = f"  old={m.old_class_acc:.3f}" if m.old_class_acc > 0 else ""
    new = f"  new={m.new_class_acc:.3f}" if m.new_class_acc > 0 else ""
    return (
        f"  {name:<36} {m.accuracy:>7.4f} {m.macro_f1:>7.4f}"
        f" {m.ece:>7.4f} {m.runtime_s:>6.2f}s{old}{new}"
    )


# ===========================================================================
# Scenario A — New Data, Same Distribution
# ===========================================================================

def benchmark_scenario_a() -> None:
    _set_seed(SEED)
    s = NewDataScenario(n_initial=1_000, n_new=500, n_test=400, n_features=20, seed=SEED)

    X_tr1, y_tr1, X_te, y_te = s.phase1()
    X_new, y_new, _, _        = s.phase2()
    X_comb, y_comb, _, _      = s.combined()

    test_loader = make_eval_loader(X_te, y_te)
    new_loader  = make_loader(X_new, y_new, batch_size=64, seed=SEED)
    old_loader  = make_loader(X_tr1, y_tr1, batch_size=64, seed=SEED)
    comb_loader = make_loader(X_comb, y_comb, batch_size=64, seed=SEED)

    champion = _make_model(20, 2)
    _train_champion(champion, X_tr1, y_tr1)

    print("\n" + "=" * 70)
    print("  SCENARIO A — New Data, Same Distribution")
    print("  Champion trained on 1,000 samples | Update: +500 same-dist samples")
    print("  Test set: 400 samples from same distribution")
    print("=" * 70)
    print(f"  {'Strategy':<36} {'Acc':>7} {'F1':>7} {'ECE':>7} {'Time':>7}")
    print("  " + "-" * 65)

    rows = []

    m = copy.deepcopy(champion)
    rows.append(("Champion (no update)", evaluate(m, test_loader, 2, "champion", "A")))

    m = copy.deepcopy(champion)
    t = FineTuner(m, mode=FineTuneMode.HEAD_ONLY, epochs=10, lr=0.01)
    r = t.fine_tune(new_loader)
    rows.append(("FineTune head-only", evaluate(m, test_loader, 2, "ft_head", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m = copy.deepcopy(champion)
    t = FineTuner(m, mode=FineTuneMode.FULL_NETWORK, epochs=10, lr=0.001)
    r = t.fine_tune(new_loader)
    rows.append(("FineTune full network", evaluate(m, test_loader, 2, "ft_full", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m = copy.deepcopy(champion)
    rt = Retrainer(m, mode=RetrainMode.WARM_NEW_ONLY, epochs=10, lr=0.005)
    r  = rt.retrain(new_loader)
    rows.append(("Retrain warm new-only", evaluate(m, test_loader, 2, "rw_new", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m = copy.deepcopy(champion)
    rt = Retrainer(m, mode=RetrainMode.WARM_COMBINED, epochs=10, lr=0.005)
    r  = rt.retrain(new_loader, old_loader=old_loader)
    rows.append(("Retrain warm combined", evaluate(m, test_loader, 2, "rw_comb", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m = copy.deepcopy(champion)
    rt = Retrainer(m, mode=RetrainMode.COLD_COMBINED, epochs=15, lr=0.01)
    r  = rt.retrain(new_loader, old_loader=old_loader)
    rows.append(("Retrain cold combined", evaluate(m, test_loader, 2, "rc_comb", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m = _make_model(20, 2)
    tr = TrainFromScratch(m, epochs=15, lr=0.01)
    r  = tr.train(comb_loader)
    rows.append(("Scratch (combined)", evaluate(m, test_loader, 2, "scratch_comb", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    for name, metrics in rows:
        print(_row(name, metrics))
    print("=" * 70)


# ===========================================================================
# Scenario B — Distribution Shift
# ===========================================================================

def benchmark_scenario_b() -> None:
    _set_seed(SEED)
    s = DistributionShiftScenario(
        n_initial=1_000, n_new=500, n_test=400,
        n_features=20, shift_magnitude=1.5, seed=SEED
    )

    X_orig, y_orig, _, _       = s.original_distribution()
    X_new,  y_new,  X_te, y_te = s.shifted_distribution()

    test_loader = make_eval_loader(X_te, y_te)
    new_loader  = make_loader(X_new, y_new, batch_size=64, seed=SEED)
    old_loader  = make_loader(X_orig, y_orig, batch_size=64, seed=SEED)

    champion = _make_model(20, 2)
    _train_champion(champion, X_orig, y_orig)

    print("\n" + "=" * 70)
    print("  SCENARIO B — Distribution Shift (shift_magnitude=1.5)")
    print("  Champion on original distribution | Test on SHIFTED distribution")
    print("=" * 70)
    print(f"  {'Strategy':<36} {'Acc':>7} {'F1':>7} {'ECE':>7} {'Time':>7}")
    print("  " + "-" * 65)

    rows = []

    m = copy.deepcopy(champion)
    rows.append(("Champion (no update)", evaluate(m, test_loader, 2, "champion", "B")))

    m = copy.deepcopy(champion)
    t = FineTuner(m, mode=FineTuneMode.HEAD_ONLY, epochs=10, lr=0.01)
    r = t.fine_tune(new_loader)
    rows.append(("FineTune head-only", evaluate(m, test_loader, 2, "ft_head", "B",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m = copy.deepcopy(champion)
    t = FineTuner(m, mode=FineTuneMode.FULL_NETWORK, epochs=10, lr=0.001)
    r = t.fine_tune(new_loader)
    rows.append(("FineTune full network", evaluate(m, test_loader, 2, "ft_full", "B",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m = copy.deepcopy(champion)
    rt = Retrainer(m, mode=RetrainMode.WARM_COMBINED, epochs=10, lr=0.005)
    r  = rt.retrain(new_loader, old_loader=old_loader)
    rows.append(("Retrain warm combined", evaluate(m, test_loader, 2, "rw_comb", "B",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m = _make_model(20, 2)
    tr = TrainFromScratch(m, epochs=15, lr=0.01)
    r  = tr.train(new_loader)
    rows.append(("Scratch (shifted data)", evaluate(m, test_loader, 2, "scratch", "B",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    for name, metrics in rows:
        print(_row(name, metrics))
    print("=" * 70)


# ===========================================================================
# Scenario C — New Classes
# ===========================================================================

def benchmark_scenario_c() -> None:
    _set_seed(SEED)
    s = NewClassScenario(
        n_per_class_train=300, n_per_class_test=100,
        n_features=20, original_classes=3, new_classes=2, seed=SEED
    )

    X_orig, y_orig, X_te_orig, y_te_orig = s.original_data()
    X_new,  y_new,  _,         _         = s.new_class_data()
    X_te_all, y_te_all                    = s.full_test_set()

    old_loader       = make_loader(X_orig, y_orig, batch_size=64, seed=SEED)
    new_only_loader  = make_loader(X_new,  y_new,  batch_size=64, seed=SEED)
    comb_X = torch.cat([X_orig, X_new]); comb_y = torch.cat([y_orig, y_new])
    comb_loader      = make_loader(comb_X, comb_y, batch_size=64, seed=SEED)
    full_test_loader = make_eval_loader(X_te_all, y_te_all)
    old_test_loader  = make_eval_loader(X_te_orig, y_te_orig)

    champion = _make_model(20, 3)
    _train_champion(champion, X_orig, y_orig)
    champ_old_acc = evaluate(champion, old_test_loader, 3, "champ", "C").accuracy

    old_ids = list(range(3)); new_ids = list(range(3, 5))

    print("\n" + "=" * 80)
    print("  SCENARIO C — New Classes (3 original → 5 total)")
    print("  Champion on 3-class task | Test: all 5 classes | 300 samples/class")
    print("=" * 80)
    print(f"  {'Strategy':<36} {'Acc':>7} {'F1':>7} {'ECE':>7} {'Time':>7}  Old acc  New acc")
    print("  " + "-" * 75)

    rows = []

    # expand head only
    m = copy.deepcopy(champion); m.expand_head(2)
    t = FineTuner(m, mode=FineTuneMode.HEAD_ONLY, epochs=12, lr=0.01)
    r = t.fine_tune(new_only_loader)
    rows.append(("Expand + head-only FT", evaluate(m, full_test_loader, 5,
        "expand_head", "C", runtime_s=r.runtime_s, params_trained=r.params_trained,
        epochs=r.epochs_trained, champion_old_acc=champ_old_acc,
        old_class_ids=old_ids, new_class_ids=new_ids)))

    # expand + full fine-tune on combined
    m = copy.deepcopy(champion); m.expand_head(2)
    t = FineTuner(m, mode=FineTuneMode.FULL_NETWORK, epochs=12, lr=0.001)
    r = t.fine_tune(comb_loader)
    rows.append(("Expand + full FT (combined)", evaluate(m, full_test_loader, 5,
        "expand_full", "C", runtime_s=r.runtime_s, params_trained=r.params_trained,
        epochs=r.epochs_trained, champion_old_acc=champ_old_acc,
        old_class_ids=old_ids, new_class_ids=new_ids)))

    # retrain warm combined on 5-class — need to expand head first
    m = copy.deepcopy(champion); m.expand_head(2)
    rt = Retrainer(m, mode=RetrainMode.WARM_COMBINED, epochs=12, lr=0.005)
    r  = rt.retrain(new_only_loader, old_loader=old_loader)
    rows.append(("Retrain warm combined", evaluate(m, full_test_loader, 5,
        "rw_comb", "C", runtime_s=r.runtime_s, params_trained=r.params_trained,
        epochs=r.epochs_trained, champion_old_acc=champ_old_acc,
        old_class_ids=old_ids, new_class_ids=new_ids)))

    # scratch on all 5 classes
    m  = _make_model(20, 5)
    tr = TrainFromScratch(m, epochs=20, lr=0.01)
    r  = tr.train(comb_loader)
    rows.append(("Scratch (all 5 classes)", evaluate(m, full_test_loader, 5,
        "scratch_all", "C", runtime_s=r.runtime_s, params_trained=r.params_trained,
        epochs=r.epochs_trained, old_class_ids=old_ids, new_class_ids=new_ids)))

    for name, metrics in rows:
        old = f"  {metrics.old_class_acc:.3f}" if metrics.old_class_acc > 0 else "    —  "
        new = f"    {metrics.new_class_acc:.3f}" if metrics.new_class_acc > 0 else "      —"
        print(
            f"  {name:<36} {metrics.accuracy:>7.4f} {metrics.macro_f1:>7.4f}"
            f" {metrics.ece:>7.4f} {metrics.runtime_s:>6.2f}s {old} {new}"
        )
    print("=" * 80)


# ===========================================================================
# Scenario D — Domain Transfer
# ===========================================================================

def benchmark_scenario_d() -> None:
    _set_seed(SEED)
    s = DomainTransferScenario(
        n_source_train=2_000, n_target_train=200, n_test=400,
        n_features=20, n_classes=3, domain_shift=2.0, seed=SEED
    )

    X_src, y_src, _, _         = s.source_data()
    X_tgt, y_tgt, X_te, y_te  = s.target_data()

    target_loader = make_loader(X_tgt, y_tgt, batch_size=32, seed=SEED)
    test_loader   = make_eval_loader(X_te, y_te)

    source_model = _make_model(20, 3)
    _train_champion(source_model, X_src, y_src)

    print("\n" + "=" * 70)
    print("  SCENARIO D — Domain Transfer (few-shot target)")
    print("  Pre-trained on 2,000 source samples | 200 target samples only")
    print("  domain_shift=2.0 | 3 classes | Test on target distribution")
    print("=" * 70)
    print(f"  {'Strategy':<36} {'Acc':>7} {'F1':>7} {'ECE':>7} {'Time':>7}")
    print("  " + "-" * 65)

    rows = []

    m = copy.deepcopy(source_model)
    rows.append(("Source model (no FT)", evaluate(m, test_loader, 3, "no_ft", "D")))

    m = copy.deepcopy(source_model)
    t = FineTuner(m, mode=FineTuneMode.HEAD_ONLY, epochs=15, lr=0.01)
    r = t.fine_tune(target_loader)
    rows.append(("FineTune head-only", evaluate(m, test_loader, 3, "ft_head", "D",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m = copy.deepcopy(source_model)
    t = FineTuner(m, mode=FineTuneMode.FULL_NETWORK, epochs=15, lr=0.001)
    r = t.fine_tune(target_loader)
    rows.append(("FineTune full network", evaluate(m, test_loader, 3, "ft_full", "D",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    m  = _make_model(20, 3)
    tr = TrainFromScratch(m, epochs=20, lr=0.01)
    r  = tr.train(target_loader)
    rows.append(("Scratch (200 target)", evaluate(m, test_loader, 3, "scratch_tgt", "D",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained)))

    for name, metrics in rows:
        print(_row(name, metrics))
    print("=" * 70)


# ===========================================================================
# Cost model summary
# ===========================================================================

def print_cost_model() -> None:
    model = CostModel(
        n_new_samples        = 500,
        n_historical_samples = 1_000,
        n_params_total       = 267_000,
        n_params_head        = 512,
        epochs               = 10,
    )
    estimates = model.estimate_all()
    print("\n" + model.comparison_table(estimates))


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD BENCHMARK: Retrain vs Fine-Tune vs Train from Scratch")
    print(f"  Seed: {SEED} | Architecture: MLP [128, 64] | Device: CPU")
    print(f"  Series: Production ML Engineering — Article 08 of 15")
    print("=" * 70)

    t_total = time.perf_counter()

    benchmark_scenario_a()
    benchmark_scenario_b()
    benchmark_scenario_c()
    benchmark_scenario_d()
    print_cost_model()

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal benchmark runtime: {elapsed:.1f}s")
    print("\nACC  = Test-set accuracy (↑ better)")
    print("F1   = Macro F1 across all classes (↑ better)")
    print("ECE  = Expected Calibration Error (↓ better)")
    print("Time = Training time in seconds")
