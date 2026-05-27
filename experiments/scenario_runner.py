"""
experiments/scenario_runner.py
--------------------------------
Runs all four experimental scenarios and prints structured results.

Each scenario simulates a real production trigger for a strategy decision:
  A — New data, same distribution  →  retrain_warm_combined vs fine_tune vs scratch
  B — Distribution shift           →  how each strategy handles drift
  C — New classes                  →  head expansion + fine-tuning
  D — Domain transfer              →  how few target-domain samples justify fine-tuning

All results are structured as StrategyMetrics records for direct comparison.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch

from data.generators import (
    NewDataScenario,
    DistributionShiftScenario,
    NewClassScenario,
    DomainTransferScenario,
)
from data.loaders import make_loader, make_eval_loader
from evaluation.metrics import evaluate, CostTracker
from models.base_model import MLP
from strategies.fine_tune import FineTuner, FineTuneMode
from strategies.retrain import Retrainer, RetrainMode
from strategies.train_from_scratch import TrainFromScratch


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_model(input_dim: int, output_dim: int) -> MLP:
    return MLP(input_dim=input_dim, hidden_dims=[128, 64], output_dim=output_dim)


def _train_champion(
    model        : MLP,
    X_tr         : torch.Tensor,
    y_tr         : torch.Tensor,
    epochs       : int   = 15,
    lr           : float = 0.01,
    verbose      : bool  = False,
) -> None:
    """Train a model to serve as the champion baseline."""
    loader  = make_loader(X_tr, y_tr, batch_size=64, seed=42)
    trainer = TrainFromScratch(model, epochs=epochs, lr=lr, verbose=verbose)
    trainer.train(loader)


# ---------------------------------------------------------------------------
# Scenario A — New Data, Same Distribution
# ---------------------------------------------------------------------------

def run_scenario_a(seed: int = 42, verbose: bool = True) -> dict:
    """
    New labelled data arrives from the same distribution.

    Champion was trained on n_initial samples.
    Update adds n_new samples from the identical DGP.

    Compared strategies:
      - Fine-tune head only on new data
      - Fine-tune full network on new data
      - Retrain warm on new data only
      - Retrain warm on combined data   ← expected winner
      - Train from scratch on new data
      - Train from scratch on combined data
    """
    _set_seed(seed)

    scenario = NewDataScenario(
        n_initial  = 1_000,
        n_new      = 500,
        n_test     = 400,
        n_features = 20,
        seed       = seed,
    )

    X_tr1, y_tr1, X_te, y_te = scenario.phase1()
    X_new, y_new, _,    _    = scenario.phase2()
    X_comb, y_comb, _,  _    = scenario.combined()

    test_loader = make_eval_loader(X_te, y_te)
    new_loader  = make_loader(X_new, y_new, batch_size=64, seed=seed)
    old_loader  = make_loader(X_tr1, y_tr1, batch_size=64, seed=seed)
    comb_loader = make_loader(X_comb, y_comb, batch_size=64, seed=seed)

    # Train champion
    champion = _make_model(input_dim=20, output_dim=2)
    _train_champion(champion, X_tr1, y_tr1, epochs=15)
    champion_acc = float(torch.no_grad()(lambda: None) or True) and \
                   evaluate(champion, test_loader, 2, "champion", "A").accuracy

    results = {}

    # ---- Fine-tune head only
    model = copy.deepcopy(champion)
    tuner = FineTuner(model, mode=FineTuneMode.HEAD_ONLY, epochs=10, lr=0.01, verbose=verbose)
    r     = tuner.fine_tune(new_loader)
    results["fine_tune_head_only"] = evaluate(
        model, test_loader, 2, "fine_tune_head_only", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    # ---- Fine-tune full network
    model = copy.deepcopy(champion)
    tuner = FineTuner(model, mode=FineTuneMode.FULL_NETWORK, epochs=10, lr=0.001, verbose=verbose)
    r     = tuner.fine_tune(new_loader)
    results["fine_tune_full_network"] = evaluate(
        model, test_loader, 2, "fine_tune_full_network", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    # ---- Retrain warm new only
    model     = copy.deepcopy(champion)
    retrainer = Retrainer(model, mode=RetrainMode.WARM_NEW_ONLY, epochs=10, lr=0.005, verbose=verbose)
    r         = retrainer.retrain(new_loader)
    results["retrain_warm_new_only"] = evaluate(
        model, test_loader, 2, "retrain_warm_new_only", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    # ---- Retrain warm combined
    model     = copy.deepcopy(champion)
    retrainer = Retrainer(model, mode=RetrainMode.WARM_COMBINED, epochs=10, lr=0.005, verbose=verbose)
    r         = retrainer.retrain(new_loader, old_loader=old_loader)
    results["retrain_warm_combined"] = evaluate(
        model, test_loader, 2, "retrain_warm_combined", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    # ---- Train from scratch (new data only)
    model   = copy.deepcopy(champion)
    trainer = TrainFromScratch(model, epochs=15, lr=0.01, verbose=verbose)
    r       = trainer.train(new_loader)
    results["scratch_new_only"] = evaluate(
        model, test_loader, 2, "scratch_new_only", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    # ---- Train from scratch (combined data)
    model   = copy.deepcopy(champion)
    trainer = TrainFromScratch(model, epochs=15, lr=0.01, verbose=verbose)
    r       = trainer.train(comb_loader)
    results["scratch_combined"] = evaluate(
        model, test_loader, 2, "scratch_combined", "A",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    return results


# ---------------------------------------------------------------------------
# Scenario B — Distribution Shift
# ---------------------------------------------------------------------------

def run_scenario_b(seed: int = 42, verbose: bool = True) -> dict:
    """
    The feature distribution has shifted.  The champion degrades on new data.

    Tests whether warm-start strategies help or hurt when the distribution
    moves significantly from the original training window.
    """
    _set_seed(seed)

    scenario = DistributionShiftScenario(
        n_initial       = 1_000,
        n_new           = 500,
        n_test          = 400,
        n_features      = 20,
        shift_magnitude = 1.5,
        seed            = seed,
    )

    X_orig, y_orig, _, _        = scenario.original_distribution()
    X_new,  y_new,  X_te, y_te  = scenario.shifted_distribution()
    X_mix,  y_mix,  _,    _     = scenario.mixed_distribution(old_fraction=0.5)

    test_loader = make_eval_loader(X_te, y_te)
    new_loader  = make_loader(X_new, y_new, batch_size=64, seed=seed)
    old_loader  = make_loader(X_orig, y_orig, batch_size=64, seed=seed)
    mix_loader  = make_loader(X_mix,  y_mix,  batch_size=64, seed=seed)

    champion = _make_model(input_dim=20, output_dim=2)
    _train_champion(champion, X_orig, y_orig, epochs=15)

    results = {}

    # ---- Fine-tune full network on new (shifted) data
    model = copy.deepcopy(champion)
    tuner = FineTuner(model, mode=FineTuneMode.FULL_NETWORK, epochs=10, lr=0.001, verbose=verbose)
    r     = tuner.fine_tune(new_loader)
    results["fine_tune_full"] = evaluate(
        model, test_loader, 2, "fine_tune_full", "B",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    # ---- Fine-tune head only
    model = copy.deepcopy(champion)
    tuner = FineTuner(model, mode=FineTuneMode.HEAD_ONLY, epochs=10, lr=0.01, verbose=verbose)
    r     = tuner.fine_tune(new_loader)
    results["fine_tune_head"] = evaluate(
        model, test_loader, 2, "fine_tune_head", "B",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    # ---- Retrain warm combined
    model     = copy.deepcopy(champion)
    retrainer = Retrainer(model, mode=RetrainMode.WARM_COMBINED, epochs=10, lr=0.005, verbose=verbose)
    r         = retrainer.retrain(new_loader, old_loader=old_loader)
    results["retrain_warm_combined"] = evaluate(
        model, test_loader, 2, "retrain_warm_combined", "B",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    # ---- Train from scratch (new data)
    model   = copy.deepcopy(champion)
    trainer = TrainFromScratch(model, epochs=15, lr=0.01, verbose=verbose)
    r       = trainer.train(new_loader)
    results["scratch"] = evaluate(
        model, test_loader, 2, "scratch", "B",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
    )

    return results


# ---------------------------------------------------------------------------
# Scenario C — New Classes
# ---------------------------------------------------------------------------

def run_scenario_c(seed: int = 42, verbose: bool = True) -> dict:
    """
    New output classes arrive that the model has never seen.

    Tests head expansion + fine-tuning against full retraining from scratch
    when the class space grows.
    """
    _set_seed(seed)

    scenario = NewClassScenario(
        n_per_class_train = 300,
        n_per_class_test  = 100,
        n_features        = 20,
        original_classes  = 3,
        new_classes       = 2,
        seed              = seed,
    )

    X_orig, y_orig, X_te_orig, y_te_orig = scenario.original_data()
    X_new,  y_new,  X_te_new,  y_te_new  = scenario.new_class_data()
    X_te_all, y_te_all                    = scenario.full_test_set()

    orig_loader      = make_loader(X_orig, y_orig, batch_size=64, seed=seed)
    new_only_loader  = make_loader(X_new,  y_new,  batch_size=64, seed=seed)
    full_test_loader = make_eval_loader(X_te_all, y_te_all)
    old_test_loader  = make_eval_loader(X_te_orig, y_te_orig)

    # Champion trained on original 3 classes
    champion = _make_model(input_dim=20, output_dim=3)
    _train_champion(champion, X_orig, y_orig, epochs=15)
    champion_old_acc = evaluate(champion, old_test_loader, 3, "champion", "C").accuracy

    results = {}
    old_ids = list(range(3))
    new_ids = list(range(3, 5))

    # ---- Fine-tune with head expansion (head only)
    model = copy.deepcopy(champion)
    model.expand_head(n_new_classes=2)
    tuner = FineTuner(model, mode=FineTuneMode.HEAD_ONLY, epochs=12, lr=0.01, verbose=verbose)
    r     = tuner.fine_tune(new_only_loader)
    results["expand_head_fine_tune"] = evaluate(
        model, full_test_loader, 5, "expand_head_fine_tune", "C",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
        champion_old_acc=champion_old_acc,
        old_class_ids=old_ids, new_class_ids=new_ids,
    )

    # ---- Fine-tune full network after head expansion
    model = copy.deepcopy(champion)
    model.expand_head(n_new_classes=2)
    # Combine old + new for full fine-tuning
    comb_X = torch.cat([X_orig, X_new])
    comb_y = torch.cat([y_orig, y_new])
    comb_loader = make_loader(comb_X, comb_y, batch_size=64, seed=seed)
    tuner = FineTuner(model, mode=FineTuneMode.FULL_NETWORK, epochs=12, lr=0.001, verbose=verbose)
    r     = tuner.fine_tune(comb_loader)
    results["expand_full_finetune"] = evaluate(
        model, full_test_loader, 5, "expand_full_finetune", "C",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
        champion_old_acc=champion_old_acc,
        old_class_ids=old_ids, new_class_ids=new_ids,
    )

    # ---- Train from scratch on all classes
    all_X = torch.cat([X_orig, X_new])
    all_y = torch.cat([y_orig, y_new])
    all_loader = make_loader(all_X, all_y, batch_size=64, seed=seed)
    model   = _make_model(input_dim=20, output_dim=5)
    trainer = TrainFromScratch(model, epochs=20, lr=0.01, verbose=verbose)
    r       = trainer.train(all_loader)
    results["scratch_all_classes"] = evaluate(
        model, full_test_loader, 5, "scratch_all_classes", "C",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
        old_class_ids=old_ids, new_class_ids=new_ids,
    )

    return results


# ---------------------------------------------------------------------------
# Scenario D — Domain Transfer
# ---------------------------------------------------------------------------

def run_scenario_d(seed: int = 42, verbose: bool = True) -> dict:
    """
    Transfer from a data-rich source domain to a data-scarce target domain.

    Key question: at what n_target sample count does fine-tuning become
    better than training from scratch on target data alone?
    """
    _set_seed(seed)

    scenario = DomainTransferScenario(
        n_source_train = 2_000,
        n_target_train = 200,
        n_test         = 400,
        n_features     = 20,
        n_classes      = 3,
        domain_shift   = 2.0,
        seed           = seed,
    )

    X_src, y_src, _, _          = scenario.source_data()
    X_tgt, y_tgt, X_te, y_te   = scenario.target_data()

    target_loader = make_loader(X_tgt, y_tgt, batch_size=32, seed=seed)
    test_loader   = make_eval_loader(X_te, y_te)

    # Pre-trained on source domain
    source_model = _make_model(input_dim=20, output_dim=3)
    _train_champion(source_model, X_src, y_src, epochs=15)

    results = {}

    # ---- Fine-tune head only (frozen trunk from source domain)
    model = copy.deepcopy(source_model)
    tuner = FineTuner(model, mode=FineTuneMode.HEAD_ONLY, epochs=15, lr=0.01, verbose=verbose)
    r     = tuner.fine_tune(target_loader)
    results["fine_tune_head_only"] = evaluate(
        model, test_loader, 3, "fine_tune_head_only", "D",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
        notes="200 target samples, frozen source trunk",
    )

    # ---- Fine-tune full network (all layers, lower LR)
    model = copy.deepcopy(source_model)
    tuner = FineTuner(model, mode=FineTuneMode.FULL_NETWORK, epochs=15, lr=0.001, verbose=verbose)
    r     = tuner.fine_tune(target_loader)
    results["fine_tune_full_network"] = evaluate(
        model, test_loader, 3, "fine_tune_full_network", "D",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
        notes="200 target samples, all layers trainable",
    )

    # ---- Train from scratch on target data only
    model   = _make_model(input_dim=20, output_dim=3)
    trainer = TrainFromScratch(model, epochs=20, lr=0.01, verbose=verbose)
    r       = trainer.train(target_loader)
    results["scratch_target_only"] = evaluate(
        model, test_loader, 3, "scratch_target_only", "D",
        runtime_s=r.runtime_s, params_trained=r.params_trained, epochs=r.epochs_trained,
        notes="200 target samples, no source knowledge",
    )

    return results


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_scenario_results(scenario_name: str, results: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  SCENARIO {scenario_name} RESULTS")
    print(f"{'=' * 70}")
    print(f"  {'Strategy':<35} {'Acc':>7} {'F1':>7} {'ECE':>7} {'Time':>7}")
    print(f"  {'-' * 65}")
    for name, m in results.items():
        old = f"  old={m.old_class_acc:.3f}" if m.old_class_acc > 0 else ""
        new = f"  new={m.new_class_acc:.3f}" if m.new_class_acc > 0 else ""
        print(
            f"  {name:<35} {m.accuracy:>7.4f} {m.macro_f1:>7.4f}"
            f" {m.ece:>7.4f} {m.runtime_s:>6.2f}s{old}{new}"
        )
    print(f"{'=' * 70}")


if __name__ == "__main__":
    print("\nRunning all four scenarios...\n")

    print("SCENARIO A — New Data, Same Distribution")
    res_a = run_scenario_a(verbose=False)
    _print_scenario_results("A", res_a)

    print("\nSCENARIO B — Distribution Shift")
    res_b = run_scenario_b(verbose=False)
    _print_scenario_results("B", res_b)

    print("\nSCENARIO C — New Classes")
    res_c = run_scenario_c(verbose=False)
    _print_scenario_results("C", res_c)

    print("\nSCENARIO D — Domain Transfer")
    res_d = run_scenario_d(verbose=False)
    _print_scenario_results("D", res_d)
