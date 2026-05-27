"""
evaluation/metrics.py
----------------------
Evaluation metrics for the Retrain vs Fine-Tune vs Scratch benchmark.

Metrics reported per strategy per scenario:

  accuracy          — Standard classification accuracy on the test set
  per_class_f1      — Per-class F1 (critical for imbalanced datasets)
  macro_f1          — Macro-average F1 across all classes
  old_class_acc     — Accuracy on original classes (backward transfer proxy)
  new_class_acc     — Accuracy on new classes (forward transfer proxy)
  forgetting        — Drop in old_class_acc relative to the champion baseline
  ece               — Expected Calibration Error (how well probabilities
                      reflect true accuracy)
  runtime_s         — Training runtime (cost proxy)
  params_trained    — Number of updated parameters (cost proxy)

The CostTracker class accumulates compute estimates across an experiment so
the total cost of each strategy over a model's lifetime can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_accuracy(
    model       : torch.nn.Module,
    loader      : DataLoader,
    device      : str = "cpu",
) -> float:
    model.eval()
    correct = total = 0
    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        preds  = model(X).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += len(y)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_per_class_accuracy(
    model       : torch.nn.Module,
    loader      : DataLoader,
    n_classes   : int,
    device      : str = "cpu",
) -> Dict[int, float]:
    """Return per-class accuracy as {class_id: accuracy}."""
    model.eval()
    correct_per = [0] * n_classes
    total_per   = [0] * n_classes

    for X, y in loader:
        X, y  = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        for c in range(n_classes):
            mask           = y == c
            correct_per[c] += (preds[mask] == c).sum().item()
            total_per[c]   += mask.sum().item()

    return {
        c: correct_per[c] / max(total_per[c], 1)
        for c in range(n_classes)
    }


@torch.no_grad()
def compute_macro_f1(
    model       : torch.nn.Module,
    loader      : DataLoader,
    n_classes   : int,
    device      : str = "cpu",
) -> float:
    """Macro-average F1 across all classes."""
    model.eval()
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    for X, y in loader:
        X, y  = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        for c in range(n_classes):
            tp[c] += ((preds == c) & (y == c)).sum().item()
            fp[c] += ((preds == c) & (y != c)).sum().item()
            fn[c] += ((preds != c) & (y == c)).sum().item()

    f1s = []
    for c in range(n_classes):
        precision = tp[c] / max(tp[c] + fp[c], 1)
        recall    = tp[c] / max(tp[c] + fn[c], 1)
        denom     = precision + recall
        f1        = 2 * precision * recall / denom if denom > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s)


@torch.no_grad()
def compute_ece(
    model       : torch.nn.Module,
    loader      : DataLoader,
    n_bins      : int = 15,
    device      : str = "cpu",
) -> float:
    """
    Expected Calibration Error.

    Measures how well the model's predicted confidence aligns with its
    actual accuracy.  A perfectly calibrated model has ECE = 0.

    ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|
    where B_b is the set of predictions in confidence bin b.
    """
    model.eval()
    all_confidences = []
    all_correct     = []

    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        probs  = F.softmax(model(X), dim=1)
        conf, preds = probs.max(dim=1)
        correct     = preds == y
        all_confidences.append(conf.cpu())
        all_correct.append(correct.cpu())

    confidences = torch.cat(all_confidences)
    correct     = torch.cat(all_correct).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean().item()
        bin_conf = confidences[mask].mean().item()
        ece += (mask.sum().item() / n) * abs(bin_acc - bin_conf)

    return ece


# ---------------------------------------------------------------------------
# Combined result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StrategyMetrics:
    """
    Complete evaluation record for one strategy on one scenario.
    """
    strategy         : str
    scenario         : str
    accuracy         : float = 0.0
    macro_f1         : float = 0.0
    ece              : float = 0.0
    old_class_acc    : float = 0.0   # accuracy on original classes
    new_class_acc    : float = 0.0   # accuracy on new classes (Scenario C)
    forgetting       : float = 0.0   # drop from champion baseline
    runtime_s        : float = 0.0
    params_trained   : int   = 0
    epochs           : int   = 0
    notes            : str   = ""

    def summary(self) -> str:
        lines = [
            f"Strategy : {self.strategy}",
            f"Scenario : {self.scenario}",
            f"Accuracy : {self.accuracy:.4f}",
            f"Macro F1 : {self.macro_f1:.4f}",
            f"ECE      : {self.ece:.4f}",
        ]
        if self.old_class_acc > 0:
            lines.append(f"Old acc  : {self.old_class_acc:.4f}")
        if self.new_class_acc > 0:
            lines.append(f"New acc  : {self.new_class_acc:.4f}")
        if self.forgetting != 0:
            lines.append(f"Forget   : {self.forgetting:+.4f}")
        lines += [
            f"Runtime  : {self.runtime_s:.2f}s",
            f"Params   : {self.params_trained:,}",
        ]
        if self.notes:
            lines.append(f"Notes    : {self.notes}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator — runs all metrics on a single model/loader pair
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model           : torch.nn.Module,
    loader          : DataLoader,
    n_classes       : int,
    strategy        : str,
    scenario        : str,
    runtime_s       : float         = 0.0,
    params_trained  : int           = 0,
    epochs          : int           = 0,
    champion_old_acc: float         = 0.0,
    old_class_ids   : Optional[List[int]] = None,
    new_class_ids   : Optional[List[int]] = None,
    device          : str           = "cpu",
    notes           : str           = "",
) -> StrategyMetrics:
    """
    Run all metrics and return a StrategyMetrics record.

    champion_old_acc is the accuracy on old classes under the current champion
    model — used to compute forgetting = old_class_acc - champion_old_acc.
    """
    accuracy  = compute_accuracy(model, loader, device)
    macro_f1  = compute_macro_f1(model, loader, n_classes, device)
    ece       = compute_ece(model, loader, device=device)

    # Per-class accuracy for backward/forward transfer
    per_class = compute_per_class_accuracy(model, loader, n_classes, device)

    old_acc = 0.0
    new_acc = 0.0
    if old_class_ids:
        old_accs = [per_class[c] for c in old_class_ids if c in per_class]
        old_acc  = sum(old_accs) / len(old_accs) if old_accs else 0.0
    if new_class_ids:
        new_accs = [per_class[c] for c in new_class_ids if c in per_class]
        new_acc  = sum(new_accs) / len(new_accs) if new_accs else 0.0

    forgetting = (old_acc - champion_old_acc) if champion_old_acc > 0 else 0.0

    return StrategyMetrics(
        strategy       = strategy,
        scenario       = scenario,
        accuracy       = accuracy,
        macro_f1       = macro_f1,
        ece            = ece,
        old_class_acc  = old_acc,
        new_class_acc  = new_acc,
        forgetting     = forgetting,
        runtime_s      = runtime_s,
        params_trained = params_trained,
        epochs         = epochs,
        notes          = notes,
    )


# ---------------------------------------------------------------------------
# Cost tracker — accumulates lifetime compute across experiments
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    """
    Tracks cumulative compute and data costs across multiple strategy runs.

    In production, you are not choosing a strategy for one update — you are
    choosing a policy that will run every time a drift trigger fires or new
    labelled data arrives.  The cost tracker makes the lifetime cost visible.

    Usage
    -----
    tracker = CostTracker()
    tracker.record("fine_tune_head_only", runtime_s=2.1, params_updated=5_120)
    tracker.record("retrain_combined",    runtime_s=8.4, params_updated=267_322)
    print(tracker.summary())
    """

    _records: List[Dict] = field(default_factory=list)

    def record(
        self,
        strategy      : str,
        runtime_s     : float,
        params_updated: int,
        data_samples  : int = 0,
        notes         : str = "",
    ) -> None:
        self._records.append({
            "strategy"      : strategy,
            "runtime_s"     : runtime_s,
            "params_updated": params_updated,
            "data_samples"  : data_samples,
            "notes"         : notes,
        })

    def summary(self) -> str:
        if not self._records:
            return "CostTracker: no records"

        lines = [
            "=" * 72,
            "  COST SUMMARY",
            "=" * 72,
            f"  {'Strategy':<30} {'Runtime':>10} {'Params':>15} {'Samples':>10}",
            "-" * 72,
        ]
        for r in self._records:
            lines.append(
                f"  {r['strategy']:<30} {r['runtime_s']:>9.2f}s"
                f" {r['params_updated']:>15,} {r['data_samples']:>10,}"
            )
        total_runtime = sum(r["runtime_s"] for r in self._records)
        lines += [
            "-" * 72,
            f"  {'TOTAL':<30} {total_runtime:>9.2f}s",
            "=" * 72,
        ]
        return "\n".join(lines)
