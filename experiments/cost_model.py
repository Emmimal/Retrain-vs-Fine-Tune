"""
experiments/cost_model.py
--------------------------
Compute and data cost estimates for each strategy.

The cost model makes the lifetime economics of each strategy visible.
In a production system that retrains N times per year, the per-update
cost difference between strategies compounds significantly.

Cost dimensions modelled:
  1. Compute cost:   proportional to (params_updated × epochs × data_samples)
  2. Data cost:      proportional to labelled samples required
  3. Forgetting risk: estimated probability of unacceptable backward transfer

These are NOT exact measurements — they are relative scaling factors that
allow strategies to be compared on a common scale.  The benchmark in
benchmarks/benchmark.py provides real runtime measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CostEstimate:
    strategy             : str
    compute_factor       : float   # relative to fine_tune_head_only = 1.0
    data_samples_required: int
    forgetting_risk      : str     # 'low' | 'medium' | 'high'
    notes                : str = ""

    def __str__(self) -> str:
        return (
            f"{self.strategy:<35}"
            f"  compute={self.compute_factor:>5.1f}x"
            f"  samples={self.data_samples_required:>8,}"
            f"  forgetting={self.forgetting_risk}"
        )


class CostModel:
    """
    Estimate and compare costs across strategies for a given scenario.

    Usage
    -----
    model = CostModel(
        n_new_samples       = 500,
        n_historical_samples= 2000,
        n_params_total      = 267_000,
        n_params_head       = 512,
        epochs              = 10,
    )
    estimates = model.estimate_all()
    print(model.comparison_table(estimates))
    """

    def __init__(
        self,
        n_new_samples        : int,
        n_historical_samples : int,
        n_params_total       : int,
        n_params_head        : int,
        epochs               : int  = 10,
    ) -> None:
        self.n_new          = n_new_samples
        self.n_historical   = n_historical_samples
        self.n_params_total = n_params_total
        self.n_params_head  = n_params_head
        self.epochs         = epochs

        # Baseline compute unit: head-only fine-tuning on new data
        self._baseline_compute = self.n_params_head * self.epochs * self.n_new

    def _compute_factor(self, params: int, epochs: int, samples: int) -> float:
        if self._baseline_compute == 0:
            return 1.0
        return (params * epochs * samples) / self._baseline_compute

    def estimate_all(self) -> List[CostEstimate]:
        estimates = []

        # Fine-tune head only
        estimates.append(CostEstimate(
            strategy              = "fine_tune_head_only",
            compute_factor        = self._compute_factor(
                self.n_params_head, self.epochs, self.n_new
            ),
            data_samples_required = self.n_new,
            forgetting_risk       = "low",
            notes                 = "Baseline — trunk frozen, head only",
        ))

        # Fine-tune full network
        estimates.append(CostEstimate(
            strategy              = "fine_tune_full_network",
            compute_factor        = self._compute_factor(
                self.n_params_total, self.epochs, self.n_new
            ),
            data_samples_required = self.n_new,
            forgetting_risk       = "medium",
            notes                 = "Lower LR recommended (10x reduction)",
        ))

        # Retrain warm new only
        estimates.append(CostEstimate(
            strategy              = "retrain_warm_new_only",
            compute_factor        = self._compute_factor(
                self.n_params_total, max(self.epochs // 2, 1), self.n_new
            ),
            data_samples_required = self.n_new,
            forgetting_risk       = "high",
            notes                 = "Warm start halves typical epoch count",
        ))

        # Retrain warm combined
        estimates.append(CostEstimate(
            strategy              = "retrain_warm_combined",
            compute_factor        = self._compute_factor(
                self.n_params_total,
                max(self.epochs // 2, 1),
                self.n_new + self.n_historical,
            ),
            data_samples_required = self.n_new + self.n_historical,
            forgetting_risk       = "low",
            notes                 = "Standard production retraining approach",
        ))

        # Retrain cold combined
        estimates.append(CostEstimate(
            strategy              = "retrain_cold_combined",
            compute_factor        = self._compute_factor(
                self.n_params_total,
                self.epochs,
                self.n_new + self.n_historical,
            ),
            data_samples_required = self.n_new + self.n_historical,
            forgetting_risk       = "low",
            notes                 = "Same as scratch on full dataset",
        ))

        # Train from scratch
        estimates.append(CostEstimate(
            strategy              = "train_from_scratch",
            compute_factor        = self._compute_factor(
                self.n_params_total, self.epochs, self.n_new
            ),
            data_samples_required = self.n_new,
            forgetting_risk       = "n/a (complete reset)",
            notes                 = "No prior knowledge used",
        ))

        return estimates

    def comparison_table(self, estimates: List[CostEstimate]) -> str:
        lines = [
            "=" * 80,
            "  COST MODEL — Relative estimates (fine_tune_head_only = 1.0x)",
            f"  Parameters total: {self.n_params_total:,} | "
            f"Head: {self.n_params_head:,} | "
            f"New samples: {self.n_new:,} | "
            f"Historical: {self.n_historical:,}",
            "=" * 80,
            f"  {'Strategy':<35}  {'Compute':>9}  {'Samples':>10}  {'Forgetting Risk'}",
            "-" * 80,
        ]
        for e in estimates:
            lines.append(
                f"  {e.strategy:<35}  {e.compute_factor:>8.1f}x  "
                f"{e.data_samples_required:>10,}  {e.forgetting_risk}"
            )
        lines.append("=" * 80)
        return "\n".join(lines)
