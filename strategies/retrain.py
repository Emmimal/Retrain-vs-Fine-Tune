"""
strategies/retrain.py
----------------------
Retraining strategies.

Retraining updates a model starting from its current weights (warm start),
but trains on a dataset that explicitly includes data from previous windows.
This is the strategy that the retraining pipeline from Article 03 executes
when a drift trigger fires.

Three retraining modes are implemented because the data composition decision
is independent of the warm-start decision and has substantial impact on results:

  Mode 1 — Warm Start on New Data Only
      Weights initialised from the prior champion.
      Training data: only the new distribution data.
      Faster convergence than scratch (warm start).
      Risk: overwrites prior distribution knowledge (catastrophic forgetting).
      Article 05 addressed how EWC and replay mitigate this in the CL setting.

  Mode 2 — Warm Start on Combined Data
      Weights initialised from the prior champion.
      Training data: old distribution data + new distribution data.
      The standard production retraining approach for slowly drifting systems.
      Risk: combined dataset size grows over time, increasing compute cost.

  Mode 3 — Cold Start on Combined Data (Full Retrain)
      Weights re-initialised.
      Training data: old + new combined.
      Equivalent to Train from Scratch on the full combined dataset.
      Use when you have reason to believe the prior weights are harmful as an
      initialisation point (e.g. the model was severely overfit to the old
      distribution) but you still want the full data history.

WHEN RETRAINING IS THE RIGHT CHOICE
-------------------------------------
  1. Moderate distribution shift — the new data is not from a completely
     different task, but the current champion is measurably degrading.
  2. You have retained labelled data from prior training windows (no GDPR
     prohibition on historical retention).
  3. The drift is gradual and combined training preserves both distributions
     adequately.
  4. The prior model architecture is still appropriate for the new task.
  5. You want to avoid the forgetting risk of fine-tuning on new data alone
     without the overhead of the EWC/replay methods from Article 05.

COST PROFILE — Warm Start New Data Only
-----------------------------------------
  Compute   : Low-Medium (warm start converges faster than scratch)
  Data      : Low (only new data required)
  Forgetting: High (same risk as fine-tuning full network on new data)

COST PROFILE — Warm Start Combined Data
-----------------------------------------
  Compute   : Medium-High (full dataset, but warm start reduces epoch count)
  Data      : High (must retain historical data)
  Forgetting: Low (prior distribution explicitly present in training signal)

COST PROFILE — Cold Start Combined Data
-----------------------------------------
  Compute   : Highest (same as train_from_scratch on combined dataset)
  Data      : High (must retain historical data)
  Forgetting: N/A (complete reset)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from models.base_model import MLP


class RetrainMode(Enum):
    WARM_NEW_ONLY  = "warm_new_only"    # warm start, new data only
    WARM_COMBINED  = "warm_combined"    # warm start, old + new data
    COLD_COMBINED  = "cold_combined"    # cold start (re-init), old + new data


@dataclass
class RetrainResult:
    """Result container for a Retrain run."""
    strategy        : str   = "retrain"
    mode            : str   = "warm_combined"
    final_train_loss: float = 0.0
    epochs_trained  : int   = 0
    runtime_s       : float = 0.0
    params_trained  : int   = 0
    warm_start      : bool  = True


class Retrainer:
    """
    Retrain a model from its current weights (warm) or re-initialised (cold).

    Parameters
    ----------
    model         : MLP — the current champion model
    mode          : RetrainMode controlling warm/cold start and data composition
    epochs        : Training epochs
    lr            : Learning rate
                    Warm start: slightly lower than scratch LR to avoid
                    overshooting the neighbourhood of the current solution
    weight_decay  : L2 regularisation
    verbose       : Print per-epoch loss

    Usage
    -----
    # Warm start on combined data (the standard production case)
    retrainer = Retrainer(model, mode=RetrainMode.WARM_COMBINED, lr=0.005)
    result    = retrainer.retrain(new_loader, old_loader=old_loader)

    # Warm start on new data only (fast but higher forgetting risk)
    retrainer = Retrainer(model, mode=RetrainMode.WARM_NEW_ONLY, lr=0.005)
    result    = retrainer.retrain(new_loader)
    """

    def __init__(
        self,
        model       : MLP,
        mode        : RetrainMode = RetrainMode.WARM_COMBINED,
        epochs      : int         = 10,
        lr          : float       = 0.005,
        weight_decay: float       = 1e-4,
        verbose     : bool        = False,
    ) -> None:
        self.model        = model
        self.mode         = mode
        self.epochs       = epochs
        self.lr           = lr
        self.weight_decay = weight_decay
        self.verbose      = verbose

        self._criterion = nn.CrossEntropyLoss()

    def retrain(
        self,
        new_loader: DataLoader,
        old_loader: DataLoader = None,
    ) -> RetrainResult:
        """
        Retrain the model.

        Parameters
        ----------
        new_loader : DataLoader for the new/current distribution data
        old_loader : DataLoader for prior distribution data
                     Required for WARM_COMBINED and COLD_COMBINED modes.
                     Ignored for WARM_NEW_ONLY.
        """
        warm_start = self.mode != RetrainMode.COLD_COMBINED

        # Cold start: re-initialise weights
        if self.mode == RetrainMode.COLD_COMBINED:
            self.model._init_weights()

        self.model.unfreeze_trunk()
        self.model.train()

        # Select training loader
        if self.mode == RetrainMode.WARM_NEW_ONLY:
            active_loader = new_loader
        else:
            # Combined: interleave old and new data each epoch
            active_loader = self._make_combined_loader(new_loader, old_loader)

        optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr           = self.lr,
            weight_decay = self.weight_decay,
        )

        t0 = time.perf_counter()
        final_loss = 0.0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in active_loader:
                optimiser.zero_grad()
                logits = self.model(X_batch)
                loss   = self._criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimiser.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss   = epoch_loss / max(n_batches, 1)
            final_loss = avg_loss
            if self.verbose:
                print(
                    f"  [retrain/{self.mode.value}] epoch {epoch+1:3d}/{self.epochs}"
                    f" — loss {avg_loss:.4f}"
                )

        runtime = time.perf_counter() - t0
        self.model.eval()

        return RetrainResult(
            mode             = self.mode.value,
            final_train_loss = final_loss,
            epochs_trained   = self.epochs,
            runtime_s        = runtime,
            params_trained   = self.model.count_total_params(),
            warm_start       = warm_start,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_combined_loader(
        self,
        new_loader: DataLoader,
        old_loader: DataLoader,
    ) -> DataLoader:
        """
        Merge new and old DataLoaders into a single combined loader.

        This avoids materialising a combined TensorDataset in memory —
        ConcatDataset streams from both sources.
        """
        if old_loader is None:
            return new_loader

        combined = ConcatDataset([new_loader.dataset, old_loader.dataset])
        return DataLoader(
            combined,
            batch_size = new_loader.batch_size,
            shuffle    = True,
            drop_last  = False,
        )
