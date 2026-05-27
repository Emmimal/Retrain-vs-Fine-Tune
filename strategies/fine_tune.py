"""
strategies/fine_tune.py
------------------------
Fine-Tuning strategies.

Fine-tuning starts from a pre-trained model's weights rather than random
initialisation.  Two variants are implemented here because the choice between
them is a second-order decision that matters as much as the first-order choice
between fine-tuning and full retraining:

  Variant 1 — Head-Only Fine-Tuning (frozen trunk)
      Only the output layer is updated.  The trunk is frozen.
      Fastest, lowest compute cost, lowest risk of catastrophic forgetting.
      The right choice when the new task is closely related to the old one and
      the trunk's representations transfer well.

  Variant 2 — Full Fine-Tuning (unfrozen trunk)
      All parameters are updated.  A lower learning rate is used to avoid
      destroying prior representations too aggressively.
      The right choice when the domain shifts enough that the trunk needs to
      adapt, but the new dataset is too small to train from scratch.

WHEN FINE-TUNING IS THE RIGHT CHOICE
--------------------------------------
  1. Small new dataset (< ~2,000 labelled examples) and the prior model was
     trained on a related task with sufficient data.
  2. Domain transfer: source domain has the same label space, target domain
     has a shifted feature distribution (see Scenario D in the benchmark).
  3. The prior trunk representations transfer well — verified by checking
     that head-only fine-tuning surpasses random initialisation quickly.
  4. New classes are being added and the class count change is small (expand_head).
  5. Inference latency is fixed and the architecture cannot be changed.

COST PROFILE — Head-Only
------------------------
  Compute   : Lowest — only one layer's gradients are computed
  Data      : Lowest — converges with very few examples
  Risk      : Low forgetting (trunk frozen), risk of underfitting if task has shifted
  Forgetting: Near-zero — frozen trunk preserves all prior representations

COST PROFILE — Full Fine-Tuning
---------------------------------
  Compute   : Medium — all gradients computed, lower LR extends training
  Data      : Medium — needs more than head-only but far less than scratch
  Risk      : Moderate forgetting if LR is too high; negative transfer if tasks diverge
  Forgetting: Moderate — depends heavily on LR and number of epochs
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base_model import MLP


class FineTuneMode(Enum):
    HEAD_ONLY    = "head_only"      # frozen trunk, trainable head
    FULL_NETWORK = "full_network"   # all parameters trainable


@dataclass
class FineTuneResult:
    """Result container for a Fine-Tuning run."""
    strategy        : str         = "fine_tune"
    mode            : str         = "head_only"
    final_train_loss: float       = 0.0
    epochs_trained  : int         = 0
    runtime_s       : float       = 0.0
    params_trained  : int         = 0
    trunk_was_frozen: bool        = True


class FineTuner:
    """
    Fine-tune a pre-trained MLP on new data.

    Parameters
    ----------
    model         : Pre-trained MLP (weights preserved from prior training)
    mode          : HEAD_ONLY or FULL_NETWORK
    epochs        : Training epochs
    lr            : Learning rate
                    Head-only: standard LR (e.g. 0.01)
                    Full network: use a lower LR (e.g. 0.001) to protect prior
                    representations from aggressive gradient updates
    weight_decay  : L2 regularisation
    verbose       : Print per-epoch loss

    Usage
    -----
    # Head-only fine-tuning (frozen trunk)
    tuner  = FineTuner(model, mode=FineTuneMode.HEAD_ONLY, lr=0.01, epochs=10)
    result = tuner.fine_tune(new_data_loader)

    # Full fine-tuning (unfrozen trunk, lower LR)
    tuner  = FineTuner(model, mode=FineTuneMode.FULL_NETWORK, lr=0.001, epochs=10)
    result = tuner.fine_tune(new_data_loader)
    """

    def __init__(
        self,
        model       : MLP,
        mode        : FineTuneMode = FineTuneMode.HEAD_ONLY,
        epochs      : int          = 10,
        lr          : float        = 0.01,
        weight_decay: float        = 1e-4,
        verbose     : bool         = False,
    ) -> None:
        self.model        = model
        self.mode         = mode
        self.epochs       = epochs
        self.lr           = lr
        self.weight_decay = weight_decay
        self.verbose      = verbose

        self._criterion = nn.CrossEntropyLoss()

    def fine_tune(self, train_loader: DataLoader) -> FineTuneResult:
        """
        Fine-tune the model on new data.

        The model is mutated in-place.  Prior weights are the starting point —
        they are NOT re-initialised.
        """
        if self.mode == FineTuneMode.HEAD_ONLY:
            self.model.freeze_trunk()
            trainable_params = self.model.head.parameters()
        else:
            self.model.unfreeze_trunk()
            trainable_params = self.model.parameters()

        trunk_frozen = self.model.trunk_is_frozen()
        self.model.train()

        optimiser = torch.optim.Adam(
            trainable_params,
            lr           = self.lr,
            weight_decay = self.weight_decay,
        )

        t0 = time.perf_counter()
        final_loss = 0.0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in train_loader:
                optimiser.zero_grad()
                logits = self.model(X_batch)
                loss   = self._criterion(logits, y_batch)
                loss.backward()

                # Only clip gradients for trainable params
                if not trunk_frozen:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                optimiser.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            final_loss = avg_loss
            if self.verbose:
                mode_str = "head-only" if trunk_frozen else "full"
                print(
                    f"  [fine_tune/{mode_str}] epoch {epoch+1:3d}/{self.epochs}"
                    f" — loss {avg_loss:.4f}"
                )

        runtime = time.perf_counter() - t0
        self.model.eval()

        # Always unfreeze after fine-tuning so the model is in a clean state
        # for subsequent evaluation or further training
        self.model.unfreeze_trunk()

        return FineTuneResult(
            mode             = self.mode.value,
            final_train_loss = final_loss,
            epochs_trained   = self.epochs,
            runtime_s        = runtime,
            params_trained   = self.model.count_trainable_params(),
            trunk_was_frozen = trunk_frozen,
        )
