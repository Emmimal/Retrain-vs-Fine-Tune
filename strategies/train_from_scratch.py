"""
strategies/train_from_scratch.py
---------------------------------
Train from Scratch strategy.

Full re-initialisation followed by full training on all available data.
No prior knowledge is used — the model starts with random weights.

WHEN THIS IS THE RIGHT CHOICE
------------------------------
  1. The task has changed so fundamentally that prior representations are
     actively harmful (negative transfer).
  2. The new data distribution is completely unrelated to the old one —
     for example, switching from NLP classification to image recognition
     within the same serving infrastructure.
  3. You have abundant new-task data and the training cost is acceptable.
  4. The prior model was known to be undertrained or poorly regularised —
     starting fresh avoids inheriting those deficiencies.
  5. The prior model's architecture no longer matches the new task's output
     space and expansion is not feasible.

COST PROFILE
------------
  Compute   : Highest — every parameter is trained from random initialisation
  Data      : Highest — needs sufficient data to converge without prior knowledge
  Risk      : Low — no risk of negative transfer from prior task representations
  Forgetting: Complete — prior task knowledge is entirely discarded by design

PARAMETERS
----------
  model       : MLP instance (will be re-initialised)
  epochs      : Training epochs per run
  lr          : Learning rate
  weight_decay: L2 regularisation
  verbose     : Whether to print per-epoch loss
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base_model import MLP


@dataclass
class ScratchResult:
    """Result container for a Train-from-Scratch run."""
    strategy       : str   = "train_from_scratch"
    final_train_loss: float = 0.0
    epochs_trained  : int  = 0
    runtime_s       : float = 0.0
    params_trained  : int  = 0


class TrainFromScratch:
    """
    Re-initialise the model and train on all provided data from epoch 0.

    Usage
    -----
    model   = MLP(input_dim=20, hidden_dims=[128, 64], output_dim=2)
    trainer = TrainFromScratch(model, epochs=10, lr=0.01)
    result  = trainer.train(train_loader)
    """

    def __init__(
        self,
        model       : MLP,
        epochs      : int   = 10,
        lr          : float = 0.01,
        weight_decay: float = 1e-4,
        verbose     : bool  = False,
    ) -> None:
        self.model        = model
        self.epochs       = epochs
        self.lr           = lr
        self.weight_decay = weight_decay
        self.verbose      = verbose

        self._criterion = nn.CrossEntropyLoss()

    def train(self, train_loader: DataLoader) -> ScratchResult:
        """
        Re-initialise weights and train on train_loader.

        The model is mutated in-place.  Call model.snapshot() before this
        if you need to preserve the pre-training state.
        """
        # Re-initialise — discard all prior knowledge
        self.model._init_weights()
        self.model.unfreeze_trunk()
        self.model.train()

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

            for X_batch, y_batch in train_loader:
                optimiser.zero_grad()
                logits = self.model(X_batch)
                loss   = self._criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            final_loss = avg_loss
            if self.verbose:
                print(f"  [scratch] epoch {epoch+1:3d}/{self.epochs} — loss {avg_loss:.4f}")

        runtime = time.perf_counter() - t0
        self.model.eval()

        return ScratchResult(
            final_train_loss = final_loss,
            epochs_trained   = self.epochs,
            runtime_s        = runtime,
            params_trained   = self.model.count_total_params(),
        )
