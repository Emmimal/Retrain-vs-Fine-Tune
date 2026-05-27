"""
models/base_model.py
--------------------
Shared MLP architecture used across all three strategies.

Architecture decisions:
  - Variable depth and width via hidden_dims list
  - BatchNorm after each hidden layer — stabilises fine-tuning when only the
    head is trainable (the trunk statistics remain useful)
  - Dropout for regularisation in the scratch-training scenario
  - expand_head() for the new-class scenario — adds output units while
    preserving existing class representations
  - freeze_trunk() / unfreeze_trunk() for fine-tuning control
  - snapshot() / restore() for cost-free rollback within an experiment
"""

from __future__ import annotations

import copy
from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with a trunk-head architecture.

    Parameters
    ----------
    input_dim   : Number of input features
    hidden_dims : List of hidden layer widths  e.g. [256, 128]
    output_dim  : Number of output classes
    dropout     : Dropout probability (0.0 = disabled)
    use_bn      : Whether to use BatchNorm after each hidden layer
    """

    def __init__(
        self,
        input_dim  : int,
        hidden_dims: List[int],
        output_dim : int,
        dropout    : float = 0.0,
        use_bn     : bool  = True,
    ) -> None:
        super().__init__()

        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim  = output_dim

        # ---- Trunk -------------------------------------------------------
        trunk_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            trunk_layers.append(nn.Linear(in_dim, h))
            if use_bn:
                trunk_layers.append(nn.BatchNorm1d(h))
            trunk_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        # ---- Head --------------------------------------------------------
        self.head = nn.Linear(in_dim, output_dim)
        self._trunk_out_dim = in_dim

        self._init_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return trunk activations before the head."""
        return self.trunk(x)

    # ------------------------------------------------------------------
    # Trunk / head control
    # ------------------------------------------------------------------

    def freeze_trunk(self) -> None:
        """Freeze trunk; head remains trainable (standard fine-tuning)."""
        for param in self.trunk.parameters():
            param.requires_grad = False

    def unfreeze_trunk(self) -> None:
        """Unfreeze trunk for full-network fine-tuning."""
        for param in self.trunk.parameters():
            param.requires_grad = True

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def trunk_is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.trunk.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Head expansion — Scenario C (new classes)
    # ------------------------------------------------------------------

    def expand_head(self, n_new_classes: int) -> None:
        """
        Add n_new_classes output units while PRESERVING existing weights.

        Naive re-initialisation destroys prior class decision boundaries.
        This method copies existing weights into the expanded layer first.
        """
        old_n  = self.head.out_features
        new_n  = old_n + n_new_classes
        in_dim = self._trunk_out_dim

        new_head = nn.Linear(in_dim, new_n)
        with torch.no_grad():
            new_head.weight[:old_n] = self.head.weight.data
            new_head.bias[:old_n]   = self.head.bias.data
            nn.init.normal_(new_head.weight[old_n:], mean=0.0, std=0.01)
            nn.init.zeros_(new_head.bias[old_n:])

        self.head       = new_head
        self.output_dim = new_n

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Deep-copy of current state dict for rollback."""
        return copy.deepcopy(self.state_dict())

    def restore(self, state: dict) -> None:
        self.load_state_dict(state)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def __repr__(self) -> str:
        return (
            f"MLP(input={self.input_dim}, hidden={self.hidden_dims}, "
            f"output={self.output_dim}, params={self.count_total_params():,})"
        )
