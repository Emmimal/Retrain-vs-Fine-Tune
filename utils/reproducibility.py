"""
utils/reproducibility.py
-------------------------
Seed management and snapshot utilities for reproducible experiments.

All benchmark runs in this article use seed=42. This module ensures
consistent seeding across PyTorch, NumPy, and Python's random module.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """
    Set all RNGs to a fixed seed for fully reproducible runs.

    Call this at the top of any script before creating models or data.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms — slight performance cost, exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def snapshot_model(model: torch.nn.Module) -> dict:
    """Deep copy of model state dict for rollback."""
    return copy.deepcopy(model.state_dict())


def restore_model(model: torch.nn.Module, state: dict) -> None:
    """Restore a model to a previously snapshotted state."""
    model.load_state_dict(state)


def count_parameters(model: torch.nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
