"""
data/loaders.py
---------------
DataLoader factory.  All loaders use a fixed generator for reproducibility —
critical for fair head-to-head comparisons across strategies where mini-batch
ordering must be identical between runs at the same seed.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(
    X          : torch.Tensor,
    y          : torch.Tensor,
    batch_size : int            = 64,
    shuffle    : bool           = True,
    seed       : int            = 42,
    drop_last  : bool           = False,
    num_workers: int            = 0,
) -> DataLoader:
    """
    Build a DataLoader from feature and label tensors.

    Parameters
    ----------
    X          : Feature tensor  [N, D]
    y          : Label tensor    [N]
    batch_size : Mini-batch size
    shuffle    : Whether to shuffle on each epoch
    seed       : Controls the RNG for shuffling — set the same seed across
                 strategies to ensure identical mini-batch ordering
    drop_last  : Whether to drop the last incomplete batch

    Returns
    -------
    DataLoader ready for a standard PyTorch training loop
    """
    dataset   = TensorDataset(X, y)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        drop_last   = drop_last,
        num_workers = num_workers,
        generator   = generator if shuffle else None,
    )


def make_eval_loader(
    X          : torch.Tensor,
    y          : torch.Tensor,
    batch_size : int = 256,
) -> DataLoader:
    """
    Build a non-shuffled DataLoader for evaluation.
    Larger batch size for faster inference.
    """
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
