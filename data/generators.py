"""
data/generators.py
------------------
Synthetic dataset generators for the Retrain vs Fine-Tune vs Train from Scratch
benchmark.

Four scenarios are modelled here, each representing a real-world trigger that
forces a retraining decision:

  Scenario A — New Data, Same Distribution
      More data from the same source arrives.  The question is whether fine-tuning
      on new data alone, retraining on the combined set, or starting fresh produces
      the best result at the lowest cost.

  Scenario B — Distribution Shift (Concept Drift)
      The feature-label relationship changes gradually or abruptly.  Mirrors the
      production scenario in Articles 05–07: what happens when the world changes
      and the current champion model no longer fits the new reality.

  Scenario C — New Task / New Classes
      A genuinely new output class appears.  The model was never trained on it.
      Choosing between scratch training and fine-tuning with head expansion is the
      core decision.

  Scenario D — Domain Transfer
      A model trained on Domain A is repurposed for Domain B, which shares
      representational structure but has different surface statistics.  This is the
      canonical transfer learning scenario.

All generators return (X_train, y_train, X_test, y_test) as torch.Tensors.
They are deterministic given a seed — set seed=42 to reproduce the benchmark.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
DataSplit = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> DataSplit:
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test,  dtype=torch.float32),
        torch.tensor(y_test,  dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Scenario A — New Data, Same Distribution
# ---------------------------------------------------------------------------

@dataclass
class NewDataScenario:
    """
    Simulates receiving a second wave of labelled data from the same distribution.

    Phase 1 (initial training): n_initial samples from a 2-class Gaussian mixture.
    Phase 2 (update):           n_new additional samples from the identical DGP.

    The test set is drawn from the same distribution and is constant across phases.

    Usage
    -----
    scenario = NewDataScenario(n_initial=1000, n_new=500, n_features=20)
    phase1 = scenario.phase1()   # (X_tr, y_tr, X_te, y_te)  — initial training set
    phase2 = scenario.phase2()   # new data only
    combined = scenario.combined()  # phase1 train + phase2 train
    """

    n_initial : int   = 1_000
    n_new     : int   = 500
    n_test    : int   = 400
    n_features: int   = 20
    n_classes : int   = 2
    class_sep : float = 1.0   # larger = easier to separate
    seed      : int   = 42

    def _make_split(self, n_samples: int, seed_offset: int) -> DataSplit:
        _set_seed(self.seed + seed_offset)
        means   = [i * self.class_sep for i in range(self.n_classes)]
        X_parts, y_parts = [], []
        per_class = n_samples // self.n_classes
        for c, mu in enumerate(means):
            X_c = np.random.randn(per_class, self.n_features) + mu
            y_c = np.full(per_class, c)
            X_parts.append(X_c)
            y_parts.append(y_c)
        X = np.vstack(X_parts).astype(np.float32)
        y = np.concatenate(y_parts).astype(np.int64)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]

    def _make_test(self) -> Tuple[np.ndarray, np.ndarray]:
        _set_seed(self.seed + 9999)
        means = [i * self.class_sep for i in range(self.n_classes)]
        X_parts, y_parts = [], []
        per_class = self.n_test // self.n_classes
        for c, mu in enumerate(means):
            X_c = np.random.randn(per_class, self.n_features) + mu
            y_c = np.full(per_class, c)
            X_parts.append(X_c)
            y_parts.append(y_c)
        X = np.vstack(X_parts).astype(np.float32)
        y = np.concatenate(y_parts).astype(np.int64)
        return X, y

    def phase1(self) -> DataSplit:
        X_tr, y_tr = self._make_split(self.n_initial, seed_offset=0)
        X_te, y_te = self._make_test()
        return _to_tensors(X_tr, y_tr, X_te, y_te)

    def phase2(self) -> DataSplit:
        """Return only the new data (no test set — use phase1 test set)."""
        X_new, y_new = self._make_split(self.n_new, seed_offset=1)
        X_te, y_te   = self._make_test()
        return _to_tensors(X_new, y_new, X_te, y_te)

    def combined(self) -> DataSplit:
        """Return phase1 + phase2 training data with the shared test set."""
        X_1, y_1 = self._make_split(self.n_initial, seed_offset=0)
        X_2, y_2 = self._make_split(self.n_new,     seed_offset=1)
        X_tr = np.vstack([X_1, X_2])
        y_tr = np.concatenate([y_1, y_2])
        X_te, y_te = self._make_test()
        return _to_tensors(X_tr, y_tr, X_te, y_te)


# ---------------------------------------------------------------------------
# Scenario B — Distribution Shift
# ---------------------------------------------------------------------------

@dataclass
class DistributionShiftScenario:
    """
    Simulates a distribution shift between the initial training window and the
    current deployment window.

    shift_magnitude controls how far the class means move.  A value of 0 means
    no shift; a value of 2.0 is a severe shift where the new distribution barely
    overlaps with the old.

    The test set is drawn exclusively from the NEW distribution to measure how
    well the updated model handles the shifted reality.
    """

    n_initial       : int   = 1_000
    n_new           : int   = 500
    n_test          : int   = 400
    n_features      : int   = 20
    n_classes       : int   = 2
    shift_magnitude : float = 1.5   # distance the means move
    seed            : int   = 42

    def _make(
        self,
        n_samples : int,
        shift     : float,
        seed_offset: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _set_seed(self.seed + seed_offset)
        X_parts, y_parts = [], []
        per_class = n_samples // self.n_classes
        for c in range(self.n_classes):
            base_mean = c * 1.0
            shifted_mean = base_mean + shift
            X_c = np.random.randn(per_class, self.n_features) + shifted_mean
            y_c = np.full(per_class, c)
            X_parts.append(X_c)
            y_parts.append(y_c)
        X = np.vstack(X_parts).astype(np.float32)
        y = np.concatenate(y_parts).astype(np.int64)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]

    def original_distribution(self) -> DataSplit:
        """Data from the original (pre-shift) distribution."""
        X_tr, y_tr = self._make(self.n_initial, shift=0.0, seed_offset=0)
        X_te, y_te = self._make(self.n_test,    shift=0.0, seed_offset=9)
        return _to_tensors(X_tr, y_tr, X_te, y_te)

    def shifted_distribution(self) -> DataSplit:
        """Data from the shifted distribution — both train and test."""
        X_tr, y_tr = self._make(self.n_new,  shift=self.shift_magnitude, seed_offset=1)
        X_te, y_te = self._make(self.n_test, shift=self.shift_magnitude, seed_offset=8)
        return _to_tensors(X_tr, y_tr, X_te, y_te)

    def mixed_distribution(self, old_fraction: float = 0.5) -> DataSplit:
        """
        Mixed training set: old_fraction of original + (1-old_fraction) of shifted.
        Useful for benchmarking retrain-on-combined against fine-tune-on-new-only.
        """
        n_old = int(self.n_initial * old_fraction)
        n_shifted = self.n_initial - n_old
        X_old, y_old = self._make(n_old,     shift=0.0,                 seed_offset=0)
        X_new, y_new = self._make(n_shifted, shift=self.shift_magnitude, seed_offset=1)
        X_tr = np.vstack([X_old, X_new])
        y_tr = np.concatenate([y_old, y_new])
        X_te, y_te = self._make(self.n_test, shift=self.shift_magnitude, seed_offset=8)
        perm = np.random.permutation(len(X_tr))
        return _to_tensors(X_tr[perm], y_tr[perm], X_te, y_te)


# ---------------------------------------------------------------------------
# Scenario C — New Task / New Classes
# ---------------------------------------------------------------------------

@dataclass
class NewClassScenario:
    """
    Simulates the arrival of a genuinely new output class that was absent from
    the original training set.

    original_classes: classes the model was trained on initially.
    new_classes:      new classes added in the update.

    The test set includes examples from ALL classes — original + new — to
    measure both backward transfer (old class performance) and forward transfer
    (new class performance) simultaneously.
    """

    n_per_class_train : int   = 300
    n_per_class_test  : int   = 100
    n_features        : int   = 20
    original_classes  : int   = 3
    new_classes       : int   = 2    # classes added in the update
    class_sep         : float = 1.2
    seed              : int   = 42

    @property
    def total_classes(self) -> int:
        return self.original_classes + self.new_classes

    def _make_classes(
        self,
        class_ids   : list,
        n_per_class : int,
        seed_offset : int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _set_seed(self.seed + seed_offset)
        X_parts, y_parts = [], []
        for c in class_ids:
            mean = c * self.class_sep
            X_c  = np.random.randn(n_per_class, self.n_features) + mean
            y_c  = np.full(n_per_class, c)
            X_parts.append(X_c)
            y_parts.append(y_c)
        X = np.vstack(X_parts).astype(np.float32)
        y = np.concatenate(y_parts).astype(np.int64)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]

    def original_data(self) -> DataSplit:
        """Training and test data using only the original classes."""
        orig_ids = list(range(self.original_classes))
        X_tr, y_tr = self._make_classes(orig_ids, self.n_per_class_train, seed_offset=0)
        X_te, y_te = self._make_classes(orig_ids, self.n_per_class_test,  seed_offset=9)
        return _to_tensors(X_tr, y_tr, X_te, y_te)

    def new_class_data(self) -> DataSplit:
        """Training and test data for the NEW classes only."""
        new_ids = list(range(self.original_classes, self.total_classes))
        X_tr, y_tr = self._make_classes(new_ids, self.n_per_class_train, seed_offset=1)
        X_te, y_te = self._make_classes(new_ids, self.n_per_class_test,  seed_offset=8)
        return _to_tensors(X_tr, y_tr, X_te, y_te)

    def full_test_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Test set covering ALL classes — original + new."""
        all_ids    = list(range(self.total_classes))
        X_te, y_te = self._make_classes(all_ids, self.n_per_class_test, seed_offset=7)
        return (
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(y_te, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Scenario D — Domain Transfer
# ---------------------------------------------------------------------------

@dataclass
class DomainTransferScenario:
    """
    Simulates transfer from a data-rich source domain to a data-scarce target
    domain.  Both domains have the same label space and task structure, but
    their feature distributions differ.

    This models the classic transfer learning scenario: a model pre-trained on
    ImageNet being fine-tuned on a medical imaging dataset.  At the MLP level,
    the source domain has a different mean and covariance structure.

    n_source_train: large — the model can be fully trained on source data.
    n_target_train: small — only a few hundred labelled examples are available.
    """

    n_source_train : int   = 2_000
    n_target_train : int   = 200     # deliberately small — few-shot regime
    n_test         : int   = 400
    n_features     : int   = 20
    n_classes      : int   = 3
    domain_shift   : float = 2.0     # distance between domain means
    seed           : int   = 42

    def _make(
        self,
        n_samples   : int,
        domain_offset: float,
        seed_offset : int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _set_seed(self.seed + seed_offset)
        X_parts, y_parts = [], []
        per_class = n_samples // self.n_classes
        for c in range(self.n_classes):
            class_mean   = c * 1.0 + domain_offset
            # Target domain also has higher variance to model domain difficulty
            scale = 1.0 if domain_offset == 0 else 1.3
            X_c  = np.random.randn(per_class, self.n_features) * scale + class_mean
            y_c  = np.full(per_class, c)
            X_parts.append(X_c)
            y_parts.append(y_c)
        X = np.vstack(X_parts).astype(np.float32)
        y = np.concatenate(y_parts).astype(np.int64)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm]

    def source_data(self) -> DataSplit:
        """Large labelled dataset from the source domain."""
        X_tr, y_tr = self._make(self.n_source_train, domain_offset=0.0, seed_offset=0)
        X_te, y_te = self._make(self.n_test,         domain_offset=0.0, seed_offset=9)
        return _to_tensors(X_tr, y_tr, X_te, y_te)

    def target_data(self) -> DataSplit:
        """Small labelled dataset from the target domain (few-shot regime)."""
        X_tr, y_tr = self._make(self.n_target_train, domain_offset=self.domain_shift, seed_offset=1)
        X_te, y_te = self._make(self.n_test,          domain_offset=self.domain_shift, seed_offset=8)
        return _to_tensors(X_tr, y_tr, X_te, y_te)
