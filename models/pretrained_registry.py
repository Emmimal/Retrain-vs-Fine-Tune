"""
models/pretrained_registry.py
------------------------------
Lightweight pre-trained model registry.

Integrates with the file-system registry from Article 04
(https://emitechlogic.com/model-versioning-in-production-machine-learning/).

The registry stores champion model state dicts on disk so they can be loaded
as starting points for fine-tuning and retraining experiments — exactly as a
production system would promote and retrieve champion artifacts.

For the Article 08 benchmark, the registry is seeded with a pre-trained
champion model from each scenario's source/initial training phase.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from models.base_model import MLP


class PretrainedRegistry:
    """
    File-system-backed registry for pre-trained model artifacts.

    Directory layout
    ----------------
    registry_dir/
      <model_id>/
        model.pt         — state dict
        metadata.json    — arch config + training metadata
    """

    def __init__(self, registry_dir: str = "registry") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(
        self,
        model_id      : str,
        model         : MLP,
        metadata      : Optional[Dict] = None,
    ) -> str:
        """
        Save a model's state dict and architecture config.

        Returns the path to the saved model directory.
        """
        model_dir = self.registry_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # State dict
        torch.save(model.state_dict(), model_dir / "model.pt")

        # Metadata
        meta = {
            "model_id"    : model_id,
            "saved_at"    : time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input_dim"   : model.input_dim,
            "hidden_dims" : model.hidden_dims,
            "output_dim"  : model.output_dim,
        }
        if metadata:
            meta.update(metadata)
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return str(model_dir)

    def load(self, model_id: str) -> MLP:
        """
        Load a pre-trained model from the registry.

        Reconstructs the architecture from saved metadata and loads the
        state dict.  Returns a model in eval mode.
        """
        model_dir = self.registry_dir / model_id
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model '{model_id}' not found in registry at {self.registry_dir}"
            )

        with open(model_dir / "metadata.json") as f:
            meta = json.load(f)

        model = MLP(
            input_dim   = meta["input_dim"],
            hidden_dims = meta["hidden_dims"],
            output_dim  = meta["output_dim"],
        )
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    def list_models(self) -> list:
        return [d.name for d in self.registry_dir.iterdir() if d.is_dir()]

    def exists(self, model_id: str) -> bool:
        return (self.registry_dir / model_id).exists()
