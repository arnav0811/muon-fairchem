"""Evaluation utilities for trained FAIR-Chem checkpoints."""
from __future__ import annotations

from typing import Any, Dict

import torch

from models.e3_model import build_model
from utils.datamodule import build_dataloaders


def evaluate(config: Dict[str, Any], checkpoint_path: str, device: str | None = None) -> Dict[str, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = build_dataloaders(config["dataset"], split="test", device=device)
    model = build_model(config["model"])
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    # TODO: hook in actual evaluation metrics once datamodule is implemented.
    return {"mae": float("nan")}
