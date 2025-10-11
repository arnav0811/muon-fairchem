"""Utilities for capturing singular value spectra during training."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch.nn import Parameter


def compute_topk_svd(matrix: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the top-k singular values and vectors for a 2D tensor."""
    if matrix.dim() != 2:
        raise ValueError("SVD expects a 2D tensor")
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    return u[:, :k], s[:k], v[:k, :]


def save_svd_snapshot(
    layer_name: str,
    weights: torch.Tensor,
    updates: torch.Tensor,
    step: int,
    out_dir: Path,
    k: int = 5,
) -> None:
    """Persist the top-k singular values for weights and updates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    u_w, s_w, v_w = compute_topk_svd(weights.detach(), k)
    u_u, s_u, v_u = compute_topk_svd(updates.detach(), k)
    torch.save(
        {
            "layer": layer_name,
            "step": step,
            "weights": dict(u=u_w, s=s_w, v=v_w),
            "updates": dict(u=u_u, s=s_u, v=v_u),
        },
        out_dir / f"{layer_name.replace('.', '_')}_step{step}.pt",
    )


def select_layers_for_svd(parameters: Iterable[Tuple[str, Parameter]], targets: Iterable[str]) -> Dict[str, Parameter]:
    """Filter named parameters to the subset we want to observe."""
    target_set = set(targets)
    return {name: param for name, param in parameters if name in target_set}
