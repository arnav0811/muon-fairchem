"""Parameter routing helpers for combining Muon with AdamW."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch.nn as nn


@dataclass
class RoutedParams:
    """Container returned by parameter routing."""

    muon: List[nn.Parameter]
    adamw: List[nn.Parameter]

    def as_optim_groups(self) -> Sequence[Iterable[nn.Parameter]]:
        return self.muon, self.adamw


def _is_candidate(name: str, param: nn.Parameter) -> bool:
    return param.requires_grad and param.dim() == 2 and name.endswith(".weight")


def split_params_for_muon(
    model: nn.Module,
    apply_layers: str = "all",
    first_k_blocks: int | None = None,
) -> RoutedParams:
    """Split parameters into Muon-eligible and AdamW groups."""
    muon_params: List[nn.Parameter] = []
    adamw_params: List[nn.Parameter] = []

    block_indices: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not _is_candidate(name, param):
            adamw_params.append(param)
            continue

        selected = False
        if apply_layers == "all":
            selected = True
        elif apply_layers == "first_n":
            if first_k_blocks is not None and first_k_blocks > 0:
                prefix = _prefix_without_param(name)
                if prefix not in block_indices:
                    block_indices[prefix] = len(block_indices)
                selected = block_indices[prefix] < first_k_blocks
        else:
            raise ValueError(f"Unknown apply_layers mode: {apply_layers}")

        if selected:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return RoutedParams(muon=muon_params, adamw=adamw_params)


def _prefix_without_param(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else name
