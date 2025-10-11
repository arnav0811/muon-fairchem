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

    for name, param in model.named_parameters():
        if _is_candidate(name, param) and _is_selected(name, apply_layers, first_k_blocks):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return RoutedParams(muon=muon_params, adamw=adamw_params)


def _is_selected(name: str, apply_layers: str, first_k_blocks: int | None) -> bool:
    if apply_layers == "all":
        return True
    if apply_layers == "first_n":
        if first_k_blocks is None or first_k_blocks <= 0:
            return False
        pieces = name.split(".")
        try:
            block_idx = next(
                int(piece.replace("block", ""))
                for piece in pieces
                if piece.startswith("block") and piece[len("block") :].isdigit()
            )
        except StopIteration:
            return False
        return block_idx < first_k_blocks
    raise ValueError(f"Unknown apply_layers mode: {apply_layers}")
