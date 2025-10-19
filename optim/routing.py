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
    muon_params: List[nn.Parameter] = []
    adamw_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not _is_candidate(name, param):
            adamw_params.append(param)
            continue

        # Try to extract the GemNet block id from the name
        block_id = None
        if "backbone.blocks." in name:
            try:
                block_id = int(name.split("backbone.blocks.")[1].split(".")[0])
            except Exception:
                block_id = None

        selected = False
        if apply_layers == "all":
            selected = True
        elif apply_layers in {"first_n", "first_k"}:
            if first_k_blocks is not None and block_id is not None:
                selected = block_id < first_k_blocks
        elif apply_layers == "last_k":
            if first_k_blocks is not None and block_id is not None:
                TOTAL = 6  # GemNet depth in this project
                selected = block_id >= (TOTAL - first_k_blocks)
        else:
            raise ValueError(f"Unknown apply_layers mode: {apply_layers}")

        (muon_params if selected else adamw_params).append(param)

    return RoutedParams(muon=muon_params, adamw=adamw_params)



def _prefix_without_param(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else name
