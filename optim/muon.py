"""Muon optimizer stub.

This file will either vendor the Muon optimizer implementation or wrap an
upstream package once we decide on the exact dependency strategy. For now we
provide a light placeholder so that the rest of the training scaffolding can be
wired up and tested incrementally.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """Placeholder Muon optimizer.

    The real update rules will be filled in once the project vendors the Muon
    algorithm. Keeping the signature stable lets us integrate parameter routing
    and configuration plumbing before the implementation lands.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2.0e-3,
        weight_decay: float = 0.0,
        eps: float = 1.0e-8,
        momentum: float = 0.9,
        orthogonalize: bool = True,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            momentum=momentum,
            orthogonalize=orthogonalize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        raise NotImplementedError(
            "Muon.step is not implemented yet. Vendor the algorithm before running training."
        )
